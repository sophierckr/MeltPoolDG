#include <deal.II/base/exceptions.h>
#include <deal.II/base/mpi.h>

#include <deal.II/fe/fe.h>

#include <deal.II/grid/filtered_iterator.h>
#include <deal.II/grid/grid_tools.h>

#include <deal.II/lac/vector.h>

#include <meltpooldg/radiative_transport/ray_marching.hpp>
#include <meltpooldg/time_integration/time_iterator.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <utility>

namespace MeltPoolDG::RadiativeTransport
{
  using namespace dealii;

  namespace
  {
    // Illinois method for finding the root
    template <typename Function, typename number>
    std::optional<number>
    find_first_intersection(const Function &f,
                            const number    t_begin,
                            const number    t_end,
                            const number    tolerance_space,
                            const number    tolerance_value)
    {
      number t0              = t_begin;
      number f0              = f(t0);
      number t1              = t_end;
      number f1              = f(t1);
      number fz              = INFINITY;
      number tz              = 0;
      number m               = 0.5;
      size_t it              = 0;
      size_t iteration_limit = 1000;

      if (f0 * f1 > 0)
        return std::nullopt;


      while (abs(t1 - t0) >= tolerance_space and abs(fz) >= tolerance_value and
             it < iteration_limit)
        {
          it++;
          number s = (f1 - f0) / (t1 - t0);
          tz       = t0 - f0 / s;
          fz       = f(tz);
          if (fz * f1 < 0)
            {
              t0 = t1;
              f0 = f1;
              t1 = tz;
              f1 = fz;
            }
          else if (fz * f1 > 0)
            {
              f0 = m * f0;
              t1 = tz;
              f1 = fz;
            }
          else
            {
              t0 = tz;
              f0 = fz;
              t1 = tz;
              f1 = fz;
            }
        }
      if (fz < INFINITY)
        {
          return tz;
        }
      else
        {
          return std::nullopt;
        }
    }



    // Interval of t for which x + t * direction lies in the box (slab test).
    template <int dim, typename number>
    std::pair<number, number>
    ray_box_interval(const BoundingBox<dim>       &box,
                     const Point<dim>             &x,
                     const Tensor<1, dim, number> &direction)
    {
      number t_entry = -std::numeric_limits<number>::max();
      number t_exit  = std::numeric_limits<number>::max();
      for (unsigned int d = 0; d < dim; ++d)
        if (direction[d] != 0)
          {
            const number t1 = (box.lower_bound(d) - x[d]) / direction[d];
            const number t2 = (box.upper_bound(d) - x[d]) / direction[d];
            t_entry         = std::max(t_entry, std::min(t1, t2));
            t_exit          = std::min(t_exit, std::max(t1, t2));
          }
      return {t_entry, t_exit};
    }



    // Level set at p in an axis-aligned box cell.
    template <int dim, typename number>
    number
    level_set_value(const FiniteElement<dim> &fe,
                    const Vector<number>     &cell_values,
                    const BoundingBox<dim>   &box,
                    const Point<dim>         &p)
    {
      const Point<dim> p_ref = box.real_to_unit(p);
      number           value = 0;
      for (unsigned int i = 0; i < fe.n_dofs_per_cell(); ++i)
        value += cell_values[i] * fe.shape_value(i, p_ref);
      return value;
    }



    // Level-set gradient at p in an axis-aligned box cell.
    template <int dim, typename number>
    Tensor<1, dim, number>
    level_set_gradient(const FiniteElement<dim> &fe,
                       const Vector<number>     &cell_values,
                       const BoundingBox<dim>   &box,
                       const Point<dim>         &p)
    {
      const Point<dim>       p_ref = box.real_to_unit(p);
      Tensor<1, dim, number> gradient;
      for (unsigned int i = 0; i < fe.n_dofs_per_cell(); ++i)
        gradient += cell_values[i] * fe.shape_grad(i, p_ref);
      for (unsigned int d = 0; d < dim; ++d) // chain rule for the scaling to the unit cell
        gradient[d] /= box.side_length(d);
      return gradient;
    }
  } // namespace



  template <int dim, typename number>
  RayMarching<dim, number>::RayMarching(const DoFHandler<dim> &level_set_dof_handler_in,
                                        const VectorType      &level_set_in,
                                        const Parameters      &parameters_in,
                                        const MPI_Comm         mpi_communicator_in)
    : mpi_communicator(mpi_communicator_in)
    , parameters(parameters_in)
    , mapping(1)
  {
    parameters.check_input_parameters();
    parameters.absolute_epsilon =
      parameters.relative_epsilon *
      GridTools::minimal_cell_diameter(level_set_dof_handler_in.get_triangulation());

    particle_handler.initialize(level_set_dof_handler_in.get_triangulation(),
                                mapping,
                                Property::n_properties);

    particle_handler.signals.particle_lost.connect(
      [this](const auto &particle, const auto & /*cell*/) {
        lost_energy += particle->get_properties()[Property::energy];
      });

    reinit(level_set_dof_handler_in, level_set_in);
  }



  template <int dim, typename number>
  void
  RayMarching<dim, number>::reinit(const DoFHandler<dim> &level_set_dof_handler_in,
                                   const VectorType      &level_set_in)
  {
    Assert(level_set_in.has_ghost_elements(), ExcMessage("Level set needs ghost values."));

    level_set_dof_handler = &level_set_dof_handler_in;
    level_set             = &level_set_in;

    const auto &tria = level_set_dof_handler_in.get_triangulation();

    // Check that cells are axis-aligned
    std::size_t wrong_cells = 0;
    for (const auto &cell : tria.active_cell_iterators())
      if (cell->is_locally_owned())
        {
          const BoundingBox<dim> box = cell->bounding_box();
          for (const unsigned int v : cell->vertex_indices())
            if (cell->vertex(v).distance(box.vertex(v)) > 1e-10 * cell->diameter())
              wrong_cells++;
        }
    AssertThrow(Utilities::MPI::max(wrong_cells, mpi_communicator) == 0,
                ExcMessage("RayMarching requires axis-aligned box cells."));

    global_bounding_boxes = Utilities::MPI::all_gather(
      mpi_communicator,
      GridTools::compute_mesh_predicate_bounding_box(tria, IteratorFilters::LocallyOwnedCell()));

    step_length = number(0.5) * GridTools::minimal_cell_diameter(tria) / std::sqrt(number(dim));

    mesh_classifier =
      std::make_unique<NonMatching::MeshClassifier<dim>>(level_set_dof_handler_in, level_set_in);
    mesh_classifier->reclassify();
  }



  template <int dim, typename number>
  std::vector<typename RayMarching<dim, number>::Intersection>
  RayMarching<dim, number>::trace(const std::vector<Ray> &rays)
  {
    Assert(level_set_dof_handler != nullptr && level_set != nullptr, ExcNotInitialized());

    // Insert rays above the energy threshold; deal.II sends them to the owning rank.
    // Rays with energy below threshold are added to lost_energy
    {
      std::vector<Point<dim>>          positions;
      std::vector<std::vector<double>> properties;

      for (const Ray &ray : rays)
        {
          if (ray.energy < parameters.optics.energy_threshold)
            {
              lost_energy += ray.energy;
              continue;
            }

          Assert(ray.direction.norm() > 0, ExcMessage("Ray direction must be non-zero."));
          Assert(ray.id < (std::uint64_t(1) << 53), ExcMessage("Ray id not exact as double."));

          std::vector<double> p(Property::n_properties, 0.0);
          for (unsigned int d = 0; d < dim; ++d)
            p[Property::direction + d] = ray.direction[d] / ray.direction.norm();
          p[Property::energy] = ray.energy;
          p[Property::ray_id] = ray.id;

          positions.push_back(ray.origin);
          properties.push_back(std::move(p));
        }

      particle_handler.insert_global_particles(positions, global_bounding_boxes, properties);
    }

    const FiniteElement<dim> &fe = level_set_dof_handler->get_fe();
    // Much smaller than eps, so that a reflected ray starts on the correct side.
    const number root_tolerance = number(1e-4) * parameters.absolute_epsilon;

    Vector<number>            cell_values(fe.n_dofs_per_cell());
    std::vector<Intersection> intersections;

    for (size_t step = 0; step < parameters.max_steps and particle_handler.n_global_particles() > 0;
         ++step)
      {
        std::vector<typename Particles::ParticleHandler<dim>::particle_iterator> to_remove;

        for (auto particle = particle_handler.begin(); particle != particle_handler.end();
             ++particle)
          {
            ArrayView<double> props = particle->get_properties();
            const Point<dim>  x     = particle->get_location();
            const auto        cell  = particle->get_surrounding_cell();

            Tensor<1, dim, number> direction;
            for (unsigned int d = 0; d < dim; ++d)
              direction[d] = props[Property::direction + d];

            const BoundingBox<dim> box   = cell->bounding_box();
            const auto [t_entry, t_exit] = ray_box_interval(box, x, direction);

            cell->as_dof_handler_iterator(*level_set_dof_handler)
              ->get_dof_values(*level_set, cell_values);

            const auto g = [&](const number t) {
              return level_set_value(fe, cell_values, box, x + t * direction) -
                     parameters.contour_value;
            };

            const number g_start = g(0);
            const number g_end   = g(t_exit);

            if (g_start * g_end <= 0)
              {
                const number t_begin =
                  props[Property::has_moved] ? t_entry - parameters.absolute_epsilon : number(0);
                const number                t_end = t_exit;
                const std::optional<number> t_hit =
                  (t_begin < t_end) ? find_first_intersection( // TODO which values for tolerance?
                                        g,
                                        t_begin,
                                        t_end,
                                        0.001 * parameters.absolute_epsilon,
                                        root_tolerance) :
                                      std::nullopt;

                if (t_hit)
                  {
                    const Point<dim>   x_hit = x + *t_hit * direction;
                    const unsigned int reflection =
                      static_cast<unsigned int>(props[Property::reflection_count]);
                    Tensor<1, dim, number> normal = level_set_gradient(fe, cell_values, box, x_hit);
                    const auto             r      = reflect(direction,
                                           normal,
                                           number(props[Property::energy]),
                                           parameters.optics.refractive_index_incident,
                                           parameters.optics.refractive_index_material);

                    intersections.push_back(
                      Intersection{Ray{x_hit,
                                       direction,
                                       r.absorbed_energy,
                                       static_cast<std::uint64_t>(props[Property::ray_id])},
                                   reflection,
                                   cell->active_cell_index()});

                    if (reflection + 1 >= parameters.optics.max_reflections ||
                        r.reflected_energy < parameters.optics.energy_threshold)
                      {
                        lost_energy += r.reflected_energy;
                        to_remove.push_back(particle);
                      }
                    else
                      {
                        for (unsigned int d = 0; d < dim; ++d)
                          props[Property::direction + d] = r.direction[d];
                        props[Property::energy]           = r.reflected_energy;
                        props[Property::reflection_count] = reflection + 1;
                        props[Property::has_moved]        = 0;
                        normal                            = normal / (normal.norm() + 1e-12);
                        if (normal * r.direction < 0)
                          normal *= -1;
                        particle->set_location(x_hit + parameters.absolute_epsilon * normal);
                      }
                    continue;
                  }
              }
            else
              {
#ifdef DEBUG
                const auto [min_it, max_it] =
                  std::minmax_element(cell_values.begin(), cell_values.end());
                const bool cell_may_be_cut =
                  not(*min_it > parameters.contour_value or *max_it < parameters.contour_value);
                if (cell_may_be_cut)
                  {
                    const auto t_test = find_first_intersection(
                      g, number(0), t_exit, 0.1 * parameters.absolute_epsilon, root_tolerance);
                    Assert(!t_test,
                           ExcMessage("The ray crosses the interface twice inside one cell."));
                  }
#endif
              }

            particle->set_location(x + (t_exit + parameters.absolute_epsilon) * direction);
            props[Property::has_moved] = 1;
          }

        particle_handler.remove_particles(to_remove);
        particle_handler.sort_particles_into_subdomains_and_cells();
      }

    const bool unfinished = particle_handler.n_global_particles() > 0;
    particle_handler.clear();
    AssertThrow(!unfinished, ExcMessage("RayMarching: max_steps reached before all rays ended."));

    return intersections;
  }



  template <int dim, typename number>
  number
  RayMarching<dim, number>::get_lost_energy() const
  {
    return Utilities::MPI::sum(lost_energy, mpi_communicator);
  }



  template class RayMarching<2, double>;
  template class RayMarching<3, double>;
} // namespace MeltPoolDG::RadiativeTransport
