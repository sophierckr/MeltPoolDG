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
// #include <cstdint>
#include <limits>
#include <optional>
#include <utility>

namespace MeltPoolDG::RadiativeTransport
{
  using namespace dealii;

  namespace
  {
    // Bisection for a sign change of f in [a, b], with f_a = f(a).
    // TODO change to more efficient approach?
    template <typename Function, typename number>
    number
    bisect(const Function &f, number a, number b, number f_a, const number tolerance)
    {
      while (b - a > tolerance)
        {
          const number m   = number(0.5) * (a + b);
          const number f_m = f(m);
          if (f_a * f_m <= 0)
            b = m;
          else
            {
              a   = m;
              f_a = f_m;
            }
        }
      return number(0.5) * (a + b);
    }



    // First root of f in [t_begin, t_end]: uniform sampling for a sign change, then bisection.
    template <typename Function, typename number>
    std::optional<number>
    first_root(const Function    &f,
               const number       t_begin,
               const number       t_end,
               const unsigned int n_samples,
               const number       tolerance)
    {
      number t0 = t_begin;
      number f0 = f(t0);
      for (unsigned int s = 1; s <= n_samples; ++s)
        {
          const number t1 = t_begin + (t_end - t_begin) * s / n_samples;
          const number f1 = f(t1);
          if (f0 * f1 <= 0)
            return bisect(f, t0, t1, f0, tolerance);
          t0 = t1;
          f0 = f1;
        }
      return std::nullopt;
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
    // TODO change to more efficient approach
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
    parameters.optics.check();

    particle_handler.initialize(level_set_dof_handler_in.get_triangulation(),
                                mapping,
                                Property::n_properties);

    particle_handler.signals.particle_lost.connect(
      [this](const auto &particle, const auto & /*cell*/) {
        escaped_energy += particle->get_properties()[Property::energy];
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

#ifdef DEBUG
    for (const auto &cell : tria.active_cell_iterators())
      if (cell->is_locally_owned())
        {
          const BoundingBox<dim> box = cell->bounding_box();
          for (const unsigned int v : cell->vertex_indices())
            Assert(cell->vertex(v).distance(box.vertex(v)) < 1e-10 * cell->diameter(),
                   ExcMessage("RayMarching requires axis-aligned box cells."));
        }
#endif

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
    {
      std::vector<Point<dim>>          positions;
      std::vector<std::vector<double>> properties;

      for (const Ray &ray : rays)
        {
          if (ray.energy < parameters.optics.energy_threshold)
            continue;

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

      escaped_energy = 0;
      particle_handler.insert_global_particles(positions,
                                               global_bounding_boxes,
                                               properties); // TODO alle nur einmal drinnen?
    }

    const FiniteElement<dim> &fe  = level_set_dof_handler->get_fe();
    const number              eps = parameters.relative_epsilon * step_length;
    // Much smaller than eps, so that a reflected ray starts on the correct side.
    const number root_tolerance = number(1e-4) * eps;

    Vector<number>            cell_values(fe.n_dofs_per_cell());
    std::vector<Intersection> intersections;

    // Pseudo-time = marched distance: one time step per marching step of length step_length.
    TimeIntegration::TimeSteppingData<number> time_data;
    time_data.start_time     = 0;
    time_data.time_step_size = step_length;
    time_data.end_time       = parameters.max_steps * step_length;
    time_data.max_n_steps    = parameters.max_steps;
    TimeIntegration::TimeIterator<number> time_iterator(time_data);

    while (!time_iterator.is_finished() && particle_handler.n_global_particles() > 0)
      {
        time_iterator.compute_next_time_increment();
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

            if (mesh_classifier->location_to_level_set(cell) ==
                NonMatching::LocationToLevelSet::intersected)
              {
                cell->as_dof_handler_iterator(*level_set_dof_handler)
                  ->get_dof_values(*level_set, cell_values);
                const BoundingBox<dim> box = cell->bounding_box();

                // Search from the previous position (or eps after the start or a hit) to the
                // next position, clipped to this cell. Together with the neighbors this covers
                // the whole path, except cells clipped between two positions.
                const auto [t_entry, t_exit] = ray_box_interval(box, x, direction);
                const number t_begin =
                  std::max(t_entry, eps - number(props[Property::has_moved]) * step_length);
                const number t_end = std::min(t_exit, step_length);

                const auto phi = [&](const number t) {
                  return level_set_value(fe, cell_values, box, x + t * direction);
                };

                const std::optional<number> t_hit =
                  (t_begin < t_end) ?
                    first_root(phi, t_begin, t_end, parameters.n_root_samples, root_tolerance) :
                    std::nullopt;

                if (t_hit)
                  {
                    const Point<dim>   x_hit = x + *t_hit * direction;
                    const unsigned int reflection =
                      static_cast<unsigned int>(props[Property::reflection_count]);
                    const auto r = reflect(direction,
                                           level_set_gradient(fe, cell_values, box, x_hit),
                                           number(props[Property::energy]),
                                           parameters.optics);

                    intersections.push_back(
                      Intersection{x_hit,
                                   direction,
                                   r.absorbed_energy,
                                   static_cast<std::uint64_t>(props[Property::ray_id]),
                                   reflection});

                    if (reflection + 1 >= parameters.optics.max_reflections ||
                        r.reflected_energy < parameters.optics.energy_threshold)
                      to_remove.push_back(particle);
                    else
                      {
                        for (unsigned int d = 0; d < dim; ++d)
                          props[Property::direction + d] = r.direction[d];
                        props[Property::energy]           = r.reflected_energy;
                        props[Property::reflection_count] = reflection + 1;
                        props[Property::has_moved]        = 0;
                        particle->set_location(x_hit);
                      }
                    continue;
                  }
              }

            particle->set_location(x + step_length * direction);
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
  RayMarching<dim, number>::get_escaped_energy() const
  {
    return Utilities::MPI::sum(escaped_energy, mpi_communicator);
  }



  template class RayMarching<2, double>;
  template class RayMarching<3, double>;
} // namespace MeltPoolDG::RadiativeTransport
