#include <deal.II/base/exceptions.h>
#include <deal.II/base/mpi.h>

#include <deal.II/grid/filtered_iterator.h>
#include <deal.II/grid/grid_tools.h>

#include <meltpooldg/radiative_transport/ray_marching.hpp>

#include <cstddef>
#include <utility>
#include <vector>

namespace MeltPoolDG::RadiativeTransport
{
  using namespace dealii;

  template <int dim, typename number>
  RayMarching<dim, number>::RayMarching(const DoFHandler<dim> &level_set_dof_handler_in,
                                        const VectorType      &level_set_in,
                                        const Parameters      &parameters_in,
                                        const MPI_Comm         mpi_communicator_in)
    : mpi_communicator(mpi_communicator_in)
    , parameters(parameters_in)
    , mapping(1) // TODO: Allow user to specify mapping degree.
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
    Assert(level_set_in.has_ghost_elements(),
           ExcMessage("The level-set vector must have ghost values."));

    level_set_dof_handler = &level_set_dof_handler_in;
    level_set             = &level_set_in;

    const auto local_boxes =
      GridTools::compute_mesh_predicate_bounding_box(level_set_dof_handler_in.get_triangulation(),
                                                     IteratorFilters::LocallyOwnedCell());

    global_bounding_boxes = Utilities::MPI::all_gather(mpi_communicator, local_boxes);
  }



  template <int dim, typename number>
  std::vector<typename RayMarching<dim, number>::Intersection>
  RayMarching<dim, number>::trace(const std::vector<Ray> &initial_rays)
  {
    Assert(level_set_dof_handler != nullptr && level_set != nullptr, ExcNotInitialized());

    // Convert rays into particle data. Rays below the energy threshold are skipped.
    std::vector<Point<dim>>            positions;
    std::vector<std::vector<double>>   properties;
    std::vector<types::particle_index> ids;

    positions.reserve(initial_rays.size());
    properties.reserve(initial_rays.size());
    ids.reserve(initial_rays.size());

    for (const Ray &ray : initial_rays)
      {
        if (ray.energy < parameters.optics.energy_threshold)
          continue;

        const number direction_norm = ray.direction.norm();
        AssertThrow(direction_norm > 0.0, ExcMessage("Ray direction must be non-zero."));
        AssertThrow(static_cast<types::particle_index>(ray.id) == ray.id,
                    ExcMessage("Ray id exceeds the range of types::particle_index."));

        std::vector<double> particle_properties(Property::n_properties);
        for (unsigned int d = 0; d < dim; ++d)
          particle_properties[Property::direction + d] = ray.direction[d] / direction_norm;
        particle_properties[Property::energy]           = ray.energy;
        particle_properties[Property::reflection_count] = 0.0;

        positions.push_back(ray.origin);
        properties.push_back(std::move(particle_properties));
        ids.push_back(static_cast<types::particle_index>(ray.id));
      }

    escaped_energy = 0;

    // Collective: every rank inserts its rays, which are sent to the rank owning their origin.
    particle_handler.insert_global_particles(positions, global_bounding_boxes, properties, ids);

    std::vector<Intersection> intersections;

    // marching loop
    //   while (particle_handler.n_global_particles() > 0 && step < parameters.max_steps)
    //     - for every particle: step, intersection test, reflect() on hit
    //     - remove_particles(collected) on all ranks
    //     - sort_particles_into_subdomains_and_cells()

    for (unsigned int step = 0;
         step < parameters.max_steps and particle_handler.n_global_particles() > 0;
         ++step)
      {
        std::vector<typename Particles::ParticleHandler<dim>::particle_iterator> to_remove;

        for (auto particle = particle_handler.begin(); particle != particle_handler.end();
             ++particle)
          {
            const ArrayView<double> props = particle->get_properties();

            Tensor<1, dim, number> direction;
            for (unsigned int d = 0; d < dim; ++d)
              direction[d] = props[Property::direction + d];

            const auto   cell        = particle->get_surrounding_cell();
            const number step_length = number(0.5) * cell->minimum_vertex_distance();

            // TODO: Treffertest; wenn der Strahl endet: to_remove.push_back(particle); continue;

            particle->set_location(particle->get_location() + step_length * direction);
          }

        particle_handler.remove_particles(to_remove);                // kollektiv
        particle_handler.sort_particles_into_subdomains_and_cells(); // kollektiv
      }
    // Ensure the handler is empty before the next mesh change.
    particle_handler.clear();

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
