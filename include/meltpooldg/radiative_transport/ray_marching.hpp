#pragma once

#include <deal.II/base/bounding_box.h>
#include <deal.II/base/mpi.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/fe/mapping_q.h>

#include <deal.II/lac/la_parallel_vector.h>

#include <deal.II/non_matching/mesh_classifier.h>

#include <deal.II/particles/particle_handler.h>

#include <meltpooldg/radiative_transport/ray_utils.hpp>

#include <memory>
#include <vector>

namespace MeltPoolDG::RadiativeTransport
{
  /**
   * Ray marcher for an interface given by the level set of a finite element field.
   *
   * Rays are deal.II particles that advance to the next cell each timestep and are moved
   * by a ParticleHandler. In cells cut by the interface, the level set is sampled along the
   * ray for a sign change, which is then refined by bisection.
   *
   * Assumes axis-aligned box cells (and refinement),
   * so that the map to the reference cell is a scaling.
   */
  template <int dim, typename number>
  class RayMarching
  {
  public:
    using VectorType   = dealii::LinearAlgebra::distributed::Vector<number>;
    using Ray          = RadiativeTransport::Ray<dim, number>;
    using Intersection = RadiativeTransport::Intersection<dim, number>;

    struct Parameters
    {
      OpticalParameters<number> optics;

      /// Safety limit on steps
      unsigned int max_steps = 10000;

      /// Offset into the next cell, relative to the step length.
      number relative_epsilon = 1e-6;

      /// Samples per search window for locating the first root.
      unsigned int n_root_samples = 6;

      /// isovalue at which the interface lies
      number contour_value = 0.0;
    };

    /// Collective. Triangulation, DoFHandler and level set must outlive this object.
    RayMarching(const dealii::DoFHandler<dim> &level_set_dof_handler,
                const VectorType              &level_set,
                const Parameters              &parameters       = Parameters{},
                const MPI_Comm                 mpi_communicator = MPI_COMM_WORLD);

    // The ParticleHandler signal captures this.
    RayMarching(const RayMarching &) = delete;
    RayMarching &
    operator=(const RayMarching &) = delete;

    /// Collective. Call after mesh or level-set changes; inputs are stored by reference.
    void
    reinit(const dealii::DoFHandler<dim> &level_set_dof_handler, const VectorType &level_set);

    /// Collective. Every ray must be passed on exactly one rank and start inside the domain
    /// (rays outside are dropped by the ParticleHandler). Returns the hits found on this rank.
    /// Energy of dropped rays is added onto lost_energy.
    std::vector<Intersection>
    trace(const std::vector<Ray> &rays);

    /// Collective. Energy of rays that left the domain during the last trace().
    number
    get_lost_energy() const;

  private:
    enum Property : unsigned int
    {
      direction = 0, // dim entries
      energy    = direction + dim,
      reflection_count,
      has_moved, // 0 at the start and after a hit, 1 after a step
      ray_id,
      n_properties
    };

    MPI_Comm   mpi_communicator;
    Parameters parameters;

    // Not owned, set by reinit().
    const dealii::DoFHandler<dim> *level_set_dof_handler = nullptr;
    const VectorType              *level_set             = nullptr;

    // Used only by the ParticleHandler, so it must be declared before it.
    dealii::MappingQ<dim> mapping;

    dealii::Particles::ParticleHandler<dim>                   particle_handler;
    std::vector<std::vector<dealii::BoundingBox<dim>>>        global_bounding_boxes;
    std::unique_ptr<dealii::NonMatching::MeshClassifier<dim>> mesh_classifier;

    number step_length = 0;
    number lost_energy = 0;
  };

} // namespace MeltPoolDG::RadiativeTransport
