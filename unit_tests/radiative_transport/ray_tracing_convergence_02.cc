// Self-convergence study of the ray tracing heat source: the rhs for a range of marching-cube
// subdivisions and rays per cell is compared against a reference with many subdivisions and
// rays on the same background mesh. The reference is computed once per run and kept in memory,
// so it always comes from the same code as the variants.

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/function.h>
#include <deal.II/base/mpi.h>

#include <deal.II/distributed/tria.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/mapping_q_generic.h>

#include <deal.II/grid/grid_generator.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/la_parallel_vector.h>

#include <deal.II/numerics/vector_tools.h>

#include "meltpooldg/heat/laser_intensity_profiles.hpp"
#include "meltpooldg/radiative_transport/ray_rhs_assembly.hpp"
#include <meltpooldg/radiative_transport/ray_data.hpp>
#include <meltpooldg/radiative_transport/ray_tracing.hpp>
#include <meltpooldg/radiative_transport/ray_utils.hpp>

#include <Kokkos_Core.hpp>

#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include "../test_utils/ray_test_level_sets.hpp"

using namespace dealii;
using namespace MeltPoolDG;
using namespace MeltPoolDG::TestUtils;

using VectorType = LinearAlgebra::distributed::Vector<double>;

// results of the comparison with the reference, one row per variant
const std::string study_filename = "ray_tracing_convergence_02.csv";

namespace
{
  constexpr unsigned int n_mesh_refinements = 4;

  struct RayRhs
  {
    VectorType   rhs;
    double       absorbed_energy;
    unsigned int n_rays;
  };

  template <int dim>
  Tensor<1, dim>
  downward_direction()
  {
    Tensor<1, dim> direction;
    direction[dim - 1] = -1.0;
    return direction;
  }

  // Traces the laser rays once with full absorption and assembles their energy into a rhs.
  template <int dim>
  RayRhs
  compute_ray_rhs(const DoFHandler<dim>           &dof_handler,
                  const VectorType                &level_set,
                  const AffineConstraints<double> &constraints,
                  const Function<dim>             &profile,
                  const Point<dim>                &center,
                  const Tensor<1, dim>            &direction,
                  const double                     a,
                  const double                     h,
                  const unsigned int               marching_cube_subdivisions,
                  const unsigned int               rays_per_cell,
                  const MPI_Comm                   mpi_comm)
  {
    const unsigned int rank = Utilities::MPI::this_mpi_process(mpi_comm);

    typename RadiativeTransport::RayData<double> parameters;
    parameters.marching_cube_subdivisions = marching_cube_subdivisions;

    // Equal refractive indices: no reflection, every ray is fully absorbed at its first hit.
    parameters.optics.refractive_index_incident = 1.0;
    parameters.optics.refractive_index_material = 1.0;
    parameters.optics.energy_threshold          = 1e-8;
    parameters.optics.max_reflections           = 1;

    RadiativeTransport::RayTracing<dim, double> ray_tracer(dof_handler,
                                                           level_set,
                                                           parameters,
                                                           mpi_comm);

    const auto local_rays =
      RadiativeTransport::make_laser_rays<dim, double>(center,
                                                       direction,
                                                       profile,
                                                       a,
                                                       h / rays_per_cell,
                                                       rank,
                                                       Utilities::MPI::n_mpi_processes(mpi_comm));

    unsigned int local_n_rays = 0; // rays below the energy threshold are dropped by the tracers
    for (const auto &ray : local_rays)
      if (ray.energy >= parameters.optics.energy_threshold)
        ++local_n_rays;
    const unsigned int n_rays = Utilities::MPI::sum(local_n_rays, mpi_comm);

    const auto intersections = ray_tracer.trace(local_rays);

    check_intersections_in_cells(dof_handler.get_triangulation(), intersections);

    // The interface closes the domain below every ray, so no ray may be lost.
    const unsigned int n_intersections =
      Utilities::MPI::sum(static_cast<unsigned int>(intersections.size()), mpi_comm);
    AssertThrow(n_intersections == n_rays,
                ExcMessage("Expected one intersection per ray, got " +
                           std::to_string(n_intersections) + " intersections for " +
                           std::to_string(n_rays) + " rays."));

    double local_absorbed_energy = 0.0;
    for (const auto &intersection : intersections)
      local_absorbed_energy += intersection.absorbed_ray.energy;
    const double absorbed_energy = Utilities::MPI::sum(local_absorbed_energy, mpi_comm);

    AssertThrow(absorbed_energy > 0.0, ExcMessage("No energy was absorbed."));

    VectorType rhs;
    rhs.reinit(dof_handler.locally_owned_dofs(),
               DoFTools::extract_locally_relevant_dofs(dof_handler),
               mpi_comm);
    RadiativeTransport::assemble_ray_energy_rhs(intersections, dof_handler, constraints, rhs, true);

    AssertThrow(std::abs(rhs.l1_norm() - absorbed_energy) < 1e-12 * absorbed_energy,
                ExcMessage("Sum of rhs " + std::to_string(rhs.l1_norm()) +
                           " does not match the energy " + std::to_string(absorbed_energy) +
                           " of the hits."));

    return {rhs, absorbed_energy, n_rays};
  }

  template <int dim>
  void
  run_study(const unsigned int               reference_subdivisions,
            const unsigned int               reference_rays_per_cell,
            const std::vector<unsigned int> &subdivisions,
            const std::vector<unsigned int> &rays_per_cell)
  {
    const MPI_Comm mpi_comm = MPI_COMM_WORLD;
    const double   a        = 100e-6; // fits with the measurements of the fixed meltpool
    const double   h        = 2.0 * a / (1u << n_mesh_refinements);
    const double   eps      = 2.0 * h;

    const unsigned int rank = Utilities::MPI::this_mpi_process(mpi_comm);

    dealii::ConditionalOStream pcout(std::cout, rank == 0);

    parallel::distributed::Triangulation<dim> triangulation(mpi_comm);
    GridGenerator::hyper_cube(triangulation, -a, a);
    triangulation.refine_global(n_mesh_refinements);

    MappingQGeneric<dim> mapping(1);
    FE_Q<dim>            fe(1);

    DoFHandler<dim> dof_handler(triangulation);
    dof_handler.distribute_dofs(fe);

    VectorType level_set;
    level_set.reinit(dof_handler.locally_owned_dofs(),
                     DoFTools::extract_locally_relevant_dofs(dof_handler),
                     mpi_comm);

    FixedMeltPoolLevelSet<dim> level_set_function(LevelSet::LevelSetType::signed_distance, eps);
    VectorTools::interpolate(mapping, dof_handler, level_set_function, level_set);
    level_set.update_ghost_values();

    AffineConstraints<double> constraints;
    constraints.reinit(dof_handler.locally_owned_dofs(),
                       DoFTools::extract_locally_relevant_dofs(dof_handler));
    constraints.close();

    const double beam_power  = 87.5;
    const double beam_radius = 70e-6;

    Point<dim> center;
    center[dim - 1] = 50e-6;

    const auto direction = downward_direction<dim>();

    const Heat::GaussProjectionIntensityProfile<dim, double> profile(beam_power,
                                                                     beam_radius,
                                                                     center,
                                                                     direction);

    const auto reference = compute_ray_rhs(dof_handler,
                                           level_set,
                                           constraints,
                                           profile,
                                           center,
                                           direction,
                                           a,
                                           h,
                                           reference_subdivisions,
                                           reference_rays_per_cell,
                                           mpi_comm);

    pcout << dim << "D reference: " << reference_subdivisions << " MC subdivisions, "
          << reference_rays_per_cell << " rays per cell, " << reference.n_rays << " rays"
          << std::endl;

    // deviation of the coarsest and the finest variant, for the sanity check below
    double coarsest_rhs_deviation = 0.0;
    double finest_rhs_deviation   = 0.0;

    for (const unsigned int n_subdivisions : subdivisions)
      for (const unsigned int n_rays_per_cell : rays_per_cell)
        {
          const auto variant = compute_ray_rhs(dof_handler,
                                               level_set,
                                               constraints,
                                               profile,
                                               center,
                                               direction,
                                               a,
                                               h,
                                               n_subdivisions,
                                               n_rays_per_cell,
                                               mpi_comm);

          // Deviations relative to the absorbed energy of the reference.
          const double energy_deviation =
            std::abs(variant.absorbed_energy - reference.absorbed_energy) /
            reference.absorbed_energy;

          VectorType difference(variant.rhs);
          difference.zero_out_ghost_values();
          difference.add(-1.0, reference.rhs);
          const double rhs_deviation = difference.l1_norm() / reference.absorbed_energy;

          if (n_subdivisions == subdivisions.front() && n_rays_per_cell == rays_per_cell.front())
            coarsest_rhs_deviation = rhs_deviation;
          if (n_subdivisions == subdivisions.back() && n_rays_per_cell == rays_per_cell.back())
            finest_rhs_deviation = rhs_deviation;

          pcout << dim << "D, " << n_subdivisions << " MC subdivisions, " << n_rays_per_cell
                << " rays per cell: energy deviation " << std::scientific << std::setprecision(1)
                << energy_deviation << ", rhs deviation " << rhs_deviation << std::defaultfloat
                << std::endl;

          if (rank == 0)
            {
              std::ofstream file(study_filename, std::ios::app);
              file << std::scientific << std::setprecision(8) << dim << "," << n_subdivisions << ","
                   << h << "," << n_rays_per_cell << "," << variant.n_rays << ","
                   << variant.absorbed_energy << "," << energy_deviation << "," << rhs_deviation
                   << "\n";
            }
        }

    // The deviations are not strictly monotone (the ray positions of different spacings are not
    // nested), so only check that refining both parameters brings the rhs closer to the reference.
    AssertThrow(finest_rhs_deviation < coarsest_rhs_deviation,
                ExcMessage("The finest variant is not closer to the reference than the coarsest."));

    // TODO: tolerances, e.g. on the energy deviation, once the study shows the convergence rates
  }
} // namespace

int
main(int argc, char *argv[])
{
  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);

  const bool initialize_kokkos = !Kokkos::is_initialized();

  if (initialize_kokkos)
    Kokkos::initialize(argc, argv);

  if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
    std::ofstream(study_filename) << "dim,mc_subdivisions,cell_size,rays_per_cell,n_rays,"
                                     "absorbed_energy,energy_deviation,rhs_deviation\n";

  // Each doubling of the subdivisions halves the edge length of the marching-cubes sub-cells.
  // The reference is several levels finer than every variant in both parameters.
  run_study<2>(1024,
               2048,
               {1, 2, 4, 8, 16, 32, 64, 128, 256, 512},
               {1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024});
  // run_study<3>(8, 16, {1, 2, 4}, {1, 2, 4});

  if (initialize_kokkos)
    Kokkos::finalize();

  return 0;
}
