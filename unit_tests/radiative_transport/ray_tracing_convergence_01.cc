// Convergence study of the ray tracing heat source against the interface projection: the rhs for a
// range of marching-cube subdivisions and rays per cell is compared against
// LaserModelType::interface_projection_sharp on the same background mesh. The background mesh and
// the projection are set up once per dimension, the ray tracer (interface reconstruction) once per
// number of subdivisions.

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

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>

#include "meltpooldg/heat/laser_intensity_profiles.hpp"
#include "meltpooldg/radiative_transport/ray_rhs_assembly.hpp"
#include <meltpooldg/core/scratch_data.hpp>
#include <meltpooldg/heat/laser_data.hpp>
#include <meltpooldg/heat/laser_heat_source_projection_based.hpp>
#include <meltpooldg/post_processing/output_data.hpp>
#include <meltpooldg/post_processing/postprocessor.hpp>
#include <meltpooldg/radiative_transport/ray_data.hpp>
#include <meltpooldg/radiative_transport/ray_tracing.hpp>
#include <meltpooldg/radiative_transport/ray_utils.hpp>
#include <meltpooldg/time_integration/time_stepping_data.hpp>

#include <Kokkos_Core.hpp>

#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <numbers>
#include <sstream>
#include <string>
#include <vector>

#include "../test_utils/ray_test_level_sets.hpp"

using namespace dealii;
using namespace MeltPoolDG;
using namespace MeltPoolDG::TestUtils;

using VectorType = LinearAlgebra::distributed::Vector<double>;

// results of the comparison with the interface projection, one row per variant
const std::string study_filename = "ray_tracing_convergence_01.csv";

namespace
{
  constexpr unsigned int n_mesh_refinements = 4;

  void
  check_close(const double       actual,
              const double       expected,
              const double       tolerance,
              const std::string &what)
  {
    std::ostringstream message;
    message << std::scientific << std::setprecision(6) << what << ": actual = " << actual
            << ", expected = " << expected << ", |diff| = " << std::abs(actual - expected)
            << " (tolerance " << tolerance << ")";

    AssertThrow(std::abs(actual - expected) < tolerance, ExcMessage(message.str()));
  }

  template <int dim>
  Tensor<1, dim>
  downward_direction()
  {
    Tensor<1, dim> direction;
    direction[dim - 1] = -1.0;
    return direction;
  }

  template <int dim>
  void
  write_background(const DoFHandler<dim> &dof_handler,
                   const VectorType      &level_set,
                   const Mapping<dim>    &mapping,
                   const MPI_Comm         mpi_comm,
                   const std::string     &basename)
  {
    DataOut<dim> data_out;

    data_out.add_data_vector(dof_handler, level_set, "level_set");

    data_out.build_patches(mapping);

    data_out.write_vtu_in_parallel(basename + ".vtu", mpi_comm);
  }

  template <int dim>
  void
  write_rhs(const DoFHandler<dim>                                            &dof_handler,
            const VectorType                                                 &level_set,
            VectorType                                                       &rhs,
            const std::vector<RadiativeTransport::Intersection<dim, double>> &intersections,
            const Mapping<dim>                                               &mapping,
            const MPI_Comm                                                    mpi_comm,
            const std::string                                                &basename)
  {
    // Energy per cell (debug output: are these the expected cells?)
    Vector<float> cell_energy(dof_handler.get_triangulation().n_active_cells());
    for (const auto &hit : intersections)
      cell_energy[hit.cell_index] += hit.absorbed_ray.energy;

    rhs.update_ghost_values(); // DataOut needs ghost values

    DataOut<dim> data_out;
    data_out.add_data_vector(dof_handler, level_set, "level_set");
    data_out.add_data_vector(dof_handler, rhs, "heat_rhs");
    data_out.add_data_vector(cell_energy, "cell_energy", DataOut<dim>::type_cell_data);
    data_out.build_patches(mapping);
    data_out.write_vtu_in_parallel(basename + ".vtu", mpi_comm);
  }

  template <int dim>
  void
  run_study(const std::vector<unsigned int> &subdivisions,
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

    // write_background(dof_handler, level_set, mapping, mpi_comm,
    //                  "ray_tracing_fixed_meltpool_" + Utilities::int_to_string(dim) +
    //                    "d_background");

    AffineConstraints<double> constraints;
    constraints.reinit(dof_handler.locally_owned_dofs(),
                       DoFTools::extract_locally_relevant_dofs(dof_handler));
    constraints.close();

    const double beam_power  = 87.5;
    const double beam_radius = 70e-6;

    Point<dim> center;
    center[dim - 1] = 50e-6;

    const auto direction = downward_direction<dim>();

    const auto profile = std::make_shared<const Heat::GaussProjectionIntensityProfile<dim, double>>(
      beam_power, beam_radius, center, direction);

    const double erf_a = std::erf(std::numbers::sqrt2 * a / beam_radius);
    double       energy_analytic;
    if constexpr (dim == 2)
      energy_analytic = beam_power * std::sqrt(2.0 / dealii::numbers::PI) / beam_radius * erf_a;
    else
      energy_analytic = beam_power * erf_a * erf_a;

    VectorType indicator(level_set);
    indicator.zero_out_ghost_values();
    indicator.add(0.5);

    Heat::LaserData<double> laser_data; // absorptivity one, beam in negative dim-1 direction
    laser_data.post(dim);

    ScratchData<dim, dim, double> scratch_data(mpi_comm, 0, false /*do_matrix_free*/);
    scratch_data.set_mapping(mapping);
    const unsigned int dof_idx =
      scratch_data.attach_dof_handler_and_constraint(dof_handler, constraints);
    scratch_data.create_partitioning();

    const Heat::LaserHeatSourceProjectionBased<dim, double> projection(
      laser_data, profile, false, laser_data.delta_approximation_phase_weighted);
    VectorType rhs_projection;
    projection.compute_interfacial_heat_source_sharp(
      rhs_projection, scratch_data, dof_idx, indicator, dof_idx);

    const double projection_energy = rhs_projection.l1_norm();

    for (const unsigned int n_subdivisions : subdivisions)
      {
        typename RadiativeTransport::RayData<double> parameters;
        parameters.marching_cube_subdivisions = n_subdivisions;

        parameters.optics.refractive_index_incident = 1.0;
        parameters.optics.refractive_index_material = 1.0;
        parameters.optics.max_reflections           = 1;

        const RadiativeTransport::RayTracing<dim, double> ray_tracer(dof_handler,
                                                                     level_set,
                                                                     parameters,
                                                                     mpi_comm);

        // RayTracing does not report lost energy (reflected remainder) yet, so total == energy_in
        // cannot be checked. Every ray interacts once and the reflectivity is smallest at normal
        // incidence, which bounds the absorbed energy.
        const double max_absorptivity =
          1.0 -
          RadiativeTransport::fresnel_reflectivity(parameters.optics.refractive_index_incident,
                                                   parameters.optics.refractive_index_material,
                                                   1.0);

        for (const unsigned int n_rays_per_cell : rays_per_cell)
          {
            const std::string basename = "ray_tracing_fixed_meltpool_" +
                                         Utilities::int_to_string(dim) + "d_mc" +
                                         Utilities::int_to_string(n_subdivisions) + "_rays" +
                                         Utilities::int_to_string(n_rays_per_cell);

            const auto local_rays =
              RadiativeTransport::make_laser_rays<dim, double>(center,
                                                               direction,
                                                               *profile,
                                                               a,
                                                               h / n_rays_per_cell,
                                                               rank,
                                                               Utilities::MPI::n_mpi_processes(
                                                                 mpi_comm));

            double       local_ray_energy = 0.0;
            unsigned int local_n_rays = 0; // rays below the energy threshold are dropped by tracers
            for (const auto &ray : local_rays)
              {
                local_ray_energy += ray.energy;
                if (ray.energy >= parameters.optics.energy_threshold)
                  ++local_n_rays;
              }
            const double       energy_in = Utilities::MPI::sum(local_ray_energy, mpi_comm);
            const unsigned int n_rays    = Utilities::MPI::sum(local_n_rays, mpi_comm);

            check_close(energy_in,
                        energy_analytic,
                        5e-3 * energy_analytic,
                        "Wrong energy of the generated rays");

            const auto intersections = ray_tracer.trace(local_rays);

            check_intersections_in_cells(triangulation, intersections);

            // The interface closes the domain below every ray and each ray interacts only once, so
            // every ray above the energy threshold must produce exactly one intersection.
            const unsigned int n_intersections =
              Utilities::MPI::sum(static_cast<unsigned int>(intersections.size()), mpi_comm);
            AssertThrow(n_intersections == n_rays,
                        ExcMessage("Expected one intersection per ray, got " +
                                   std::to_string(n_intersections) + " intersections for " +
                                   std::to_string(n_rays) + " rays."));

            OutputData<double> output_data;
            output_data.point.enable   = true;
            output_data.point.filename = basename + "_intersections";
            const TimeIntegration::TimeSteppingData<double> time_data;
            Postprocessor<dim, double>                      postprocessor(
              mpi_comm, output_data, time_data, mapping, triangulation, pcout);
            postprocessor.process_points(0, 0.0, intersections);

            double local_absorbed_energy = 0.0;
            for (const auto &intersection : intersections)
              local_absorbed_energy += intersection.absorbed_ray.energy;
            const double absorbed_energy = Utilities::MPI::sum(local_absorbed_energy, mpi_comm);

            AssertThrow(absorbed_energy > 0.0, ExcMessage("No energy was absorbed."));
            AssertThrow(absorbed_energy <= max_absorptivity * energy_in * (1.0 + 1e-10),
                        ExcMessage("More energy absorbed than physically possible."));

            VectorType rhs;
            rhs.reinit(dof_handler.locally_owned_dofs(),
                       DoFTools::extract_locally_relevant_dofs(dof_handler),
                       mpi_comm);
            RadiativeTransport::assemble_ray_energy_rhs(
              intersections, dof_handler, constraints, rhs, true);

            AssertThrow(std::abs(rhs.l1_norm() - absorbed_energy) < 1e-12 * absorbed_energy,
                        ExcMessage("Sum of rhs " + std::to_string(rhs.l1_norm()) +
                                   " does not match the energy " + std::to_string(absorbed_energy) +
                                   " of the hits."));

            // Deviations relative to the absorbed ray energy. The rays are point sources, so the
            // nodal deviation decreases with the number of rays per cell.
            const double energy_deviation =
              std::abs(projection_energy - absorbed_energy) / absorbed_energy;

            // copy, the reference is reused for the next variant
            VectorType difference(rhs_projection);
            difference.zero_out_ghost_values();
            difference.add(-1.0, rhs);
            const double rhs_deviation = difference.l1_norm() / absorbed_energy;

            pcout << dim << "D, " << n_subdivisions << " MC subdivisions, " << n_rays_per_cell
                  << " rays per cell: energy deviation " << std::scientific << std::setprecision(1)
                  << energy_deviation << ", rhs deviation " << rhs_deviation << std::defaultfloat
                  << std::endl;

            if (rank == 0)
              {
                std::ofstream file(study_filename, std::ios::app);
                file << std::scientific << std::setprecision(8) << dim << "," << n_subdivisions
                     << "," << h << "," << n_rays_per_cell << "," << n_rays << ","
                     << absorbed_energy << "," << energy_deviation << "," << rhs_deviation << "\n";
              }

            // TODO: tolerances are estimates
            /*check_close(energy_deviation, 0.0, 1e-2, "Total energy of the interface projection");
            check_close(rhs_deviation,
                        0.0,
                        0.5 / n_rays_per_cell,
                        "L1 difference between the ray rhs and the interface projection");*/

            // write_rhs(dof_handler, level_set, rhs, intersections, mapping, mpi_comm,
            //            basename + "_rhs");
          }
      }
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

  // the projection uses Marching Cubes with three subdivisions
  run_study<2>({3}, {64});
  // run_study<2>({0,3,10,50,100,200}, {32,64,128,256,512});
  //  run_study<3>({1, 2, 3, 4, 8}, {1, 2, 4, 8, 16});

  if (initialize_kokkos)
    Kokkos::finalize();

  return 0;
}
