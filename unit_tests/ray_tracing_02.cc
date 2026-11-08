#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/function.h>
#include <deal.II/base/mpi.h>

#include <deal.II/distributed/tria.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/mapping_q_generic.h>

#include <deal.II/grid/grid_generator.h>

#include <deal.II/lac/la_parallel_vector.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>

#include "meltpooldg/heat/laser_intensity_profiles.hpp"
#include "meltpooldg/radiative_transport/ray_rhs_assembly.hpp"
#include <meltpooldg/post_processing/output_data.hpp>
#include <meltpooldg/post_processing/postprocessor.hpp>
#include <meltpooldg/radiative_transport/ray_tracing.hpp> // changed: was ray_marching.hpp
#include <meltpooldg/radiative_transport/ray_utils.hpp>
#include <meltpooldg/time_integration/time_stepping_data.hpp>

#include <Kokkos_Core.hpp>

#include <array>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "test_utils/ray_test_level_sets.hpp"

using namespace dealii;
using namespace MeltPoolDG;

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


using VectorType = LinearAlgebra::distributed::Vector<double>;
using namespace MeltPoolDG::TestUtils;

namespace
{

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
    // Energie pro Zelle (Debug: Sind es die erwarteten Zellen?)
    Vector<float> cell_energy(dof_handler.get_triangulation().n_active_cells());
    for (const auto &hit : intersections)
      cell_energy[hit.cell_index] += hit.energy;

    rhs.update_ghost_values(); // DataOut braucht Ghost-Werte

    DataOut<dim> data_out;
    data_out.add_data_vector(dof_handler, level_set, "level_set");
    data_out.add_data_vector(dof_handler, rhs, "heat_rhs");
    data_out.add_data_vector(cell_energy, "cell_energy", DataOut<dim>::type_cell_data);
    data_out.build_patches(mapping);
    data_out.write_vtu_in_parallel(basename + ".vtu", mpi_comm);
  }


  template <int dim>
  void
  run_test()
  {
    const MPI_Comm     mpi_comm      = MPI_COMM_WORLD;
    const double       a             = 100e-6; // fits with the measurements of the fixed meltpool
    const unsigned int n_refinements = 5;
    const double       h             = 2.0 * a / (1u << n_refinements);
    const double       eps           = 2.0 * h;


    const unsigned int rank = Utilities::MPI::this_mpi_process(mpi_comm);

    dealii::ConditionalOStream pcout(std::cout, rank == 0);

    const std::string basename =
      "ray_tracing_fixed_meltpool_" + Utilities::int_to_string(dim) + "d";

    parallel::distributed::Triangulation<dim> triangulation(mpi_comm);

    GridGenerator::hyper_cube(triangulation, -a, a);

    triangulation.refine_global(n_refinements);

    MappingQGeneric<dim> mapping(1);
    FE_Q<dim>            fe(1);

    DoFHandler<dim> dof_handler(triangulation);
    // TODO Test with local refinement

    dof_handler.distribute_dofs(fe);

    VectorType level_set;

    level_set.reinit(dof_handler.locally_owned_dofs(),
                     DoFTools::extract_locally_relevant_dofs(dof_handler),
                     mpi_comm);

    FixedMeltPoolLevelSet<dim> level_set_function(LevelSet::LevelSetType::signed_distance, eps);
    VectorTools::interpolate(mapping, dof_handler, level_set_function, level_set);
    level_set.update_ghost_values();



    write_background(dof_handler, level_set, mapping, mpi_comm, basename + "_background");

    using RayTracer = RadiativeTransport::RayTracing<dim, double>; // changed

    typename RayTracer::Parameters parameters;

    parameters.optics.refractive_index_incident = 1.0;

    parameters.optics.refractive_index_material = 1.5;

    // changed: RayTracing has no contour_value parameter and always reconstructs the zero level
    // set. Make sure the test level set is actually zero at the interface.
    AssertThrow(interface_contour_value<double>(LevelSet::LevelSetType::signed_distance) == 0.0,
                ExcMessage("RayTracing reconstructs the zero contour only."));

    parameters.optics.energy_threshold = 1e-4;

    // changed: max_reflections is the maximum number of interface interactions per ray in
    // RayTracing, so 1 gives the single hit per ray that max_reflections = 0 gives in RayMarching.
    parameters.optics.max_reflections = 1;

    RayTracer ray_tracer(dof_handler, level_set, parameters, mpi_comm); // changed

    ray_tracer.write_surface(basename + "_surface"); // changed: debug output of the reconstruction

    const double beam_power  = 87.5;
    const double beam_radius = 70e-6;

    Point<dim> center;
    center[dim - 1] = 50e-6;

    const auto direction = downward_direction<dim>();

    const Heat::GaussProjectionIntensityProfile<dim, double> profile(beam_power,
                                                                     beam_radius,
                                                                     center,
                                                                     direction);

    const auto local_rays = RadiativeTransport::make_laser_rays<dim, double>(
      center, direction, profile, a, h, rank, Utilities::MPI::n_mpi_processes(mpi_comm));

    double local_ray_energy = 0.0;
    for (const auto &ray : local_rays)
      local_ray_energy += ray.energy;
    const double energy_in = Utilities::MPI::sum(local_ray_energy, mpi_comm);

    const auto intersections = ray_tracer.trace(local_rays); // changed

    OutputData<double> output_data;
    output_data.point.enable   = true;
    output_data.point.filename = basename + "_intersections";
    const TimeIntegration::TimeSteppingData<double> time_data;
    Postprocessor<dim, double>                      postprocessor(
      mpi_comm, output_data, time_data, mapping, triangulation, pcout);
    postprocessor.process_points(0, 0.0, intersections);

    double intersection_energy = 0.0;

    for (const auto &intersection : intersections)
      intersection_energy += intersection.energy;



    AffineConstraints<double> constraints;
    constraints.reinit(dof_handler.locally_owned_dofs(),
                       DoFTools::extract_locally_relevant_dofs(dof_handler));
    constraints.close();
    VectorType rhs;
    rhs.reinit(dof_handler.locally_owned_dofs(),
               DoFTools::extract_locally_relevant_dofs(dof_handler),
               mpi_comm);

    RadiativeTransport::assemble_ray_energy_rhs(intersections, dof_handler, constraints, rhs, true);



    const auto total_intersection_energy =
      Utilities::MPI::sum(static_cast<double>(intersection_energy), mpi_comm);

    const double erf_a = std::erf(std::numbers::sqrt2 * a / beam_radius);
    double       energy_analytic;
    if constexpr (dim == 2)
      energy_analytic = beam_power * std::sqrt(2.0 / dealii::numbers::PI) / beam_radius * erf_a;
    else
      energy_analytic = beam_power * erf_a * erf_a;
    check_close(energy_in,
                energy_analytic,
                5e-3 * energy_analytic,
                "Wrong energy of the generated rays");

    // changed: RayTracing does not report lost energy (missed rays, reflected remainder), so
    // total == energy_in cannot be checked. Absorbed energy must at least be positive and not
    // exceed the incoming energy.
    AssertThrow(total_intersection_energy > 0.0, ExcMessage("No energy was absorbed."));
    AssertThrow(total_intersection_energy <= energy_in * (1.0 + 1e-10),
                ExcMessage("More energy absorbed than emitted."));

    AssertThrow(std::abs(rhs.l1_norm() - total_intersection_energy) <
                  1e-12 * total_intersection_energy,
                ExcMessage("Sum of rhs " + std::to_string(rhs.l1_norm()) +
                           " does not match the energy " +
                           std::to_string(total_intersection_energy) + " of the hits."));
    write_rhs(dof_handler, level_set, rhs, intersections, mapping, mpi_comm, basename + "_rhs");
  }
} // namespace

int
main(int argc, char *argv[])
{
  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);

  const bool initialize_kokkos = !Kokkos::is_initialized();

  if (initialize_kokkos)
    Kokkos::initialize(argc, argv);

  {
    run_test<2>();
    run_test<3>();
  }

  if (initialize_kokkos)
    Kokkos::finalize();

  return 0;
}
