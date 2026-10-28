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

#include <meltpooldg/post_processing/output_data.hpp>
#include <meltpooldg/post_processing/postprocessor.hpp>
#include <meltpooldg/radiative_transport/ray_marching.hpp>
#include <meltpooldg/radiative_transport/ray_utils.hpp>
#include <meltpooldg/time_integration/time_stepping_data.hpp>

#include <Kokkos_Core.hpp>

#include <cmath>
#include <iostream>
#include <string>
#include <vector>

using namespace dealii;
using namespace MeltPoolDG;

using VectorType = LinearAlgebra::distributed::Vector<double>;

namespace
{
  template <int dim>
  class TwoParallelSurfacesLevelSet : public Function<dim>
  {
  public:
    TwoParallelSurfacesLevelSet()
      : Function<dim>(1)
    {}

    double
    value(const Point<dim> &point, const unsigned int component = 0) const override
    {
      AssertIndexRange(component, 1);

      const double coordinate = point[dim - 1];

      if (coordinate <= 0.5)
        return coordinate - 0.3;

      return 0.7 - coordinate;
    }
  };

  template <int dim>
  Tensor<1, dim>
  downward_direction()
  {
    Tensor<1, dim> direction;
    direction[dim - 1] = -1.0;
    return direction;
  }

  template <int dim>
  Point<dim>
  first_ray_origin()
  {
    Point<dim> point;

    point[0] = 0.47;

    if constexpr (dim == 3)
      point[1] = 0.43;

    point[dim - 1] = 0.5;

    return point;
  }

  template <int dim>
  Point<dim>
  second_ray_origin()
  {
    Point<dim> point;

    point[0] = 0.53;

    if constexpr (dim == 3)
      point[1] = 0.57;

    point[dim - 1] = 0.5;

    return point;
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
  run_test()
  {
    const MPI_Comm mpi_comm = MPI_COMM_WORLD;

    const unsigned int rank = Utilities::MPI::this_mpi_process(mpi_comm);

    dealii::ConditionalOStream pcout(std::cout, rank == 0);

    const std::string basename = "ray_marching_" + Utilities::int_to_string(dim) + "d";

    parallel::distributed::Triangulation<dim> triangulation(mpi_comm);

    GridGenerator::hyper_cube(triangulation, 0.0, 1.0);

    triangulation.refine_global(3);

    MappingQGeneric<dim> mapping(1);
    FE_Q<dim>            fe(1);

    DoFHandler<dim> dof_handler(triangulation);

    dof_handler.distribute_dofs(fe);

    VectorType level_set;

    level_set.reinit(dof_handler.locally_owned_dofs(),
                     DoFTools::extract_locally_relevant_dofs(dof_handler),
                     mpi_comm);

    TwoParallelSurfacesLevelSet<dim> level_set_function;

    VectorTools::interpolate(mapping, dof_handler, level_set_function, level_set);

    level_set.update_ghost_values();

    write_background(dof_handler, level_set, mapping, mpi_comm, basename + "_background");

    using RayTracer = RadiativeTransport::RayMarching<dim, double>;

    typename RayTracer::Parameters parameters;

    parameters.optics.refractive_index_incident = 1.0;

    parameters.optics.refractive_index_material = 1.5;

    /*
     * For normal incidence with n1=1 and n2=1.5:
     *
     * R = 0.04.
     *
     * Remaining ray energy:
     *
     * 1.0
     * 0.04
     * 0.0016
     * 0.000064
     *
     * The fourth value is below the threshold.
     */
    parameters.optics.energy_threshold = 1e-4;

    parameters.optics.max_reflections = 10;

    RayTracer ray_tracer(dof_handler, level_set, parameters, mpi_comm);

    std::vector<typename RayTracer::Ray> local_rays;

    if (rank == 0)
      {
        local_rays.emplace_back(
          typename RayTracer::Ray{first_ray_origin<dim>(), downward_direction<dim>(), 1.0, 0});

        /*
         * This ray is below the energy threshold and must therefore be
         * removed before marching.
         */
        local_rays.emplace_back(
          typename RayTracer::Ray{second_ray_origin<dim>(), downward_direction<dim>(), 0.5e-4, 1});
      }

    const auto intersections = ray_tracer.trace(local_rays);

    OutputData<double> output_data;
    output_data.point.enable   = true;
    output_data.point.filename = basename + "_intersections";
    const TimeIntegration::TimeSteppingData<double> time_data;
    Postprocessor<dim, double>                      postprocessor(
      mpi_comm, output_data, time_data, mapping, triangulation, pcout);
    postprocessor.process_points(0, 0.0, intersections);


    const unsigned int n_intersections =
      Utilities::MPI::sum(static_cast<unsigned int>(intersections.size()), mpi_comm);

    AssertThrow(n_intersections == 3, ExcMessage("Expected exactly three intersections."));

    if (rank == 0)
      {
        AssertDimension(intersections.size(), 3);

        constexpr double position_tolerance = 1e-8;

        AssertThrow(std::abs(intersections[0].point[dim - 1] - 0.3) < position_tolerance,
                    ExcMessage("Wrong first intersection position."));

        AssertThrow(std::abs(intersections[1].point[dim - 1] - 0.7) < position_tolerance,
                    ExcMessage("Wrong second intersection position."));

        AssertThrow(std::abs(intersections[2].point[dim - 1] - 0.3) < position_tolerance,
                    ExcMessage("Wrong third intersection position."));

        constexpr double energy_tolerance = 1e-10;

        AssertThrow(std::abs(intersections[0].energy - 0.96) < energy_tolerance,
                    ExcMessage("Wrong absorbed energy at first intersection."));

        AssertThrow(std::abs(intersections[1].energy - 0.0384) < energy_tolerance,
                    ExcMessage("Wrong absorbed energy at second intersection."));

        AssertThrow(std::abs(intersections[2].energy - 0.001536) < energy_tolerance,
                    ExcMessage("Wrong absorbed energy at third intersection."));

        for (unsigned int i = 0; i < intersections.size(); ++i)
          {
            AssertThrow(intersections[i].ray_id == 0,
                        ExcMessage("Ray below the energy threshold generated an intersection."));

            AssertThrow(intersections[i].reflection == i,
                        ExcMessage("Incorrect reflection index."));

            const double direction_norm = intersections[i].ray_direction.norm();

            AssertThrow(std::abs(direction_norm - 1.0) < 1e-12,
                        ExcMessage("Stored ray direction is not normalized."));
          }

        AssertThrow(intersections[0].ray_direction[dim - 1] < 0.0,
                    ExcMessage("First ray direction should point downward."));

        AssertThrow(intersections[1].ray_direction[dim - 1] > 0.0,
                    ExcMessage("Second ray direction should point upward."));

        AssertThrow(intersections[2].ray_direction[dim - 1] < 0.0,
                    ExcMessage("Third ray direction should point downward."));
      }

    pcout << dim << "D intersections: " << n_intersections << std::endl;

    pcout << dim << "D intersection points: OK" << std::endl;

    pcout << dim << "D Fresnel energies: OK" << std::endl;

    pcout << dim << "D ray directions: OK" << std::endl;

    pcout << dim << "D energy threshold: OK" << std::endl;
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
