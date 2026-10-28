#include <meltpooldg/radiative_transport/ray_source.hpp>

#ifdef DEAL_II_WITH_ARBORX

#  include <deal.II/base/exceptions.h>
#  include <deal.II/base/utilities.h>

#  include <array>
#  include <cmath>
#  include <cstdint>

namespace MeltPoolDG::RadiativeTransport
{
  using namespace dealii;

  namespace
  {
    /**
     * Orthonormal basis of the plane perpendicular to @p direction.
     *
     * The first axis is the projection of e_x onto the plane (or of e_y if
     * the beam runs along x). In 3D, the second axis is e_1 x direction.
     * For a beam along -e_{dim-1}, this yields exactly the global x (and y)
     * axes, also for beams tilted in the x-z plane.
     */
    template <int dim, typename number>
    std::array<Tensor<1, dim, number>, dim - 1>
    beam_plane_basis(const Tensor<1, dim, number> &direction)
    {
      std::array<Tensor<1, dim, number>, dim - 1> basis;

      Tensor<1, dim, number> e_x;
      e_x[0] = 1.0;

      basis[0] = e_x - (e_x * direction) * direction;

      if (basis[0].norm() < 1e-6)
        {
          Tensor<1, dim, number> e_y;
          e_y[1] = 1.0;

          basis[0] = e_y - (e_y * direction) * direction;
        }

      basis[0] /= basis[0].norm();

      if constexpr (dim == 3)
        basis[1] = cross_product_3d(basis[0], direction);

      return basis;
    }
  } // namespace



  template <int beam_dim>
  GaussianBeamProfile<beam_dim>::GaussianBeamProfile(const double beam_radius)
    : Function<beam_dim>(1)
    , beam_radius(beam_radius)
  {
    AssertThrow(beam_radius > 0.0, ExcMessage("The beam radius must be positive."));
  }



  template <int beam_dim>
  double
  GaussianBeamProfile<beam_dim>::value(const Point<beam_dim> &point,
                                       const unsigned int     component) const
  {
    AssertIndexRange(component, 1);
    (void)component;

    return std::exp(-2.0 * point.norm_square() / (beam_radius * beam_radius));
  }



  template <int dim, typename number>
  std::vector<typename RayTracing<dim, number>::Ray>
  generate_laser_rays(const LaserRayParameters<dim, number> &parameters,
                      const Function<dim - 1>               &intensity_profile,
                      const MPI_Comm                         mpi_communicator)
  {
    AssertThrow(dim == 2 or dim == 3, ExcMessage("Only implemented for dim = 2 and dim = 3."));

    using Ray = typename RayTracing<dim, number>::Ray;

    AssertThrow(parameters.power >= 0.0, ExcMessage("The laser power must be non-negative."));
    AssertThrow(parameters.cutoff_radius > 0.0, ExcMessage("The cutoff radius must be positive."));
    AssertThrow(parameters.ray_spacing > 0.0, ExcMessage("The ray spacing must be positive."));
    AssertThrow(parameters.direction.norm() > 0.0,
                ExcMessage("The beam direction must not be zero."));

    const Tensor<1, dim, number> direction = parameters.direction / parameters.direction.norm();

    const auto basis = beam_plane_basis<dim, number>(direction);

    const unsigned int rank    = Utilities::MPI::this_mpi_process(mpi_communicator);
    const unsigned int n_ranks = Utilities::MPI::n_mpi_processes(mpi_communicator);

    // Grid points per side of the beam center and in total.
    const std::uint64_t n_half =
      static_cast<std::uint64_t>(std::floor(parameters.cutoff_radius / parameters.ray_spacing));
    const std::uint64_t n_per_direction = 2 * n_half + 1;

    std::uint64_t n_grid_points = 1;
    for (unsigned int d = 0; d < dim - 1; ++d)
      n_grid_points *= n_per_direction;

    std::vector<Ray> rays;

    double total_weight = 0.0;

    for (std::uint64_t k = 0; k < n_grid_points; ++k)
      {
        // Local position of grid point k in the start plane.
        Point<dim - 1> local_point;

        std::uint64_t remainder = k;
        for (unsigned int d = 0; d < dim - 1; ++d)
          {
            const std::int64_t index = static_cast<std::int64_t>(remainder % n_per_direction) -
                                       static_cast<std::int64_t>(n_half);
            remainder /= n_per_direction;

            local_point[d] = index * parameters.ray_spacing;
          }

        if (local_point.norm() > parameters.cutoff_radius * (1.0 + 1e-12))
          continue;

        const double weight = intensity_profile.value(local_point);

        AssertThrow(weight >= 0.0, ExcMessage("The intensity profile must be non-negative."));

        if (weight == 0.0)
          continue;

        // Every rank sums over all grid points in the same order, so the
        // normalization is identical on all ranks without communication.
        total_weight += weight;

        const bool is_my_ray =
          parameters.distribute_over_ranks ? (k % n_ranks == rank) : (rank == 0);

        if (!is_my_ray)
          continue;

        Ray ray;

        ray.origin = parameters.beam_center;
        for (unsigned int d = 0; d < dim - 1; ++d)
          ray.origin += local_point[d] * basis[d];

        ray.direction = direction;
        ray.energy    = weight; // normalized below
        ray.id        = k;

        rays.push_back(ray);
      }

    AssertThrow(total_weight > 0.0,
                ExcMessage("The intensity profile vanishes inside the cutoff radius."));

    for (auto &ray : rays)
      ray.energy *= parameters.power / total_weight;

    return rays;
  }



  template class GaussianBeamProfile<1>;
  template class GaussianBeamProfile<2>;

  template std::vector<RayTracing<2, double>::Ray>
  generate_laser_rays<2, double>(const LaserRayParameters<2, double> &,
                                 const Function<1> &,
                                 const MPI_Comm);

  template std::vector<RayTracing<3, double>::Ray>
  generate_laser_rays<3, double>(const LaserRayParameters<3, double> &,
                                 const Function<2> &,
                                 const MPI_Comm);
} // namespace MeltPoolDG::RadiativeTransport

#endif
