#pragma once

#include <deal.II/base/function.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/point.h>
#include <deal.II/base/tensor.h>

#include <meltpooldg/radiative_transport/ray_tracing.hpp>

#include <vector>

#ifdef DEAL_II_WITH_ARBORX

namespace MeltPoolDG::RadiativeTransport
{
  /**
   * Parameters of a collimated laser beam that is discretized into rays.
   *
   * The rays start on the plane through @p beam_center perpendicular to
   * @p direction. They are placed on a uniform Cartesian grid with spacing
   * @p ray_spacing in this plane, and only grid points within
   * @p cutoff_radius around the beam center are kept.
   *
   * @tparam dim Spatial dimension (2 or 3).
   * @tparam number Floating-point number type.
   */
  template <int dim, typename number = double>
  struct LaserRayParameters
  {
    LaserRayParameters()
    {
      direction[dim - 1] = -1.0;
    }

    /**
     * Total laser power. The energies of all generated rays (over all MPI
     * ranks) sum up exactly to this value.
     */
    number power = 1.0;

    /**
     * Center of the beam in the start plane of the rays. Must lie in the
     * gas phase, i.e., in front of the material surface.
     */
    dealii::Point<dim> beam_center;

    /**
     * Propagation direction. Normalized internally. Default: -e_{dim-1}.
     */
    dealii::Tensor<1, dim, number> direction;

    /**
     * Radius around the beam center outside of which no rays are generated.
     */
    number cutoff_radius = 0.0;

    /**
     * Distance between neighboring rays in the start plane. Should be chosen
     * relative to the mesh size, e.g., at most half a cell in the focal spot,
     * so that the deposited energy is resolved on the mesh.
     */
    number ray_spacing = 0.0;

    /**
     * If true, the rays are distributed round-robin over all ranks of the
     * communicator according to their id. If false, rank 0 owns all rays.
     */
    bool distribute_over_ranks = true;
  };

  /**
   * Gaussian intensity profile I(r) = exp(-2 r^2 / w^2), where w is the
   * 1/e^2 beam radius. The profile does not need to be normalized, since
   * generate_laser_rays() rescales all ray energies to the laser power.
   *
   * @tparam beam_dim Dimension of the beam cross section, i.e., dim - 1.
   */
  template <int beam_dim>
  class GaussianBeamProfile : public dealii::Function<beam_dim>
  {
  public:
    explicit GaussianBeamProfile(const double beam_radius);

    double
    value(const dealii::Point<beam_dim> &point, const unsigned int component = 0) const override;

  private:
    const double beam_radius;
  };

  /**
   * Discretize a collimated laser beam into rays for RayTracing::trace().
   *
   * The intensity profile is evaluated in local coordinates of the start
   * plane relative to the beam center. For a beam along -e_{dim-1}, the
   * local axes coincide with the global x (and y) axes. The profile only
   * needs to be non-negative; its scaling is irrelevant.
   *
   * Each ray carries the energy (power * I_i / sum_j I_j). Since all rays
   * belong to a uniform grid, the area element is identical for every ray
   * and drops out of the normalization.
   *
   * The id of a ray is its index on the full ray grid. It is therefore
   * unique and independent of the number of MPI ranks, which keeps results
   * reproducible for different rank counts. Ids are not contiguous, since
   * grid points outside the cutoff radius or with zero intensity are skipped.
   *
   * No MPI communication is performed: every rank loops over all grid points
   * and computes the same normalization, but only stores its own rays.
   */
  template <int dim, typename number>
  std::vector<typename RayTracing<dim, number>::Ray>
  generate_laser_rays(const LaserRayParameters<dim, number> &parameters,
                      const dealii::Function<dim - 1>       &intensity_profile,
                      const MPI_Comm                         mpi_communicator);
} // namespace MeltPoolDG::RadiativeTransport

#endif
