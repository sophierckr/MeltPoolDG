#pragma once

#include <deal.II/base/mpi.h>
#include <deal.II/base/point.h>
#include <deal.II/base/tensor.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/fe/mapping_q_generic.h>

#include <deal.II/grid/cell_data.h>
#include <deal.II/grid/tria.h>

#include <deal.II/lac/la_parallel_vector.h>

#include <deal.II/numerics/data_component_interpretation.h>

#include <deal.II/particles/particle_handler.h>

#include <meltpooldg/radiative_transport/ray_utils.hpp>

#ifdef DEAL_II_WITH_ARBORX
#  include <ArborX.hpp>
#  include <ArborX_Ray.hpp>
#  include <ArborX_Segment.hpp>
#  include <ArborX_Triangle.hpp>
#  include <Kokkos_Core.hpp>
#endif

#include <limits>
#include <memory>
#include <string>
#include <type_traits>
#include <vector>

namespace MeltPoolDG::RadiativeTransport
{
#ifdef DEAL_II_WITH_ARBORX
  /**
   * Distributed ray tracer for interfaces represented by a level-set field.
   *
   * The zero level set is reconstructed using deal.II's marching-square
   * algorithm in two dimensions and marching-cube algorithm in three
   * dimensions. Only interface primitives generated from locally owned
   * background cells are stored on each MPI process. No complete interface
   * mesh is gathered across MPI processes.
   *
   * In two dimensions, the interface is represented by line segments. In
   * three dimensions, it is represented by triangles.
   *
   * ArborX::DistributedTree is used to perform the distributed geometric
   * search. In two dimensions, a finite ArborX segment is chosen sufficiently
   * long to contain the part of the ray intersecting the global background-mesh
   * bounding box. In three dimensions, a native infinite ArborX ray is used.
   *
   * At every interface intersection, the ray is specularly reflected and its
   * energy is reduced according to Fresnel's law for unpolarized light and
   * real-valued refractive indices. The absorbed fraction of the incoming
   * energy is returned together with the intersection position and incoming
   * ray direction.
   *
   * A ray is removed if
   *
   * - it does not intersect the interface,
   * - its reflected energy falls below Parameters::energy_threshold, or
   * - Parameters::max_reflections is reached.
   *
   * @tparam dim Spatial dimension. Supported values are 2 and 3.
   * @tparam number Floating-point number type.
   */
  template <int dim, typename number>
  class RayTracing
  {
  public:
    using VectorType   = dealii::LinearAlgebra::distributed::Vector<number>;
    using Ray          = RadiativeTransport::Ray<dim, number>;
    using Intersection = RadiativeTransport::Intersection<dim, number>;

    struct Parameters
    {
      OpticalParameters<number> optics;
      unsigned int              marching_cube_subdivisions = 2;
      number                    marching_cube_tolerance    = 1e-10;
      number                    ray_epsilon                = 1e-10;
    };

    /**
     * Construct the ray tracer and reconstruct the current level-set
     * interface.
     *
     * Construction is collective and must be performed by every MPI rank in
     * @p mpi_communicator.
     *
     * @param level_set_dof_handler DoFHandler associated with the level-set
     * field.
     * @param level_set Distributed level-set vector.
     * @param parameters Ray-tracing parameters.
     * @param mpi_communicator MPI communicator used by the distributed ArborX
     * tree.
     */
    RayTracing(const dealii::DoFHandler<dim> &level_set_dof_handler,
               const VectorType              &level_set,
               const Parameters              &parameters       = Parameters{},
               const MPI_Comm                 mpi_communicator = MPI_COMM_WORLD);

    /**
     * Reconstruct the local interface and rebuild the distributed ArborX tree.
     *
     * This function should be called after the level-set geometry changes.
     *
     * @param level_set_dof_handler DoFHandler associated with the current
     * level-set field.
     * @param level_set Distributed level-set vector.
     */
    void
    reinit(const dealii::DoFHandler<dim> &level_set_dof_handler, const VectorType &level_set);

    /**
     * Trace locally owned rays through the distributed interface.
     *
     * Rays remain associated with the MPI rank on which they are supplied.
     * ArborX forwards geometric queries to ranks whose local interface
     * primitives may be intersected by a ray.
     *
     * The ArborX distributed query is collective. Consequently, this function
     * must be called by all MPI ranks in the communicator, including ranks
     * that own no rays.
     *
     * Rays whose initial energy is below Parameters::energy_threshold are
     * removed before the first intersection query.
     *
     * @param initial_rays Rays owned by the current MPI process.
     *
     * @return Intersection history for the locally owned rays. The energy
     * stored in every entry is the energy absorbed at that intersection.
     */
    std::vector<Intersection>
    trace(const std::vector<Ray> &initial_rays) const;

    /**
     * Write the locally reconstructed interface to one VTU file per MPI rank
     * and create a PVTU master file.
     *
     * In two dimensions the interface files contain line elements. In three
     * dimensions they contain triangular elements.
     *
     * No interface geometry is communicated between ranks for output.
     *
     * Example:
     *
     * @code
     * ray_tracing_surface.pvtu
     * ray_tracing_surface_rank_0000.vtu
     * ray_tracing_surface_rank_0001.vtu
     * @endcode
     *
     * Ranks with an empty local interface do not generate a VTU piece and are
     * not listed in the PVTU file.
     *
     * @param basename Base filename without extension.
     */
    void
    write_surface(const std::string &basename) const;

    /**
     * Return the number of interface primitives stored on this MPI rank.
     *
     * The primitives are line segments for dim==2 and triangles for dim==3.
     */
    unsigned int
    n_local_interface_cells() const;

  private:
    using ExecutionSpace = Kokkos::DefaultHostExecutionSpace;
    using MemorySpace    = typename ExecutionSpace::memory_space;

    /**
     * A dimension-independent finite segment is used as the ArborX query
     * geometry.
     */
    using ArborXQuery = std::conditional_t<dim == 2,
                                           ArborX::Experimental::Segment<2, number>,
                                           ArborX::Experimental::Ray<number>>;
    /**
     * Interface primitive used by the local ArborX tree.
     *
     * dim==2: line segment
     * dim==3: triangle
     */
    using ArborXPrimitive = std::conditional_t<dim == 2,
                                               ArborX::Experimental::Segment<2, number>,
                                               ArborX::Triangle<3, number>>;

    using Predicate = decltype(ArborX::intersects(ArborXQuery{}));
    using Tree      = ArborX::DistributedTree<MemorySpace, ArborXPrimitive>;

    /**
     * Minimal information returned from a rank owning an intersected
     * interface primitive.
     */
    struct HitCandidate
    {
      number distance;
      number normal_x;
      number normal_y;
      number normal_z;
    };

    /**
     * Closest positive intersection for one ray.
     */
    struct FirstHit
    {
      bool                           hit      = false;
      number                         distance = std::numeric_limits<number>::infinity();
      dealii::Point<dim>             point;
      dealii::Tensor<1, dim, number> normal;
    };

    /**
     * Callback evaluated on the MPI rank owning the candidate primitive.
     *
     * In two dimensions an exact segment-segment intersection is computed.
     * In three dimensions the finite query segment is converted into an
     * ArborX ray and an exact ray-triangle intersection is evaluated.
     *
     * Only the intersection distance and interface normal are returned to the
     * MPI rank owning the ray.
     */
    struct IntersectionCallback
    {
      number minimum_distance;

      template <typename PredicateType, typename PrimitiveType, typename OutputFunctor>
      KOKKOS_FUNCTION void
      operator()(const PredicateType &predicate,
                 const PrimitiveType &primitive,
                 const OutputFunctor &out) const;
    };

    /**
     * Determine the closest positive interface intersection for each locally
     * owned ray.
     */
    std::vector<FirstHit>
    compute_first_hits(const std::vector<Ray> &rays) const;


    MPI_Comm mpi_communicator;

    Parameters parameters;

    ExecutionSpace execution_space;

    /**
     * Linear mapping used for marching-square/cube reconstruction.
     */
    dealii::MappingQGeneric<dim> mapping;

    /**
     * Global axis-aligned bounding box of the background mesh.
     */
    dealii::Point<dim> domain_min;
    dealii::Point<dim> domain_max;

    /**
     * Rank-local marching-square/cube vertices.
     */
    std::vector<dealii::Point<dim>> local_surface_vertices;

    /**
     * Rank-local marching-square/cube cells.
     */
    std::vector<dealii::CellData<dim - 1>> local_surface_cells;

    /**
     * Rank-local ArborX interface primitives.
     */
    Kokkos::View<ArborXPrimitive *, MemorySpace> local_primitives;

    /**
     * Distributed ArborX tree.
     */
    std::unique_ptr<Tree> tree;
  };
#endif
} // namespace MeltPoolDG::RadiativeTransport
