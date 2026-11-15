#pragma once

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/fe/mapping_q.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/la_parallel_vector.h>

#include <meltpooldg/radiative_transport/ray_utils.hpp>

#include <vector>

namespace MeltPoolDG::RadiativeTransport
{
  /**
   * Assemble the weak form of a point-wise energy source, i.e. for every hit
   *
   *   rhs_i += E_hit * N_i(xi_hit),
   *
   * into @p rhs. Shared by all ray tracers.
   *
   * Preconditions:
   * - Every hit lies in a cell owned by this rank. Intersection::cell_index is the
   *   active_cell_index() of that cell in the triangulation of @p dof_handler, and
   *   Intersection::reference_point its position in the reference cell.
   * - @p rhs is initialized for @p dof_handler (e.g. by ScratchData::initialize_dof_vector).
   * - @p constraints are the hanging-node constraints of @p dof_handler, without
   *   Dirichlet constraints (otherwise energy is lost).
   *
   * Collective: every rank has to call this function, also ranks without hits.
   *
   * @param hits         Hits of this rank (not modified).
   * @param dof_handler  DoFHandler of the heat field (scalar FE).
   * @param constraints  Constraints used when writing into @p rhs.
   * @param rhs          Output; contributions are added.
   * @param zero_out     If true, @p rhs is set to zero first.
   */
  template <int dim, typename number>
  void
  assemble_ray_energy_rhs(const std::vector<Intersection<dim, number>>       &hits,
                          const dealii::DoFHandler<dim>                      &dof_handler,
                          const dealii::AffineConstraints<number>            &constraints,
                          dealii::LinearAlgebra::distributed::Vector<number> &rhs,
                          const bool                                          zero_out = false,
                          const dealii::MappingQGeneric<dim>                 &mapping  = 1);
} // namespace MeltPoolDG::RadiativeTransport
