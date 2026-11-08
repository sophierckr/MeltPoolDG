#include <deal.II/base/exceptions.h>

#include <deal.II/dofs/dof_accessor.h>

#include <deal.II/fe/fe.h>
#include <deal.II/fe/mapping_q.h>

#include <deal.II/lac/vector.h>

#include <meltpooldg/radiative_transport/ray_rhs_assembly.hpp>

#include <algorithm>
#include <numeric>
#include <vector>

namespace MeltPoolDG::RadiativeTransport
{
  using namespace dealii;

  template <int dim, typename number>
  void
  assemble_ray_energy_rhs(const std::vector<Intersection<dim, number>> &hits,
                          const DoFHandler<dim>                        &dof_handler,
                          const AffineConstraints<number>              &constraints,
                          LinearAlgebra::distributed::Vector<number>   &rhs,
                          const bool                                    zero_out,
                          const dealii::MappingQGeneric<dim>           &mapping)
  {
    const auto &fe = dof_handler.get_fe();

    AssertDimension(rhs.size(), dof_handler.n_dofs());

    if (zero_out)
      rhs = number(0);

    // Hit indices sorted by cell, so that the hits stay untouched.
    std::vector<unsigned int> order(hits.size());
    std::iota(order.begin(), order.end(), 0u);
    std::sort(order.begin(), order.end(), [&](const unsigned int a, const unsigned int b) {
      return hits[a].cell_index < hits[b].cell_index;
    });

    const unsigned int dofs_per_cell = fe.n_dofs_per_cell();

    Vector<number>                       cell_rhs(dofs_per_cell);
    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

    // Walk through the cells in ascending active_cell_index() and consume the sorted hits.
    std::size_t k = 0;
    for (const auto &cell : dof_handler.active_cell_iterators())
      {
        if (k == order.size())
          break;

        const unsigned int cell_index = cell->active_cell_index();
        if (hits[order[k]].cell_index != cell_index)
          continue;

        Assert(cell->is_locally_owned(), ExcMessage("Hit lies in a cell not owned by this rank."));

        cell_rhs = number(0);
        for (; k < order.size() && hits[order[k]].cell_index == cell_index; ++k)
          {
            const auto      &hit = hits[order[k]];
            const Point<dim> xi  = mapping.transform_real_to_unit_cell(cell, hit.point);

            for (unsigned int i = 0; i < dofs_per_cell; ++i)
              cell_rhs[i] += hit.energy * static_cast<number>(fe.shape_value(i, xi));
          }

        cell->get_dof_indices(local_dof_indices);
        constraints.distribute_local_to_global(cell_rhs, local_dof_indices, rhs);
      }

    // Energy conservation: every hit has to be assigned to a cell.
    AssertThrow(k == order.size(),
                ExcMessage("Hits could not be assigned to cells (invalid cell_index)."));

    rhs.compress(VectorOperation::add);
  }

  template void
  assemble_ray_energy_rhs<2, float>(const std::vector<Intersection<2, float>> &,
                                    const DoFHandler<2> &,
                                    const AffineConstraints<float> &,
                                    LinearAlgebra::distributed::Vector<float> &,
                                    const bool,
                                    const dealii::MappingQGeneric<2> &);
  template void
  assemble_ray_energy_rhs<2, double>(const std::vector<Intersection<2, double>> &,
                                     const DoFHandler<2> &,
                                     const AffineConstraints<double> &,
                                     LinearAlgebra::distributed::Vector<double> &,
                                     const bool,
                                     const dealii::MappingQGeneric<2> &);
  template void
  assemble_ray_energy_rhs<3, float>(const std::vector<Intersection<3, float>> &,
                                    const DoFHandler<3> &,
                                    const AffineConstraints<float> &,
                                    LinearAlgebra::distributed::Vector<float> &,
                                    const bool,
                                    const dealii::MappingQGeneric<3> &);
  template void
  assemble_ray_energy_rhs<3, double>(const std::vector<Intersection<3, double>> &,
                                     const DoFHandler<3> &,
                                     const AffineConstraints<double> &,
                                     LinearAlgebra::distributed::Vector<double> &,
                                     const bool,
                                     const dealii::MappingQGeneric<3> &);
} // namespace MeltPoolDG::RadiativeTransport
