#pragma once

#include <deal.II/base/exceptions.h>
#include <deal.II/base/function.h>
#include <deal.II/base/point.h>

#include <meltpooldg/level_set/level_set_type.hpp>
#include <meltpooldg/utilities/characteristic_functions.hpp>

#include <algorithm>
#include <cmath>

namespace MeltPoolDG::TestUtils
{
  /**
   * Value of the level set at the interface for the given representation, i.e. the
   * contour value required by the ray tracers.
   */
  template <typename number = double>
  inline number
  interface_contour_value(const LevelSet::LevelSetType type)
  {
    switch (type)
      {
        case LevelSet::LevelSetType::signed_distance:
        case LevelSet::LevelSetType::tanh:
          return 0;
        case LevelSet::LevelSetType::smoothed_heaviside:
          return 0.5;
        default:
          AssertThrow(false, dealii::ExcNotImplemented());
      }
    return 0;
  }



  /**
   * Signed distance to two parallel planes at z = 0.3 and z = 0.7 (negative below 0.3, positive in
   * between).
   */
  template <int dim, typename number = double>
  class TwoParallelSurfacesLevelSet : public dealii::Function<dim, number>
  {
  public:
    TwoParallelSurfacesLevelSet()
      : dealii::Function<dim, number>(1)
    {}

    number
    value(const dealii::Point<dim> &point, const unsigned int component = 0) const override
    {
      AssertIndexRange(component, 1);

      const number z = point[dim - 1];
      return z <= 0.5 ? z - 0.3 : 0.7 - z;
    }
  };



  /**
   * Copy of Simulation::FixedMeltPool::FixedMeltPoolGeometry (applications/mp-heat-transfer/cases/
   * fixed_melt_pool_geometry), kept here because unit tests only link against the library.
   *
   * Fixed melt pool benchmark from Much et al., Adv. Model. and Simul. in Eng. Sci. 11(1), 16
   * (2024). The signed distance (eq. 37) is positive in the metal and negative in the gas, i.e. the
   * opposite of TwoParallelSurfacesLevelSet; the ray tracers do not depend on this orientation.
   *
   * 3D extension (not part of the original, which is 2D only)
   *
   * @param type representation of the level set
   * @param eps  interface thickness parameter, only for tanh and smoothed_heaviside
   */
  template <int dim, typename number = double>
  class FixedMeltPoolLevelSet : public dealii::Function<dim, number>
  {
    static_assert(dim >= 2, "The fixed melt pool geometry requires at least two dimensions.");

  public:
    FixedMeltPoolLevelSet(const LevelSet::LevelSetType type, const number eps = 0)
      : dealii::Function<dim, number>(1)
      , type(type)
      , eps(eps)
    {}

    number
    value(const dealii::Point<dim> &p, const unsigned int /*component*/ = 0) const override
    {
      const number d = signed_distance(p);

      switch (type)
        {
          case LevelSet::LevelSetType::tanh:
            return CharacteristicFunctions::tanh_characteristic_function(d, eps);
          case LevelSet::LevelSetType::smoothed_heaviside:
            return CharacteristicFunctions::smoothed_heaviside(d, eps);
          case LevelSet::LevelSetType::signed_distance:
            return d;
          default:
            AssertThrow(false, dealii::ExcNotImplemented());
        }
      return 0;
    }

  private:
    static constexpr number y_level       = 10e-6; // top level of the interface
    static constexpr number centre_radius = 50e-6; // pool radius, centre at the origin
    static constexpr number bead_radius   = 10e-6;
    static constexpr number bead_centre_x = centre_radius + bead_radius;

    number
    signed_distance(const dealii::Point<dim> &p) const
    {
      // Profile coordinates: height h and distance r from the vertical axis (r = |x| in 2D).
      const number h  = p[dim - 1];
      number       r2 = 0;
      for (unsigned int d = 0; d < dim - 1; ++d)
        r2 += p[d] * p[d];
      const number r = std::sqrt(r2);

      const number d_level     = -h + y_level;
      const bool   inside_pool = r <= bead_centre_x;

      if (h >= 0) // above the centre line
        return inside_pool ? bead_radius - std::hypot(r - bead_centre_x, h) : d_level;

      const number d_pool = std::hypot(r, h) - centre_radius; // below the centre line
      return inside_pool ? d_pool : std::min(d_pool, d_level);
    }

    const LevelSet::LevelSetType type;
    const number                 eps;
  };
} // namespace MeltPoolDG::TestUtils
