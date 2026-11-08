#pragma once

#include <deal.II/base/exceptions.h>
#include <deal.II/base/function.h>
#include <deal.II/base/point.h>
#include <deal.II/base/tensor.h>

#include <deal.II/numerics/data_component_interpretation.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <string>
#include <utility>
#include <vector>

/**
 * Types and functions shared by the ray tracers in MeltPoolDG::RadiativeTransport
 * (RayTracing: marching cubes + ArborX, RayMarching: deal.II ParticleHandler).
 *
 * This header must not depend on ArborX, Kokkos or any mesh data structure, so that
 * every tracer can use it independently of its own dependencies.
 */
namespace MeltPoolDG::RadiativeTransport
{
  using namespace dealii;
  /**
   * Optical parameters common to all ray tracers.
   */
  template <typename number>
  struct OpticalParameters
  {
    /**
     * Refractive index of the medium in which the ray propagates before
     * reaching the interface.
     */
    number refractive_index_incident = 1.0;

    /**
     * Refractive index of the material represented by the level-set
     * interface. The current implementation assumes a real-valued refractive index.
     */
    number refractive_index_material = 1.5;

    /**
     * Minimum reflected ray energy retained for subsequent reflections.
     */
    number energy_threshold = 1e-4;

    /**
     * Maximum number of interface interactions considered for each ray.
     */
    unsigned int max_reflections = 20;

    /**
     * Throw if a parameter is outside its valid range.
     */
    void
    check() const
    {
      AssertThrow(refractive_index_incident > 0.0,
                  dealii::ExcMessage("Incident refractive index must be positive."));
      AssertThrow(refractive_index_material > 0.0,
                  dealii::ExcMessage("Material refractive index must be positive."));
      AssertThrow(energy_threshold >= 0.0,
                  dealii::ExcMessage("Energy threshold must be non-negative."));
    }
  };

  /**
   * Representation of a ray.
   */
  template <int dim, typename number>
  struct Ray
  {
    /**
     * Current ray origin.
     */
    dealii::Point<dim> origin;

    /**
     * Propagation direction. It does not have to be normalized when supplied
     * to a tracer's trace() function.
     */
    dealii::Tensor<1, dim, number> direction;

    /**
     * Current ray energy.
     */
    number energy = 0.0;

    /**
     * User-defined identifier preserved throughout all reflections.
     */
    std::size_t id = 0;
  };

  /**
   * Information associated with one ray-interface interaction.
   * Interface for Postprocessor::process_points().
   */
  template <int dim, typename number>
  struct Intersection
  {
    dealii::Point<dim>             point;
    dealii::Tensor<1, dim, number> ray_direction;

    /**
     * Energy absorbed at this interface interaction. If E_in denotes the
     * incoming ray energy and R the Fresnel reflectivity, this value is
     *
     * @f[
     *   E_\mathrm{abs} = (1-R) E_\mathrm{in}.
     * @f]
     */
    number energy = 0.0;

    std::size_t ray_id = 0;

    unsigned int reflection = 0; // Reflection index. The first intersection has index zero.

    unsigned int cell_index; // active_cell_index of the cell containing the hit, rank-local

    // dealii::Point<dim> reference_location; // position of the hit in the reference cell

    /**
     * Names and interpretations of the properties written by fill_output(). Components of
     * ray_direction share one name so that they are written as a vector.
     */
    static std::pair<std::vector<std::string>,
                     std::vector<dealii::DataComponentInterpretation::DataComponentInterpretation>>
    get_property_names_and_component_interpretation()
    {
      namespace DCI = dealii::DataComponentInterpretation;

      std::vector<std::string> names{"energy", "ray_id", "reflection"};
      names.insert(names.end(), dim, "ray_direction");

      std::vector<DCI::DataComponentInterpretation> interpretations(3, DCI::component_is_scalar);
      interpretations.insert(interpretations.end(), dim, DCI::component_is_part_of_vector);

      return {names, interpretations};
    }

    /**
     * Write the location and the properties of this intersection, in the order of
     * get_property_names_and_component_interpretation(), to @p location and @p properties.
     */
    void
    fill_output(dealii::Point<dim> &location, double *properties) const
    {
      location      = point;
      properties[0] = static_cast<double>(energy);
      properties[1] = static_cast<double>(ray_id);
      properties[2] = static_cast<double>(reflection);
      for (unsigned int d = 0; d < dim; ++d)
        properties[3 + d] = static_cast<double>(ray_direction[d]);
    }

    template <class Archive>
    void
    serialize(Archive &ar, const unsigned int /*version*/)
    {
      ar &point &ray_direction &energy &ray_id &reflection &cell_index;
    }
  };

  /**
   * Fresnel reflectivity for unpolarized light and real-valued refractive indices.
   *
   * @param n1 refractive index of the incident medium
   * @param n2 refractive index of the material
   * @param cos_theta_i cosine of the angle of incidence, clamped to [0, 1]
   */
  template <typename number>
  inline number
  fresnel_reflectivity(const number n1, const number n2, const number cos_theta_i)
  {
    if (n1 == n2)
      return 0.0;

    const number cos_i  = std::clamp(cos_theta_i, number(0), number(1));
    const number sin2_i = std::max(number(0), number(1) - cos_i * cos_i);

    const number eta    = n1 / n2;
    const number sin2_t = eta * eta * sin2_i;

    // total internal reflection
    if (sin2_t >= 1.0)
      return 1.0;

    const number cos_t = std::sqrt(number(1) - sin2_t);

    const number r_s = (n1 * cos_i - n2 * cos_t) / (n1 * cos_i + n2 * cos_t);
    const number r_p = (n2 * cos_i - n1 * cos_t) / (n2 * cos_i + n1 * cos_t);

    return number(0.5) * (r_s * r_s + r_p * r_p);
  }

  /**
   * Result of one specular reflection at the interface.
   */
  template <int dim, typename number>
  struct Reflection
  {
    /**
     * Normalized reflected direction.
     */
    dealii::Tensor<1, dim, number> direction;

    /**
     * Energy carried on by the reflected ray, R * E_in.
     */
    number reflected_energy;

    /**
     * Energy absorbed at the interface, (1 - R) * E_in.
     */
    number absorbed_energy;
  };

  /**
   * Specular reflection with Fresnel energy split.
   *
   * The orientation of @p normal is irrelevant: it is flipped to face the incoming ray.
   *
   * @param direction normalized incoming ray direction
   * @param normal interface normal (any length > 0, any orientation)
   * @param energy incoming ray energy
   */
  template <int dim, typename number>
  inline Reflection<dim, number>
  reflect(const dealii::Tensor<1, dim, number> &direction,
          dealii::Tensor<1, dim, number>        normal,
          const number                          energy,
          const OpticalParameters<number>      &parameters)
  {
    normal /= normal.norm();
    if (direction * normal > 0.0)
      normal *= -1.0;

    const number cos_theta_i = std::clamp(-(direction * normal), number(0), number(1));

    const number reflectivity = fresnel_reflectivity(parameters.refractive_index_incident,
                                                     parameters.refractive_index_material,
                                                     cos_theta_i);

    Reflection<dim, number> result;
    result.direction = direction - number(2) * (direction * normal) * normal;
    result.direction /= result.direction.norm();
    result.reflected_energy = reflectivity * energy;
    result.absorbed_energy  = (number(1) - reflectivity) * energy;
    return result;
  }

  template <int dim, typename number>
  std::vector<Ray<dim, number>>
  make_laser_rays(const Point<dim>             &center,
                  const Tensor<1, dim, number> &beam_direction,
                  const Function<dim, number>  &intensity_profile,
                  const number                  radius,
                  const number                  spacing,
                  const unsigned int            rank,
                  const unsigned int            n_ranks)
  {
    AssertThrow(radius > 0, ExcMessage("The radius must be positive."));
    AssertThrow(spacing > 0, ExcMessage("The ray spacing must be positive."));
    AssertThrow(n_ranks > 0 && rank < n_ranks, ExcMessage("Invalid rank."));
    AssertThrow(beam_direction.norm() > 0, ExcMessage("The beam direction must be non-zero."));

    const Tensor<1, dim, number> direction = beam_direction / beam_direction.norm();

    // Orthonormal basis
    std::array<Tensor<1, dim, number>, dim - 1> basis;
    if constexpr (dim == 2)
      {
        basis[0][0] = -direction[1];
        basis[0][1] = direction[0];
      }
    else
      {
        // Helper axis: the coordinate axis that is most perpendicular to the beam.
        unsigned int axis = 0;
        for (unsigned int d = 1; d < dim; ++d)
          if (std::abs(direction[d]) < std::abs(direction[axis]))
            axis = d;

        Tensor<1, dim, number> helper;
        helper[axis] = 1;

        basis[0] = cross_product_3d(direction, helper);
        basis[0] /= basis[0].norm();
        basis[1] = cross_product_3d(direction, basis[0]);
      }
    // Regular grid over [-R, R] per lateral direction
    const std::size_t n_per_direction =
      std::max<std::size_t>(1, static_cast<std::size_t>(std::ceil(2 * radius / spacing)));
    const number delta = 2 * radius / n_per_direction;

    std::size_t n_rays  = 1;
    number      measure = 1; // area (3D) or length (2D) represented by one ray
    for (unsigned int k = 0; k < dim - 1; ++k)
      {
        n_rays *= n_per_direction;
        measure *= delta;
      }

    std::vector<Ray<dim, number>> rays;
    rays.reserve(n_rays / n_ranks + 1);

    // Round-robin over the ranks: ray i is created on rank i % n_ranks
    for (std::size_t id = rank; id < n_rays; id += n_ranks)
      {
        Point<dim>  origin = center;
        std::size_t rest   = id;
        for (unsigned int k = 0; k < dim - 1; ++k)
          {
            const number offset = (rest % n_per_direction + number(0.5)) * delta - radius;
            origin += offset * basis[k];
            rest /= n_per_direction;
          }

        rays.push_back({origin, direction, intensity_profile.value(origin) * measure, id});
      }

    return rays;
  }

  template std::vector<Ray<2, float>>
  make_laser_rays<2, float>(const Point<2> &,
                            const Tensor<1, 2, float> &,
                            const Function<2, float> &,
                            const float,
                            const float,
                            const unsigned int,
                            const unsigned int);
  template std::vector<Ray<2, double>>
  make_laser_rays<2, double>(const Point<2> &,
                             const Tensor<1, 2, double> &,
                             const Function<2, double> &,
                             const double,
                             const double,
                             const unsigned int,
                             const unsigned int);
  template std::vector<Ray<3, float>>
  make_laser_rays<3, float>(const Point<3> &,
                            const Tensor<1, 3, float> &,
                            const Function<3, float> &,
                            const float,
                            const float,
                            const unsigned int,
                            const unsigned int);
  template std::vector<Ray<3, double>>
  make_laser_rays<3, double>(const Point<3> &,
                             const Tensor<1, 3, double> &,
                             const Function<3, double> &,
                             const double,
                             const double,
                             const unsigned int,
                             const unsigned int);


} // namespace MeltPoolDG::RadiativeTransport
