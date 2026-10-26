#pragma once

#include <deal.II/base/exceptions.h>
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
    /**
     * Physical intersection position.
     */
    dealii::Point<dim> point;

    /**
     * Normalized incoming ray direction at the intersection.
     */
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

    /**
     * Identifier of the original ray.
     */
    std::size_t ray_id = 0;

    /**
     * Reflection index. The first intersection has index zero.
     */
    unsigned int reflection = 0;

    /**
     * Names and interpretations of the properties written by fill_output(). Components of
     * ray_direction share one name so that they are written as a vector.
     */
    static std::pair<std::vector<std::string>,
                     std::vector<dealii::DataComponentInterpretation::DataComponentInterpretation>>
    get_property_names_and_component_interpretation(){
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
    fill_output(dealii::Point<dim> &location, double *properties) const{
    location      = point;
    properties[0] = static_cast<double>(energy);
    properties[1] = static_cast<double>(ray_id);
    properties[2] = static_cast<double>(reflection);
    for (unsigned int d = 0; d < dim; ++d)
      properties[3 + d] = static_cast<double>(ray_direction[d]);
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

} // namespace MeltPoolDG::RadiativeTransport