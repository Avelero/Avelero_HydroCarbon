/**
 * @file emissions_calc.h
 * @brief Core transport emissions calculation
 *
 * @author Moussa Ouallaf
 * @date 2025-12-01
 * @version 2.0
 */

#ifndef EMISSIONS_CALC_H
#define EMISSIONS_CALC_H

#include "transport_types.h"

/**
 * @brief Calculate total CO2 emissions for a transport trip
 */
double calculate_transport_emissions(const Trip *trip, const EmissionData *data);

/**
 * @brief Calculate emissions with detailed output
 */
double calculate_transport_emissions_detailed(
    const Trip *trip,
    const EmissionData *data,
    ModalSplit *modal_split,
    double *weighted_ef
);

/**
 * @brief Calculate emissions per tonne-km
 */
double calculate_emissions_intensity(double distance_km, const EmissionData *data);

/**
 * @brief Validate trip parameters
 */
bool validate_trip(const Trip *trip);

#endif /* EMISSIONS_CALC_H */
