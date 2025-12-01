/**
 * @file emission_data.h
 * @brief Emission data loading and management
 *
 * @author Moussa Ouallaf
 * @date 2025-12-01
 * @version 2.0
 */

#ifndef EMISSION_DATA_H
#define EMISSION_DATA_H

#include "transport_types.h"

/**
 * @brief Initialize emission data from CSV files
 */
StatusCode init_emission_data(
    EmissionData *data,
    const char *emission_factors_file,
    const char *utility_params_file
);

/**
 * @brief Load emission factors from CSV file
 */
StatusCode load_emission_factors(EmissionData *data, const char *filepath);

/**
 * @brief Load utility parameters from CSV file
 */
StatusCode load_utility_parameters(EmissionData *data, const char *filepath);

/**
 * @brief Map mode name string to TransportMode enum
 */
TransportMode map_mode_name(const char *mode_name);

/**
 * @brief Validate that all emission data is complete
 */
bool validate_emission_data(const EmissionData *data);

#endif /* EMISSION_DATA_H */
