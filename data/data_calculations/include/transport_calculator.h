/**
 * @file transport_calculator.h
 * @brief Main API for transport carbon footprint calculation
 *
 * @author Moussa Ouallaf
 * @date 2025-12-01
 * @version 2.0
 */

#ifndef TRANSPORT_CALCULATOR_H
#define TRANSPORT_CALCULATOR_H

#include "transport_types.h"

/**
 * @brief Transport calculator instance
 */
typedef struct {
    EmissionData emission_data;
    bool ready;
} TransportCalculator;

/**
 * @brief Initialize the transport calculator
 */
StatusCode init_transport_calculator(
    TransportCalculator *calculator,
    const char *emission_factors_file,
    const char *utility_params_file
);

/**
 * @brief Calculate transport carbon footprint
 */
double calculate_transport_footprint(
    const TransportCalculator *calculator,
    double distance_km,
    double weight_kg
);

/**
 * @brief Calculate transport footprint with detailed output
 */
double calculate_transport_footprint_detailed(
    const TransportCalculator *calculator,
    double distance_km,
    double weight_kg,
    ModalSplit *modal_split,
    double *weighted_ef
);

/**
 * @brief Get emissions intensity at a given distance
 */
double get_emissions_intensity(
    const TransportCalculator *calculator,
    double distance_km
);

/**
 * @brief Check if calculator is ready
 */
bool is_calculator_ready(const TransportCalculator *calculator);

/**
 * @brief Clean up transport calculator
 */
void cleanup_transport_calculator(TransportCalculator *calculator);

#endif /* TRANSPORT_CALCULATOR_H */
