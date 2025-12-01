/**
 * @file transport_calculator.c
 * @brief Implementation of main transport calculator API
 *
 * @author Moussa Ouallaf
 * @date 2025-12-01
 * @version 2.0
 */

#include <stdio.h>
#include <string.h>
#include "transport_calculator.h"
#include "emission_data.h"
#include "emissions_calc.h"

StatusCode init_transport_calculator(
    TransportCalculator *calculator,
    const char *emission_factors_file,
    const char *utility_params_file
) {
    if (calculator == NULL) {
        fprintf(stderr, "Error: NULL calculator pointer\n");
        return STATUS_INVALID_INPUT;
    }

    memset(calculator, 0, sizeof(TransportCalculator));
    calculator->ready = false;

    StatusCode status = init_emission_data(
        &calculator->emission_data,
        emission_factors_file,
        utility_params_file
    );

    if (status != STATUS_SUCCESS) {
        fprintf(stderr, "Error: Failed to initialize emission data\n");
        return status;
    }

    calculator->ready = true;
    return STATUS_SUCCESS;
}

bool is_calculator_ready(const TransportCalculator *calculator) {
    if (calculator == NULL) {
        return false;
    }

    return calculator->ready && calculator->emission_data.initialized;
}

double calculate_transport_footprint_detailed(
    const TransportCalculator *calculator,
    double distance_km,
    double weight_kg,
    ModalSplit *modal_split,
    double *weighted_ef
) {
    if (!is_calculator_ready(calculator)) {
        return -2.0;
    }

    Trip trip = {
        .distance_km = distance_km,
        .weight_kg = weight_kg
    };

    double emissions = calculate_transport_emissions_detailed(
        &trip,
        &calculator->emission_data,
        modal_split,
        weighted_ef
    );

    return emissions;
}

double calculate_transport_footprint(
    const TransportCalculator *calculator,
    double distance_km,
    double weight_kg
) {
    return calculate_transport_footprint_detailed(
        calculator,
        distance_km,
        weight_kg,
        NULL,
        NULL
    );
}

double get_emissions_intensity(
    const TransportCalculator *calculator,
    double distance_km
) {
    if (!is_calculator_ready(calculator)) {
        return -1.0;
    }

    return calculate_emissions_intensity(distance_km, &calculator->emission_data);
}

void cleanup_transport_calculator(TransportCalculator *calculator) {
    if (calculator == NULL) {
        return;
    }

    memset(calculator, 0, sizeof(TransportCalculator));
    calculator->ready = false;
}
