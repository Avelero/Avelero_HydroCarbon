/**
 * @file emissions_calc.c
 * @brief Implementation of transport emissions calculations
 *
 * @author Moussa Ouallaf
 * @date 2025-12-01
 * @version 2.0
 */

#include <stdio.h>
#include <math.h>
#include "emissions_calc.h"
#include "modal_split.h"

bool validate_trip(const Trip *trip) {
    if (trip == NULL) {
        return false;
    }

    if (trip->distance_km <= 0.0 || isnan(trip->distance_km) || isinf(trip->distance_km)) {
        return false;
    }

    if (trip->weight_kg <= 0.0 || isnan(trip->weight_kg) || isinf(trip->weight_kg)) {
        return false;
    }

    return true;
}

double calculate_emissions_intensity(double distance_km, const EmissionData *data) {
    if (data == NULL || !data->initialized) {
        return -1.0;
    }

    if (distance_km <= 0.0) {
        return -1.0;
    }

    ModalSplit split;
    StatusCode status = calculate_modal_split(data, distance_km, &split);
    if (status != STATUS_SUCCESS) {
        return -1.0;
    }

    double weighted_ef = calculate_weighted_emission_factor(data, &split);
    double intensity = weighted_ef / 1000.0;

    return intensity;
}

double calculate_transport_emissions_detailed(
    const Trip *trip,
    const EmissionData *data,
    ModalSplit *modal_split,
    double *weighted_ef
) {
    if (!validate_trip(trip)) {
        return -1.0;
    }

    if (data == NULL || !data->initialized) {
        return -1.0;
    }

    ModalSplit split;
    StatusCode status = calculate_modal_split(data, trip->distance_km, &split);
    if (status != STATUS_SUCCESS) {
        return -2.0;
    }

    if (modal_split != NULL) {
        *modal_split = split;
    }

    double ef_weighted = calculate_weighted_emission_factor(data, &split);

    if (weighted_ef != NULL) {
        *weighted_ef = ef_weighted;
    }

    double weight_tonnes = trip->weight_kg / 1000.0;
    double ef_kg_per_tkm = ef_weighted / 1000.0;
    double total_emissions = weight_tonnes * trip->distance_km * ef_kg_per_tkm;

    if (total_emissions < 0.0 || isnan(total_emissions) || isinf(total_emissions)) {
        return -2.0;
    }

    return total_emissions;
}

double calculate_transport_emissions(const Trip *trip, const EmissionData *data) {
    return calculate_transport_emissions_detailed(trip, data, NULL, NULL);
}
