/**
 * @file modal_split.c
 * @brief Implementation of modal split calculations
 *
 * @author Moussa Ouallaf
 * @date 2025-12-01
 * @version 2.0
 */

#include <stdio.h>
#include <math.h>
#include "modal_split.h"

#define PROB_SUM_TOLERANCE 1e-6

double calculate_utility(const UtilityParameters *params, double distance_km) {
    if (params == NULL || distance_km <= 0.0) {
        return 0.0;
    }

    double utility = params->beta0 + params->beta1 * log(distance_km);
    return utility;
}

StatusCode calculate_modal_split(
    const EmissionData *data,
    double distance_km,
    ModalSplit *split
) {
    if (data == NULL || split == NULL) {
        return STATUS_INVALID_INPUT;
    }

    if (!data->initialized) {
        fprintf(stderr, "Error: Emission data not initialized\n");
        return STATUS_ERROR;
    }

    if (distance_km <= 0.0) {
        fprintf(stderr, "Error: Invalid distance: %.2f km\n", distance_km);
        return STATUS_INVALID_INPUT;
    }

    double utilities[NUM_TRANSPORT_MODES];
    for (int i = 0; i < NUM_TRANSPORT_MODES; i++) {
        utilities[i] = calculate_utility(&data->utility_params[i], distance_km);
    }

    double exp_utilities[NUM_TRANSPORT_MODES];
    for (int i = 0; i < NUM_TRANSPORT_MODES; i++) {
        exp_utilities[i] = exp(utilities[i]);

        if (isinf(exp_utilities[i]) || isnan(exp_utilities[i])) {
            fprintf(stderr, "Error: Numerical overflow in exp(utility) for mode %d\n", i);
            return STATUS_ERROR;
        }
    }

    double denominator = 0.0;
    for (int i = 0; i < NUM_TRANSPORT_MODES; i++) {
        denominator += exp_utilities[i];
    }

    if (denominator <= 0.0 || isinf(denominator) || isnan(denominator)) {
        fprintf(stderr, "Error: Invalid denominator in modal split calculation\n");
        return STATUS_ERROR;
    }

    for (int i = 0; i < NUM_TRANSPORT_MODES; i++) {
        split->probabilities[i] = exp_utilities[i] / denominator;
    }

    if (!validate_modal_split(split)) {
        fprintf(stderr, "Error: Modal split validation failed\n");
        return STATUS_ERROR;
    }

    return STATUS_SUCCESS;
}

double calculate_weighted_emission_factor(
    const EmissionData *data,
    const ModalSplit *split
) {
    if (data == NULL || split == NULL) {
        return 0.0;
    }

    double weighted_ef = 0.0;

    for (int i = 0; i < NUM_TRANSPORT_MODES; i++) {
        double probability = split->probabilities[i];
        double emission_factor = data->emission_factors[i].ef_gco2e_per_tkm;

        weighted_ef += probability * emission_factor;
    }

    return weighted_ef;
}

bool validate_modal_split(const ModalSplit *split) {
    if (split == NULL) {
        return false;
    }

    double sum = 0.0;

    for (int i = 0; i < NUM_TRANSPORT_MODES; i++) {
        double prob = split->probabilities[i];

        if (prob < 0.0 || prob > 1.0) {
            fprintf(stderr, "Validation error: Probability %d = %.6f (out of range)\n",
                    i, prob);
            return false;
        }

        if (isnan(prob) || isinf(prob)) {
            fprintf(stderr, "Validation error: Probability %d is NaN or Inf\n", i);
            return false;
        }

        sum += prob;
    }

    if (fabs(sum - 1.0) > PROB_SUM_TOLERANCE) {
        fprintf(stderr, "Validation error: Probabilities sum to %.9f (expected 1.0)\n", sum);
        return false;
    }

    return true;
}
