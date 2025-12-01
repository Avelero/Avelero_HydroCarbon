/**
 * @file modal_split.h
 * @brief Modal split calculation using multinomial logit model
 *
 * @author Moussa Ouallaf
 * @date 2025-12-01
 * @version 2.0
 */

#ifndef MODAL_SPLIT_H
#define MODAL_SPLIT_H

#include "transport_types.h"

/**
 * @brief Calculate utility function for a transport mode
 */
double calculate_utility(const UtilityParameters *params, double distance_km);

/**
 * @brief Calculate modal split probabilities for all modes
 */
StatusCode calculate_modal_split(
    const EmissionData *data,
    double distance_km,
    ModalSplit *split
);

/**
 * @brief Calculate weighted average emission factor
 */
double calculate_weighted_emission_factor(
    const EmissionData *data,
    const ModalSplit *split
);

/**
 * @brief Validate modal split probabilities
 */
bool validate_modal_split(const ModalSplit *split);

#endif /* MODAL_SPLIT_H */
