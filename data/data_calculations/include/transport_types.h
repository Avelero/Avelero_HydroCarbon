/**
 * @file transport_types.h
 * @brief Core data structures and type definitions for transport emissions calculator
 *
 * @author Moussa Ouallaf
 * @date 2025-12-01
 * @version 2.0
 */

#ifndef TRANSPORT_TYPES_H
#define TRANSPORT_TYPES_H

#include <stdbool.h>

/* Maximum number of transport modes supported */
#define NUM_TRANSPORT_MODES 5

/* Maximum string length for mode names */
#define MAX_MODE_NAME_LEN 32

/**
 * @brief Enumeration of transport modes
 */
typedef enum {
    MODE_ROAD = 0,           /**< Road freight transport (reference mode) */
    MODE_RAIL = 1,           /**< Rail freight transport */
    MODE_INLAND_WATERWAY = 2,/**< Inland waterway transport (IWW) */
    MODE_SEA = 3,            /**< Sea freight transport */
    MODE_AIR = 4             /**< Air freight transport */
} TransportMode;

/**
 * @brief Emission factor data for a transport mode
 */
typedef struct {
    TransportMode mode;
    char name[MAX_MODE_NAME_LEN];
    double ef_gco2e_per_tkm;  /**< Emission factor in g CO2e per tonne-km */
} EmissionFactor;

/**
 * @brief Utility function parameters for multinomial logit model
 */
typedef struct {
    TransportMode mode;
    char name[MAX_MODE_NAME_LEN];
    double beta0;            /**< Intercept (baseline attractiveness) */
    double beta1;            /**< Log-distance coefficient */
} UtilityParameters;

/**
 * @brief Modal split probabilities for all transport modes
 */
typedef struct {
    double probabilities[NUM_TRANSPORT_MODES];
} ModalSplit;

/**
 * @brief Trip information for emission calculation
 */
typedef struct {
    double distance_km;
    double weight_kg;
} Trip;

/**
 * @brief Complete emission data configuration
 */
typedef struct {
    EmissionFactor emission_factors[NUM_TRANSPORT_MODES];
    UtilityParameters utility_params[NUM_TRANSPORT_MODES];
    bool initialized;
} EmissionData;

/**
 * @brief Status codes for function return values
 */
typedef enum {
    STATUS_SUCCESS = 0,
    STATUS_ERROR = -1,
    STATUS_FILE_ERROR = -2,
    STATUS_PARSE_ERROR = -3,
    STATUS_INVALID_INPUT = -4,
    STATUS_MEMORY_ERROR = -5
} StatusCode;

#endif /* TRANSPORT_TYPES_H */
