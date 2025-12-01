/**
 * @file json_parser.h
 * @brief JSON parsing utilities for material composition
 *
 * This module provides utilities for parsing JSON strings containing
 * material composition data from product CSV files.
 *
 * @author Moussa Ouallaf
 * @date 2025-12-01
 * @version 2.0
 */

#ifndef JSON_PARSER_H
#define JSON_PARSER_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_MATERIALS 20

/**
 * @brief Material-percentage pair structure
 */
typedef struct {
    char material_name[128];
    double percentage;
} MaterialEntry;

/**
 * @brief Material composition structure
 */
typedef struct {
    MaterialEntry *entries;
    int count;
} MaterialComposition;

/**
 * @brief Parse material composition JSON string
 *
 * Expected format: {"material_name": percentage, "material_name": percentage, ...}
 *
 * @param[in] json_str JSON string containing material composition
 * @return Pointer to MaterialComposition, or NULL on error
 */
MaterialComposition* parse_material_json(const char *json_str);

/**
 * @brief Free material composition structure
 *
 * @param[in,out] comp MaterialComposition to free
 */
void free_material_composition(MaterialComposition *comp);

#endif /* JSON_PARSER_H */
