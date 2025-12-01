/**
 * @file material_carbon.h
 * @brief Header for material carbon footprint calculations
 *
 * @author Moussa Ouallaf
 * @date 2025-12-01
 * @version 2.0
 */

#ifndef MATERIAL_CARBON_H
#define MATERIAL_CARBON_H

#include "json_parser.h"

/**
 * @brief Load material carbon footprint dataset from CSV
 */
int load_material_carbon_dataset(const char *filename);

/**
 * @brief Get carbon database count
 */
int get_carbon_db_count(void);

/**
 * @brief Lookup carbon footprint value for a material
 */
double lookup_material_carbon_footprint(const char *material_name);

/**
 * @brief Calculate total material carbon footprint for a product
 */
double calculate_material_carbon_footprint(double product_weight_kg, MaterialComposition *comp);

#endif /* MATERIAL_CARBON_H */
