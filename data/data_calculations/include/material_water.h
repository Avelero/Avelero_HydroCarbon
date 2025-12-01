/**
 * @file material_water.h
 * @brief Header for material water footprint calculations
 *
 * @author Moussa Ouallaf
 * @date 2025-12-01
 * @version 2.0
 */

#ifndef MATERIAL_WATER_H
#define MATERIAL_WATER_H

#include "json_parser.h"

/**
 * @brief Load material water footprint dataset from CSV
 */
int load_material_water_dataset(const char *filename);

/**
 * @brief Get water database count
 */
int get_water_db_count(void);

/**
 * @brief Lookup water footprint value for a material
 */
double lookup_material_water_footprint(const char *material_name);

/**
 * @brief Calculate total material water footprint for a product
 */
double calculate_material_water_footprint(double product_weight_kg, MaterialComposition *comp);

#endif /* MATERIAL_WATER_H */
