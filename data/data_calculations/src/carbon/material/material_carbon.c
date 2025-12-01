/**
 * @file material_carbon.c
 * @brief Carbon footprint calculator for product materials
 *
 * @author Moussa Ouallaf
 * @date 2025-12-01
 * @version 2.0
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "csv_parser.h"
#include "json_parser.h"

/* Undefine MAX_MATERIALS from json_parser.h to use our own value */
#undef MAX_MATERIALS
#define MAX_MATERIALS 100

/* Material footprint lookup structure */
typedef struct {
    char material_name[128];
    double carbon_footprint_kgco2e;
} MaterialCarbonFootprint;

/* Global material carbon footprint database */
static MaterialCarbonFootprint carbon_db[MAX_MATERIALS];
static int carbon_db_count = 0;

/**
 * @brief Load material carbon footprint dataset from CSV
 */
int load_material_carbon_dataset(const char *filename) {
    FILE *fp = fopen(filename, "r");
    if (!fp) {
        fprintf(stderr, "Error: Cannot open material dataset file: %s\n", filename);
        return -1;
    }

    skip_header(fp);

    char line[MAX_LINE_LENGTH];
    carbon_db_count = 0;

    while (fgets(line, sizeof(line), fp) && carbon_db_count < MAX_MATERIALS) {
        CSVRow *row = parse_csv_line(line);
        if (!row) continue;

        if (row->field_count >= 8) {
            char *material_name = get_field(row, 0);  /* Column 0: material */
            char *carbon_footprint_str = get_field(row, 7);  /* Column 7: carbon_footprint_kgCO2e */

            if (material_name && carbon_footprint_str && strlen(carbon_footprint_str) > 0) {
                strncpy(carbon_db[carbon_db_count].material_name, material_name, 127);
                carbon_db[carbon_db_count].material_name[127] = '\0';
                carbon_db[carbon_db_count].carbon_footprint_kgco2e = atof(carbon_footprint_str);
                carbon_db_count++;
            }
        }

        free_csv_row(row);
    }

    fclose(fp);
    return 0;
}

/**
 * @brief Get carbon database count
 */
int get_carbon_db_count(void) {
    return carbon_db_count;
}

/**
 * @brief Lookup carbon footprint value for a material
 */
double lookup_material_carbon_footprint(const char *material_name) {
    if (!material_name || strlen(material_name) == 0) {
        return 0.0;
    }

    for (int i = 0; i < carbon_db_count; i++) {
        if (strcmp(carbon_db[i].material_name, material_name) == 0) {
            return carbon_db[i].carbon_footprint_kgco2e;
        }
    }

    return 0.0;
}

/**
 * @brief Calculate total material carbon footprint for a product
 */
double calculate_material_carbon_footprint(double product_weight_kg, MaterialComposition *comp) {
    double total_footprint = 0.0;

    for (int i = 0; i < comp->count; i++) {
        const char *material_name = comp->entries[i].material_name;
        double percentage = comp->entries[i].percentage;
        double material_footprint = lookup_material_carbon_footprint(material_name);

        double contribution = product_weight_kg * percentage * material_footprint;
        total_footprint += contribution;
    }

    return total_footprint;
}
