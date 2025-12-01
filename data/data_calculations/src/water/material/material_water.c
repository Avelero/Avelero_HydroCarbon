/**
 * @file material_water.c
 * @brief Water footprint calculator for product materials
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

#undef MAX_MATERIALS
#define MAX_MATERIALS 100

typedef struct {
    char material_name[128];
    double water_footprint_liters;
} MaterialWaterFootprint;

static MaterialWaterFootprint water_db[MAX_MATERIALS];
static int water_db_count = 0;

int load_material_water_dataset(const char *filename) {
    FILE *fp = fopen(filename, "r");
    if (!fp) {
        fprintf(stderr, "Error: Cannot open material dataset file: %s\n", filename);
        return -1;
    }

    skip_header(fp);

    char line[MAX_LINE_LENGTH];
    water_db_count = 0;

    while (fgets(line, sizeof(line), fp) && water_db_count < MAX_MATERIALS) {
        CSVRow *row = parse_csv_line(line);
        if (!row) continue;

        if (row->field_count >= 11) {
            char *material_name = get_field(row, 0);  /* Column 0: material */
            char *water_footprint_str = get_field(row, 10);  /* Column 10: water_footprint_liters */

            if (material_name && water_footprint_str && strlen(water_footprint_str) > 0) {
                strncpy(water_db[water_db_count].material_name, material_name, 127);
                water_db[water_db_count].material_name[127] = '\0';
                water_db[water_db_count].water_footprint_liters = atof(water_footprint_str);
                water_db_count++;
            }
        }

        free_csv_row(row);
    }

    fclose(fp);
    return 0;
}

int get_water_db_count(void) {
    return water_db_count;
}

double lookup_material_water_footprint(const char *material_name) {
    if (!material_name || strlen(material_name) == 0) {
        return 0.0;
    }

    for (int i = 0; i < water_db_count; i++) {
        if (strcmp(water_db[i].material_name, material_name) == 0) {
            return water_db[i].water_footprint_liters;
        }
    }

    return 0.0;
}

double calculate_material_water_footprint(double product_weight_kg, MaterialComposition *comp) {
    double total_footprint = 0.0;

    for (int i = 0; i < comp->count; i++) {
        const char *material_name = comp->entries[i].material_name;
        double percentage = comp->entries[i].percentage;
        double material_water_footprint = lookup_material_water_footprint(material_name);

        double contribution = product_weight_kg * percentage * material_water_footprint;
        total_footprint += contribution;
    }

    return total_footprint;
}
