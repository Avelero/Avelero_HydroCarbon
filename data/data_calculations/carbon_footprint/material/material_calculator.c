#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "utils/csv_parser.h"
#include "utils/json_parser.h"

#define MAX_MATERIALS 100

// Material footprint lookup structure
typedef struct {
    char material_name[128];
    double carbon_footprint_kgco2e;
} MaterialFootprint;

// Global material footprint database
MaterialFootprint material_db[MAX_MATERIALS];
int material_db_count = 0;

// Load material dataset from CSV
int load_material_dataset(const char *filename) {
    FILE *fp = fopen(filename, "r");
    if (!fp) {
        fprintf(stderr, "Error: Cannot open material dataset file: %s\n", filename);
        return -1;
    }

    // Skip header
    skip_header(fp);

    char line[MAX_LINE_LENGTH];
    material_db_count = 0;

    while (fgets(line, sizeof(line), fp) && material_db_count < MAX_MATERIALS) {
        CSVRow *row = parse_csv_line(line);
        if (!row) continue;

        if (row->field_count >= 8) {
            char *material_name = get_field(row, 0);  // Column 0: material
            char *carbon_footprint_str = get_field(row, 7);  // Column 7: carbon_footprint_kgCO2e

            if (material_name && carbon_footprint_str && strlen(carbon_footprint_str) > 0) {
                strncpy(material_db[material_db_count].material_name, material_name, 127);
                material_db[material_db_count].material_name[127] = '\0';
                material_db[material_db_count].carbon_footprint_kgco2e = atof(carbon_footprint_str);


                material_db_count++;
            }
        }

        free_csv_row(row);
    }

    fclose(fp);
    printf("Loaded %d materials from dataset\n", material_db_count);
    return 0;
}

/**
 * @brief Lookup carbon footprint value for a material
 *
 * Searches the material database for a matching material name and returns
 * its carbon footprint coefficient in kgCO2e per kg of material.
 *
 * @param[in] material_name Name of the material to lookup
 * @return Carbon footprint coefficient in kgCO2e/kg, or 0.0 if not found
 */
double lookup_material_footprint(const char *material_name) {
    if (!material_name || strlen(material_name) == 0) {
        return 0.0;
    }

    for (int i = 0; i < material_db_count; i++) {
        if (strcmp(material_db[i].material_name, material_name) == 0) {
            return material_db[i].carbon_footprint_kgco2e;
        }
    }

    // Material not found - could be a typo or missing from dataset
    // Return 0.0 as conservative default
    return 0.0;
}

/**
 * @brief Calculate total material carbon footprint for a product
 *
 * Calculates the carbon footprint by summing the contributions of each
 * material component. For each material, the contribution is:
 * product_weight * material_percentage * carbon_footprint_coefficient
 *
 * @param[in] product_weight_kg Total weight of the product in kilograms
 * @param[in] comp Material composition with percentages
 * @return Total material carbon footprint in kgCO2e
 */
double calculate_material_footprint(double product_weight_kg, MaterialComposition *comp) {
    double total_footprint = 0.0;

    for (int i = 0; i < comp->count; i++) {
        const char *material_name = comp->entries[i].material_name;
        double percentage = comp->entries[i].percentage;
        double material_footprint = lookup_material_footprint(material_name);

        // Formula: material_weight * percentage * carbon_footprint_kgCO2e
        double contribution = product_weight_kg * percentage * material_footprint;
        total_footprint += contribution;
    }

    return total_footprint;
}

int main(int argc, char *argv[]) {
    const char *product_file = "../input/Product_data_final.csv";
    const char *material_dataset = "../input/material_dataset_final.csv";
    const char *output_file = "../output/material_footprints.csv";
    
    // Load material dataset
    if (load_material_dataset(material_dataset) != 0) {
        return 1;
    }
    
    // Open product data file
    FILE *fp = fopen(product_file, "r");
    if (!fp) {
        fprintf(stderr, "Error: Cannot open product file: %s\n", product_file);
        return 1;
    }
    
    // Open output file
    FILE *out = fopen(output_file, "w");
    if (!out) {
        fprintf(stderr, "Error: Cannot create output file: %s\n", output_file);
        fclose(fp);
        return 1;
    }
    
    // Write output header
    fprintf(out, "product_name,material_carbon_footprint_kg\n");
    
    // Skip header in product file
    skip_header(fp);
    
    char line[MAX_LINE_LENGTH];
    int processed = 0;
    int errors = 0;
    
    printf("Processing products...\n");
    
    while (fgets(line, sizeof(line), fp)) {
        CSVRow *row = parse_csv_line(line);
        if (!row) {
            errors++;
            continue;
        }
        
        // Validate field count
        if (row->field_count < 7) {
            errors++;
            continue;
        }

        char *product_name = get_field(row, 0);  // Column 0: product_name
        char *materials_json = get_field(row, 5);  // Column 5: materials
        char *weight_str = get_field(row, 6);  // Column 6: weight_kg

        // Validate required fields are present
        if (!product_name || !materials_json || !weight_str || strlen(materials_json) == 0) {
            errors++;
            continue;
        }

        double weight_kg = atof(weight_str);

        MaterialComposition *comp = parse_material_json(materials_json);
        if (comp) {
            double footprint = calculate_material_footprint(weight_kg, comp);
            fprintf(out, "\"%s\",%.6f\n", product_name, footprint);
            free_material_composition(comp);
            processed++;

            if (processed % 50000 == 0) {
                printf("Processed %d products...\n", processed);
            }
        } else {
            errors++;
        }
        
        free_csv_row(row);
    }
    
    fclose(fp);
    fclose(out);
    
    printf("\nMaterial Calculator Complete!\n");
    printf("Processed: %d products\n", processed);
    printf("Errors: %d\n", errors);
    printf("Output: %s\n", output_file);
    
    return 0;
}
