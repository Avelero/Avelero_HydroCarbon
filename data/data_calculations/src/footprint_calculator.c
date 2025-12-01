/**
 * @file footprint_calculator.c
 * @brief Unified footprint calculator - combines carbon and water footprints
 *
 * This is the main handler that processes product data and calculates:
 * - Material carbon footprint (kgCO2e)
 * - Transport carbon footprint (kgCO2e)
 * - Total carbon footprint (kgCO2e)
 * - Total water footprint (liters)
 *
 * Output: A single CSV file containing all original product data plus
 * the calculated footprint columns.
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
#include "material_carbon.h"
#include "material_water.h"
#include "transport_calculator.h"

#define MAX_PRODUCTS 1000000
#define MAX_PRODUCT_LINE 8192

/* Product with all footprint data */
typedef struct {
    char original_line[MAX_PRODUCT_LINE];
    char product_name[512];
    double weight_kg;
    double distance_km;
    double carbon_material;
    double carbon_transport;
    double carbon_total;
    double water_total;
} ProductFootprint;

/**
 * @brief Load product data and calculate all footprints
 */
int process_products(
    const char *product_file,
    const char *material_dataset,
    const char *output_file,
    TransportCalculator *transport_calc
) {
    /* Load material datasets */
    printf("Loading material carbon dataset...\n");
    if (load_material_carbon_dataset(material_dataset) != 0) {
        fprintf(stderr, "Error: Failed to load material carbon dataset\n");
        return -1;
    }
    printf("Loaded %d materials for carbon footprint\n", get_carbon_db_count());

    printf("Loading material water dataset...\n");
    if (load_material_water_dataset(material_dataset) != 0) {
        fprintf(stderr, "Error: Failed to load material water dataset\n");
        return -1;
    }
    printf("Loaded %d materials for water footprint\n", get_water_db_count());

    /* Open product data file */
    FILE *fp = fopen(product_file, "r");
    if (!fp) {
        fprintf(stderr, "Error: Cannot open product file: %s\n", product_file);
        return -1;
    }

    /* Open output file */
    FILE *out = fopen(output_file, "w");
    if (!out) {
        fprintf(stderr, "Error: Cannot create output file: %s\n", output_file);
        fclose(fp);
        return -1;
    }

    /* Write output header - original columns plus footprint columns */
    fprintf(out, "product_name,gender,parent_category,category,manufacturer_country,materials,weight_kg,total_distance_km,carbon_material,carbon_transport,carbon_total,water_total\n");

    /* Skip header in product file */
    skip_header(fp);

    char line[MAX_PRODUCT_LINE];
    int processed = 0;
    int errors = 0;

    printf("Processing products...\n");

    while (fgets(line, sizeof(line), fp)) {
        /* Remove trailing newline */
        size_t len = strlen(line);
        while (len > 0 && (line[len-1] == '\n' || line[len-1] == '\r')) {
            line[len-1] = '\0';
            len--;
        }

        CSVRow *row = parse_csv_line(line);
        if (!row) {
            errors++;
            continue;
        }

        /* Validate field count - need at least 8 columns */
        if (row->field_count < 8) {
            errors++;
            free_csv_row(row);
            continue;
        }

        /* Extract fields */
        char *product_name = get_field(row, 0);      /* Column 0: product_name */
        char *materials_json = get_field(row, 5);    /* Column 5: materials */
        char *weight_str = get_field(row, 6);        /* Column 6: weight_kg */
        char *distance_str = get_field(row, 7);      /* Column 7: total_distance_km */

        if (!product_name || !materials_json || !weight_str || !distance_str || 
            strlen(materials_json) == 0) {
            errors++;
            free_csv_row(row);
            continue;
        }

        double weight_kg = atof(weight_str);
        double distance_km = atof(distance_str);

        /* Parse material composition */
        MaterialComposition *comp = parse_material_json(materials_json);
        if (!comp) {
            errors++;
            free_csv_row(row);
            continue;
        }

        /* Calculate footprints */
        double carbon_material = calculate_material_carbon_footprint(weight_kg, comp);
        double carbon_transport = 0.0;
        
        /* Only calculate transport if distance > 0 */
        if (distance_km > 0.0 && is_calculator_ready(transport_calc)) {
            carbon_transport = calculate_transport_footprint(transport_calc, distance_km, weight_kg);
            if (carbon_transport < 0.0) {
                carbon_transport = 0.0;  /* Handle error gracefully */
            }
        }
        
        double carbon_total = carbon_material + carbon_transport;
        double water_total = calculate_material_water_footprint(weight_kg, comp);

        /* Write output line: original data + footprints */
        fprintf(out, "%s,%.6f,%.6f,%.6f,%.6f\n",
                line,
                carbon_material,
                carbon_transport,
                carbon_total,
                water_total);

        free_material_composition(comp);
        free_csv_row(row);
        processed++;

        /* Progress indicator */
        if (processed % 50000 == 0) {
            printf("Processed %d products...\n", processed);
        }
    }

    fclose(fp);
    fclose(out);

    printf("\n=== Footprint Calculation Complete! ===\n");
    printf("Processed: %d products\n", processed);
    printf("Errors: %d\n", errors);
    printf("Output: %s\n", output_file);

    return 0;
}

int main(void) {
    printf("=== Unified Footprint Calculator ===\n\n");

    /* File paths - relative to build directory when running from there */
    const char *product_file = "../input/Product_data_final.csv";
    const char *material_dataset = "../input/material_dataset_final.csv";
    const char *emission_factors = "../input/transport_emission_factors_generalised.csv";
    const char *utility_params = "../input/utility_attractiveness.csv";
    const char *output_file = "../output/Product_data_with_footprints.csv";

    /* Initialize transport calculator */
    printf("Initializing transport calculator...\n");
    TransportCalculator transport_calc;
    StatusCode status = init_transport_calculator(
        &transport_calc, 
        emission_factors, 
        utility_params
    );

    if (status != STATUS_SUCCESS) {
        fprintf(stderr, "Warning: Transport calculator initialization failed. Transport footprints will be 0.\n");
    } else {
        printf("Transport calculator initialized successfully\n");
    }

    /* Process all products */
    printf("\n");
    int result = process_products(
        product_file, 
        material_dataset, 
        output_file, 
        &transport_calc
    );

    /* Cleanup */
    cleanup_transport_calculator(&transport_calc);

    if (result == 0) {
        printf("\nOutput columns:\n");
        printf("  - product_name, gender, parent_category, category, manufacturer_country\n");
        printf("  - materials, weight_kg, total_distance_km (original data)\n");
        printf("  - carbon_material (material carbon footprint in kgCO2e)\n");
        printf("  - carbon_transport (transport carbon footprint in kgCO2e)\n");
        printf("  - carbon_total (total carbon footprint in kgCO2e)\n");
        printf("  - water_total (total water footprint in liters)\n");
    }

    return result;
}
