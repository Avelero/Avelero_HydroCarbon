#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "utils/csv_parser.h"

// Transport carbon footprint calculator (TEMPLATE)
// TODO: Implement proper transport formula in future iteration

double calculate_transport_footprint(double distance_km, double weight_kg) {
    // TEMPLATE PLACEHOLDER
    // Future formula considerations:
    // - Distance in km
    // - Product weight
    // - Transport mode (air, sea, land)
    // - Emissions factor per ton-km
    
    // For now, return 0.0 as placeholder
    return 0.0;
}

int main(int argc, char *argv[]) {
    const char *product_file = "../input/Product_data_final.csv";
    const char *output_file = "../output/transport_footprints.csv";
    
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
    fprintf(out, "product_name,transport_carbon_footprint_kg\n");
    
    // Skip header in product file
    skip_header(fp);
    
    char line[MAX_LINE_LENGTH];
    int processed = 0;
    
    printf("Processing transport footprints (template mode)...\n");
    
    while (fgets(line, sizeof(line), fp)) {
        CSVRow *row = parse_csv_line(line);
        if (!row) continue;
        
        if (row->field_count >= 8) {
            char *product_name = get_field(row, 0);  // Column 0: product_name
            char *weight_str = get_field(row, 6);  // Column 6: weight_kg
            char *distance_str = get_field(row, 7);  // Column 7: total_distance_km
            
            if (product_name && weight_str && distance_str) {
                double weight_kg = atof(weight_str);
                double distance_km = atof(distance_str);
                
                // Calculate transport footprint (currently returns 0.0)
                double footprint = calculate_transport_footprint(distance_km, weight_kg);
                
                fprintf(out, "\"%s\",%.6f\n", product_name, footprint);
                processed++;
                
                if (processed % 50000 == 0) {
                    printf("Processed %d products...\n", processed);
                }
            }
        }
        
        free_csv_row(row);
    }
    
    fclose(fp);
    fclose(out);
    
    printf("\nTransport Calculator Complete (Template)!\n");
    printf("Processed: %d products\n", processed);
    printf("Output: %s\n", output_file);
    printf("Note: All transport footprints = 0.0 (awaiting formula implementation)\n");
    
    return 0;
}
