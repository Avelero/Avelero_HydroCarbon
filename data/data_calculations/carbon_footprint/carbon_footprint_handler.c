#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/wait.h>
#include <unistd.h>
#include "utils/csv_parser.h"

#define MAX_PRODUCTS 1000000
#define MAX_PRODUCT_LINE 8192

// Product with footprint data
typedef struct {
    char original_line[MAX_PRODUCT_LINE];  // Store entire original CSV line
    char product_name[512];
    double material_footprint;
    double transport_footprint;
    double total_footprint;
} ProductData;

// Load original product data
int load_product_data(const char *product_file, ProductData *products, int *count) {
    FILE *fp = fopen(product_file, "r");
    if (!fp) {
        fprintf(stderr, "Error: Cannot open product file: %s\n", product_file);
        return -1;
    }
    
    // Skip header (we'll write it separately)
    skip_header(fp);
    
    char line[MAX_PRODUCT_LINE];
    *count = 0;
    
    while (fgets(line, sizeof(line), fp) && *count < MAX_PRODUCTS) {
        // Remove trailing newline and carriage return
        size_t len = strlen(line);
        while (len > 0 && (line[len-1] == '\n' || line[len-1] == '\r')) {
            line[len-1] = '\0';
            len--;
        }
        
        // Parse to get product name
        CSVRow *row = parse_csv_line(line);
        if (!row) continue;
        
        if (row->field_count >= 1) {
            char *product_name = get_field(row, 0);
            if (product_name) {
                // Store original line and product name
                strncpy(products[*count].original_line, line, MAX_PRODUCT_LINE - 1);
                products[*count].original_line[MAX_PRODUCT_LINE - 1] = '\0';
                strncpy(products[*count].product_name, product_name, 511);
                products[*count].product_name[511] = '\0';
                
                // Initialize footprints to 0
                products[*count].material_footprint = 0.0;
                products[*count].transport_footprint = 0.0;
                products[*count].total_footprint = 0.0;
                
                (*count)++;
            }
        }
        
        free_csv_row(row);
    }
    
    fclose(fp);
    return 0;
}

// Match and merge footprints into product data
int merge_footprints(ProductData *products, int count, 
                     const char *material_file, const char *transport_file) {
    FILE *mat_fp = fopen(material_file, "r");
    FILE *trans_fp = fopen(transport_file, "r");
    
    if (!mat_fp || !trans_fp) {
        fprintf(stderr, "Error: Cannot open footprint files\n");
        if (mat_fp) fclose(mat_fp);
        if (trans_fp) fclose(trans_fp);
        return -1;
    }
    
    // Skip headers
    skip_header(mat_fp);
    skip_header(trans_fp);
    
    char mat_line[MAX_LINE_LENGTH];
    char trans_line[MAX_LINE_LENGTH];
    int index = 0;
    
    while (fgets(mat_line, sizeof(mat_line), mat_fp) && 
           fgets(trans_line, sizeof(trans_line), trans_fp) &&
           index < count) {
        
        CSVRow *mat_row = parse_csv_line(mat_line);
        CSVRow *trans_row = parse_csv_line(trans_line);
        
        if (mat_row && trans_row) {
            if (mat_row->field_count >= 2 && trans_row->field_count >= 2) {
                char *mat_footprint_str = get_field(mat_row, 1);
                char *trans_footprint_str = get_field(trans_row, 1);
                
                if (mat_footprint_str && trans_footprint_str) {
                    products[index].material_footprint = atof(mat_footprint_str);
                    products[index].transport_footprint = atof(trans_footprint_str);
                    products[index].total_footprint = 
                        products[index].material_footprint + products[index].transport_footprint;
                }
            }
        }
        
        if (mat_row) free_csv_row(mat_row);
        if (trans_row) free_csv_row(trans_row);
        index++;
    }
    
    fclose(mat_fp);
    fclose(trans_fp);
    
    return 0;
}

// Write enriched output with all original columns plus footprints
int write_enriched_output(const char *output_file, ProductData *products, int count) {
    FILE *out = fopen(output_file, "w");
    if (!out) {
        fprintf(stderr, "Error: Cannot create output file: %s\n", output_file);
        return -1;
    }
    
    // Write header with original columns plus new footprint columns
    fprintf(out, "product_name,gender,parent_category,category,manufacturer_country,materials,weight_kg,total_distance_km,material_carbon_footprint_kg,transport_carbon_footprint_kg,total_carbon_footprint_kg\n");
    
    // Write data - original columns plus footprints
    for (int i = 0; i < count; i++) {
        // Write original line
        fprintf(out, "%s", products[i].original_line);
        
        // Append footprint columns
        fprintf(out, ",%.6f,%.6f,%.6f\n",
                products[i].material_footprint,
                products[i].transport_footprint,
                products[i].total_footprint);
    }
    
    fclose(out);
    return 0;
}

int main(int argc, char *argv[]) {
    printf("=== Carbon Footprint Calculator Handler ===\n\n");
    
    // Step 1: Run Material Calculator
    printf("Step 1/3: Running Material Calculator...\n");
    int status = system("./build/material_calculator");
    if (status != 0) {
        fprintf(stderr, "Error: Material calculator failed\n");
        return 1;
    }
    printf("\n");
    
    // Step 2: Run Transport Calculator
    printf("Step 2/3: Running Transport Calculator...\n");
    status = system("./build/transport_calculator");
    if (status != 0) {
        fprintf(stderr, "Error: Transport calculator failed\n");
        return 1;
    }
    printf("\n");
    
    // Step 3: Merge Results with Original Product Data
    printf("Step 3/3: Merging Results with Product Data...\n");
    
    ProductData *products = (ProductData*)malloc(MAX_PRODUCTS * sizeof(ProductData));
    if (!products) {
        fprintf(stderr, "Error: Memory allocation failed\n");
        return 1;
    }
    
    // Load original product data
    int count = 0;
    printf("Loading original product data...\n");
    if (load_product_data("../input/Product_data_final.csv", products, &count) != 0) {
        free(products);
        return 1;
    }
    printf("Loaded %d products\n", count);
    
    // Merge footprints
    printf("Merging footprint calculations...\n");
    if (merge_footprints(products, count, 
                        "../output/material_footprints.csv", 
                        "../output/transport_footprints.csv") != 0) {
        free(products);
        return 1;
    }
    
    // Write enriched output
    const char *output_file = "../output/carbon_footprint_results.csv";
    printf("Writing enriched output...\n");
    if (write_enriched_output(output_file, products, count) != 0) {
        free(products);
        return 1;
    }
    
    free(products);
    
    printf("\n=== Carbon Footprint Calculation Complete! ===\n");
    printf("Output file: %s\n", output_file);
    printf("Total products processed: %d\n", count);
    printf("Output includes all original columns plus:\n");
    printf("  - material_carbon_footprint_kg\n");
    printf("  - transport_carbon_footprint_kg\n");
    printf("  - total_carbon_footprint_kg\n");
    
    return 0;
}

