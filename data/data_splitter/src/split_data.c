#define _POSIX_C_SOURCE 200809L

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <ctype.h>

#define MAX_LINE_LENGTH 4096
#define MAX_FIELD_LENGTH 2048
#define INITIAL_CAPACITY 1000
#define TRAIN_RATIO 0.75
#define RANDOM_SEED 42
#define NUM_MATERIALS 34

// All 34 materials in alphabetical order
static const char *MATERIALS[NUM_MATERIALS] = {
    "acrylic", "cashmere", "coated_fabric_pu", "cotton_conventional", 
    "cotton_organic", "cotton_recycled", "down_feather", "down_synthetic",
    "elastane", "eva", "hemp", "jute",
    "leather_bovine", "leather_ovine", "leather_synthetic", "linen_flax",
    "lyocell_tencel", "metal_brass", "metal_gold", "metal_silver",
    "metal_steel", "modal", "natural_rubber", "polyamide_6",
    "polyamide_66", "polyamide_recycled", "polyester_recycled", "polyester_virgin",
    "rubber_synthetic", "silk", "tpu", "viscose",
    "wool_generic", "wool_merino"
};

typedef struct {
    char **lines;
    size_t count;
    size_t capacity;
} Dataset;

// Initialize dataset
void init_dataset(Dataset *ds) {
    ds->lines = malloc(INITIAL_CAPACITY * sizeof(char *));
    if (!ds->lines) {
        fprintf(stderr, "Error: Memory allocation failed\n");
        exit(1);
    }
    ds->count = 0;
    ds->capacity = INITIAL_CAPACITY;
}

// Add line to dataset
void add_line(Dataset *ds, const char *line) {
    if (ds->count >= ds->capacity) {
        ds->capacity *= 2;
        char **new_lines = realloc(ds->lines, ds->capacity * sizeof(char *));
        if (!new_lines) {
            fprintf(stderr, "Error: Memory reallocation failed\n");
            exit(1);
        }
        ds->lines = new_lines;
    }
    
    ds->lines[ds->count] = malloc(strlen(line) + 1);
    if (!ds->lines[ds->count]) {
        fprintf(stderr, "Error: Memory allocation failed for line\n");
        exit(1);
    }
    strcpy(ds->lines[ds->count], line);
    ds->count++;
}

// Free dataset memory
void free_dataset(Dataset *ds) {
    for (size_t i = 0; i < ds->count; i++) {
        free(ds->lines[i]);
    }
    free(ds->lines);
}

// Fisher-Yates shuffle algorithm
void shuffle_dataset(Dataset *ds, unsigned int seed) {
    srand(seed);
    for (size_t i = ds->count - 1; i > 0; i--) {
        size_t j = rand() % (i + 1);
        // Swap lines[i] and lines[j]
        char *temp = ds->lines[i];
        ds->lines[i] = ds->lines[j];
        ds->lines[j] = temp;
    }
}

// Find material index by name
int find_material_index(const char *material_name) {
    for (int i = 0; i < NUM_MATERIALS; i++) {
        if (strcmp(material_name, MATERIALS[i]) == 0) {
            return i;
        }
    }
    return -1;
}

// Parse materials JSON and populate percentages array
// Handles format after CSV stripping: {polyester_virgin:1.0,cotton_conventional:0.5}
void parse_materials(const char *json_str, double *percentages) {
    // Initialize all percentages to 0.0
    for (int i = 0; i < NUM_MATERIALS; i++) {
        percentages[i] = 0.0;
    }
    
    if (!json_str || strlen(json_str) < 2) return;
    
    char *str_copy = strdup(json_str);
    if (!str_copy) return;
    
    // Start after the opening brace
    char *ptr = strchr(str_copy, '{');
    if (!ptr) {
        free(str_copy);
        return;
    }
    ptr++; // Skip the {
    
    // Parse each material:value pair
    while (*ptr) {
        // Skip whitespace
        while (*ptr && isspace(*ptr)) ptr++;
        if (!*ptr || *ptr == '}') break;
        
        // Extract material name (until colon)
        char material[128];
        int mat_idx = 0;
        while (*ptr && *ptr != ':' && *ptr != ',' && *ptr != '}') {
            if (mat_idx < 127) {
                material[mat_idx++] = *ptr;
            }
            ptr++;
        }
        material[mat_idx] = '\0';
        
        if (*ptr != ':') break;
        ptr++; // Skip :
        
        // Parse percentage value
        double percentage = atof(ptr);
        
        // Find material index and set percentage
        int idx = find_material_index(material);
        if (idx >= 0) {
            percentages[idx] = percentage;
        }
        
        // Move to next comma or end
        while (*ptr && *ptr != ',' && *ptr != '}') ptr++;
        if (*ptr == ',') ptr++;
    }
    
    free(str_copy);
}

// Parse CSV line handling quoted fields properly
int parse_csv_line(const char *line, char fields[][MAX_FIELD_LENGTH], int max_fields) {
    int field_count = 0;
    int pos = 0;
    int in_quotes = 0;
    int field_pos = 0;
    
    while (line[pos] && field_count < max_fields) {
        char c = line[pos];
        
        if (c == '"') {
            in_quotes = !in_quotes;
            // Don't include quotes in the field
        } else if (c == ',' && !in_quotes) {
            fields[field_count][field_pos] = '\0';
            field_count++;
            field_pos = 0;
        } else {
            fields[field_count][field_pos++] = c;
            if (field_pos >= MAX_FIELD_LENGTH - 1) {
                fprintf(stderr, "Error: Field too long\n");
                return -1;
            }
        }
        pos++;
    }
    
    // Add last field
    fields[field_count][field_pos] = '\0';
    field_count++;
    
    return field_count;
}

// Process CSV line and convert materials JSON to one-hot encoding
void process_line(const char *input_line, char *output_line) {
    char fields[20][MAX_FIELD_LENGTH];
    int num_fields = parse_csv_line(input_line, fields, 20);
    
    if (num_fields < 12) {
        strcpy(output_line, input_line);
        return;
    }
    
    // Parse materials (field 5, 0-indexed)
    double percentages[NUM_MATERIALS];
    parse_materials(fields[5], percentages);
    
    // Build output line: fields 0-4, 6-11, then all material percentages
    char temp[MAX_LINE_LENGTH];
    int offset = 0;
    
    // Add fields 0-4 (before materials)
    for (int i = 0; i < 5; i++) {
        offset += snprintf(temp + offset, MAX_LINE_LENGTH - offset, "%s,", fields[i]);
    }
    
    // Add fields 6-11 (after materials)
    for (int i = 6; i < 12; i++) {
        offset += snprintf(temp + offset, MAX_LINE_LENGTH - offset, "%s,", fields[i]);
    }
    
    // Add all material percentages
    for (int i = 0; i < NUM_MATERIALS; i++) {
        if (i < NUM_MATERIALS - 1) {
            offset += snprintf(temp + offset, MAX_LINE_LENGTH - offset, "%.6f,", percentages[i]);
        } else {
            offset += snprintf(temp + offset, MAX_LINE_LENGTH - offset, "%.6f", percentages[i]);
        }
    }
    
    strcpy(output_line, temp);
}

// Read CSV file and process lines
int read_and_process_csv(const char *filename, Dataset *ds, char **header) {
    FILE *file = fopen(filename, "r");
    if (!file) {
        fprintf(stderr, "Error: Could not open file '%s'\n", filename);
        return 0;
    }
    
    char line[MAX_LINE_LENGTH];
    int first_line = 1;
    
    while (fgets(line, sizeof(line), file)) {
        // Remove trailing newline
        line[strcspn(line, "\n")] = 0;
        line[strcspn(line, "\r")] = 0;
        
        if (first_line) {
            // Build new header with material columns
            char new_header[MAX_LINE_LENGTH];
            int offset = snprintf(new_header, MAX_LINE_LENGTH,
                "product_name,gender,parent_category,category,manufacturer_country,"
                "weight_kg,total_distance_km,carbon_material,carbon_transport,carbon_total,water_total");
            
            // Add all material column names
            for (int i = 0; i < NUM_MATERIALS; i++) {
                offset += snprintf(new_header + offset, MAX_LINE_LENGTH - offset, ",%s", MATERIALS[i]);
            }
            
            *header = strdup(new_header);
            first_line = 0;
        } else {
            char processed_line[MAX_LINE_LENGTH];
            process_line(line, processed_line);
            add_line(ds, processed_line);
        }
    }
    
    fclose(file);
    return 1;
}

// Write dataset to CSV file
int write_csv(const char *filename, const char *header, Dataset *ds, size_t start, size_t end) {
    FILE *file = fopen(filename, "w");
    if (!file) {
        fprintf(stderr, "Error: Could not open file '%s' for writing\n", filename);
        return 0;
    }
    
    // Write header
    fprintf(file, "%s\n", header);
    
    // Write data rows
    for (size_t i = start; i < end && i < ds->count; i++) {
        fprintf(file, "%s\n", ds->lines[i]);
    }
    
    fclose(file);
    return 1;
}

int main() {
    const char *input_file = "../data_calculations/output/Product_data_with_footprints.csv";
    const char *train_file = "output/train.csv";
    const char *validate_file = "output/validate.csv";
    
    Dataset dataset;
    char *header = NULL;
    
    printf("Data Splitter - Shuffling and splitting dataset with one-hot encoding\n");
    printf("========================================================================\n\n");
    
    // Initialize dataset
    init_dataset(&dataset);
    
    // Read and process input CSV
    printf("Reading and preprocessing input file: %s\n", input_file);
    if (!read_and_process_csv(input_file, &dataset, &header)) {
        free_dataset(&dataset);
        if (header) free(header);
        return 1;
    }
    printf("Loaded %zu data rows\n", dataset.count);
    printf("Converted materials JSON to %d one-hot encoded columns\n\n", NUM_MATERIALS);
    
    // Shuffle dataset
    printf("Shuffling dataset with seed %d\n", RANDOM_SEED);
    shuffle_dataset(&dataset, RANDOM_SEED);
    
    // Calculate split point
    size_t train_size = (size_t)(dataset.count * TRAIN_RATIO);
    size_t validate_size = dataset.count - train_size;
    
    printf("Split: %.0f%% train (%zu rows), %.0f%% validate (%zu rows)\n\n",
           TRAIN_RATIO * 100, train_size, 
           (1.0 - TRAIN_RATIO) * 100, validate_size);
    
    // Write training set
    printf("Writing training set: %s\n", train_file);
    if (!write_csv(train_file, header, &dataset, 0, train_size)) {
        free_dataset(&dataset);
        free(header);
        return 1;
    }
    
    // Write validation set
    printf("Writing validation set: %s\n", validate_file);
    if (!write_csv(validate_file, header, &dataset, train_size, dataset.count)) {
        free_dataset(&dataset);
        free(header);
        return 1;
    }
    
    printf("\nData splitting completed successfully!\n");
    printf("Total columns in output: %d (11 original + %d materials)\n", 11 + NUM_MATERIALS, NUM_MATERIALS);
    
    // Cleanup
    free_dataset(&dataset);
    free(header);
    
    return 0;
}
