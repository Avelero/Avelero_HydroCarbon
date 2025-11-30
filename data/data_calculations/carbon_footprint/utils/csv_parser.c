#define _POSIX_C_SOURCE 200809L
#include "csv_parser.h"
#include <ctype.h>

// Parse a CSV line handling quoted fields and escaped quotes
CSVRow* parse_csv_line(const char *line) {
    CSVRow *row = (CSVRow*)malloc(sizeof(CSVRow));
    if (!row) return NULL;
    
    row->fields = (char**)malloc(MAX_FIELDS * sizeof(char*));
    if (!row->fields) {
        free(row);
        return NULL;
    }
    
    row->field_count = 0;
    
    char buffer[MAX_FIELD_LENGTH];
    int buffer_pos = 0;
    int in_quotes = 0;
    int i = 0;
    
    while (line[i] != '\0' && line[i] != '\n' && line[i] != '\r') {
        char c = line[i];
        
        if (c == '"') {
            // Check for escaped quote (two consecutive quotes)
            if (in_quotes && line[i + 1] == '"') {
                buffer[buffer_pos++] = '"';
                i += 2;
                continue;
            }
            // Toggle quote state
            in_quotes = !in_quotes;
            i++;
            continue;
        }
        
        if (c == ',' && !in_quotes) {
            // End of field
            buffer[buffer_pos] = '\0';
            row->fields[row->field_count] = strdup(buffer);
            row->field_count++;
            buffer_pos = 0;
            i++;
            continue;
        }
        
        // Add character to buffer
        if (buffer_pos < MAX_FIELD_LENGTH - 1) {
            buffer[buffer_pos++] = c;
        }
        i++;
    }
    
    // Add last field
    buffer[buffer_pos] = '\0';
    row->fields[row->field_count] = strdup(buffer);
    row->field_count++;
    
    return row;
}

// Free memory allocated for CSV row
void free_csv_row(CSVRow *row) {
    if (!row) return;
    
    for (int i = 0; i < row->field_count; i++) {
        free(row->fields[i]);
    }
    free(row->fields);
    free(row);
}

// Get field by index
char* get_field(CSVRow *row, int index) {
    if (!row || index < 0 || index >= row->field_count) {
        return NULL;
    }
    return row->fields[index];
}

// Skip header line in CSV file
int skip_header(FILE *fp) {
    char line[MAX_LINE_LENGTH];
    if (fgets(line, sizeof(line), fp)) {
        return 0; // Success
    }
    return -1; // Error
}
