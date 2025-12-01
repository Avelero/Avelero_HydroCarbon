/**
 * @file csv_parser.c
 * @brief Implementation of unified CSV parsing utilities
 *
 * @author Moussa Ouallaf
 * @date 2025-12-01
 * @version 2.0
 */

#define _POSIX_C_SOURCE 200809L
#include "csv_parser.h"
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

char* trim_whitespace(char *str) {
    if (str == NULL) {
        return NULL;
    }

    /* Trim leading whitespace */
    while (isspace((unsigned char)*str)) {
        str++;
    }

    /* Handle empty string */
    if (*str == '\0') {
        return str;
    }

    /* Trim trailing whitespace */
    char *end = str + strlen(str) - 1;
    while (end > str && isspace((unsigned char)*end)) {
        end--;
    }

    /* Null-terminate after last non-whitespace character */
    *(end + 1) = '\0';

    return str;
}

CSVRow* parse_csv_line(const char *line) {
    if (line == NULL) {
        return NULL;
    }

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
            /* Check for escaped quote (two consecutive quotes) */
            if (in_quotes && line[i + 1] == '"') {
                buffer[buffer_pos++] = '"';
                i += 2;
                continue;
            }
            /* Toggle quote state */
            in_quotes = !in_quotes;
            i++;
            continue;
        }
        
        if (c == ',' && !in_quotes) {
            /* End of field */
            buffer[buffer_pos] = '\0';
            row->fields[row->field_count] = strdup(buffer);
            row->field_count++;
            buffer_pos = 0;
            i++;
            continue;
        }
        
        /* Add character to buffer */
        if (buffer_pos < MAX_FIELD_LENGTH - 1) {
            buffer[buffer_pos++] = c;
        }
        i++;
    }
    
    /* Add last field */
    buffer[buffer_pos] = '\0';
    row->fields[row->field_count] = strdup(buffer);
    row->field_count++;
    
    return row;
}

void free_csv_row(CSVRow *row) {
    if (!row) return;
    
    for (int i = 0; i < row->field_count; i++) {
        free(row->fields[i]);
    }
    free(row->fields);
    free(row);
}

char* get_field(CSVRow *row, int index) {
    if (!row || index < 0 || index >= row->field_count) {
        return NULL;
    }
    return row->fields[index];
}

int skip_header(FILE *fp) {
    char line[MAX_LINE_LENGTH];
    if (fgets(line, sizeof(line), fp)) {
        return 0; /* Success */
    }
    return -1; /* Error */
}
