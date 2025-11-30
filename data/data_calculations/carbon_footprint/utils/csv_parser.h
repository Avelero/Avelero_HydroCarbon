#ifndef CSV_PARSER_H
#define CSV_PARSER_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_LINE_LENGTH 8192
#define MAX_FIELDS 20
#define MAX_FIELD_LENGTH 4096

// CSV field structure
typedef struct {
    char **fields;
    int field_count;
} CSVRow;

// Function declarations
CSVRow* parse_csv_line(const char *line);
void free_csv_row(CSVRow *row);
char* get_field(CSVRow *row, int index);
int skip_header(FILE *fp);

#endif // CSV_PARSER_H
