/**
 * @file csv_parser.h
 * @brief Unified CSV file parsing utilities
 *
 * This module provides utilities for parsing CSV files containing
 * product data, emission factors, and utility parameters.
 *
 * @author Moussa Ouallaf
 * @date 2025-12-01
 * @version 2.0
 */

#ifndef CSV_PARSER_H
#define CSV_PARSER_H

#include <stdio.h>
#include <stdbool.h>

/* Maximum line length in CSV file */
#define MAX_LINE_LENGTH 8192
#define MAX_CSV_LINE MAX_LINE_LENGTH

/* Maximum number of fields per CSV row */
#define MAX_FIELDS 32
#define MAX_CSV_FIELDS MAX_FIELDS

/* Maximum field length */
#define MAX_FIELD_LENGTH 4096

/**
 * @brief Parsed CSV row structure
 *
 * Contains an array of string pointers to each field in a CSV row.
 * Memory is managed internally and must be freed using free_csv_row().
 */
typedef struct {
    char **fields;           /**< Array of field strings */
    int field_count;         /**< Number of fields in this row */
} CSVRow;

/**
 * @brief Parse a single CSV line into fields
 *
 * Splits a CSV line on commas, handling quoted fields that may contain
 * commas and escaped quotes.
 *
 * @param[in] line CSV line string (null-terminated)
 * @return Pointer to allocated CSVRow, or NULL on error
 */
CSVRow* parse_csv_line(const char *line);

/**
 * @brief Free memory allocated for CSV row
 *
 * @param[in,out] row CSVRow to free
 */
void free_csv_row(CSVRow *row);

/**
 * @brief Get field value by index
 *
 * @param[in] row Parsed CSV row
 * @param[in] index Field index (0-based)
 * @return Pointer to field string, or NULL if index out of bounds
 */
char* get_field(CSVRow *row, int index);

/**
 * @brief Skip CSV header line
 *
 * @param[in,out] fp File pointer to CSV file
 * @return 0 on success, -1 on error
 */
int skip_header(FILE *fp);

/**
 * @brief Trim whitespace from string
 *
 * @param[in,out] str String to trim
 * @return Pointer to trimmed string
 */
char* trim_whitespace(char *str);

#endif /* CSV_PARSER_H */
