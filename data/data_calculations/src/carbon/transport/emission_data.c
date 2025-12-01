/**
 * @file emission_data.c
 * @brief Implementation of emission data loading and management
 *
 * @author Moussa Ouallaf
 * @date 2025-12-01
 * @version 2.0
 */

#define _POSIX_C_SOURCE 200809L

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include "emission_data.h"
#include "csv_parser.h"

TransportMode map_mode_name(const char *mode_name) {
    if (mode_name == NULL) {
        return MODE_ROAD;
    }

    if (strcasecmp(mode_name, "road") == 0) {
        return MODE_ROAD;
    } else if (strcasecmp(mode_name, "rail") == 0) {
        return MODE_RAIL;
    } else if (strcasecmp(mode_name, "inland_waterway") == 0) {
        return MODE_INLAND_WATERWAY;
    } else if (strcasecmp(mode_name, "sea") == 0) {
        return MODE_SEA;
    } else if (strcasecmp(mode_name, "air") == 0) {
        return MODE_AIR;
    }

    return MODE_ROAD;
}

StatusCode load_emission_factors(EmissionData *data, const char *filepath) {
    if (data == NULL || filepath == NULL) {
        return STATUS_INVALID_INPUT;
    }

    FILE *fp = fopen(filepath, "r");
    if (fp == NULL) {
        fprintf(stderr, "Error: Cannot open emission factors file: %s\n", filepath);
        return STATUS_FILE_ERROR;
    }

    skip_header(fp);

    char line[MAX_LINE_LENGTH];
    int modes_loaded = 0;

    while (fgets(line, sizeof(line), fp) != NULL) {
        CSVRow *row = parse_csv_line(line);
        if (row == NULL || row->field_count < 2) {
            free_csv_row(row);
            continue;
        }

        const char *mode_name = get_field(row, 0);
        const char *ef_str = get_field(row, 1);

        if (mode_name == NULL || ef_str == NULL) {
            free_csv_row(row);
            continue;
        }

        TransportMode mode = map_mode_name(mode_name);
        double ef = atof(ef_str);

        if (ef <= 0.0) {
            free_csv_row(row);
            continue;
        }

        data->emission_factors[mode].mode = mode;
        strncpy(data->emission_factors[mode].name, mode_name, MAX_MODE_NAME_LEN - 1);
        data->emission_factors[mode].name[MAX_MODE_NAME_LEN - 1] = '\0';
        data->emission_factors[mode].ef_gco2e_per_tkm = ef;

        modes_loaded++;
        free_csv_row(row);
    }

    fclose(fp);
    return STATUS_SUCCESS;
}

StatusCode load_utility_parameters(EmissionData *data, const char *filepath) {
    if (data == NULL || filepath == NULL) {
        return STATUS_INVALID_INPUT;
    }

    FILE *fp = fopen(filepath, "r");
    if (fp == NULL) {
        fprintf(stderr, "Error: Cannot open utility parameters file: %s\n", filepath);
        return STATUS_FILE_ERROR;
    }

    skip_header(fp);

    char line[MAX_LINE_LENGTH];

    for (int i = 0; i < NUM_TRANSPORT_MODES; i++) {
        data->utility_params[i].mode = i;
        data->utility_params[i].beta0 = 0.0;
        data->utility_params[i].beta1 = 0.0;
    }

    while (fgets(line, sizeof(line), fp) != NULL) {
        CSVRow *row = parse_csv_line(line);
        if (row == NULL || row->field_count < 3) {
            free_csv_row(row);
            continue;
        }

        const char *mode_name = get_field(row, 0);
        const char *param_name = get_field(row, 1);
        const char *value_str = get_field(row, 2);

        if (mode_name == NULL || param_name == NULL || value_str == NULL) {
            free_csv_row(row);
            continue;
        }

        TransportMode mode = map_mode_name(mode_name);
        double value = atof(value_str);

        if (data->utility_params[mode].name[0] == '\0') {
            strncpy(data->utility_params[mode].name, mode_name, MAX_MODE_NAME_LEN - 1);
            data->utility_params[mode].name[MAX_MODE_NAME_LEN - 1] = '\0';
        }

        if (strcasecmp(param_name, "beta0") == 0) {
            data->utility_params[mode].beta0 = value;
        } else if (strcasecmp(param_name, "beta1") == 0) {
            data->utility_params[mode].beta1 = value;
        }

        free_csv_row(row);
    }

    fclose(fp);
    return STATUS_SUCCESS;
}

bool validate_emission_data(const EmissionData *data) {
    if (data == NULL) {
        return false;
    }

    for (int i = 0; i < NUM_TRANSPORT_MODES; i++) {
        if (data->emission_factors[i].ef_gco2e_per_tkm <= 0.0) {
            fprintf(stderr, "Validation error: Invalid emission factor for mode %d\n", i);
            return false;
        }
    }

    return true;
}

StatusCode init_emission_data(
    EmissionData *data,
    const char *emission_factors_file,
    const char *utility_params_file
) {
    if (data == NULL || emission_factors_file == NULL || utility_params_file == NULL) {
        return STATUS_INVALID_INPUT;
    }

    memset(data, 0, sizeof(EmissionData));
    data->initialized = false;

    StatusCode status = load_emission_factors(data, emission_factors_file);
    if (status != STATUS_SUCCESS) {
        fprintf(stderr, "Error: Failed to load emission factors\n");
        return status;
    }

    status = load_utility_parameters(data, utility_params_file);
    if (status != STATUS_SUCCESS) {
        fprintf(stderr, "Error: Failed to load utility parameters\n");
        return status;
    }

    if (!validate_emission_data(data)) {
        fprintf(stderr, "Error: Emission data validation failed\n");
        return STATUS_ERROR;
    }

    data->initialized = true;
    return STATUS_SUCCESS;
}
