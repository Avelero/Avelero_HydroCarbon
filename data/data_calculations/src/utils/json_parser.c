/**
 * @file json_parser.c
 * @brief Implementation of JSON parsing utilities for material composition
 *
 * @author Moussa Ouallaf
 * @date 2025-12-01
 * @version 2.0
 */

#include "json_parser.h"
#include <ctype.h>

/* Helper function to skip whitespace */
static const char* skip_whitespace(const char *str) {
    while (*str && isspace(*str)) str++;
    return str;
}

/**
 * @brief Extract string between quotes from JSON
 */
static char* extract_string(const char **str) {
    const char *p = *str;
    p = skip_whitespace(p);

    static char buffer[256];
    int i = 0;
    int quote_count = 0;

    /* Count consecutive quotes to determine escaping level */
    const char *temp = p;
    while (*temp == '"' && quote_count < 4) {
        quote_count++;
        temp++;
    }

    /* Determine format based on quote count */
    if (quote_count >= 2) {
        /* Double-escaped format: ""material"" */
        p += 2;
    } else if (quote_count == 1) {
        /* Standard format: "material" */
        p++;
    } else {
        return NULL;
    }

    /* Extract the material name until we hit closing quotes */
    while (*p && i < 255) {
        if (quote_count >= 2) {
            if (*p == '"' && *(p + 1) == '"') {
                p += 2;
                break;
            }
        } else {
            if (*p == '"') {
                p++;
                break;
            }
        }

        buffer[i++] = *p++;
    }
    buffer[i] = '\0';

    *str = p;
    return buffer;
}

/* Helper function to extract number */
static double extract_number(const char **str) {
    const char *p = *str;
    p = skip_whitespace(p);
    
    char *endptr;
    double value = strtod(p, &endptr);
    
    if (endptr == p) return 0.0;
    
    *str = endptr;
    return value;
}

MaterialComposition* parse_material_json(const char *json_str) {
    if (!json_str) return NULL;

    MaterialComposition *comp = (MaterialComposition*)malloc(sizeof(MaterialComposition));
    if (!comp) return NULL;

    comp->entries = (MaterialEntry*)malloc(MAX_MATERIALS * sizeof(MaterialEntry));
    if (!comp->entries) {
        free(comp);
        return NULL;
    }
    comp->count = 0;

    const char *p = json_str;
    p = skip_whitespace(p);

    /* Skip opening brace */
    if (*p == '{') p++;

    while (*p && *p != '}' && comp->count < MAX_MATERIALS) {
        p = skip_whitespace(p);

        if (*p == '}') break;

        /* Extract material name */
        char *material = extract_string(&p);
        if (!material || strlen(material) == 0) {
            while (*p && *p != ',' && *p != '}') p++;
            if (*p == ',') p++;
            continue;
        }

        /* Copy material name */
        strncpy(comp->entries[comp->count].material_name, material, 127);
        comp->entries[comp->count].material_name[127] = '\0';

        p = skip_whitespace(p);

        /* Skip colon */
        if (*p == ':') p++;

        /* Extract percentage */
        double percentage = extract_number(&p);
        comp->entries[comp->count].percentage = percentage;

        if (strlen(comp->entries[comp->count].material_name) > 0) {
            comp->count++;
        }

        p = skip_whitespace(p);

        /* Skip comma */
        if (*p == ',') p++;
    }

    return comp;
}

void free_material_composition(MaterialComposition *comp) {
    if (!comp) return;
    free(comp->entries);
    free(comp);
}
