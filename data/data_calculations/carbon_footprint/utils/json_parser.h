#ifndef JSON_PARSER_H
#define JSON_PARSER_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_MATERIALS 20

// Material-percentage pair structure
typedef struct {
    char material_name[128];
    double percentage;
} MaterialEntry;

// Material composition structure
typedef struct {
    MaterialEntry *entries;
    int count;
} MaterialComposition;

// Function declarations
MaterialComposition* parse_material_json(const char *json_str);
void free_material_composition(MaterialComposition *comp);

#endif // JSON_PARSER_H
