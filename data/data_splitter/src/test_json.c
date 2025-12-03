#define _POSIX_C_SOURCE 200809L

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

// Test JSON parsing
int find_material_index(const char *material_name) {
    const char *materials[] = {"viscose", "polyester_virgin", "cotton_conventional"};
    for (int i = 0; i < 3; i++) {
        if (strcmp(material_name, materials[i]) == 0) {
            return i;
        }
    }
    return -1;
}

void parse_materials(const char *json_str, double *percentages) {
    printf("Input JSON: [%s]\n", json_str);
    
    for (int i = 0; i < 3; i++) {
        percentages[i] = 0.0;
    }
    
    if (!json_str || strlen(json_str) < 2) {
        printf("JSON string too short or NULL\n");
        return;
    }
    
    char *str_copy = strdup(json_str);
    printf("After strdup: [%s]\n", str_copy);
    
    char *start = str_copy;
    while (*start && (*start == '"' || *start == '{'  || isspace(*start))) start++;
    printf("After removing leading: [%s]\n", start);
    
    char *end = start + strlen(start) - 1;
    while (end > start && (*end == '"' || *end == '}' || isspace(*end))) {
        *end = '\0';
        end--;
    }
    printf("After removing trailing: [%s]\n", start);
    
    char *token = start;
    while (*token) {
        char *mat_start = strstr(token, "\"\"");
        if (!mat_start) {
            printf("No more material markers found\n");
            break;
        }
        mat_start += 2;
        printf("Found material start at: [%s]\n", mat_start);
        
        char *mat_end = strstr(mat_start, "\"\"");
        if (!mat_end) {
            printf("No material end found\n");
            break;
        }
        
        size_t mat_len = mat_end - mat_start;
        char material[128];
        strncpy(material, mat_start, mat_len);
        material[mat_len] = '\0';
        printf("Extracted material: [%s]\n", material);
        
        char *colon = strchr(mat_end, ':');
        if (!colon) {
            printf("No colon found\n");
            break;
        }
        colon++;
        
        double percentage = atof(colon);
        printf("Extracted percentage: %.6f\n", percentage);
        
        int idx = find_material_index(material);
        printf("Material index: %d\n", idx);
        if (idx >= 0) {
            percentages[idx] = percentage;
        }
        
        token = strchr(colon, ',');
        if (!token) break;
        token++;
    }
    
    free(str_copy);
}

int main() {
    const char *test = "{\"\"viscose\"\":0.72,\"\"polyester_virgin\"\":0.28}";
    double percentages[3];
    
    printf("Test 1: Direct JSON\n");
    printf("===================\n");
    parse_materials(test, percentages);
    printf("\nResults: viscose=%.2f, polyester_virgin=%.2f\n\n", percentages[0], percentages[1]);
    
    return 0;
}
