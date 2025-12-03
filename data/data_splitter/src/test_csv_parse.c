#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <string.h>

#define MAX_FIELD_LENGTH 2048

int parse_csv_line(const char *line, char fields[][MAX_FIELD_LENGTH], int max_fields) {
    int field_count = 0;
    int pos = 0;
    int in_quotes = 0;
    int field_pos = 0;
    
    while (line[pos] && field_count < max_fields) {
        char c = line[pos];
        
        if (c == '"') {
            in_quotes = !in_quotes;
        } else if (c == ',' && !in_quotes) {
            fields[field_count][field_pos] = '\0';
            field_count++;
            field_pos = 0;
        } else {
            fields[field_count][field_pos++] = c;
        }
        pos++;
    }
    
    fields[field_count][field_pos] = '\0';
    field_count++;
    
    return field_count;
}

int main() {
    const char *line = "Drawstring Hem Crop Hoodie,Female,Tops,Crop Tops,WS,\"{\"\"polyester_virgin\"\":1.0}\",0.296,14031.53,0.607196,0.070728,0.677924,17.760000";
    char fields[20][MAX_FIELD_LENGTH];
    
    int n = parse_csv_line(line, fields, 20);
    printf("Parsed %d fields:\n", n);
    for (int i = 0; i < n; i++) {
        printf("Field %d: [%s]\n", i, fields[i]);
    }
    
    printf("\nField 5 (materials): [%s]\n", fields[5]);
    
    return 0;
}
