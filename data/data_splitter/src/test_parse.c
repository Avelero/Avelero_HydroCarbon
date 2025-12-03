#define _POSIX_C_SOURCE 20080 9L

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Test parsing a single line
int main() {
    const char *test_line = "Boho Floral Print Maxi Skirt,Female,Bottoms,Maxi Skirts,MQ,\"{\"\"viscose\"\":0.72,\"\"polyester_virgin\"\":0.28}\",0.587,14321.63,0.971118,0.142511,1.113629,1277.781600";
    
    char fields[20][2048];
    int field_count = 0;
    int pos = 0;
    int in_quotes = 0;
    int field_pos = 0;
    
    while (test_line[pos] && field_count < 20) {
        char c = test_line[pos];
        
        if (c == '"') {
            in_quotes = !in_quotes;
            // DON'T include quotes in the field
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
    
    printf("Parsed %d fields:\n", field_count);
    for (int i = 0; i < field_count; i++) {
        printf("Field %d: [%s]\n", i, fields[i]);
    }
    
    return 0;
}
