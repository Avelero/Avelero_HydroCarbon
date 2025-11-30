# Carbon Footprint Calculator Makefile

CC = gcc
CFLAGS = -Wall -Wextra -O2 -std=c11
INCLUDES = -I. -Iutils

# Directories
BUILD_DIR = build
UTILS_DIR = utils
OUTPUT_DIR = output

# Shared object files in build directory
SHARED_OBJS = $(BUILD_DIR)/csv_parser.o $(BUILD_DIR)/json_parser.o

# Executables
MATERIAL_CALC = $(BUILD_DIR)/material_calculator
TRANSPORT_CALC = $(BUILD_DIR)/transport_calculator
HANDLER = $(BUILD_DIR)/carbon_footprint_handler

# Targets
all: $(MATERIAL_CALC) $(TRANSPORT_CALC) $(HANDLER)

# Shared utilities
$(BUILD_DIR)/csv_parser.o: $(UTILS_DIR)/csv_parser.c $(UTILS_DIR)/csv_parser.h | $(BUILD_DIR)
	$(CC) $(CFLAGS) $(INCLUDES) -c $(UTILS_DIR)/csv_parser.c -o $(BUILD_DIR)/csv_parser.o

$(BUILD_DIR)/json_parser.o: $(UTILS_DIR)/json_parser.c $(UTILS_DIR)/json_parser.h | $(BUILD_DIR)
	$(CC) $(CFLAGS) $(INCLUDES) -c $(UTILS_DIR)/json_parser.c -o $(BUILD_DIR)/json_parser.o

# Material calculator
$(MATERIAL_CALC): material/material_calculator.c $(SHARED_OBJS) | $(BUILD_DIR)
	$(CC) $(CFLAGS) $(INCLUDES) material/material_calculator.c $(SHARED_OBJS) -o $(MATERIAL_CALC)

# Transport calculator
$(TRANSPORT_CALC): transport/transport_calculator.c $(BUILD_DIR)/csv_parser.o | $(BUILD_DIR)
	$(CC) $(CFLAGS) $(INCLUDES) transport/transport_calculator.c $(BUILD_DIR)/csv_parser.o -o $(TRANSPORT_CALC)

# Main handler
$(HANDLER): carbon_footprint_handler.c $(BUILD_DIR)/csv_parser.o | $(BUILD_DIR)
	$(CC) $(CFLAGS) $(INCLUDES) carbon_footprint_handler.c $(BUILD_DIR)/csv_parser.o -o $(HANDLER)

# Create build directory
$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

# Create output directory
$(OUTPUT_DIR):
	mkdir -p $(OUTPUT_DIR)

# Clean build artifacts
clean:
	rm -rf $(BUILD_DIR)
	rm -f $(OUTPUT_DIR)/material_footprints.csv
	rm -f $(OUTPUT_DIR)/transport_footprints.csv

# Rebuild everything
rebuild: clean all

# Run the complete pipeline
run: all | $(OUTPUT_DIR)
	$(HANDLER)

.PHONY: all clean rebuild run
