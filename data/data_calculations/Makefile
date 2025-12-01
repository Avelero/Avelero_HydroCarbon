# Unified Footprint Calculator - Makefile
# Author: Moussa Ouallaf
# Date: 2025-12-01
# Version: 2.0
#
# This Makefile builds the complete footprint calculation system including:
# - Carbon footprint (material + transport)
# - Water footprint (material)
# - Unified handler that produces a single output file

# Compiler and flags
CC = gcc
CFLAGS = -Wall -Wextra -Werror -pedantic -std=c11 -O2
INCLUDES = -Iinclude
LDFLAGS = -lm

# Directories
SRC_DIR = src
INC_DIR = include
BUILD_DIR = build
BIN_DIR = bin
OUTPUT_DIR = output

# Source files
UTILS_SRC = $(SRC_DIR)/utils/csv_parser.c $(SRC_DIR)/utils/json_parser.c
CARBON_MATERIAL_SRC = $(SRC_DIR)/carbon/material/material_carbon.c
CARBON_TRANSPORT_SRC = $(wildcard $(SRC_DIR)/carbon/transport/*.c)
WATER_MATERIAL_SRC = $(SRC_DIR)/water/material/material_water.c
HANDLER_SRC = $(SRC_DIR)/footprint_calculator.c

# Object files
UTILS_OBJ = $(BUILD_DIR)/utils/csv_parser.o $(BUILD_DIR)/utils/json_parser.o
CARBON_MATERIAL_OBJ = $(BUILD_DIR)/carbon/material/material_carbon.o
CARBON_TRANSPORT_OBJ = $(BUILD_DIR)/carbon/transport/emission_data.o \
                       $(BUILD_DIR)/carbon/transport/modal_split.o \
                       $(BUILD_DIR)/carbon/transport/emissions_calc.o \
                       $(BUILD_DIR)/carbon/transport/transport_calculator.o
WATER_MATERIAL_OBJ = $(BUILD_DIR)/water/material/material_water.o
HANDLER_OBJ = $(BUILD_DIR)/footprint_calculator.o

ALL_OBJ = $(UTILS_OBJ) $(CARBON_MATERIAL_OBJ) $(CARBON_TRANSPORT_OBJ) $(WATER_MATERIAL_OBJ) $(HANDLER_OBJ)

# Main executable
CALCULATOR = $(BUILD_DIR)/footprint_calculator

# Default target
.PHONY: all
all: directories $(CALCULATOR)

# Create necessary directories
.PHONY: directories
directories:
	@mkdir -p $(BUILD_DIR)/utils
	@mkdir -p $(BUILD_DIR)/carbon/material
	@mkdir -p $(BUILD_DIR)/carbon/transport
	@mkdir -p $(BUILD_DIR)/water/material
	@mkdir -p $(BIN_DIR)
	@mkdir -p $(OUTPUT_DIR)

# ============================================================================
# Utility Objects (shared by all modules)
# ============================================================================

$(BUILD_DIR)/utils/csv_parser.o: $(SRC_DIR)/utils/csv_parser.c $(INC_DIR)/csv_parser.h
	@echo "Compiling: csv_parser.c"
	$(CC) $(CFLAGS) $(INCLUDES) -c $< -o $@

$(BUILD_DIR)/utils/json_parser.o: $(SRC_DIR)/utils/json_parser.c $(INC_DIR)/json_parser.h
	@echo "Compiling: json_parser.c"
	$(CC) $(CFLAGS) $(INCLUDES) -c $< -o $@

# ============================================================================
# Carbon Material Calculator
# ============================================================================

$(BUILD_DIR)/carbon/material/material_carbon.o: $(SRC_DIR)/carbon/material/material_carbon.c $(INC_DIR)/material_carbon.h
	@echo "Compiling: material_carbon.c"
	$(CC) $(CFLAGS) $(INCLUDES) -c $< -o $@

# ============================================================================
# Carbon Transport Calculator
# ============================================================================

$(BUILD_DIR)/carbon/transport/emission_data.o: $(SRC_DIR)/carbon/transport/emission_data.c $(INC_DIR)/emission_data.h
	@echo "Compiling: emission_data.c"
	$(CC) $(CFLAGS) $(INCLUDES) -c $< -o $@

$(BUILD_DIR)/carbon/transport/modal_split.o: $(SRC_DIR)/carbon/transport/modal_split.c $(INC_DIR)/modal_split.h
	@echo "Compiling: modal_split.c"
	$(CC) $(CFLAGS) $(INCLUDES) -c $< -o $@

$(BUILD_DIR)/carbon/transport/emissions_calc.o: $(SRC_DIR)/carbon/transport/emissions_calc.c $(INC_DIR)/emissions_calc.h
	@echo "Compiling: emissions_calc.c"
	$(CC) $(CFLAGS) $(INCLUDES) -c $< -o $@

$(BUILD_DIR)/carbon/transport/transport_calculator.o: $(SRC_DIR)/carbon/transport/transport_calculator.c $(INC_DIR)/transport_calculator.h
	@echo "Compiling: transport_calculator.c"
	$(CC) $(CFLAGS) $(INCLUDES) -c $< -o $@

# ============================================================================
# Water Material Calculator
# ============================================================================

$(BUILD_DIR)/water/material/material_water.o: $(SRC_DIR)/water/material/material_water.c $(INC_DIR)/material_water.h
	@echo "Compiling: material_water.c"
	$(CC) $(CFLAGS) $(INCLUDES) -c $< -o $@

# ============================================================================
# Main Footprint Calculator
# ============================================================================

$(BUILD_DIR)/footprint_calculator.o: $(SRC_DIR)/footprint_calculator.c
	@echo "Compiling: footprint_calculator.c"
	$(CC) $(CFLAGS) $(INCLUDES) -c $< -o $@

$(CALCULATOR): $(ALL_OBJ)
	@echo "Linking: footprint_calculator"
	$(CC) $(CFLAGS) $^ $(LDFLAGS) -o $@
	@echo "✓ Footprint calculator built: $@"

# ============================================================================
# Run and Test Targets
# ============================================================================

# Run the complete footprint calculation
.PHONY: run
run: all
	@echo "Running footprint calculator..."
	@cd $(BUILD_DIR) && ./footprint_calculator

# Clean all build artifacts
.PHONY: clean
clean:
	rm -rf $(BUILD_DIR)
	rm -rf $(BIN_DIR)

# Clean and rebuild
.PHONY: rebuild
rebuild: clean all

# Show help
.PHONY: help
help:
	@echo "Unified Footprint Calculator - Available targets:"
	@echo ""
	@echo "  all      - Build the footprint calculator (default)"
	@echo "  run      - Build and execute the calculator"
	@echo "  clean    - Remove all build artifacts"
	@echo "  rebuild  - Clean and rebuild everything"
	@echo "  help     - Show this help message"
	@echo ""
	@echo "Output:"
	@echo "  The calculator produces: output/Product_data_with_footprints.csv"
	@echo "  This file contains all original product data plus:"
	@echo "    - carbon_material (material carbon footprint in kgCO2e)"
	@echo "    - carbon_transport (transport carbon footprint in kgCO2e)"
	@echo "    - carbon_total (total carbon footprint in kgCO2e)"
	@echo "    - water_total (total water footprint in liters)"
