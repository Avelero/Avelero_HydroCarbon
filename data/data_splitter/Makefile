# Data Splitter Makefile
# Compiles and runs the data splitter program

CC = gcc
CFLAGS = -Wall -Wextra -O2 -std=c99
SRC_DIR = src
BIN_DIR = bin
OUTPUT_DIR = output

# Source files
SOURCES = $(SRC_DIR)/split_data.c
TARGET = $(BIN_DIR)/split_data

# Default target: compile the program
all: $(TARGET)

# Compile the C program
$(TARGET): $(SOURCES) | $(BIN_DIR)
	@echo "Compiling data splitter..."
	$(CC) $(CFLAGS) $(SOURCES) -o $(TARGET)
	@echo "Compilation successful!"

# Create bin directory if it doesn't exist
$(BIN_DIR):
	mkdir -p $(BIN_DIR)

# Create output directory if it doesn't exist
$(OUTPUT_DIR):
	mkdir -p $(OUTPUT_DIR)

# Run the program
run: $(TARGET) | $(OUTPUT_DIR)
	@echo ""
	@echo "Running data splitter..."
	@echo ""
	./$(TARGET)

# Build and run in one step
build-run: all run

# Clean generated files
clean:
	@echo "Cleaning up..."
	rm -f $(TARGET)
	rm -f $(OUTPUT_DIR)/*.csv
	@echo "Clean complete!"

# Clean everything including directories
distclean: clean
	rm -rf $(BIN_DIR) $(OUTPUT_DIR)

# Show help
help:
	@echo "Data Splitter Makefile"
	@echo "======================"
	@echo ""
	@echo "Targets:"
	@echo "  make          - Compile the program"
	@echo "  make run      - Run the compiled program"
	@echo "  make build-run- Build and run in one step"
	@echo "  make clean    - Remove generated files"
	@echo "  make distclean- Remove all generated files and directories"
	@echo "  make help     - Show this help message"

.PHONY: all run build-run clean distclean help
