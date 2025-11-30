# Carbon Footprint Calculator

C-based carbon footprint calculator for processing product data and calculating material, transport, and total carbon footprints.

## Directory Structure

```
carbon_footprint/
├── build/                          # Compiled executables and object files
│   ├── carbon_footprint_handler    # Main orchestrator executable
│   ├── material_calculator         # Material footprint calculator
│   ├── transport_calculator        # Transport footprint calculator
│   └── *.o                         # Object files
├── utils/                          # Helper/utility libraries
│   ├── csv_parser.c/h              # CSV parsing utilities
│   └── json_parser.c/h             # JSON parsing utilities
├── material/                       # Material calculator source
│   └── material_calculator.c
├── transport/                      # Transport calculator source
│   └── transport_calculator.c
├── output/                         # Output CSV files
│   ├── material_footprints.csv
│   ├── transport_footprints.csv
│   └── (final results in ../output/)
├── carbon_footprint_handler.c      # Main handler source
└── Makefile                        # Build configuration
```

## Building

Compile all components:

```bash
make all
```

Clean and rebuild:

```bash
make rebuild
```

## Running

Execute the complete pipeline:

```bash
make run
```

Or run the handler directly:

```bash
./build/carbon_footprint_handler
```

## Components

### Material Calculator
- Calculates material carbon footprint using: `sum(weight × percentage × material_footprint)`
- Input: Product data + Material dataset
- Output: `output/material_footprints.csv`

### Transport Calculator
- Template structure for transport footprint calculations
- Currently returns 0.0 placeholder values
- Output: `output/transport_footprints.csv`

### Handler
- Orchestrates material and transport calculators
- Combines results into unified dataset
- Final output: `../output/carbon_footprint_results.csv`

## Output Format

Final CSV contains **all original columns** from `Product_data_final.csv` plus:
- `material_carbon_footprint_kg`
- `transport_carbon_footprint_kg`
- `total_carbon_footprint_kg`

## Processing Statistics

- **Products Processed**: 901,558
- **Materials in Dataset**: 35
- **Output File Size**: ~51 MB
