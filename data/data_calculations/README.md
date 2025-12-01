# Unified Footprint Calculator

A comprehensive carbon and water footprint calculation system for product lifecycle assessment.

## Overview

This system calculates environmental footprints for products based on:
- **Material composition**: Carbon and water footprints from raw materials
- **Transport distance**: Carbon footprint from transportation using a multinomial logit modal split model

## Output

The calculator produces a single unified output file: `output/Product_data_with_footprints.csv`

This file contains all original product data columns plus:
| Column | Description | Unit |
|--------|-------------|------|
| `carbon_material` | Carbon footprint from materials | kgCO2e |
| `carbon_transport` | Carbon footprint from transport | kgCO2e |
| `carbon_total` | Total carbon footprint | kgCO2e |
| `water_total` | Total water footprint | liters |

## Quick Start

```bash
# Build the calculator
make

# Run the calculation
make run

# Or run directly
./build/footprint_calculator
```

## Project Structure

```
data_calculations/
├── input/                          # Input data files
│   ├── Product_data_final.csv      # Product data with materials
│   ├── material_dataset_final.csv  # Material footprint factors
│   ├── transport_emission_factors_generalised.csv
│   └── utility_attractiveness.csv
├── output/                         # Output directory
│   └── Product_data_with_footprints.csv  # Unified output
├── src/
│   ├── carbon/                     # Carbon footprint calculators
│   │   ├── material/               # Material carbon calculations
│   │   └── transport/              # Transport carbon calculations
│   ├── water/                      # Water footprint calculators
│   │   └── material/               # Material water calculations
│   ├── utils/                      # Shared utilities
│   │   ├── csv_parser.c
│   │   └── json_parser.c
│   └── footprint_calculator.c      # Main unified handler
├── include/                        # Header files
├── build/                          # Compiled objects and executables
├── docs/                           # Documentation
├── tests/                          # Test files
└── Makefile                        # Build system
```

## Input Data Format

### Product Data (`Product_data_final.csv`)
```csv
product_name,gender,parent_category,category,manufacturer_country,materials,weight_kg,total_distance_km
"Example Product",Female,Bottoms,Maxi Skirts,CN,"{""cotton"":0.8,""polyester"":0.2}",0.5,15000
```

### Material Dataset (`material_dataset_final.csv`)
Must contain columns:
- Column 0: `material` (material name)
- Column 7: `carbon_footprint_kgCO2e` (kg CO2e per kg of material)
- Column 10: `water_footprint_liters` (liters per kg of material)

## Calculation Methods

### Material Carbon Footprint
```
carbon_material = Σ (product_weight_kg × material_percentage × material_carbon_factor)
```

### Transport Carbon Footprint
Uses a multinomial logit model to determine modal split probabilities:
```
P_m(D) = exp(U_m(D)) / Σ exp(U_k(D))
U_m(D) = β0_m + β1_m × ln(D)

carbon_transport = (weight_kg / 1000) × distance_km × weighted_EF / 1000
weighted_EF = Σ P_m(D) × EF_m
```

### Material Water Footprint
```
water_total = Σ (product_weight_kg × material_percentage × material_water_factor)
```

## Building

### Requirements
- GCC with C11 support
- Make
- Math library (libm)

### Build Commands
```bash
make          # Build the calculator
make clean    # Remove build artifacts
make rebuild  # Clean and rebuild
make help     # Show available targets
```

## Authors
- Moussa Ouallaf

## Version
2.0 - Unified footprint calculation system
