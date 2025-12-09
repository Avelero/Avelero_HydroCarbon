# HydroCarbon Footprint Predictor - Preview Mode

A terminal-based preview integration for the trained XGBoost model that predicts carbon and water footprints for fashion products.

## Quick Start

### Interactive Mode
```bash
python preview.py
```

This will guide you through entering product details step by step.

### Single Product Prediction (CSV format)
```bash
python preview.py --csv 'Boho Floral Print Maxi Skirt,Female,Bottoms,Maxi Skirts,MQ,"{""viscose"":0.72,""polyester_virgin"":0.28}",0.587,14321.63'
```

### Quiet Mode (only output)
```bash
python preview.py -q --csv 'Product Name,Female,Bottoms,Maxi Skirts,CN,"{""cotton_conventional"":0.95,""elastane"":0.05}",0.5,15000'
```

Output format: `carbon_material,carbon_transport,carbon_total,water_total`

## Input Fields

| Field | Description | Example |
|-------|-------------|---------|
| `product_name` | Name of the product (for display only) | `Boho Floral Print Maxi Skirt` |
| `gender` | Gender category | `Female`, `Male`, `Unisex` |
| `parent_category` | Top-level category | `Tops`, `Bottoms`, `Dresses`, `Outerwear`, `Footwear`, `Accessories` |
| `category` | Specific category | `T-Shirts`, `Maxi Skirts`, `Jeans`, `Sneakers`, `Jackets` |
| `manufacturer_country` | 2-letter country code | `CN`, `BD`, `VN`, `IN`, `TR`, `MQ`, `TG` |
| `materials` | JSON dict of material percentages (must sum to ~1.0) | `{"viscose":0.72,"polyester_virgin":0.28}` |
| `weight_kg` | Product weight in kilograms | `0.587` |
| `total_distance_km` | Total transport distance in kilometers | `14321.63` |

## Output Fields

| Field | Description | Unit |
|-------|-------------|------|
| `carbon_material` | Carbon footprint from material production | kgCO2e |
| `carbon_transport` | Carbon footprint from transportation | kgCO2e |
| `carbon_total` | Total carbon footprint | kgCO2e |
| `water_total` | Total water footprint | liters |

## Available Materials

View the list of valid material names:
```bash
python preview.py --list-materials
```

Common materials include:
- **Cotton**: `cotton_conventional`, `cotton_organic`, `cotton_recycled`
- **Polyester**: `polyester_virgin`, `polyester_recycled`
- **Polyamide/Nylon**: `polyamide_6`, `polyamide_66`, `polyamide_recycled`
- **Natural fibers**: `wool_generic`, `wool_merino`, `silk`, `linen_flax`, `hemp`
- **Synthetic**: `acrylic`, `elastane`, `viscose`, `modal`, `lyocell_tencel`
- **Leather**: `leather_bovine`, `leather_ovine`, `leather_synthetic`
- **Other**: `cashmere`, `down_feather`, `down_synthetic`, `rubber_synthetic`

## Batch Processing

Process a CSV file with multiple products:
```bash
python preview.py --batch input.csv --output results.csv
```

Input CSV must have headers:
```
product_name,gender,parent_category,category,manufacturer_country,materials,weight_kg,total_distance_km
```

## Model Selection

By default, the baseline model is used. To use the robustness model:
```bash
python preview.py --model trained_model/robustness
```

## Examples

### Example 1: Viscose Maxi Skirt
```bash
python preview.py -q --csv 'Pleated Viscose Maxi Skirt,Female,Bottoms,Maxi Skirts,TG,"{""viscose"":0.94,""elastane"":0.06}",0.413,17894.27'
```

### Example 2: Cotton T-Shirt
```bash
python preview.py -q --csv 'Basic Cotton Tee,Unisex,Tops,T-Shirts,BD,"{""cotton_conventional"":0.95,""elastane"":0.05}",0.2,12000'
```

### Example 3: Leather Jacket
```bash
python preview.py -q --csv 'Leather Biker Jacket,Male,Outerwear,Jackets,IT,"{""leather_bovine"":0.85,""polyester_virgin"":0.15}",1.8,8000'
```

## Technical Details

The preview mode uses:
- **Model**: XGBoost multi-output regressor trained on ~120,000 fashion products
- **Preprocessing**: Full preprocessing pipeline including:
  - Categorical encoding (label + target encoding)
  - Material-based fallback features
  - Category-based statistics
  - Formula-based physics features
- **Targets**: Log-transformed during training for better accuracy on skewed distributions

## Directory Structure

```
Trained-Implementation/
├── preview.py              # Main preview script
├── README.md               # This file
└── trained_model/
    ├── baseline/           # Baseline model (default)
    │   ├── xgb_model.json  # Trained XGBoost model
    │   ├── preprocessor.pkl # Fitted preprocessor
    │   ├── trainer_config.pkl # Trainer configuration
    │   └── evaluation/     # Evaluation metrics
    └── robustness/         # Robustness model (handles missing data better)
        ├── xgb_model.json
        ├── preprocessor.pkl
        ├── trainer_config.pkl
        └── evaluation/
```
