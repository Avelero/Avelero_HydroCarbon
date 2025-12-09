# Datasets Directory

This directory contains all datasets used in the Bulk Product Generator project, organized by data type and processing stage. All datasets are tracked using Git LFS for efficient version control.

---

## Directory Structure

```
datasets/
├── raw/                    # Original generated product data
├── processed/              # Corrected and enriched product data
├── reference/              # Reference datasets for calculations
├── splits/                 # Train/validation splits for ML models
└── model_outputs/          # ML model predictions and evaluations
    ├── baseline/           # Baseline model results
    └── robustness/         # Robustness-trained model results
```

---

## Dataset Descriptions

### 1. Raw Data (`raw/`)

Original product data generated using Google Gemini API before any correction or enrichment.

#### `Product_data.csv`
- **Description**: Main generated product dataset
- **Rows**: ~878,000 fashion products
- **Generation Method**: Google Gemini 2.5 Flash API with structured prompts
- **Use Case**: Starting point for data correction pipeline

**Schema:**
| Column | Type | Description | Example |
|--------|------|-------------|---------|
| `product_name` | string | Product name | "Boho Floral Print Maxi Skirt" |
| `gender` | string | Target gender | "Female", "Male" |
| `parent_category` | string | High-level category | "Tops", "Bottoms", "Outerwear" |
| `category` | string | Specific product category | "Maxi Skirts", "T-Shirts", "Jackets" |
| `manufacturer_country` | string | ISO 3166-1 alpha-2 country code | "CN", "IN", "BD" |
| `materials` | JSON string | Material composition | `{"viscose": 0.72, "polyester_virgin": 0.28}` |
| `weight_kg` | float | Product weight in kilograms | 0.587 |
| `total_distance_km` | float | Transport distance from manufacturer | 14321.63 |

**Data Quality Notes:**
- Generated data may contain inconsistencies (corrected in processed datasets)
- Material percentages sum to 1.0
- Weights and distances are realistic estimates based on product type

#### `fashion_products_20251126_055821.csv`
- **Description**: Timestamped snapshot of raw generated data
- **Purpose**: Historical backup of generation run
- **Schema**: Identical to `Product_data.csv`

---

### 2. Processed Data (`processed/`)

Cleaned, validated, and enriched product data ready for analysis and modeling.

#### `Product_data_final.csv`
- **Description**: Corrected product dataset after validation and cleanup
- **Rows**: ~902,000 products (invalid entries removed)
- **Processing Steps**:
  1. Gender validation and correction
  2. Category hierarchy validation
  3. Country code validation (ISO 3166-1 alpha-2)
  4. Material composition validation (sum = 1.0)
  5. Weight and distance range validation
  6. Duplicate removal
- **Use Case**: Input for footprint calculations and ML training

**Schema:** Same as `Product_data.csv` (see Raw Data section)

**Validation Rules:**
- Gender: Must be "Female" or "Male"
- Categories: Must match predefined category hierarchy
- Manufacturer country: Must be valid ISO 3166-1 alpha-2 code
- Materials: JSON format, percentages sum to 1.0 ± 0.01
- Weight: 0.05 kg ≤ weight ≤ 5.0 kg
- Distance: 100 km ≤ distance ≤ 25,000 km

#### `Product_data_with_footprints.csv`
- **Description**: Final dataset with calculated environmental footprints
- **Rows**: ~902,000 products
- **Processing**: Combines corrected product data with carbon and water footprint calculations
- **Use Case**: Complete dataset for environmental impact analysis and ML modeling

**Schema:** All columns from `Product_data_final.csv` plus:

| Column | Type | Unit | Description |
|--------|------|------|-------------|
| `carbon_material` | float | kgCO2e | Carbon footprint from material production |
| `carbon_transport` | float | kgCO2e | Carbon footprint from transportation |
| `carbon_total` | float | kgCO2e | Total carbon footprint (material + transport) |
| `water_total` | float | liters | Total water footprint from materials |

**Calculation Methods:**
- **Material Carbon**: `Σ (weight_kg × material_percentage × material_carbon_factor)`
- **Transport Carbon**: Multinomial logit modal split model (see reference data)
- **Water**: `Σ (weight_kg × material_percentage × material_water_factor)`

**Data Statistics:**
- Carbon total range: 0.5 - 25 kgCO2e
- Water total range: 50 - 15,000 liters
- Average carbon footprint: ~5.2 kgCO2e per product
- Average water footprint: ~2,800 liters per product

---

### 3. Reference Data (`reference/`)

Lookup tables and factors used for environmental footprint calculations.

#### `material_dataset_final.csv`
- **Description**: Material-level carbon and water footprint factors
- **Source**: Idemat 2026 database + literature review
- **Rows**: 34 fashion materials
- **Use Case**: Material footprint calculations

**Schema:**
| Column | Type | Description |
|--------|------|-------------|
| `material` | string | Material name (normalized) |
| `source` | string | Data source |
| `idemat_row_number` | int | Row reference in Idemat database |
| `process_id_name` | string | Idemat process identifier |
| `category` | string | Material category |
| `unit` | string | Measurement unit (typically "kg") |
| `total_ecocost_euro` | float | Total eco-cost in euros |
| `carbon_footprint_kgCO2e` | float | Carbon footprint per kg of material |
| `notes` | string | Additional information |
| `source_kgCO2` | string | Source reference for carbon data |
| `water_footprint_liters` | float | Water footprint per kg of material |
| `source_url_water` | string | Source URL for water data |

**Material Examples:**
- Cotton: 5.5 kgCO2e/kg, 10,000 L/kg
- Polyester (virgin): 6.0 kgCO2e/kg, 50 L/kg
- Wool: 17.0 kgCO2e/kg, 125,000 L/kg
- Viscose: 3.0 kgCO2e/kg, 600 L/kg

#### `transport_emission_factors_generalised.csv`
- **Description**: Transport emission factors by mode
- **Source**: GLEC Framework v3.0, DEFRA 2023
- **Rows**: 5 transport modes
- **Use Case**: Transport carbon footprint calculation

**Schema:**
| Column | Type | Unit | Description |
|--------|------|------|-------------|
| `mode` | string | - | Transport mode name |
| `ef_gco2e_per_tkm` | float | gCO2e/tkm | Emission factor (grams CO2e per tonne-kilometer) |

**Transport Modes:**
| Mode | Emission Factor (gCO2e/tkm) |
|------|----------------------------|
| Road | 62.0 |
| Rail | 22.0 |
| Inland Waterway | 31.0 |
| Sea | 10.5 |
| Air | 602.0 |

#### `utility_attractiveness.csv`
- **Description**: Modal split model parameters
- **Source**: Freight transport choice literature
- **Rows**: 5 transport modes
- **Use Case**: Calculate probability distribution across transport modes

**Schema:**
| Column | Type | Description |
|--------|------|-------------|
| `mode` | string | Transport mode name |
| `beta_0` | float | Intercept parameter in utility function |
| `beta_1` | float | Distance coefficient in utility function |

**Multinomial Logit Model:**
```
U_m(D) = β0_m + β1_m × ln(D)
P_m(D) = exp(U_m(D)) / Σ exp(U_k(D))
```
Where:
- `U_m(D)` = Utility of mode m at distance D
- `P_m(D)` = Probability of choosing mode m
- Road is reference mode with U_road(D) = 0

---

### 4. Data Splits (`splits/`)

Train and validation splits for machine learning model development.

#### `train.csv`
- **Description**: Training dataset (75% of total data)
- **Rows**: ~676,000 products
- **Split Method**: Random stratified split by category
- **Schema**: Same as `Product_data_with_footprints.csv`

#### `validate.csv`
- **Description**: Validation dataset (25% of total data)
- **Rows**: ~225,000 products
- **Split Method**: Random stratified split by category
- **Schema**: Same as `Product_data_with_footprints.csv`

**Split Characteristics:**
- Stratified by `category` to maintain distribution
- Random seed: 42 (reproducible)
- No data leakage between splits
- Both splits contain all material types and countries

---

### 5. Model Outputs (`model_outputs/`)

Results from machine learning models predicting environmental footprints.

#### Baseline Model (`model_outputs/baseline/`)

Standard neural network trained on complete data.

##### `baseline_predictions.csv`
- **Description**: Model predictions on validation set
- **Rows**: ~225,000 products
- **Model**: Feed-forward neural network (3 layers, 128-64-32 units)
- **Training**: 2000 epochs, Adam optimizer, MSE loss

**Schema:** All columns from validation data plus:
| Column | Type | Description |
|--------|------|-------------|
| `predicted_carbon_material` | float | Predicted material carbon footprint |
| `predicted_carbon_transport` | float | Predicted transport carbon footprint |
| `predicted_carbon_total` | float | Predicted total carbon footprint |
| `predicted_water_total` | float | Predicted water footprint |

##### `robustness_results.csv`
- **Description**: Model performance under various data corruption scenarios
- **Rows**: Multiple corruption levels per metric
- **Purpose**: Evaluate model resilience to missing data

**Schema:**
| Column | Type | Description |
|--------|------|-------------|
| `corruption_level` | float | Percentage of features randomly masked (0.0 - 0.5) |
| `mae_carbon_total` | float | Mean absolute error for carbon total |
| `mae_water_total` | float | Mean absolute error for water total |
| `r2_carbon_total` | float | R² score for carbon total |
| `r2_water_total` | float | R² score for water total |

#### Robustness Model (`model_outputs/robustness/`)

Neural network trained with data augmentation for robustness to missing features.

##### `baseline_predictions.csv`
- **Description**: Robustness model predictions on validation set
- **Model**: Same architecture as baseline, trained with dropout augmentation
- **Training**: 2000 epochs with 20% feature dropout during training
- **Schema**: Same as baseline predictions

##### `robustness_results.csv`
- **Description**: Robustness model performance under data corruption
- **Expected Behavior**: Better performance than baseline under high corruption
- **Schema**: Same as baseline robustness results

**Performance Comparison (Corruption = 30%):**
| Model | MAE Carbon | MAE Water | R² Carbon | R² Water |
|-------|-----------|----------|-----------|----------|
| Baseline | ~1.2 kgCO2e | ~450 L | ~0.65 | ~0.58 |
| Robustness | ~0.9 kgCO2e | ~320 L | ~0.78 | ~0.71 |

---

## Data Formats

### JSON Material Composition
Materials are stored as JSON strings with normalized material names and percentage values:

```json
{
  "cotton": 0.60,
  "polyester_virgin": 0.35,
  "elastane": 0.05
}
```

**Rules:**
- Keys: Lowercase, underscores for spaces
- Values: Floats between 0.0 and 1.0
- Sum of values: 1.0 ± 0.01 (allowing rounding error)

### Country Codes
ISO 3166-1 alpha-2 format (2-letter codes):
- CN = China
- IN = India
- BD = Bangladesh
- VN = Vietnam
- etc.

---

## Data Usage Guidelines

### For Kaggle
1. Use `Product_data_with_footprints.csv` as main dataset
2. Reference `material_dataset_final.csv` for material explanations
3. Use `splits/` for reproducible model training

### For Research
1. Cite data sources (Idemat 2026, GLEC Framework)
2. Acknowledge AI-generated product data (Google Gemini)
3. Report data validation and correction steps

### For Model Development
1. Use provided train/validation splits for fair comparison
2. Evaluate on robustness tests for real-world applicability
3. Consider both carbon and water footprint predictions

---

## Data Quality and Limitations

### Strengths
- Large-scale dataset (900k products)
- Realistic product attributes generated using state-of-the-art LLM
- Scientifically validated footprint calculations
- Well-documented data lineage

### Limitations
- Product data is AI-generated, not real market data
- Material compositions are estimated, not actual measurements
- Transport distances use straight-line approximations
- Water footprint data has limited source coverage
- No scope 2 or scope 3 (use phase) emissions included

### Known Issues
- Some rare materials may have uncertain footprint factors
- Transport modal split model calibrated for European freight (may not generalize globally)
- Products from small island nations may have unrealistic distances

---

## Data Sources and Citations

### Material Footprints
- **Idemat 2026**: TU Delft material database
  - URL: https://www.ecocostsvalue.com/idemat/
- **Water Footprint Network**: Textile industry water consumption
  - URL: https://waterfootprint.org/

### Transport Emissions
- **GLEC Framework v3.0**: Global Logistics Emissions Council
  - URL: https://www.smartfreightcentre.org/glec/
- **DEFRA 2023**: UK Department for Environment, Food & Rural Affairs
  - URL: https://www.gov.uk/government/publications/greenhouse-gas-reporting-conversion-factors-2023

### Product Generation
- **Google Gemini 2.5 Flash**: Large language model for product data generation
  - URL: https://ai.google.dev/

---

## Version History

### Version 1.0 (2025-12-09)
- Initial dataset release
- 900,000 fashion products
- Complete footprint calculations
- Train/validation splits
- Baseline and robustness model outputs

---

## Contact and Support

For questions about the datasets:
- GitHub Issues: [bulk_product_generator/issues](https://github.com/yourusername/bulk_product_generator/issues)
- Documentation: See main README.md

---

## License

The datasets in this directory are released under the same license as the parent project. Please refer to the LICENSE file in the root directory.

**Note**: While the code and generated data are open, please respect the original sources' licenses when using reference data (Idemat, GLEC, etc.).
