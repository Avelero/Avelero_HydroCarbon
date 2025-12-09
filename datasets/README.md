# Datasets Directory

This directory contains all datasets used in the **Avelero HydroCarbon** environmental footprint prediction project. Datasets are organized by processing stage and purpose, with all large files tracked using Git LFS.

---

## Quick Reference

| Dataset | Rows | Size | Description |
|---------|------|------|-------------|
| **Raw Products** | 878k | ~150 MB | AI-generated product data |
| **Processed Products** | 902k | ~180 MB | Validated + corrected data |
| **Complete Dataset** | 902k | ~360 MB | Products with footprints |
| **Training Set** | 676k | ~270 MB | 75% stratified split |
| **Validation Set** | 225k | ~90 MB | 25% stratified split |
| **Material Reference** | 34 | 10 KB | Carbon & water factors |
| **Transport Reference** | 5 | 1 KB | Emission factors by mode |

> **Note**: Processed data (902k) contains more products than raw data (878k) because additional products were generated for underrepresented categories (e.g., Dresses, Gowns) to ensure balanced distribution across all 86 fashion categories.

---

## Directory Structure

```
datasets/
├── README.md               # This file
├── raw/                    # Original AI-generated product data
│   ├── Product_data.csv
│   └── fashion_products_*.csv
├── processed/              # Validated and enriched data
│   ├── Product_data_final.csv
│   └── Product_data_with_footprints.csv
├── reference/              # Lookup tables for calculations
│   ├── material_dataset_final.csv
│   ├── transport_emission_factors_generalised.csv
│   └── utility_attractiveness.csv
├── splits/                 # ML train/validation splits
│   ├── train.csv
│   └── validate.csv
└── model_outputs/          # Trained model predictions
    ├── baseline/
    │   ├── baseline_predictions.csv
    │   └── robustness_results.csv
    └── robustness/
        ├── baseline_predictions.csv
        └── robustness_results.csv
```

---

## Dataset Descriptions

### 1. Raw Data (`raw/`)

Original product data generated using Google Gemini 2.0 Flash API.

#### `Product_data.csv`

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| `product_name` | string | Creative product name | "Boho Floral Print Maxi Skirt" |
| `gender` | string | Target gender | "Female", "Male" |
| `parent_category` | string | High-level category | "Tops", "Bottoms", "Outerwear" |
| `category` | string | Specific category (86 types) | "Maxi Skirts", "T-Shirts" |
| `manufacturer_country` | string | ISO 3166-1 alpha-2 code | "CN", "IN", "BD" |
| `materials` | JSON | Material composition | `{"viscose": 0.72, "polyester_virgin": 0.28}` |
| `weight_kg` | float | Product weight (kg) | 0.587 |
| `total_distance_km` | float | Transport distance (km) | 14321.63 |

**Generation Details**:
- **API**: Google Gemini 2.0 Flash
- **Products per category**: ~10,000
- **Categories**: 86 fashion product types
- **Materials**: 34 unique materials

---

### 2. Processed Data (`processed/`)

Cleaned and validated product data with calculated environmental footprints.

#### `Product_data_final.csv`

Same schema as raw data, after applying validation rules:

| Validation | Rule | Action |
|------------|------|--------|
| Gender | Must be "Female" or "Male" | Correct or remove |
| Categories | Must match hierarchy | Validate against list |
| Country | Valid ISO 3166-1 alpha-2 | Remove invalid |
| Materials | JSON, sum = 1.0 ± 0.01 | Normalize or remove |
| Weight | 0.05 ≤ w ≤ 5.0 kg | Flag outliers |
| Distance | 100 ≤ d ≤ 25,000 km | Flag outliers |

#### `Product_data_with_footprints.csv`

Complete dataset with calculated environmental footprints. Includes all columns from `Product_data_final.csv` plus:

| Column | Type | Unit | Description |
|--------|------|------|-------------|
| `carbon_material` | float | kgCO2e | Footprint from material production |
| `carbon_transport` | float | kgCO2e | Footprint from transportation |
| `carbon_total` | float | kgCO2e | Total carbon (material + transport) |
| `water_total` | float | liters | Water footprint from materials |

**Footprint Statistics**:
| Metric | Min | Mean | Median | Max |
|--------|-----|------|--------|-----|
| Carbon Material | 0.15 | 3.2 | 2.8 | 18.5 kgCO2e |
| Carbon Transport | 0.05 | 2.0 | 1.6 | 12.3 kgCO2e |
| Carbon Total | 0.25 | 5.2 | 4.5 | 25.8 kgCO2e |
| Water Total | 50 | 2,800 | 1,950 | 15,000 L |

---

### 3. Reference Data (`reference/`)

Lookup tables for environmental footprint calculations.

#### `material_dataset_final.csv`

Carbon and water footprint factors for 34 fashion materials.

| Column | Description |
|--------|-------------|
| `material` | Normalized material name |
| `carbon_footprint_kgCO2e` | Carbon footprint per kg |
| `water_footprint_liters` | Water footprint per kg |
| `source` | Data source (Idemat, Research, Proxy) |
| `source_url_water` | URL for water footprint source |

**Top Materials by Carbon Footprint**:
| Material | Carbon (kgCO2e/kg) | Water (L/kg) |
|----------|-------------------|--------------|
| Wool (merino) | 61.0 | 170,000 |
| Cashmere | 14.0 | 34,160 |
| Leather (bovine) | 15.2 | 17,100 |
| Polyester (virgin) | 2.05 | 60 |
| Cotton (conventional) | 0.94 | 9,113 |
| Cotton (organic) | 0.32 | 7,837 |

#### `transport_emission_factors_generalised.csv`

Emission factors by transport mode (source: CE Delft STREAM 2020).

| Mode | EF (gCO2e/tkm) | Typical Use |
|------|---------------|-------------|
| Sea | 10.3 | Intercontinental shipping |
| Rail | 22.0 | Continental freight |
| Inland Waterway | 31.0 | River/canal transport |
| Road | 72.9 | Last-mile delivery |
| Air | 782.0 | Urgent/high-value items |

#### `utility_attractiveness.csv`

Multinomial logit model parameters for transport mode choice.

```
Model: U_m(D) = β0_m + β1_m × ln(D)
       P_m(D) = exp(U_m(D)) / Σ exp(U_k(D))
```

Road is the reference mode (U_road = 0). Parameters calibrated from CE Delft (2011) Rhine corridor freight data.

---

### 4. Data Splits (`splits/`)

Train/validation splits for ML model development.

| Split | Rows | Percentage | Purpose |
|-------|------|------------|---------|
| `train.csv` | 676,178 | 75% | Model training |
| `validate.csv` | 225,393 | 25% | Model evaluation |

**Split Characteristics**:
- **Method**: Stratified by `category` (maintains distribution)
- **Random seed**: 42 (reproducible)
- **No leakage**: Proper separation, no duplicate products
- **Schema**: Same as `Product_data_with_footprints.csv`

---

### 5. Model Outputs (`model_outputs/`)

Predictions and evaluation metrics from trained XGBoost models.

#### Baseline Model (`baseline/`)

XGBoost multi-output regressor optimized for maximum accuracy on complete data.

**`baseline_predictions.csv`**: Model predictions on validation set
- All columns from validation data
- Plus: `predicted_carbon_material`, `predicted_carbon_transport`, `predicted_carbon_total`, `predicted_water_total`

**Performance (Complete Data)**:
| Target | MAE | R² |
|--------|-----|-----|
| Carbon Material | 0.041 kgCO2e | 0.9999 |
| Carbon Transport | 0.0008 kgCO2e | 0.9998 |
| Carbon Total | 0.044 kgCO2e | 0.9999 |
| Water Total | 115.3 L | 0.9998 |

#### Robustness Model (`robustness/`)

XGBoost trained with 20% feature dropout augmentation for missing data resilience.

**`robustness_results.csv`**: Performance under simulated data corruption

| Missing % | Baseline R² (Carbon) | Robustness R² (Carbon) |
|-----------|---------------------|----------------------|
| 0% | 0.9999 | 0.9999 |
| 20% | 0.001 | **0.968** |
| 40% | -0.991 | **0.936** |

> **Key insight**: Robustness model maintains R² > 0.93 even with 40% missing features, while baseline collapses completely.

---

## Data Formats

### Material Composition (JSON)

```json
{
  "cotton_conventional": 0.65,
  "polyester_virgin": 0.30,
  "elastane": 0.05
}
```

- **Keys**: Lowercase with underscores (matches `material` column in reference data)
- **Values**: Floats 0.0–1.0, sum = 1.0 ± 0.01
- **Materials**: 34 supported types (see `material_dataset_final.csv`)

### Country Codes

ISO 3166-1 alpha-2 format:
- CN = China, IN = India, BD = Bangladesh, VN = Vietnam
- Full list: 277 countries represented in dataset

---

## Usage Examples

### Loading Data (Python)

```python
import pandas as pd

# Load complete dataset
df = pd.read_csv('datasets/processed/Product_data_with_footprints.csv')

# Load with material parsing
import json
df['materials_dict'] = df['materials'].apply(json.loads)

# Load reference data
materials = pd.read_csv('datasets/reference/material_dataset_final.csv')
```

### Loading Splits for ML

```python
# Load train/validation splits
train = pd.read_csv('datasets/splits/train.csv')
val = pd.read_csv('datasets/splits/validate.csv')

# Features and targets
TARGETS = ['carbon_material', 'carbon_transport', 'carbon_total', 'water_total']
X_train = train.drop(columns=TARGETS)
y_train = train[TARGETS]
```

---

## Data Quality Notes

### Strengths
- ✅ Large-scale dataset (900k products)
- ✅ Realistic attributes from state-of-the-art LLM
- ✅ Scientifically validated footprint calculations
- ✅ Comprehensive material coverage (34 types)
- ✅ Well-documented data lineage

### Limitations
- ⚠️ Product data is AI-generated, not real market data
- ⚠️ Material compositions are estimated, not measured
- ⚠️ Transport distances use straight-line approximations
- ⚠️ Modal split model calibrated for European freight
- ⚠️ No scope 2/3 (use phase) emissions included

---

## Data Sources

### Material Footprints
| Source | Usage | License |
|--------|-------|---------|
| [TU Delft Idemat 2026](https://www.ecocostsvalue.com/idemat/) | Carbon footprints | Non-commercial free |
| [Water Footprint Network](https://waterfootprint.org/) | Water footprints | Open Access |
| [MDPI Journals](https://www.mdpi.com/) | Research values | CC BY 4.0 |
| [Carbonfact](https://www.carbonfact.com/) | Vegan leather, synthetics | Public blog |

### Transport Emissions
| Source | Usage | License |
|--------|-------|---------|
| [CE Delft STREAM 2020](https://cedelft.eu/) | Emission factors | Academic use |
| [Eurostat](https://ec.europa.eu/eurostat) | EU freight statistics | Open Data |
| [IATA](https://www.iata.org/) | Air cargo data | Public reports |

### Product Generation
| Source | Usage |
|--------|-------|
| [Google Gemini 2.0 Flash](https://ai.google.dev/) | Product data generation |

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0 | 2025-12-09 | Initial release: 900k products, complete footprints, baseline + robustness models |

---

## License

Datasets are released under the same license as the parent project (MIT). 

**Third-party data**: Please respect original source licenses when using reference data. See main README for detailed attribution requirements.

---

## Related Documentation

- [Main README](../README.md) — Project overview and usage
- [C Calculator README](../data/data_calculations/README.md) — Footprint calculation details
- [Model Training](../models/) — ML model architecture and training

---

**Questions?** Open an issue on [GitHub](https://github.com/Avelero/Avelero_HydroCarbon/issues) or see the main project README.
