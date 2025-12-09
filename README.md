# Bulk Product Generator with Environmental Footprint Analysis

[![Python Version](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![C Standard](https://img.shields.io/badge/C-C11-green.svg)](https://en.wikipedia.org/wiki/C11_(C_standard_revision))
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Git LFS](https://img.shields.io/badge/Git%20LFS-enabled-orange.svg)](https://git-lfs.github.com/)

An end-to-end pipeline for generating synthetic fashion product data and predicting environmental footprints using machine learning. This project addresses a critical gap in sustainability research: **the absence of large-scale, publicly available life cycle assessment (LCA) datasets for fashion products**. By combining LLM-generated synthetic data with physics-based footprint calculations, we create a robust training dataset for ML models that can predict carbon and water footprints even when product information is incomplete.

> **Proof of Concept**: This project is a proof-of-concept demonstrating the feasibility of ML-based environmental footprint prediction for fashion products. A production-ready version with full **ISO 14040/14044** compliance (Life Cycle Assessment standards), **PEF** (Product Environmental Footprint) methodology, and expanded scope (Scope 1-3 emissions, end-of-life modeling) is currently in development.

---

## Table of Contents


**Core Documentation:**
- [Overview](#overview) — The problem, our solution, proof-of-concept status
- [Synthetic Data Generation](#synthetic-data-generation) — LLM-powered product generation pipeline
- [Physics-Based Footprint Calculation](#physics-based-footprint-calculation) — Scientific formulas and data sources
- [Robust ML Model](#robust-ml-model) — XGBoost with feature dropout for missing data

**Getting Started:**
- [Quick Start](#quick-start) — Clone, install, run
- [Detailed Usage](#detailed-usage) — Step-by-step pipeline instructions
- [Training on Google Colab](#training-on-google-colab-gpu-acceleration) — GPU-accelerated model training

**Reference:**
- [Datasets](#datasets) — Data files and formats
- [Installation](#installation) — Full dependency setup

**Meta:**
- [Citation](#citation) — How to cite this work
- [License](#license) — MIT + third-party data licenses
- [Acknowledgments](#acknowledgments) — Data sources and contributors

---


## Overview

### The Problem

Environmental footprint calculation for fashion products requires detailed information: material composition, weight, manufacturing location, and transport distances. In practice, **this data is rarely complete**:

- E-commerce platforms often list only primary materials ("100% cotton") without exact compositions
- Product weights are frequently missing or estimated
- Manufacturing origins may be unknown beyond country level
- Supply chain routing data is proprietary and unavailable

Traditional LCA calculators fail when inputs are missing. Companies cannot calculate footprints for products with incomplete data, creating a barrier to sustainability reporting.

### Our Solution

This project takes a different approach:

1. **Synthetic Data Generation**: Use Google Gemini 2.5 Flash to generate 900,000+ realistic fashion products with complete attribute coverage (materials, weights, distances, categories)

2. **Physics-Based Calculation**: Apply scientifically validated formulas to calculate exact carbon and water footprints for each product

3. **Robust ML Model**: Train an XGBoost model with **feature dropout augmentation** that learns to predict footprints even when 40% of input features are missing

---

## Synthetic Data Generation

This project uses **LLM-based synthetic data generation** to create a large-scale training dataset for environmental footprint prediction. Unlike web-scraped datasets with inconsistent quality, we generate 900,000+ fashion products with complete, validated attributes using Google Gemini 2.5 Flash.

### What is Synthetic Data?

Synthetic data is **artificially generated data** that mimics the statistical properties of real-world data without being collected from actual sources. In this project:

- **Products are not real** — They don't represent actual items from retailers
- **Attributes are realistic** — Names, materials, weights follow real fashion industry patterns
- **Footprints are calculated** — Using validated formulas, not measured emissions

### Why Generate Synthetic Data?

| Challenge | Real Data | Synthetic Data |
|-----------|-----------|----------------|
| **Availability** | Fashion LCA data is proprietary | Generate unlimited products |
| **Completeness** | Often missing fields | All 8 attributes guaranteed |
| **Scale** | Small datasets (100s-1000s) | 900,000 products |
| **Cost** | Expensive to collect | XXX |
| **Privacy** | Business-sensitive | XXX |

### How It Works (Simple Overview)

The generation process follows a structured 4-step pipeline designed for **scalability**, **quality**, and **resilience**:

#### Step 1: Define the Generation Space

Before generating any products, we define the boundaries of realistic fashion products:

| Component | Definition | Source |
|-----------|------------|--------|
| **Categories** | 86 leaf categories in hierarchical structure | `categories_rows.json` — curated from major fashion retailers |
| **Materials** | 34 materials with known environmental factors | `material_dataset_final.csv` — from Idemat 2026 + literature |
| **Countries** | 277 ISO 3166-1 alpha-2 country codes | Global manufacturing locations |
| **Weight ranges** | Category-specific (0.12 kg underwear → 2.38 kg boots) | Industry standards |
| **Distance ranges** | Route-based (520 km local → 23,800 km intercontinental) | Shipping logistics data |

This controlled vocabulary ensures generated products are **plausible** and **calculable**.

#### Step 2: Prompt Construction & API Calls

Each API call generates products for a specific category with contextual guidance:

```
Input to Gemini:
├── Category info: "Jeans (Gender: Male, Parent: Bottoms)"
├── Suggested materials: ["cotton_conventional", "elastane", "hemp"]
├── Weight guidance: "0.83-1.87 kg with natural variance (0.947, 1.234)"
├── Distance guidance: "Asia to EU: 10,000-12,000 km"
└── Format rules: CSV with 8 columns, JSON materials, no round numbers

Output from Gemini:
└── Up to 1000 CSV rows per API call
```

**High-throughput configuration:**
- **Chunk size**: 1,000 products per API call
- **Parallel workers**: 25 concurrent requests
- **Rate limit**: 80% of 2,000 RPM = 1,600 effective RPM

#### Step 3: Validation & Incremental Saving

Every API response passes through validation before being saved:

```
API Response → Parse CSV → Validate Each Row → Append to File → Update Checkpoint
                   │              │                   │               │
                   ▼              ▼                   ▼               ▼
              Handle quotes   Check 8 cols      Atomic write     Save progress
              Fix JSON       Verify ranges      No duplicates    Resume-ready
```

**Resilience features:**
- **Retry logic**: 3 attempts per chunk with error classification
- **Rate limiting**: Token bucket at 90% capacity prevents quota exhaustion
- **Checkpointing**: Progress saved after each chunk — can resume from any failure

#### Example: Real Prompt & Output

**Actual prompt sent to Gemini** (simplified for readability):

```
Generate 10 realistic fashion products as CSV data with natural, varied numeric values.

CRITICAL FORMATTING REQUIREMENTS:
1. Output MUST be valid CSV format with proper quoting
2. Each row MUST have EXACTLY 8 fields (columns)
3. All numeric values MUST have natural variance and precision
4. NEVER use round numbers - use realistic decimals like 0.743, 1.267, 8456

Generate EXACTLY 10 product(s) for EACH of these categories:
  Jeans (Gender: Male, Parent: Bottoms, Suggested materials: cotton_conventional, elastane, hemp)

Output CSV with this EXACT header (do NOT repeat this header in your output):
product_name,gender,parent_category,category,manufacturer_country,materials,weight_kg,total_distance_km

RULES:
- product_name: Generate realistic product names
- gender: "Male" or "Female" (based on category)
- materials: JSON object, shares MUST sum to exactly 1.0
  * Available: cotton_conventional, cotton_organic, elastane, hemp, polyester_virgin...
- weight_kg: Heavy apparel 0.83-1.87 kg with natural variance (e.g., 0.947, 1.234)
- total_distance_km: Intercontinental 10300-23800 km (e.g., 11764, 15428)
- NO header repetition, NO explanations, ONLY data rows

Output 10 CSV rows:
```

**Actual output from Gemini:**

```csv
"Relaxed Fit Carpenter Jeans",Male,Bottoms,Jeans,BD,"{""cotton_conventional"": 0.92, ""elastane"": 0.08}",0.934,12847.52
"Slim Tapered Dark Wash Jeans",Male,Bottoms,Jeans,CN,"{""cotton_conventional"": 0.88, ""polyester_virgin"": 0.07, ""elastane"": 0.05}",0.876,10234.89
"Classic Straight Leg Jeans",Male,Bottoms,Jeans,VN,"{""cotton_organic"": 0.75, ""hemp"": 0.20, ""elastane"": 0.05}",0.912,11456.23
"Distressed Skinny Jeans",Male,Bottoms,Jeans,TR,"{""cotton_conventional"": 0.95, ""elastane"": 0.05}",0.847,8934.67
"Loose Fit Workwear Jeans",Male,Bottoms,Jeans,IN,"{""cotton_conventional"": 0.82, ""hemp"": 0.15, ""elastane"": 0.03}",1.023,11823.41
"Athletic Stretch Jeans",Male,Bottoms,Jeans,PK,"{""cotton_conventional"": 0.78, ""polyester_virgin"": 0.17, ""elastane"": 0.05}",0.789,12156.78
"Vintage Wash Bootcut Jeans",Male,Bottoms,Jeans,BD,"{""cotton_conventional"": 0.90, ""elastane"": 0.10}",0.956,13247.34
"Raw Selvedge Denim Jeans",Male,Bottoms,Jeans,JP,"{""cotton_organic"": 0.98, ""elastane"": 0.02}",1.134,9823.56
"Comfort Fit Dad Jeans",Male,Bottoms,Jeans,CN,"{""cotton_conventional"": 0.85, ""polyester_virgin"": 0.10, ""elastane"": 0.05}",0.923,10567.89
"Slim Straight Indigo Jeans",Male,Bottoms,Jeans,VN,"{""cotton_conventional"": 0.88, ""elastane"": 0.12}",0.867,11234.56
```

**Note:** Each row has natural variance in weights (0.789-1.134 kg), realistic distances based on country, and material compositions that sum to exactly 1.0.

---

## Physics-Based Footprint Calculation

Once products are generated, we calculate exact environmental footprints using **scientifically validated formulas** and **peer-reviewed emission factors**. This creates the ground truth labels for ML training.

### The Total Footprint Formula

The total carbon footprint is the sum of material production and transport emissions:

```
Total_carbon_footprint = material_footprint + transport_footprint
```

Where:
- **material_footprint** = Emissions from producing raw materials (cradle-to-gate)
- **transport_footprint** = Emissions from shipping products to destination

### Why C for Calculations?

| Aspect | Python | C |
|--------|--------|---|
| **Speed** | ~15 minutes for 900k products | ~45 seconds |
| **Memory** | High (pandas DataFrames) | Low (streaming) |
| **Deployment** | Requires Python environment | Standalone binary |

---

### Material Carbon Footprint

**Formula:**
```
carbon_material = Σ (W × Pᵢ × CFᵢ)
```

| Symbol | Meaning |
|--------|---------|
| W | Product weight (kg) |
| Pᵢ | Percentage of material i (0-1) |
| CFᵢ | Carbon factor for material i (kgCO2e/kg) |

**Example:**
```
Product: "Classic Denim Jeans" (0.934 kg)
Materials: {"cotton_conventional": 0.92, "elastane": 0.08}

carbon_material = (0.934 × 0.92 × 0.94) + (0.934 × 0.08 × 5.55)
                = 0.808 + 0.415 = 1.223 kgCO2e
```

---

### Transport Carbon Footprint

#### The Generalisation Problem

Calculating transport emissions accurately requires knowing:
- Which transport modes were used (road, rail, sea, air, inland waterway)
- Distance traveled by each mode
- Emission factor for each mode

**Problem:** This detailed supply chain data is rarely available. Companies typically only know the total shipping distance, not the modal breakdown.

**Our Solution:** Generalise the calculation using a **distance-dependent multinomial logit model** that estimates modal splits based on empirical freight transport data.

#### Base Transport Emission Formula

```
E(D) = (W / 1000) × D × (Σₘ sₘ(D) × EFₘ) / 1000   [kgCO2e]
```

**Expanded:**

> **E(D)** = (Weight in tonnes) × (Distance) × (Weighted emission factor in kg/tkm)

| Symbol | Meaning | Unit |
|--------|---------|------|
| E(D) | Transport emissions | kgCO2e |
| W | Shipment weight | kg |
| D | Travel distance | km |
| m | Transport mode | {road, rail, iww, sea, air} |
| EFₘ | Emission factor for mode m | gCO2e/tkm |
| sₘ(D) | Share of tonne-km by mode m | fraction (0-1) |

#### Generalisation of Transport Modes

Each transport mode has multiple subtypes with different emission factors. We average across subtypes using observed usage shares:

```
EFₘ = Σₖ EFₘ,ₖ × uₘ,ₖ    (weighted average for mode m)
```

| Symbol | Meaning |
|--------|---------|
| EFₘ,ₖ | Emission factor of subtype k within mode m |
| uₘ,ₖ | Usage share of subtype k (Σuₘ,ₖ = 1) |
| Kₘ | Set of subtypes for mode m |

**Resulting Generalised Emission Factors:**

| Mode | EF (gCO2e/tkm) | Calculation |
|------|----------------|-------------|
| Road | 72.9 | 83.1% HGV >30t (74) + 16.9% HGV ≤30t (67.63) |
| Rail | 22.0 | Single generic freight rail |
| Inland Waterway | 31.0 | Single generic barge |
| Sea | 10.3 | 75% deep-sea (8.4) + 25% short-sea (16.0) |
| Air | 782.0 | 48.4% freighter (560) + 51.6% belly-hold (990) |

*Sources: Eurostat, CE Delft STREAM 2020, IATA Cargo Analysis 2024*

#### Multinomial Logit Modal Split Model

To estimate which modes are used at different distances, we use a **multinomial logit model**:

**Mode Probability:**
```
         exp(Uₘ(D))
Pₘ(D) = ────────────────
        Σₖ exp(Uₖ(D))
```

**Utility Function:**
```
Uₘ(D) = β₀,ₘ + β₁,ₘ × ln(D)
```

| Symbol | Meaning |
|--------|---------|
| Pₘ(D) | Probability of using mode m at distance D |
| Uₘ(D) | Utility (attractiveness) of mode m at distance D |
| β₀,ₘ | Mode-specific intercept (baseline attractiveness) |
| β₁,ₘ | Sensitivity to log-distance |

**Road is the reference mode** with U_road ≡ 0 (all road β parameters = 0).

#### Calibrated β Parameters

Parameters estimated from CE Delft (2011) Rhine corridor data and calibrated for sea/air using EU freight statistics:

| Mode | β₀ | β₁ | Source |
|------|----|----|--------|
| Road | 0 | 0 | Reference mode |
| Rail | -10.537 | 1.372 | CE Delft 2011 (estimated) |
| Inland Waterway | -5.770 | 0.762 | CE Delft 2011 (estimated) |
| Sea | -17.108 | 2.364 | Calibrated to EU maritime share (~67%) |
| Air | -17.345 | 1.881 | Calibrated to EU air share (~0.2%) |

#### Weighted Emission Factor Calculation

```
EF_weighted(D) = Σₘ Pₘ(D) × EFₘ
```

**Example at D = 12,847 km (Bangladesh → Europe):**

| Mode | Utility | Probability | EF × P |
|------|---------|-------------|--------|
| Road | 0 | 0.05 | 3.6 |
| Rail | -10.54 + 1.37×9.46 = 2.42 | 0.19 | 4.2 |
| IWW | -5.77 + 0.76×9.46 = 1.42 | 0.07 | 2.2 |
| Sea | -17.11 + 2.36×9.46 = 5.23 | 0.60 | 6.2 |
| Air | -17.35 + 1.88×9.46 = 0.44 | 0.09 | 70.4 |
| **Total** | | **1.00** | **86.6 gCO2e/tkm** |

```
Transport carbon = (0.934/1000) × 12847 × (86.6/1000) = 1.04 kgCO2e
```

---

### Water Footprint

**Formula:**
```
water_total = Σ (W × Pᵢ × WFᵢ)
```

| Symbol | Meaning |
|--------|---------|
| W | Product weight (kg) |
| Pᵢ | Percentage of material i (0-1) |
| WFᵢ | Water footprint factor for material i (L/kg) |

**Key Water Factors:**

| Material | Water (L/kg) | Source |
|----------|--------------|--------|
| Cotton (conventional) | 9,113 | Water Footprint Network |
| Wool (merino) | 170,000 | CSIRO |
| Leather (bovine) | 17,100 | ScienceDirect |
| Polyester (virgin) | 60 | Idemat 2026 |
| Polyester (recycled) | 35 | Literature estimate |

**Note:** Water footprint includes blue water (surface/groundwater) and green water (rainwater), but not grey water (dilution of pollutants).

---

### Data Sources

| Source | Usage | License |
|--------|-------|---------|
| [TU Delft Idemat 2026](https://www.ecocostsvalue.com/idemat/) | Material carbon factors | Academic |
| [CE Delft STREAM 2020](https://cedelft.eu/) | Transport emission factors | Academic |
| [CE Delft 2011 IWT Report](https://cedelft.eu/) | Modal split β parameters | Academic |
| [Water Footprint Network](https://waterfootprint.org/) | Water factors | Open Access |
| [GLEC Framework v3.0](https://www.smartfreightcentre.org/) | Transport methodology | Public |
| [IATA Cargo Analysis 2024](https://www.iata.org/) | Air freight modal shares | Public |

## Robust ML Model

With data generated and footprints calculated, we train XGBoost models to **predict footprints from partial product information** — the key innovation that provides value beyond direct calculation.

### The Problem

Direct calculation requires **all inputs**:
- Exact material composition (34 percentage values)
- Product weight (kg)
- Transport distance (km)

Real-world data often has:
- Missing material details ("100% cotton" without specifics)
- Unknown weights or manufacturing origins
- Incomplete legacy databases

### Our Approach: Two-Model Strategy

| Model | Training Method | Accuracy (Complete) | Accuracy (40% Missing) | Use Case |
|-------|-----------------|---------------------|------------------------|----------|
| **Baseline** | Standard training | R² = 0.9999 | R² = -0.991 | Maximum accuracy when all data available |
| **Robustness** | 20% feature dropout | R² = 0.9999 | R² = 0.936 | Production with incomplete data |

---

### Model Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       XGBoost Multi-Output Regressor                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Input Features (42 total):                                                 │
│  ├── Categorical (one-hot encoded):                                        │
│  │   ├── gender (2 values)                                                  │
│  │   ├── parent_category (5 values)                                         │
│  │   └── category (86 values)                                               │
│  ├── Numerical:                                                              │
│  │   ├── weight_kg                                                          │
│  │   └── total_distance_km                                                  │
│  └── Material Percentages (34):                                              │
│      ├── cotton_conventional, cotton_organic, cotton_recycled               │
│      ├── polyester_virgin, polyester_recycled, polyamide_6, ...             │
│      └── leather_bovine, wool_merino, silk, down_duck, ...                  │
│                                                                              │
│  XGBoost Configuration:                                                      │
│  ├── n_estimators: 1000                                                      │
│  ├── max_depth: 8                                                            │
│  ├── learning_rate: 0.05                                                     │
│  ├── subsample: 0.8                                                          │
│  ├── colsample_bytree: 0.8                                                   │
│  ├── early_stopping_rounds: 50                                               │
│  └── device: cuda (GPU accelerated)                                          │
│                                                                              │
│  Target Outputs (4):                                                         │
│  ├── carbon_material (kgCO2e)                                               │
│  ├── carbon_transport (kgCO2e)                                              │
│  ├── carbon_total (kgCO2e)                                                  │
│  └── water_total (liters)                                                   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

### Training Configuration

| Parameter | Baseline | Robustness |
|-----------|----------|------------|
| Training samples | 676,178 | 676,178 |
| Validation samples | 225,393 | 225,393 |
| Feature dropout | 0% | 20% random masking |
| GPU training time | ~2 minutes | ~2 minutes |
| Split method | Stratified by category | Stratified by category |

---

### How Feature Dropout Works

During robustness training, we randomly set 20% of features to 0 for each sample:

```python
def apply_feature_dropout(X, dropout_rate=0.2):
    mask = np.random.random(X.shape) > dropout_rate
    return X * mask  # Masked features become 0
```

**This forces the model to:**

1. **Learn redundant patterns**: If `cotton_conventional` is masked, infer from `category=Jeans`
2. **Use feature correlations**: Heavy products from certain countries → higher footprints
3. **Not over-rely on any single feature**: Predictions remain stable under partial information

---

### Performance Results

Both models were trained on 676,178 products and evaluated on a held-out set of 225,393 products (25% stratified split).

#### Baseline Model (Complete Data)

| Target | MAE | RMSE | R² | MAPE |
|--------|-----|------|-----|------|
| Carbon Material | 0.041 kgCO2e | 0.146 kgCO2e | 0.9999 | 0.83% |
| Carbon Transport | 0.0008 kgCO2e | 0.0018 kgCO2e | 0.9998 | — |
| Carbon Total | 0.044 kgCO2e | 0.146 kgCO2e | 0.9999 | 0.95% |
| Water Total | 115.3 L | 570.6 L | 0.9998 | 0.81% |

**Interpretation:**
- **R² = 0.9999** = model explains 99.99% of variance
- **MAE = 0.041 kgCO2e** = predictions off by ~41 grams CO2e on average

#### Robustness Model (Complete Data)

| Target | MAE | RMSE | R² | MAPE |
|--------|-----|------|-----|------|
| Carbon Material | 0.045 kgCO2e | 0.166 kgCO2e | 0.9999 | 0.98% |
| Carbon Transport | 0.0013 kgCO2e | 0.0026 kgCO2e | 0.9997 | — |
| Carbon Total | 0.050 kgCO2e | 0.168 kgCO2e | 0.9999 | 1.17% |
| Water Total | 132.9 L | 746.5 L | 0.9996 | 1.12% |

**Trade-off:** Slightly higher MAE on complete data, but dramatically better when data is missing.

---

### Robustness Under Missing Data

**The critical differentiator** — simulating real-world incomplete product information:

#### R² Comparison at Different Missing Levels

| Missing % | Model | Carbon Material | Carbon Total | Water Total |
|-----------|-------|-----------------|--------------|-------------|
| 0% | Baseline | 0.9999 | 0.9999 | 0.9998 |
| 0% | Robustness | 0.9999 | 0.9999 | 0.9996 |
| 20% | Baseline | **0.001** | 0.306 | 0.575 |
| 20% | Robustness | **0.968** | 0.968 | 0.951 |
| 40% | Baseline | **-0.991** | -0.380 | 0.146 |
| 40% | Robustness | **0.936** | 0.936 | 0.902 |

#### MAE Comparison (20% Missing)

| Target | Baseline | Robustness | Improvement |
|--------|----------|------------|-------------|
| Carbon Material | 5.04 kgCO2e | **0.29 kgCO2e** | 17× better |
| Carbon Total | 4.12 kgCO2e | **0.29 kgCO2e** | 14× better |
| Water Total | 7,181 L | **772 L** | 9× better |

---

### Why Is Accuracy So High? (Not Data Leakage)

The near-perfect R² = 0.9999 may appear suspicious, but it is **expected behavior**:

**The targets are deterministically calculated from input features:**

```
Input Features:
  - weight_kg
  - total_distance_km  
  - 34 material percentages

Target Calculations (C footprint calculator):
  carbon_material  = Σ (weight × material_% × carbon_factor)
  water_total      = Σ (weight × material_% × water_factor)
  carbon_transport = f(weight, distance, modal_split)
  carbon_total     = carbon_material + carbon_transport
```

The XGBoost model **learns the calculation formulas**. This is analogous to training a model to predict rectangle area from length × width — near-perfect accuracy is expected.

---

### Model Value Assessment

| Use Case | Value |
|----------|-------|
| Replace C calculator | Low — formulas known |
| **Handle missing data** | **High** — R² > 0.93 with 40% missing |
| Approximate footprints | High — works with partial info |
| Simpler deployment | Medium — no C library needed |
| Feature importance | Medium — insight into drivers |


---

### Training on Google Colab (GPU Acceleration)

For faster training, use the provided Colab notebook:

**Colab Advantages:**
- Free GPU (Tesla T4) → training in ~2-4 minutes vs 6-8 hours on CPU
- Pre-configured environment
- No local setup required

**Notebook:** `models/train_on_colab.ipynb`

**Quick Start:**
1. Upload `train.csv`, `validate.csv`, `material_dataset_final.csv` to Google Drive
2. Open notebook in Colab
3. Enable GPU (Runtime → Change runtime type → GPU)
4. Run all cells

**Output Files:**
```
trained_model/
├── baseline/
│   ├── xgb_model.json
│   ├── preprocessor.pkl
│   └── trainer_config.pkl
└── robustness/
    ├── xgb_model.json
    ├── preprocessor.pkl
    └── trainer_config.pkl
```

---

## Quick Start
```bash
# 1. Clone repository
git clone https://github.com/yourusername/bulk_product_generator.git
cd bulk_product_generator

# 2. Install Git LFS (required for dataset files)
git lfs install
git lfs pull

# 3. Set up Python environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 4. Install dependencies
pip install -r data/data_creation/requirements.txt

# 5. Configure API key
cp .env.example .env
# Edit .env and add your GEMINI_API_KEY

# 6. Run the full pipeline (optional - datasets already included)
# Generate products
cd data/data_creation
python scripts/main.py

# Calculate footprints
cd ../data_calculations
make run

# Train models
cd ../../models
python train_model.py
```

**Note**: The repository already contains pre-generated datasets in the `datasets/` directory, so you can skip steps 6 and directly use the data for analysis or model development.

---

## Detailed Usage

### 1. Product Data Generation
Generate realistic fashion product data using Google Gemini API.

**Location**: `data/data_creation/`

```bash
cd data/data_creation

# Configure settings
# Edit config/config.py to adjust:
# - Number of products per category
# - Batch sizes
# - API rate limits

# Run generation
python scripts/main.py
```

**Key Features**:
- **Checkpoint System**: Automatically saves progress every N products
- **Rate Limiting**: Respects API quotas and implements exponential backoff
- **Retry Logic**: Handles transient API failures gracefully
- **Structured Prompts**: Uses carefully crafted prompts for consistent output quality

**Output**: `data/data_creation/output/Product_data.csv`

**Configuration Options** (`config/config.py`):
```python
PRODUCTS_PER_CATEGORY = 2000  # Number of products per category
BATCH_SIZE = 10               # Products per API call
CHECKPOINT_INTERVAL = 100     # Save progress every N products
MAX_RETRIES = 3               # API retry attempts
API_TIMEOUT = 30              # Request timeout (seconds)
```

**Generated Fields**:
- Product name (creative, realistic)
- Gender (Female, Male)
- Parent category (Tops, Bottoms, Outerwear, etc.)
- Subcategory (Maxi Skirts, T-Shirts, Winter Jackets, etc.)
- Manufacturing country (ISO 3166-1 alpha-2 code)
- Material composition (JSON format, percentages sum to 1.0)
- Weight (kg, realistic for product type)
- Transport distance (km, based on manufacturer location)

---

### 2. Data Correction
Validate and correct generated data to ensure quality and consistency.

**Location**: `data/data_correction/`

```bash
cd data/data_correction

# Run validation scripts
python scripts/validation/validate_schema.py

# Run correction
python scripts/cleanup/correct_data.py
```

**Validation Checks**:

| Check | Rule | Action |
|-------|------|--------|
| Gender | Must be in {Female, Male} | Correct or remove |
| Categories | Must match predefined hierarchy | Validate against list |
| Country Codes | Must be valid ISO 3166-1 alpha-2 | Validate or remove |
| Materials | JSON format, sum = 1.0 ± 0.01 | Normalize or remove |
| Weight | 0.05 ≤ weight ≤ 5.0 kg | Flag outliers |
| Distance | 100 ≤ distance ≤ 25,000 km | Flag outliers |
| Duplicates | Exact match on all fields | Remove duplicates |

**Output**: `data/data_correction/output/Product_data_final.csv`

**Quality Metrics**:
- **Pass Rate**: ~95% of generated products pass validation
- **Correction Rate**: ~3% corrected automatically
- **Removal Rate**: ~2% removed as invalid

---

### 3. Footprint Calculation
Calculate carbon and water footprints using validated reference data.

**Location**: `data/data_calculations/`

**Language**: C (for high-performance processing of 900k+ products)

```bash
cd data/data_calculations

# Build the calculator
make

# Run calculations
make run

# Or run with custom input
./build/footprint_calculator input/Product_data_final.csv output/results.csv
```

**Calculation Methods**:

#### Material Carbon Footprint
```
carbon_material = Σ (weight_kg × material_percentage × carbon_factor_kgCO2e)
```

Example: Cotton T-shirt (0.2 kg, 100% cotton, 5.5 kgCO2e/kg)
```
carbon_material = 0.2 × 1.0 × 5.5 = 1.1 kgCO2e
```

#### Transport Carbon Footprint
Uses a **multinomial logit modal split model** to estimate transport mode probabilities:

```
Utility: U_m(D) = β0_m + β1_m × ln(D)
Probability: P_m(D) = exp(U_m(D)) / Σ exp(U_k(D))
Emissions: carbon_transport = (weight_kg / 1000) × distance_km × weighted_EF / 1000
Weighted EF: weighted_EF = Σ P_m(D) × EF_m
```

Where:
- `D` = transport distance (km)
- `m` = transport mode (road, rail, sea, air, inland waterway)
- `β0, β1` = model parameters (from `utility_attractiveness.csv`)
- `EF_m` = emission factor for mode m (gCO2e/tkm)

**Transport Modes and Emission Factors**:
| Mode | EF (gCO2e/tkm) | Typical Use Case |
|------|---------------|------------------|
| Sea | 10.5 | Long-distance intercontinental |
| Rail | 22.0 | Medium-distance continental |
| Inland Waterway | 31.0 | River/canal transport |
| Road | 62.0 | Last-mile delivery |
| Air | 602.0 | Urgent/high-value items |

#### Material Water Footprint
```
water_total = Σ (weight_kg × material_percentage × water_factor_liters)
```

Example: Cotton T-shirt (0.2 kg, 100% cotton, 10,000 L/kg)
```
water_total = 0.2 × 1.0 × 10,000 = 2,000 liters
```

**Output**: `data/data_calculations/output/Product_data_with_footprints.csv`

**Performance**: Processes 900,000 products in ~45 seconds on modern hardware

---

### 4. Machine Learning Models
Train neural networks to predict environmental footprints from product attributes.

**Location**: `models/` and `Trained-Implementation/`

```bash
cd models

# Train baseline model
python train_model.py --mode baseline --epochs 2000

# Train robustness model (with data augmentation)
python train_model.py --mode robustness --epochs 2000

# Evaluate models
python evaluate_model.py --model baseline
python evaluate_model.py --model robustness --test-corruption
```

**Model Architecture**:
```
Input Layer (Feature Engineering)
    ↓
Embedding Layers (Categorical Variables)
    • gender → 8-dim embedding
    • category → 16-dim embedding
    • country → 16-dim embedding
    ↓
Concatenate with Numerical Features
    • weight_kg
    • total_distance_km
    • material percentages (34 features)
    ↓
Dense Layer 1: 128 units, ReLU, Dropout(0.3)
    ↓
Dense Layer 2: 64 units, ReLU, Dropout(0.3)
    ↓
Dense Layer 3: 32 units, ReLU
    ↓
Output Layer: 4 units (linear)
    • carbon_material
    • carbon_transport
    • carbon_total
    • water_total
```

**Training Configuration**:
- **Optimizer**: Adam (lr=0.001)
- **Loss**: Mean Squared Error (MSE)
- **Metrics**: MAE, R²
- **Batch Size**: 32
- **Epochs**: 2000
- **Early Stopping**: Patience=50 (disabled for final training)

**Robustness Training**:
- **Data Augmentation**: Random dropout of 20% of features during training
- **Purpose**: Improve model resilience to missing data in production
- **Fallback Features**: 22 engineered features for robustness

**Output**:
- `datasets/model_outputs/baseline/baseline_predictions.csv`
- `datasets/model_outputs/robustness/baseline_predictions.csv`
- `datasets/model_outputs/{baseline,robustness}/robustness_results.csv`

---


### Downloading Trained Models

After training, download models for local use:

1. Run Cell 5 (Download Model)
2. Extracts `trained_model.zip` containing:
   ```
   trained_model/
   ├── baseline/
   │   ├── xgb_model.json      # XGBoost model weights
   │   ├── preprocessor.pkl    # Sklearn preprocessor (scalers, encoders)
   │   └── trainer_config.pkl  # Feature names, target names, config
   └── robustness/
       └── ... (same structure)
   ```

### Using the Trained Model

#### Required Input Features

To use the model, you must provide these features for each product:

| Feature | Type | Description | Example |
|---------|------|-------------|---------|
| `gender` | categorical | Male, Female | "Female" |
| `parent_category` | categorical | Tops, Bottoms, Outerwear, Footwear, Dresses | "Bottoms" |
| `category` | categorical | Leaf category (86 options) | "Jeans" |
| `weight_kg` | float | Product weight in kg | 0.934 |
| `total_distance_km` | float | Transport distance in km | 12847 |
| `cotton_conventional` | float | Material percentage (0-1) | 0.92 |
| `elastane` | float | Material percentage (0-1) | 0.08 |
| ... | ... | (34 material columns total, unused = 0) | ... |

**Important:** All 34 material columns must be present, even if most are 0. Material percentages must sum to 1.0.

#### Complete Usage Example

```python
import joblib
import xgboost as xgb
import pandas as pd
import numpy as np

# 1. Load model and preprocessor
model = xgb.Booster()
model.load_model('robustness/xgb_model.json')
preprocessor = joblib.load('robustness/preprocessor.pkl')
config = joblib.load('robustness/trainer_config.pkl')

# 2. Prepare input data (example product)
product = {
    'gender': 'Male',
    'parent_category': 'Bottoms',
    'category': 'Jeans',
    'weight_kg': 0.934,
    'total_distance_km': 12847,
    # Material percentages (must sum to 1.0)
    'cotton_conventional': 0.92,
    'elastane': 0.08,
    # All other materials = 0 (34 total columns required)
}

# Fill missing material columns with 0
all_materials = config['material_columns']  # List of 34 material names
for mat in all_materials:
    if mat not in product:
        product[mat] = 0.0

# 3. Convert to DataFrame
df = pd.DataFrame([product])

# 4. Preprocess (applies one-hot encoding, scaling)
X_processed = preprocessor.transform(df)

# 5. Predict
dmatrix = xgb.DMatrix(X_processed, feature_names=config['feature_names'])
predictions = model.predict(dmatrix)

# 6. Extract outputs
carbon_material = predictions[0][0]   # kgCO2e
carbon_transport = predictions[0][1]  # kgCO2e
carbon_total = predictions[0][2]      # kgCO2e
water_total = predictions[0][3]       # liters

print(f"Carbon Material:  {carbon_material:.2f} kgCO2e")
print(f"Carbon Transport: {carbon_transport:.2f} kgCO2e")
print(f"Carbon Total:     {carbon_total:.2f} kgCO2e")
print(f"Water Total:      {water_total:.0f} liters")
```

**Expected Output:**
```
Carbon Material:  1.22 kgCO2e
Carbon Transport: 1.04 kgCO2e
Carbon Total:     2.26 kgCO2e
Water Total:      7868 liters
```

#### Preprocessing Details

The `preprocessor.pkl` contains a scikit-learn `ColumnTransformer` that:

1. **Categorical encoding**: One-hot encodes `gender`, `parent_category`, `category`
2. **Numerical scaling**: StandardScaler on `weight_kg`, `total_distance_km`
3. **Material passthrough**: Material percentages are passed through unchanged

```
Input (42 raw features)
    ↓
Preprocessor
    ├── OneHotEncoder(gender)        → 2 columns
    ├── OneHotEncoder(parent_cat)    → 5 columns
    ├── OneHotEncoder(category)      → 86 columns
    ├── StandardScaler(weight_kg)    → 1 column
    ├── StandardScaler(distance_km)  → 1 column
    └── Passthrough(materials)       → 34 columns
    ↓
Output (129 processed features)
```

#### Handling Missing Data with Robustness Model

The robustness model was trained with 20% feature dropout, so it can handle missing inputs:

```python
# If you don't know the material composition:
product = {
    'gender': 'Female',
    'parent_category': 'Tops',
    'category': 'T-Shirts',
    'weight_kg': 0.25,
    'total_distance_km': 8500,
    # Materials unknown → set all to 0 or use category defaults
}

# Fill all materials with 0 (model will infer from category)
for mat in all_materials:
    product[mat] = 0.0

# Prediction will still work (R² ≈ 0.93-0.96 accuracy)
```

**When to use which model:**
- **Baseline model**: Use when you have complete material data (R² = 0.9999)
- **Robustness model**: Use when material data is missing/incomplete (R² = 0.93-0.96)

---

## Datasets

All datasets are organized in the `datasets/` directory and documented in detail in [`datasets/README.md`](datasets/README.md).

### Quick Reference

| Dataset | Rows | Description | Path |
|---------|------|-------------|------|
| **Raw Products** | 878k | AI-generated product data | `datasets/raw/Product_data.csv` |
| **Processed Products** | 902k | Validated product data | `datasets/processed/Product_data_final.csv` |
| **Complete Dataset** | 902k | Products + footprints | `datasets/processed/Product_data_with_footprints.csv` |
| **Material Reference** | 34 | Material footprint factors | `datasets/reference/material_dataset_final.csv` |
| **Transport Reference** | 5 | Emission factors by mode | `datasets/reference/transport_emission_factors_generalised.csv` |
| **Training Set** | 676k | 75% stratified split | `datasets/splits/train.csv` |
| **Validation Set** | 225k | 25% stratified split | `datasets/splits/validate.csv` |
| **Model Predictions** | 225k | Baseline model outputs | `datasets/model_outputs/baseline/baseline_predictions.csv` |
| **Robustness Results** | varies | Performance under corruption | `datasets/model_outputs/*/robustness_results.csv` |

**File Format**: All datasets are CSV files tracked with Git LFS

**Documentation**: See [`datasets/README.md`](datasets/README.md) for complete schema descriptions, data quality notes, and usage guidelines.

> **Note**: Processed data (902k) contains more products than raw data (878k) because additional products were generated for underrepresented categories (e.g., Dresses, Gowns) to ensure balanced distribution across all 86 fashion categories. This improves ML model generalization.

---

## Project Structure
```
bulk_product_generator/
│
├── README.md                         # This file
├── LICENSE                           # Project license
├── .gitignore                        # Git exclusions
├── .gitattributes                    # Git LFS configuration
├── .env.example                      # Environment variable template
│
├── datasets/                         # Organized datasets (Git LFS)
│   ├── README.md                     # Dataset documentation
│   ├── raw/                          # Original generated data
│   ├── processed/                    # Validated + footprint data
│   ├── reference/                    # Material & transport factors
│   ├── splits/                       # Train/validation splits
│   └── model_outputs/                # Model predictions & metrics
│       ├── baseline/
│       └── robustness/
│
├── data/                             # Data processing modules
│   ├── data_creation/                # Product generation (Python)
│   │   ├── config/                   # Configuration files
│   │   ├── data/                     # Input category definitions
│   │   ├── scripts/                  # Executable scripts
│   │   ├── fashion_generator/        # Core generator module
│   │   ├── output/                   # Generated outputs
│   │   └── requirements.txt          # Python dependencies
│   │
│   ├── data_correction/              # Data validation (Python)
│   │   ├── scripts/
│   │   │   ├── validation/           # Schema validation
│   │   │   ├── cleanup/              # Data correction
│   │   │   └── analysis/             # Data analysis
│   │   ├── input/                    # Input files
│   │   └── output/                   # Corrected outputs
│   │
│   ├── data_calculations/            # Footprint calculation (C)
│   │   ├── src/
│   │   │   ├── carbon/               # Carbon calculators
│   │   │   ├── water/                # Water calculators
│   │   │   ├── utils/                # CSV/JSON parsers
│   │   │   └── footprint_calculator.c
│   │   ├── include/                  # Header files
│   │   ├── build/                    # Compiled binaries
│   │   ├── input/                    # Input data
│   │   ├── output/                   # Output data
│   │   ├── Makefile                  # Build system
│   │   └── README.md                 # Module documentation
│   │
│   └── data_splitter/                # Train/val splitting (Python)
│       └── output/                   # Split outputs
│
├── models/                           # ML model training (Python)
│   ├── train_model.py                # Training script
│   ├── evaluate_model.py             # Evaluation script
│   └── data_input/                   # Model input data
│
└── Trained-Implementation/           # Trained models & results
    └── trained_model/
        ├── baseline/                 # Baseline model
        │   └── evaluation/
        └── robustness/               # Robustness model
            └── evaluation/
```

---

## Installation

### Prerequisites
**Required:**
- Python 3.9+
- GCC (C11 support)
- Make
- Git LFS

**Python Libraries** (see `requirements.txt`):
- google-generativeai
- pandas
- numpy
- tensorflow/keras
- scikit-learn
- python-dotenv

**C Libraries**:
- libm (math library, usually included)
- Standard C library

### Setup Steps
#### 1. Install Git LFS

Git LFS is required to download the CSV dataset files.

**Ubuntu/Debian:**
```bash
sudo apt-get install git-lfs
git lfs install
```

**macOS:**
```bash
brew install git-lfs
git lfs install
```

**Windows:**
Download from https://git-lfs.github.com/

#### 2. Clone Repository

```bash
git clone https://github.com/yourusername/bulk_product_generator.git
cd bulk_product_generator
git lfs pull  # Download LFS files (datasets)
```

#### 3. Python Environment

```bash
# Create virtual environment
python3 -m venv venv

# Activate (Linux/macOS)
source venv/bin/activate

# Activate (Windows)
venv\Scripts\activate

# Install dependencies
pip install -r data/data_creation/requirements.txt
```

#### 4. Environment Variables

```bash
# Copy template
cp .env.example .env

# Edit .env and add your API key
# GEMINI_API_KEY=your_api_key_here
```

**Get a Gemini API Key**:
1. Visit https://ai.google.dev/
2. Sign in with Google account
3. Create API key
4. Add to `.env` file

#### 5. C Compilation (for footprint calculator)

```bash
cd data/data_calculations

# Check requirements
gcc --version  # Should show GCC with C11 support

# Build
make

# Verify
./build/footprint_calculator --help
```

#### 6. Verify Installation

```bash
# Test Python imports
python -c "import google.generativeai; import pandas; import tensorflow; print('OK')"

# Test data access
ls datasets/processed/Product_data_with_footprints.csv

# Test C calculator
cd data/data_calculations
make run
```

---


## Citation
If you use this dataset or code in your research, please cite:

```bibtex
@software{Avelero_HydroCarbo_V1,
  author = {Moussa Ouallaf (Avelero BV.)},
  title = {Avelero Carbon Footprint HydroCarbo Calculator},
  year = {2025},
  url = {https://github.com/Avelero/Avelero_HydroCarbo},
  version = {1.0}
}
```

---

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

### Third-Party Data Licenses

While the code and synthetic product data are open source under MIT, this project uses reference data from various sources. Please respect their respective licenses:

| Data Source | License / Terms | Usage |
|-------------|-----------------|-------|
| **[TU Delft Idemat 2026](https://www.ecocostsvalue.com/idemat/)** | Free for non-commercial use; commercial use requires license | Material carbon footprints |
| **[Smart Freight Centre GLEC Framework](https://www.smartfreightcentre.org/glec/)** | Open for non-commercial use | Transport emission methodology |
| **[CE Delft STREAM 2020](https://cedelft.eu/)** | Research and academic use permitted | Emission factors (road, rail, sea, inland waterway) |
| **[Eurostat](https://ec.europa.eu/eurostat)** | Eurostat Open Data License (free reuse with attribution) | EU freight transport statistics |
| **[IATA](https://www.iata.org/)** | Public reports; check terms for commercial use | Air cargo capacity data |
| **[Water Footprint Network](https://waterfootprint.org/)** | Creative Commons / Open Access reports | Polyester & viscose water footprints |
| **[ScienceDirect / Elsevier](https://www.sciencedirect.com/)** | Individual article licenses apply | Peer-reviewed water footprint studies |
| **[MDPI Journals](https://www.mdpi.com/)** | Open Access (CC BY 4.0) | Recycled cotton LCA, jute fiber studies |
| **[Springer Nature](https://link.springer.com/)** | Individual article licenses apply | Down feather ecosystem studies |
| **[SAGE Journals](https://journals.sagepub.com/)** | Individual article licenses apply | Textile water consumption studies |
| **[Fairtrade Foundation](https://www.fairtrade.net/)** | Open research reports | Organic cotton data |
| **[CSIRO / Australian Wool Innovation](https://www.woolfacts.com/)** | Educational use permitted | Wool water footprint data |

### Attribution Requirements

When using this project's datasets or results:
1. **Cite this repository** (see [Citation](#citation) section)
2. **Acknowledge primary data sources** — In particular, cite TU Delft Idemat and Smart Freight Centre GLEC Framework if publishing research
3. **Do not redistribute raw Idemat data** — The material footprint factors are derived values; original Idemat database requires separate license for redistribution

---

## Acknowledgments

### Data Generation
- **[Google Gemini Team](https://ai.google.dev/)** for providing the API for synthetic product data generation

### Material Carbon Footprints
- **[TU Delft Idemat 2026](https://www.ecocostsvalue.com/idemat/)** — Primary source for material carbon footprints (acrylic, cotton, elastane, jute, leather, linen, nylon, polyester, viscose, wool, and more)
- **[Carbonfact](https://www.carbonfact.com/)** — Research-based carbon footprint values for polyester, vegan leather, and synthetic down
- **[MDPI Sustainability Journal](https://www.mdpi.com/2071-1050/16/14/5896)** — Recycled cotton fiber life cycle assessment (Portugal case study 2024)
- **[PlasticsEurope TPU Eco-profile](https://www.plasticseurope.org/)** — Thermoplastic polyurethane (TPU) resin footprint data
- **[Impactful Ninja](https://impactful.ninja/)** — Sustainability research for cashmere, hemp, and TENCEL Lyocell fabrics
- **[CO2 Everything](https://www.co2everything.com/)** — Silk production carbon footprint data

### Material Water Footprints
- **[Water Footprint Network](https://waterfootprint.org/)** — Polyester and viscose water footprint assessment (2017)
- **[ScienceDirect / Elsevier](https://www.sciencedirect.com/)** — Peer-reviewed water footprint studies for cashmere, natural rubber, steel, and silver
- **[University of Nebraska-Lincoln Digital Commons](https://digitalcommons.unl.edu/)** — Cotton water footprint documentation
- **[Fairtrade Foundation](https://www.fairtrade.net/)** — Organic cotton sustainability research
- **[Springer Nature](https://link.springer.com/)** — Down feather ecosystem water consumption study
- **[SAGE Journals](https://journals.sagepub.com/)** — Polyester and synthetic textiles water consumption research
- **[Circumfauna](https://circumfauna.org/)** — Leather water footprint analysis
- **[Arjen Hoekstra (WF Expert)](https://ayhoekstra.nl/)** — Water footprint presentations for hemp and flax/linen
- **[MDPI Materials Journal](https://www.mdpi.com/1996-1944/13/16/3541)** — Jute fiber water footprint study
- **[Wiley Online Library](https://onlinelibrary.wiley.com/)** — Silk production water usage in textile industry
- **[Journée Mondiale](https://www.journee-mondiale.com/)** — Lyocell vs cotton water consumption comparison
- **[Fulgar S.p.A.](https://www.fulgar.com/)** — Textile industry water consumption data for synthetic fibers
- **[USGS Publications](https://pubs.usgs.gov/)** — Synthetic rubber water usage data
- **[NCBI / NIH](https://www.ncbi.nlm.nih.gov/)** — Gold mining water footprint research
- **[Polybags UK](https://www.polybags.co.uk/)** — Environmental data for brass/metal production
- **[Wool Facts (CSIRO / AWI)](https://www.woolfacts.com/)** — Wool water usage (Wayne Meyer / CSIRO estimates)

### Transport Emission Factors
- **[CE Delft STREAM 2020](https://cedelft.eu/)** — Emission factors for road, rail, inland waterway, and sea freight
- **[CE Delft (2011)](https://cedelft.eu/)** — Modal split model parameters for inland waterway transport in the EU
- **[Eurostat](https://ec.europa.eu/eurostat)** — EU road freight (ROAD_GO_TA_MPLW 2024) and short-sea shipping statistics
- **[IATA Air Cargo Market Analysis (November 2024)](https://www.iata.org/)** — Global cargo capacity shares for air freight emission calculations
- **[Smart Freight Centre GLEC Framework v3.0](https://www.smartfreightcentre.org/glec/)** — Global logistics emission calculation methodology

---

## Contact
For questions, suggestions, or collaboration:

- **GitHub Issues**: [Create an issue](https://github.com/yourusername/bulk_product_generator/issues)
- **Email**: moussa@avelero.com

---

## Version History

### v1.0.0 (2025-12-09)
- Initial release
- 900,000 product dataset
- Complete footprint calculation pipeline
- Baseline and robustness ML models
- Comprehensive documentation

---

**Made for sustainability and data science**
