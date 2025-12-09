# Bulk Product Generator with Environmental Footprint Analysis

[![Python Version](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![C Standard](https://img.shields.io/badge/C-C11-green.svg)](https://en.wikipedia.org/wiki/C11_(C_standard_revision))
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Git LFS](https://img.shields.io/badge/Git%20LFS-enabled-orange.svg)](https://git-lfs.github.com/)

An end-to-end pipeline for generating synthetic fashion product data and predicting environmental footprints using machine learning. This project addresses a critical gap in sustainability research: **the absence of large-scale, publicly available life cycle assessment (LCA) datasets for fashion products**. By combining LLM-generated synthetic data with physics-based footprint calculations, we create a robust training dataset for ML models that can predict carbon and water footprints even when product information is incomplete.

> **Proof of Concept**: This project is a proof-of-concept demonstrating the feasibility of ML-based environmental footprint prediction for fashion products. A production-ready version with full **ISO 14040/14044** compliance (Life Cycle Assessment standards), **PEF** (Product Environmental Footprint) methodology, and expanded scope (Scope 1-3 emissions, end-of-life modeling) is currently in development.

---

## Table of Contents
- [Overview](#overview)
- [Synthetic Data Generation](#synthetic-data-generation)
- [Key Features](#key-features)
- [Pipeline Architecture](#pipeline-architecture)
- [Quick Start](#quick-start)
- [Detailed Usage](#detailed-usage)
  - [1. Product Data Generation](#1-product-data-generation)
  - [2. Data Correction](#2-data-correction)
  - [3. Footprint Calculation](#3-footprint-calculation)
  - [4. Machine Learning Models](#4-machine-learning-models)
- [Training on Google Colab (GPU Acceleration)](#training-on-google-colab-gpu-acceleration)
- [Datasets](#datasets)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Results and Performance](#results-and-performance)
- [Citation](#citation)
- [License](#license)
- [Acknowledgments](#acknowledgments)

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
                    ⎛  W  ⎞         ⎛ Σₘ sₘ(D) × EFₘ ⎞
E(D) [kgCO2e] =    ⎜────⎟  ×  D  × ⎜─────────────────⎟
                    ⎝1000⎠         ⎝      1000       ⎠
```

| Symbol | Meaning |
|--------|---------|
| D | Travel distance (km) |
| W | Shipment weight (kg) |
| m | Transport mode ∈ {road, rail, iww, sea, air} |
| EFₘ | Emission factor for mode m (gCO2e/tkm) |
| sₘ(D) | Share of tonne-km by mode m (function of distance) |

#### Generalisation of Transport Modes

Each transport mode has multiple subtypes with different emission factors. We average across subtypes using observed usage shares:

```
EFₘ = Σₖ∈Kₘ EFₘ,ₖ × uₘ,ₖ
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
| **Baseline** | Standard training | R² = 0.9999 | R² = -0.991 ❌ | Maximum accuracy when all data available |
| **Robustness** | 20% feature dropout | R² = 0.9999 | R² = 0.936 ✓ | Production with incomplete data |

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
| 20% | Baseline | **0.001** ❌ | 0.306 | 0.575 |
| 20% | Robustness | **0.968** ✓ | 0.968 | 0.951 |
| 40% | Baseline | **-0.991** ❌ | -0.380 | 0.146 |
| 40% | Robustness | **0.936** ✓ | 0.936 | 0.902 |

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

**Data Leakage Verification:**

| Check | Status |
|-------|--------|
| Target values in inputs? | ❌ No |
| Train/test contamination? | ❌ No (stratified split) |
| Future information used? | ❌ No |
| Deterministic relationship? | ✅ Yes (explains high R²) |

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

### When to Use Which Model

```
┌─────────────────────────────────────┐
│   Do you have complete data?        │
└──────────────┬──────────────────────┘
               │
       ┌───────┴───────┐
       ▼               ▼
      YES              NO
       │               │
       ▼               ▼
┌──────────────┐ ┌──────────────────┐
│  Use Direct  │ │ Use Robustness   │
│ C Calculation│ │     Model        │
│  (Exact)     │ │ (R² = 0.93-0.99) │
└──────────────┘ └──────────────────┘
```

**Production Recommendations:**

1. **E-commerce platforms** → Robustness model (users provide partial data)
2. **Internal LCA tools** → Baseline model (complete data guaranteed)
3. **Quick estimates** → Robustness model (fast approximation)
4. **Legacy database analysis** → Robustness model (incomplete records)

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

## Key Features
- **LLM-powered Synthetic Data Generation**: Creates large, diverse, and realistic fashion product datasets.
- **Physics-Based Footprint Calculation**: Provides accurate carbon and water footprint labels using validated scientific methods.
- **Robust ML Models**: Predicts environmental footprints even with significant missing input data (up to 40% feature dropout).
- **High Performance**: C-based footprint calculator processes 900,000 products in ~45 seconds.
- **Scalable Architecture**: Designed for large-scale data generation and processing.
- **Comprehensive Data Validation**: Ensures data quality and consistency throughout the pipeline.

---

## Pipeline Architecture

The project is structured into four main stages:

1.  **Data Generation**: Uses Google Gemini 2.5 Flash to create synthetic product data.
2.  **Data Correction**: Validates and cleans the generated data.
3.  **Footprint Calculation**: Computes carbon and water footprints using C-based algorithms.
4.  **Machine Learning**: Trains models to predict footprints from product attributes.

```mermaid
graph TD
    A[Define Generation Space] --> B(Prompt Construction & API Calls)
    B --> C{Validation & Incremental Saving}
    C --> D[Synthetic Product Data .csv]
    D --> E[Data Correction & Cleaning]
    E --> F[Cleaned Product Data .csv]
    F --> G[C-based Footprint Calculator]
    G --> H[Product Data with Footprints .csv]
    H --> I[ML Model Training (XGBoost)]
    I --> J[Trained ML Model]
    J --> K[Footprint Prediction API]
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

## Training on Google Colab (GPU Acceleration)

For faster model training with GPU acceleration, we provide a Colab notebook that handles the entire setup process automatically.

### Why Use Colab?
**Local Training Limitations:**
- **CPU Training**: 676k samples on CPU takes ~6-8 hours
- **Memory**: Large datasets may exceed laptop RAM
- **Interruptions**: Long training sessions vulnerable to system crashes

**Colab Advantages:**
- **Free GPU Access**: Tesla T4 GPU accelerates training to ~2-4 minutes
- **No Setup Required**: Pre-configured environment with all dependencies
- **Cloud Storage**: Use Google Drive to store large CSV files
- **Reproducible**: Same environment every time

### Colab Notebook: `models/train_on_colab.ipynb`
The notebook provides a complete training pipeline specifically designed for Colab's environment and constraints.

#### Features

1. **Automatic Setup**
   - Clones latest code from GitHub (skips LFS files to save bandwidth)
   - Mounts Google Drive for CSV file access
   - Installs all Python dependencies
   - Verifies GPU availability

2. **Two Training Modes**
   - **Baseline Only**: Maximum accuracy on complete data (R² = 0.9999)
   - **With Robustness**: Also trains model with data augmentation for missing value handling

3. **Smart Data Loading**
   - Bypasses Git LFS bandwidth limits by loading CSVs from Google Drive
   - Required files (~360 MB total):
     - `train.csv` (269 MB, 676k products)
     - `validate.csv` (90 MB, 225k products)
     - `material_dataset_final.csv` (5 KB, 34 materials)

4. **Comprehensive Evaluation**
   - Tests models on clean data
   - Simulates missing data scenarios (0%, 20%, 40% corruption)
   - Generates robustness curves
   - Saves detailed evaluation reports

#### Usage Instructions

**Step 1: Prepare Google Drive**

Upload these 3 files to your Google Drive (e.g., `MyDrive/data/`):
```
/content/drive/MyDrive/data/
├── train.csv              # From datasets/splits/train.csv
├── validate.csv           # From datasets/splits/validate.csv
└── material_dataset_final.csv  # From datasets/reference/
```

You can download these files directly from the GitHub repository (tracked with Git LFS).

**Step 2: Open Colab Notebook**

1. Go to [Google Colab](https://colab.research.google.com/)
2. File → Open Notebook → GitHub
3. Enter repository URL: `https://github.com/Avelero/Avelero_HydroCarbon`
4. Select: `models/train_on_colab.ipynb`

**Step 3: Enable GPU**

1. Runtime → Change runtime type
2. Hardware accelerator → **GPU**
3. Save

**Step 4: Configure Training**

Edit the configuration cell:
```python
# Configuration
QUICK_TEST = False  # True = 10K samples (~30 sec), False = 676K samples (~4 min)
TRAIN_ROBUST = True  # True = train both baseline + robustness models
```

**Training Mode Options:**
| Mode | What It Does | Training Time | Models Produced |
|------|-------------|---------------|-----------------|
| Baseline only | Max accuracy on complete data | ~2 min | 1 model (baseline) |
| With robustness | Baseline + data augmentation | ~4 min | 2 models (baseline + robust) |
| Quick test | 10K sample test run | ~30 sec | Testing only |

**Step 5: Run Training**

Execute all cells in order:

1. **Cell 1**: Clone code & setup environment (~30 sec)
   - Clones latest code from GitHub
   - Mounts Google Drive
   - Copies CSV files from Drive to Colab
   - Installs dependencies

2. **Cell 2**: Check GPU (~5 sec)
   - Verifies GPU is available
   - Shows GPU name and memory

3. **Cell 3**: Train model (~2-4 min)
   - Trains baseline model (R² = 0.9999)
   - Optionally trains robustness model
   - Evaluates on validation set
   - Tests robustness to missing data

4. **Cell 4**: View results (~instant)
   - Displays final metrics
   - Shows model locations

5. **Cell 5**: Download models (optional)
   - Creates ZIP file
   - Downloads to local machine

### Expected Results
**Baseline Model (Complete Data):**
```
carbon_material:  R² = 0.9999, MAE = 0.04 kgCO2e
carbon_transport: R² = 0.9998, MAE = 0.001 kgCO2e
carbon_total:     R² = 0.9999, MAE = 0.04 kgCO2e
water_total:      R² = 0.9998, MAE = 115 L
```

**Robustness Test (30% Missing Data):**
| Model | Carbon R² | Water R² |
|-------|-----------|----------|
| Baseline (not trained for missing data) | ~0.30 | ~0.25 |
| Robustness (trained with augmentation) | **~0.85** | **~0.80** |

### Notebook Architecture
The notebook internally calls the Python training script:
```bash
python train_max_accuracy.py [OPTIONS]
```

This allows the same training code to work both locally and on Colab, with the notebook handling:
- Environment setup
- Data loading from Google Drive
- GPU configuration
- Result visualization


### Downloading Trained Models
After training, download models for local use:

1. Run Cell 5 (Download Model)
2. Extracts `trained_model.zip` containing:
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

3. Use in your own projects:
   ```python
   import joblib
   import xgboost as xgb

   # Load model
   model = xgb.Booster()
   model.load_model('baseline/xgb_model.json')

   # Load preprocessor
   preprocessor = joblib.load('baseline/preprocessor.pkl')
   ```

### Cost Considerations
**Colab Free Tier:**
- GPU runtime: Limited to ~12 hours per day
- Our training: ~4 minutes per run
- Sufficient for multiple experiments per day

**Colab Pro (if needed):**
- $9.99/month
- Faster GPUs (V100, A100)
- Longer runtime limits
- Priority access

**Recommendation**: Free tier is perfectly adequate for this project.

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

## Dataset Statistics

### Product Distribution

| Metric | Value |
|--------|-------|
| Total Products | 901,571 |
| Categories | 86 fashion categories |
| Materials | 34 unique materials |
| Countries | 277 (global coverage) |
| Gender Distribution | Female 61.5%, Male 38.2%, Other 0.3% |

### Environmental Footprints

| Metric | Min | Mean | Median | Max | Unit |
|--------|-----|------|--------|-----|------|
| Carbon (Material) | 0.15 | 3.2 | 2.8 | 18.5 | kgCO2e |
| Carbon (Transport) | 0.05 | 2.0 | 1.6 | 12.3 | kgCO2e |
| Carbon (Total) | 0.25 | 5.2 | 4.5 | 25.8 | kgCO2e |
| Water | 50 | 2,800 | 1,950 | 15,000 | liters |

### Highest Impact Materials (per kg)

| Material | Carbon | Water |
|----------|--------|-------|
| Wool | 17.0 kgCO2e | 125,000 L |
| Cashmere | 15.2 kgCO2e | 105,000 L |
| Leather | 14.0 kgCO2e | 17,000 L |
| Polyester (virgin) | 6.0 kgCO2e | 50 L |
| Cotton | 5.5 kgCO2e | 10,000 L |



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
