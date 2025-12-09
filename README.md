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

2. **Physics-Based Calculation**: Apply scientifically validated formulas (Idemat 2026, GLEC Framework) to calculate exact carbon and water footprints for each product

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


### Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        Data Generation Pipeline                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────────┐    ┌──────────────────┐    ┌──────────────────┐      │
│  │  categories_     │    │   vocabularies   │    │     prompts      │      │
│  │   rows.json      │───▶│      .py        │───▶│       .py        │      │
│  │  (86 categories) │    │  (34 materials)  │    │ (dynamic builder)│      │
│  └──────────────────┘    └──────────────────┘    └────────┬─────────┘      │
│                                                            │                 │
│                                                            ▼                 │
│  ┌──────────────────┐    ┌──────────────────┐    ┌──────────────────┐      │
│  │   generator.py   │◀───│  Gemini 2.5      │◀───│  Rate Limiter    │      │
│  │ (parallel chunks)│    │   Flash API      │    │  (Token Bucket)  │      │
│  └────────┬─────────┘    └──────────────────┘    └──────────────────┘      │
│           │                                                                  │
│           ▼                                                                  │
│  ┌──────────────────┐    ┌──────────────────┐                               │
│  │  csv_writer.py   │───▶│  checkpoint.py   │                               │
│  │ (incremental)    │    │ (resume support) │                               │
│  └──────────────────┘    └──────────────────┘                               │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Generation Components

#### 1. Hierarchical Category System

Categories are loaded from `categories_rows.json` with parent-child relationships:

```
Male/Female (Gender)
  └── Bottoms/Tops/Outerwear/Footwear/Dresses (Parent Category)
        └── Jeans/T-Shirts/Jackets/Sneakers/Maxi Dresses (Leaf Category)
```

- **86 leaf categories** across 5 parent categories
- Each category has a `parent_id` enabling hierarchy traversal
- `get_hierarchy_info()` extracts gender, parent, and category from any leaf node

#### 2. Material Vocabulary System

Materials are defined in `vocabularies.py` with category-specific combinations:

```python
MATERIAL_COMBINATIONS = {
    "Jeans": ["cotton_conventional", "cotton_organic", "elastane", "hemp"],
    "Leather Jackets": ["leather_bovine", "leather_ovine", "viscose", "metal_brass"],
    "Sneakers": ["leather_bovine", "polyester_virgin", "eva", "natural_rubber"],
    # ... 86 categories total
}
```

- **34 unique materials** with verified carbon/water footprint factors
- Category-appropriate suggestions guide realistic compositions
- Materials extracted dynamically: `MATERIAL_VOCAB = set(all materials used)`

#### 3. Dynamic Prompt Builder

The `build_generation_prompt_csv()` function constructs prompts dynamically:

```python
def build_generation_prompt_csv(categories_to_generate, countries_subset, n_per_category):
    # Build category info with hierarchy and material suggestions
    for cat in categories_to_generate:
        info = get_hierarchy_info(cat)
        mat_suggestions = MATERIAL_COMBINATIONS.get(info['category'], [])
        # Include: "Jeans (Gender: Male, Parent: Bottoms, Suggested: cotton, elastane)"
    
    # Construct prompt with rules for:
    # - Product naming (creative, realistic)
    # - Material JSON format (shares sum to 1.0)
    # - Weight ranges by product type (0.12-2.38 kg)
    # - Distance ranges by route (520-23,800 km)
    # - Natural variance (NO round numbers)
```

**Key prompt features:**
- Category-specific material suggestions
- Realistic weight ranges by product type (light apparel: 0.12-0.28 kg, heavy footwear: 1.23-2.38 kg)
- Distance ranges by shipping route (local: 520-1,940 km, intercontinental: 10,300-23,800 km)
- Explicit instruction to use natural variance (e.g., 0.847, not 0.85)

#### 4. Chunk-Based Parallel Generation

Products are generated in configurable chunks to maximize throughput:

```python
# Configuration
CHUNK_SIZE = 5              # Products per API call
PRODUCTS_PER_CATEGORY = 10500  # ~10,500 per category × 86 = ~900k total
PARALLEL_WORKERS = 4        # Concurrent API calls

# Execution
work_queue = [
    {"category": "Jeans", "chunk_index": 0, "products": 5},
    {"category": "Jeans", "chunk_index": 1, "products": 5},
    # ... thousands of chunks
]

with ThreadPoolExecutor(max_workers=4) as executor:
    futures = {executor.submit(generate_chunk, work) for work in queue}
```

#### 5. Rate Limiting & Resilience

Token bucket rate limiter prevents API quota exhaustion:

```python
# Tier 1 limits: 15 RPM, 1M TPM
effective_rpm = int(TIER_1_RPM_LIMIT * 0.9)  # 90% buffer
rate_limiter = TokenBucketRateLimiter(max_rpm=effective_rpm)

# Before each API call
wait_time = rate_limiter.acquire(tokens=1)
if wait_time > 0:
    time.sleep(wait_time)
```

**Resilience features:**
- Automatic retry on transient failures (3 attempts per chunk)
- Rate limit detection (429 errors) with graceful pause
- Checkpoint saving on any failure

#### 6. Checkpoint & Resume System

Long generation runs can be interrupted and resumed:

```python
# Checkpoint saved after each successful chunk
checkpoint = {
    "session_id": "20251126_055821",
    "output_file": "fashion_products_20251126.csv",
    "completed_chunks": [(0, 0), (0, 1), (0, 2), ...],
    "failed_chunks": [],
    "total_products": 45230
}

# On resume
checkpoint_manager = CheckpointManager.load_existing(checkpoint_path)
work_queue = [chunk for chunk in all_chunks if not completed]
```

### Quality Assurance

#### Generation-Time Validation

Each API response is validated before saving:

1. **CSV parsing** with proper quote handling for JSON fields
2. **Field count check** (exactly 8 columns)
3. **Line-by-line validation** with malformed row rejection

#### Post-Generation Validation (Separate Pipeline)

```
Product_data.csv → data_correction/ → Product_data_final.csv
                        │
                        ├── Gender validation (Male/Female only)
                        ├── Category hierarchy check
                        ├── ISO 3166-1 country code validation
                        ├── Material JSON parsing (sum = 1.0 ± 0.01)
                        ├── Weight range check (0.05-5.0 kg)
                        ├── Distance range check (100-25,000 km)
                        └── Duplicate removal
```

**Pass rate**: ~97% of generated products pass all validation

### Generation Statistics

| Metric | Value |
|--------|-------|
| Total products generated | 902,000 |
| Categories covered | 86 |
| Unique materials | 34 |
| Countries represented | 277 |
| Generation time | ~7 hours |
| API cost | ~$50 (Gemini 2.5 Flash) |
| Validation pass rate | 97% |


### Why This Matters
Synthetic data generation democratizes access to large-scale datasets for sustainability research. Researchers, students, and developers can now:
- Experiment with environmental impact models without proprietary data
- Train machine learning algorithms on realistic, diverse product data
- Develop and test sustainability tools before accessing real data
- Understand trade-offs in material choices and supply chains

This project demonstrates that **LLM-based synthetic data generation is a viable approach for sustainability research**, opening new possibilities for data-driven environmental analysis.

---

## Key Features

### Data Generation
- **AI-Powered Generation**: Uses Google Gemini 2.5 Flash API to generate realistic product attributes
- **Diverse Categories**: 86 fashion categories (dresses, shirts, jackets, shoes, accessories)
- **Realistic Materials**: 34 different materials with accurate composition percentages
- **Global Manufacturing**: 277 countries (comprehensive global coverage) with realistic transport distances

### Environmental Analysis
- **Carbon Footprint Calculation**:
  - Material production emissions (cradle-to-gate)
  - Transportation emissions using multinomial logit modal split model
  - Scientifically validated emission factors (GLEC Framework, DEFRA 2023)

- **Water Footprint Calculation**:
  - Material-level water consumption
  - Based on Idemat 2026 database and textile industry research

### Machine Learning
- **Predictive Models**: Neural networks to predict carbon and water footprints from product attributes
- **Robustness Training**: Models trained to handle missing or incomplete data
- **Evaluation Framework**: Comprehensive testing including data corruption scenarios

### Data Quality
- **Multi-Stage Validation**: Gender, category, country code, material composition checks
- **Automated Correction**: Intelligent correction of common AI generation errors
- **Checkpoint System**: Fault-tolerant processing for large-scale operations

---

## Pipeline Architecture
```
┌─────────────────────────────────────────────────────────────────────┐
│                     BULK PRODUCT GENERATOR PIPELINE                  │
└─────────────────────────────────────────────────────────────────────┘

┌──────────────────┐
│  1. GENERATION   │  Language: Python
│  (Gemini API)    │  Input: Category definitions, material database
├──────────────────┤  Output: ~900k raw products (CSV)
│ - Product names  │  Tools: Google Gemini 2.5 Flash
│ - Categories     │
│ - Materials      │
│ - Weights        │
│ - Countries      │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  2. CORRECTION   │  Language: Python
│  (Validation)    │  Input: Raw product data
├──────────────────┤  Output: Validated product data
│ - Gender check   │  Processing:
│ - Category check │   • Schema validation
│ - Country codes  │   • Constraint enforcement
│ - Material sum   │   • Duplicate removal
│ - Range checks   │   • Error correction
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ 3. CALCULATION   │  Language: C (high performance)
│  (Footprints)    │  Input: Validated products + reference data
├──────────────────┤  Output: Products with footprints
│ • Material       │  Calculations:
│   Carbon & Water │   • Σ(weight × % × factor) for materials
│ • Transport      │   • Modal split model for transport
│   Carbon         │   • Distance-based emissions
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  4. SPLITTING    │  Language: Python
│  (Train/Val)     │  Input: Complete dataset
├──────────────────┤  Output: 75/25 stratified splits
│ - Stratified     │  Method: Random stratified by category
│   by category    │
│ - Reproducible   │
│   (seed=42)      │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ 5. ML TRAINING   │  Language: Python (TensorFlow/Keras)
│  (Prediction)    │  Input: Train/validation splits
├──────────────────┤  Output: Trained models + predictions
│ • Neural Network │  Models:
│ • Robustness     │   • Baseline: Standard training
│   Training       │   • Robustness: Dropout augmentation
│ • Evaluation     │
└──────────────────┘

OUTPUTS:
├── datasets/raw/                   → Raw generated products
├── datasets/processed/             → Validated + footprint data
├── datasets/splits/                → Train/validation sets
└── datasets/model_outputs/         → Predictions + metrics
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

## Results and Performance

### Dataset Statistics
**Product Distribution**:
- **Total Products**: 901,571
- **Categories**: 86 fashion categories
- **Materials**: 34 unique materials
- **Countries**: 277 countries (comprehensive global coverage)
- **Gender Distribution**: Female (61.5%), Male (38.2%), Other (0.3%)

**Environmental Footprints**:
| Metric | Min | Mean | Median | Max | Unit |
|--------|-----|------|--------|-----|------|
| Carbon (Material) | 0.15 | 3.2 | 2.8 | 18.5 | kgCO2e |
| Carbon (Transport) | 0.05 | 2.0 | 1.6 | 12.3 | kgCO2e |
| Carbon (Total) | 0.25 | 5.2 | 4.5 | 25.8 | kgCO2e |
| Water | 50 | 2,800 | 1,950 | 15,000 | liters |

**Highest Impact Materials** (per kg):
1. Wool: 17.0 kgCO2e, 125,000 L
2. Cashmere: 15.2 kgCO2e, 105,000 L
3. Leather: 14.0 kgCO2e, 17,000 L
4. Polyester (virgin): 6.0 kgCO2e, 50 L
5. Cotton: 5.5 kgCO2e, 10,000 L

### Model Performance

Both models were trained on 676,178 products and evaluated on a held-out validation set of 225,393 products (25% stratified split by category).

#### Baseline Model (XGBoost Multi-Output Regression)

Optimized for maximum accuracy on complete data. Achieves near-perfect predictions when all input features are available.

**Validation Set Metrics**:
| Target | MAE | RMSE | R² | MAPE |
|--------|-----|------|----|----|
| **Carbon Material** | 0.041 kgCO2e | 0.146 kgCO2e | 0.9999 | 0.83% |
| **Carbon Transport** | 0.0008 kgCO2e | 0.0018 kgCO2e | 0.9998 | — |
| **Carbon Total** | 0.044 kgCO2e | 0.146 kgCO2e | 0.9999 | 0.95% |
| **Water Total** | 115.3 L | 570.6 L | 0.9998 | 0.81% |

**Interpretation**:
- **R² = 0.9999** means the model explains 99.99% of variance in the target variable
- **MAE of 0.041 kgCO2e** for carbon material means predictions are off by ~41 grams CO2e on average
- **Constraint Violation Rate**: 1.8% of predictions violate physics constraint (carbon_total ≠ carbon_material + carbon_transport)

#### Robustness Model (XGBoost with Feature Dropout Augmentation)

Trained with 20% random feature masking during training to improve resilience when input data is incomplete or missing.

**Validation Set Metrics (Complete Data)**:
| Target | MAE | RMSE | R² | MAPE |
|--------|-----|------|----|----|
| **Carbon Material** | 0.045 kgCO2e | 0.166 kgCO2e | 0.9999 | 0.98% |
| **Carbon Transport** | 0.0013 kgCO2e | 0.0026 kgCO2e | 0.9997 | — |
| **Carbon Total** | 0.050 kgCO2e | 0.168 kgCO2e | 0.9999 | 1.17% |
| **Water Total** | 132.9 L | 746.5 L | 0.9996 | 1.12% |

**Trade-off**: Slightly lower accuracy on complete data (R² 0.9999 vs 0.9999), but dramatically better performance when data is missing.

#### Robustness Under Missing Data

The key differentiator between models becomes apparent when input features are randomly masked (simulating real-world incomplete product information):

**Performance Comparison at Different Missing Data Levels**:

| Missing % | Model | Carbon Material R² | Carbon Total R² | Water Total R² |
|-----------|-------|-------------------|-----------------|----------------|
| **0%** | Baseline | 0.9999 | 0.9999 | 0.9998 |
| **0%** | Robustness | 0.9999 | 0.9999 | 0.9996 |
| **20%** | Baseline | 0.001 | 0.306 | 0.575 |
| **20%** | Robustness | **0.968** ✓ | **0.968** | **0.951** |
| **40%** | Baseline | -0.991 | -0.380 | 0.146 |
| **40%** | Robustness | **0.936** ✓ | **0.936** | **0.902** |

**MAE Comparison (20% Missing Data)**:
| Target | Baseline MAE | Robustness MAE | Improvement |
|--------|-------------|----------------|-------------|
| Carbon Material | 5.04 kgCO2e | **0.29 kgCO2e** | 17× better |
| Carbon Total | 4.12 kgCO2e | **0.29 kgCO2e** | 14× better |
| Water Total | 7,181 L | **772 L** | 9× better |

#### Key Insights

1. **Baseline model is fragile**: Performance collapses catastrophically when even 20% of features are missing (R² drops from 0.9999 to 0.001)

2. **Robustness model degrades gracefully**: Maintains R² > 0.93 even with 40% of features missing

3. **Real-world recommendation**: 
   - Use **Baseline** when you have guaranteed complete product data
   - Use **Robustness** for production systems where missing fields are possible (e.g., user-submitted products, incomplete databases)

4. **Physics constraint violations**: Both models occasionally predict carbon_total ≠ carbon_material + carbon_transport. Baseline: 1.8%, Robustness: 2.9%. Post-processing correction recommended for production use

#### Why Is Accuracy So High? (Not Data Leakage)

The near-perfect R² = 0.9999 may appear suspicious, but it is **expected behavior** — not data leakage. Here's why:

**The targets are deterministically calculated from the input features:**

```
Input Features:
  - weight_kg
  - total_distance_km  
  - 34 material percentage columns (cotton_conventional, polyester_virgin, etc.)

Target Calculations (from C footprint calculator):
  carbon_material = Σ (weight_kg × material_percentage × carbon_factor)
  water_total     = Σ (weight_kg × material_percentage × water_factor)
  carbon_transport = f(weight_kg, distance_km, modal_split_model)
  carbon_total    = carbon_material + carbon_transport
```

The XGBoost model is essentially **learning the calculation formulas** that generated the targets. This is analogous to training a model to predict rectangle area from length and width — near-perfect accuracy is expected because the relationship is deterministic.

**Why this is NOT data leakage:**

| Leakage Check | Status |
|---------------|--------|
| Target values in input features? | ❌ No — inputs are only product attributes |
| Train/test contamination? | ❌ No — proper stratified split |
| Future information used? | ❌ No — all features exist before target calculation |
| Deterministic formula-based relationship? | ✅ Yes — this explains the high accuracy |

**The real value of the ML model:**

| Use Case | Value |
|----------|-------|
| Replace the C calculator | Low — formulas are known |
| **Handle missing data** | **High** — Robustness model maintains R² > 0.93 with 40% missing features |
| Approximate footprints when exact composition unknown | High |
| Simpler deployment (no C library needed) | Medium |
| Feature importance analysis | Medium |

**Key takeaway**: The baseline model memorizes the exact formulas, while the robustness model learns generalized patterns that work even when inputs are incomplete. For real-world applications where product data is often missing or approximate, the robustness model provides significant value.


---

## Citation
If you use this dataset or code in your research, please cite:

```bibtex
@software{bulk_product_generator_2025,
  author = {Moussa Ouallaf},
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
