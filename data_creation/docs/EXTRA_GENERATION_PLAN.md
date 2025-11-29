# Extra Data Generation Plan

**Created**: 2025-11-29
**Purpose**: Add critical missing materials and boost underrepresented materials to ~35,000 products each

---

## Executive Summary

**Total Products to Generate**: ~390,000 products

**Objectives**:
1. Add 8 critical missing materials with ~35,000 products each
2. Boost 15 underrepresented materials to 35,000 products each
3. Maintain uniform distribution across categories, parents, and genders
4. Reuse existing data_creation infrastructure

---

## 1. Current State Analysis

### Materials Needing Boost (< 35,000 products)

| Priority | Material | Current | Needed | Category Focus |
|----------|----------|---------|--------|----------------|
| **CRITICAL** | shearling_faux | 2 | 34,998 | Footwear (Boots), Outerwear |
| **CRITICAL** | cotton_recycled | 4 | 34,996 | Tops, Bottoms |
| **CRITICAL** | polyamide_recycled | 28 | 34,972 | Tops (Sweaters), Outerwear |
| **CRITICAL** | jute | 58 | 34,942 | Footwear (Sandals) |
| **CRITICAL** | jute_sole | 83 | 34,917 | Footwear (Flats, Sandals) |
| **HIGH** | rubber_synthetic | 1,668 | 33,332 | Footwear (Sneakers, Flats) |
| **HIGH** | cashmere | 4,730 | 30,270 | Tops (Sweaters), Outerwear (Cloaks, Coats) |
| **HIGH** | wool_merino | 7,030 | 27,970 | Tops (Sweaters & Knitwear) |
| **HIGH** | metal_steel | 8,573 | 26,427 | Footwear (Boots), Outerwear (Jackets) |
| **MEDIUM** | tpu | 13,309 | 21,691 | Footwear (Heels, Sneakers, Athletic) |
| **MEDIUM** | silk | 19,441 | 15,559 | Dresses (Gowns), Tops (Blouses) |
| **MEDIUM** | coated_fabric_pu | 19,626 | 15,374 | Outerwear (Raincoats), Footwear |
| **LOW** | linen_flax | 33,793 | 1,207 | Tops (Shirts), Bottoms |
| **LOW** | membrane_tpu | 34,338 | 662 | Outerwear (Rain Jackets, Raincoats) |
| **LOW** | cotton_organic | 34,431 | 569 | Bottoms (Jeans, Denim) |

**Subtotal**: 347,886 products

### New Materials to Add

| Material | Target | Primary Categories | Ecoinvent Code |
|----------|--------|-------------------|----------------|
| down_feather | 35,000 | Down Jackets, Parkas, Heavy Coats | down_feather_production |
| down_synthetic | 35,000 | Down Jackets, Parkas, Vests | polyester_fiber_synthetic_fill |
| lyocell_tencel | 35,000 | Dresses, Tops, Bottoms | lyocell_fiber_production |
| hemp | 35,000 | Bottoms (Jeans, Casual Pants), Tops | hemp_fiber_production |
| leather_ovine | 35,000 | Footwear, Outerwear (Jackets) | leather_ovine_production |
| suede | 35,000 | Footwear, Outerwear | leather_suede_production |
| nubuck | 35,000 | Footwear (Boots, Casual Shoes) | leather_nubuck_production |
| patent_leather | 35,000 | Footwear (Heels, Dress Shoes) | leather_patent_production |

**Subtotal**: 280,000 products (8 materials × 35,000)

---

## 2. Generation Strategy

### 2.1 Material Distribution Preservation

**Gender Distribution** (maintain current ratios):
- Overall: 61.75% Female, 38.25% Male
- Material-specific ratios will follow existing patterns

**Parent Category Distribution** (maintain proportions):
- Tops: 25.10%
- Outerwear: 24.55%
- Bottoms: 23.69%
- Footwear: 16.27%
- Dresses: 10.39%

### 2.2 Category Selection Algorithm

For each material, select categories based on:
1. **Existing usage patterns** (from analysis above)
2. **Realistic material-category compatibility**
3. **Ecoinvent LCA availability**
4. **Industry standards**

Example for `cashmere` (need 30,270):
- Sweaters & Knitwear (Male): 40% = 12,108 products
- Sweaters (Female): 30% = 9,081 products
- Cloaks & Capes (Male): 15% = 4,541 products
- Coats (Female): 10% = 3,027 products
- Heavy Coats (Male): 5% = 1,514 products

### 2.3 Material Combinations

**Principle**: Materials rarely appear alone. Use realistic blends.

Examples:
```python
# Cashmere sweater
{"cashmere": 0.85, "wool_merino": 0.15}

# Down jacket
{"down_feather": 0.30, "polyamide_66": 0.65, "elastane": 0.05}

# Hemp jeans
{"hemp": 0.60, "cotton_organic": 0.38, "elastane": 0.02}

# Suede boots
{"suede": 0.75, "synthetic_rubber_sbr": 0.20, "eva": 0.05}
```

---

## 3. Implementation Plan

### Phase 1: Update Vocabularies (30 min)

**File**: `data_creation/src/vocabularies.py`

**Changes**:
1. Add 8 new materials to `MATERIAL_VOCAB`
2. Update `MATERIAL_COMBINATIONS` for all 74 categories
3. Add new material combinations for critical materials

**New Material Definitions**:
```python
# Add to MATERIAL_VOCAB (maintain alphabetical order)
NEW_MATERIALS = [
    "down_feather",           # Natural down filling
    "down_synthetic",         # Synthetic down (polyester fill)
    "hemp",                   # Hemp fiber
    "leather_ovine",          # Sheep/lamb leather
    "lyocell_tencel",        # Lyocell/Tencel fiber
    "nubuck",                # Nubuck leather
    "patent_leather",        # Patent leather
    "suede",                 # Suede leather
]

# Update MATERIAL_COMBINATIONS
MATERIAL_COMBINATIONS = {
    # ... existing combinations ...

    # DOWN JACKETS - Critical fix!
    "Down Jackets": [
        "down_feather",        # Natural down
        "down_synthetic",      # Synthetic alternative
        "polyamide_66",        # Shell
        "polyester_virgin",    # Shell
        "membrane_tpu"         # Waterproofing
    ],

    # PARKAS
    "Parkas": [
        "down_feather",
        "down_synthetic",
        "polyamide_6",
        "polyester_virgin"
    ],

    # HEAVY COATS
    "Heavy Coats": [
        "wool_generic",
        "wool_merino",
        "cashmere",
        "down_feather",
        "polyester_virgin"
    ],

    # SWEATERS & KNITWEAR
    "Sweaters & Knitwear": [
        "wool_merino",
        "cashmere",
        "acrylic",
        "cotton_conventional"
    ],

    # BOOTS
    "Boots": [
        "leather_bovine",
        "leather_ovine",
        "suede",
        "nubuck",
        "shearling_faux",
        "synthetic_rubber_sbr",
        "eva"
    ],

    # HEELS & DRESS SHOES
    "Heels": [
        "leather_bovine",
        "patent_leather",
        "suede",
        "eva",
        "tpu"
    ],

    "Dress Shoes": [
        "leather_bovine",
        "leather_ovine",
        "patent_leather",
        "natural_rubber"
    ],

    # JEANS
    "Jeans": [
        "cotton_conventional",
        "cotton_organic",
        "cotton_recycled",
        "hemp",
        "polyester_virgin",
        "elastane"
    ],

    # DRESSES (add lyocell)
    "Maxi Dresses": [
        "cotton_conventional",
        "viscose",
        "modal",
        "lyocell_tencel"
    ],

    "Midi Dresses": [
        "viscose",
        "polyester_virgin",
        "lyocell_tencel",
        "elastane"
    ],

    # ... etc for all 74 categories
}
```

### Phase 2: Create Generation Configuration (1 hour)

**New File**: `data_creation/config/extra_generation_config.py`

```python
"""
Configuration for extra generation to boost underrepresented materials.

This file defines exactly which categories to generate for each material
to reach the 35,000 target while maintaining distribution uniformity.
"""

# Materials needing boost with their target distribution
BOOST_MATERIALS = {
    "shearling_faux": {
        "target": 35000,
        "current": 2,
        "categories": {
            "Boots": {"count": 17500, "gender_split": {"Female": 0.7, "Male": 0.3}},
            "Heavy Coats": {"count": 10500, "gender_split": {"Female": 0.6, "Male": 0.4}},
            "Jackets": {"count": 7000, "gender_split": {"Female": 0.65, "Male": 0.35}},
        }
    },

    "cotton_recycled": {
        "target": 35000,
        "current": 4,
        "categories": {
            "Hoodies": {"count": 8750, "gender_split": {"Female": 0.55, "Male": 0.45}},
            "Sweatshirts": {"count": 8750, "gender_split": {"Female": 0.60, "Male": 0.40}},
            "Jeans": {"count": 7000, "gender_split": {"Female": 0.65, "Male": 0.35}},
            "Casual Pants": {"count": 5250, "gender_split": {"Male": 0.75, "Female": 0.25}},
            "Short Sleeve Shirts": {"count": 5250, "gender_split": {"Female": 0.55, "Male": 0.45}},
        }
    },

    "down_feather": {
        "target": 35000,
        "current": 0,
        "categories": {
            "Down Jackets": {"count": 21000, "gender_split": {"Female": 0.60, "Male": 0.40}},
            "Parkas": {"count": 7000, "gender_split": {"Male": 0.55, "Female": 0.45}},
            "Heavy Coats": {"count": 7000, "gender_split": {"Male": 0.52, "Female": 0.48}},
        }
    },

    "down_synthetic": {
        "target": 35000,
        "current": 0,
        "categories": {
            "Down Jackets": {"count": 21000, "gender_split": {"Female": 0.60, "Male": 0.40}},
            "Parkas": {"count": 7000, "gender_split": {"Male": 0.55, "Female": 0.45}},
            "Vests": {"count": 7000, "gender_split": {"Male": 0.45, "Female": 0.55}},
        }
    },

    "lyocell_tencel": {
        "target": 35000,
        "current": 0,
        "categories": {
            "Maxi Dresses": {"count": 8750, "gender_split": {"Female": 1.0}},
            "Midi Dresses": {"count": 8750, "gender_split": {"Female": 1.0}},
            "Blouses": {"count": 7000, "gender_split": {"Female": 1.0}},
            "Short Sleeve Shirts": {"count": 5250, "gender_split": {"Female": 0.60, "Male": 0.40}},
            "Casual Pants": {"count": 5250, "gender_split": {"Female": 0.50, "Male": 0.50}},
        }
    },

    "hemp": {
        "target": 35000,
        "current": 0,
        "categories": {
            "Jeans": {"count": 14000, "gender_split": {"Female": 0.60, "Male": 0.40}},
            "Casual Pants": {"count": 10500, "gender_split": {"Male": 0.65, "Female": 0.35}},
            "Short Sleeve Shirts": {"count": 7000, "gender_split": {"Male": 0.50, "Female": 0.50}},
            "Denim": {"count": 3500, "gender_split": {"Male": 0.70, "Female": 0.30}},
        }
    },

    "leather_ovine": {
        "target": 35000,
        "current": 0,
        "categories": {
            "Leather Jackets": {"count": 10500, "gender_split": {"Female": 0.60, "Male": 0.40}},
            "Boots": {"count": 10500, "gender_split": {"Female": 0.70, "Male": 0.30}},
            "Dress Shoes": {"count": 7000, "gender_split": {"Male": 0.80, "Female": 0.20}},
            "Casual Shoes": {"count": 7000, "gender_split": {"Male": 0.55, "Female": 0.45}},
        }
    },

    "suede": {
        "target": 35000,
        "current": 0,
        "categories": {
            "Boots": {"count": 14000, "gender_split": {"Female": 0.65, "Male": 0.35}},
            "Casual Shoes": {"count": 7000, "gender_split": {"Male": 0.55, "Female": 0.45}},
            "Loafers": {"count": 7000, "gender_split": {"Male": 0.75, "Female": 0.25}},
            "Jackets": {"count": 7000, "gender_split": {"Female": 0.55, "Male": 0.45}},
        }
    },

    "nubuck": {
        "target": 35000,
        "current": 0,
        "categories": {
            "Boots": {"count": 17500, "gender_split": {"Female": 0.65, "Male": 0.35}},
            "Casual Shoes": {"count": 10500, "gender_split": {"Male": 0.60, "Female": 0.40}},
            "Sneakers": {"count": 7000, "gender_split": {"Male": 0.70, "Female": 0.30}},
        }
    },

    "patent_leather": {
        "target": 35000,
        "current": 0,
        "categories": {
            "Heels": {"count": 21000, "gender_split": {"Female": 1.0}},
            "Dress Shoes": {"count": 10500, "gender_split": {"Male": 0.85, "Female": 0.15}},
            "Flats": {"count": 3500, "gender_split": {"Female": 1.0}},
        }
    },

    "cashmere": {
        "target": 35000,
        "current": 4730,
        "categories": {
            "Sweaters & Knitwear": {"count": 12000, "gender_split": {"Male": 1.0}},
            "Sweaters": {"count": 9000, "gender_split": {"Female": 1.0}},
            "Cloaks & Capes": {"count": 4500, "gender_split": {"Male": 1.0}},
            "Coats": {"count": 3000, "gender_split": {"Female": 1.0}},
            "Heavy Coats": {"count": 1770, "gender_split": {"Male": 1.0}},
        }
    },

    "wool_merino": {
        "target": 35000,
        "current": 7030,
        "categories": {
            "Sweaters & Knitwear": {"count": 16788, "gender_split": {"Male": 1.0}},
            "Sweaters": {"count": 5594, "gender_split": {"Female": 1.0}},
            "Cloaks & Capes": {"count": 2797, "gender_split": {"Male": 1.0}},
            "Long Sleeve Shirts": {"count": 2797, "gender_split": {"Female": 1.0}},
            "Boots": {"count": 994, "gender_split": {"Female": 0.90, "Male": 0.10}},
        }
    },

    "metal_steel": {
        "target": 35000,
        "current": 8573,
        "categories": {
            "Leather Jackets": {"count": 7900, "gender_split": {"Female": 0.73, "Male": 0.27}},
            "Boots": {"count": 7900, "gender_split": {"Female": 0.85, "Male": 0.15}},
            "Heels": {"count": 5283, "gender_split": {"Female": 1.0}},
            "Denim Jackets": {"count": 2642, "gender_split": {"Female": 0.55, "Male": 0.45}},
            "Dress Shoes": {"count": 2642, "gender_split": {"Male": 1.0}},
            "Casual Shoes": {"count": 60, "gender_split": {"Male": 0.60, "Female": 0.40}},
        }
    },

    "tpu": {
        "target": 35000,
        "current": 13309,
        "categories": {
            "Heels": {"count": 5538, "gender_split": {"Female": 1.0}},
            "Athletic Shoes": {"count": 4338, "gender_split": {"Male": 0.70, "Female": 0.30}},
            "Sneakers": {"count": 4338, "gender_split": {"Male": 0.65, "Female": 0.35}},
            "Sandals": {"count": 4338, "gender_split": {"Female": 0.55, "Male": 0.45}},
            "Casual Shoes": {"count": 2769, "gender_split": {"Female": 0.55, "Male": 0.45}},
            "Boots": {"count": 370, "gender_split": {"Female": 0.80, "Male": 0.20}},
        }
    },

    "silk": {
        "target": 35000,
        "current": 19441,
        "categories": {
            "Gowns": {"count": 7779, "gender_split": {"Female": 1.0}},
            "Blouses": {"count": 4668, "gender_split": {"Female": 1.0}},
            "Maxi Dresses": {"count": 1556, "gender_split": {"Female": 1.0}},
            "Button-Ups": {"count": 778, "gender_split": {"Female": 1.0}},
            "Long Sleeve Shirts": {"count": 778, "gender_split": {"Female": 1.0}},
        }
    },

    "coated_fabric_pu": {
        "target": 35000,
        "current": 19626,
        "categories": {
            "Raincoats": {"count": 6147, "gender_split": {"Male": 1.0}},
            "Flats": {"count": 1537, "gender_split": {"Female": 1.0}},
            "Boots": {"count": 1537, "gender_split": {"Female": 0.97, "Male": 0.03}},
            "Heels": {"count": 1537, "gender_split": {"Female": 1.0}},
            "Bombers": {"count": 1537, "gender_split": {"Female": 0.58, "Male": 0.42}},
            "Rain Jackets": {"count": 3079, "gender_split": {"Female": 1.0}},
        }
    },

    "rubber_synthetic": {
        "target": 35000,
        "current": 1668,
        "categories": {
            "Sneakers": {"count": 14999, "gender_split": {"Male": 0.94, "Female": 0.06}},
            "Flats": {"count": 9999, "gender_split": {"Female": 1.0}},
            "Heels": {"count": 6666, "gender_split": {"Female": 1.0}},
            "Casual Shoes": {"count": 1668, "gender_split": {"Male": 1.0}},
        }
    },

    "jute_sole": {
        "target": 35000,
        "current": 83,
        "categories": {
            "Flats": {"count": 17459, "gender_split": {"Female": 1.0}},
            "Sandals": {"count": 17458, "gender_split": {"Female": 1.0}},
        }
    },

    "jute": {
        "target": 35000,
        "current": 58,
        "categories": {
            "Sandals": {"count": 34942, "gender_split": {"Female": 1.0}},
        }
    },

    "polyamide_recycled": {
        "target": 35000,
        "current": 28,
        "categories": {
            "Sweaters & Knitwear": {"count": 24987, "gender_split": {"Male": 1.0}},
            "Parkas": {"count": 6994, "gender_split": {"Male": 1.0}},
            "Bombers": {"count": 2991, "gender_split": {"Male": 1.0}},
        }
    },

    "linen_flax": {
        "target": 35000,
        "current": 33793,
        "categories": {
            "Short Sleeve Shirts": {"count": 604, "gender_split": {"Female": 0.51, "Male": 0.49}},
            "Button-Ups": {"count": 603, "gender_split": {"Female": 0.94, "Male": 0.06}},
        }
    },

    "membrane_tpu": {
        "target": 35000,
        "current": 34338,
        "categories": {
            "Rain Jackets": {"count": 331, "gender_split": {"Female": 1.0}},
            "Raincoats": {"count": 331, "gender_split": {"Male": 1.0}},
        }
    },

    "cotton_organic": {
        "target": 35000,
        "current": 34431,
        "categories": {
            "Jeans": {"count": 285, "gender_split": {"Female": 0.90, "Male": 0.10}},
            "Denim": {"count": 284, "gender_split": {"Male": 1.0}},
        }
    },
}

# Calculate total products needed
TOTAL_PRODUCTS_TO_GENERATE = sum(
    data["target"] - data["current"]
    for data in BOOST_MATERIALS.values()
)

print(f"Total products to generate: {TOTAL_PRODUCTS_TO_GENERATE:,}")
# Expected: ~390,000 products
```

### Phase 3: Create Extra Generation Script (2 hours)

**New File**: `data_creation/scripts/generate_extra.py`

This script will:
1. Import existing generator, vocabularies, prompts
2. Load extra_generation_config
3. Generate products batch by batch
4. Save with checkpoints every 10,000 products
5. Merge with Product_data_final.csv

**Architecture** (reuses 90% of existing code):
```python
#!/usr/bin/env python3
"""
Extra Product Generation - Boost Underrepresented Materials

Generates additional products to bring all materials to ~35,000 product count.
Maintains distribution uniformity across categories, parents, and genders.

REUSES:
- src/generator.py (GeminiGenerator)
- src/vocabularies.py (materials, categories)
- src/prompts.py (build_generation_prompt_csv)
- src/csv_writer.py (CSVWriter)
- src/checkpoint.py (CheckpointManager)
- src/rate_limiter.py (RateLimiter)
"""

import sys
from pathlib import Path

# Add data_creation to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.generator import GeminiGenerator
from src.csv_writer import CSVWriter
from src.checkpoint import CheckpointManager
from src.rate_limiter import RateLimiter
from config.extra_generation_config import BOOST_MATERIALS, TOTAL_PRODUCTS_TO_GENERATE
from src.vocabularies import CATEGORIES_BY_ID

# Configuration
OUTPUT_CSV = "data_creation/output/Product_data_extra.csv"
CHECKPOINT_DIR = "data_creation/output/checkpoints/extra"
BATCH_SIZE = 50  # Products per API call
CHECKPOINT_INTERVAL = 10000  # Save every 10k products

def main():
    print("=" * 90)
    print("EXTRA PRODUCT GENERATION - MATERIAL BOOSTING")
    print("=" * 90)
    print()
    print(f"Total products to generate: {TOTAL_PRODUCTS_TO_GENERATE:,}")
    print(f"Materials to boost: {len(BOOST_MATERIALS)}")
    print()

    # Initialize components
    generator = GeminiGenerator()
    csv_writer = CSVWriter(OUTPUT_CSV)
    checkpoint_mgr = CheckpointManager(CHECKPOINT_DIR, CHECKPOINT_INTERVAL)
    rate_limiter = RateLimiter()

    # Generate for each material
    total_generated = 0

    for material, config in BOOST_MATERIALS.items():
        needed = config["target"] - config["current"]
        print(f"\nGenerating {needed:,} products for {material}...")

        # Generate for each category
        for category_name, cat_config in config["categories"].items():
            count = cat_config["count"]
            gender_split = cat_config["gender_split"]

            # Find category object
            category = next(
                (c for c in CATEGORIES_BY_ID.values() if c["name"] == category_name),
                None
            )

            if not category:
                print(f"  ⚠️  Category '{category_name}' not found, skipping...")
                continue

            # Generate products in batches
            # ... implementation continues ...

if __name__ == "__main__":
    main()
```

### Phase 4: Merge and Validate (1 hour)

**New File**: `data_creation/scripts/merge_extra_with_final.py`

This script will:
1. Load Product_data_final.csv (901,573 products)
2. Load Product_data_extra.csv (~390,000 products)
3. Merge both datasets
4. Validate material distribution
5. Save as Product_data_complete_v2.csv (~1,291,573 products)
6. Run comprehensive analysis

---

## 4. Material Combinations for New Materials

### Down Feather (Natural)
```python
# Down jacket - 70/30 down to shell
{
    "down_feather": 0.30,      # Down fill
    "polyamide_66": 0.65,      # Shell fabric
    "elastane": 0.05           # Stretch
}

# Parka - heavier down
{
    "down_feather": 0.35,
    "polyamide_6": 0.60,
    "membrane_tpu": 0.05       # Waterproof layer
}
```

### Down Synthetic
```python
# Synthetic down jacket
{
    "down_synthetic": 0.30,    # Polyester fill
    "polyamide_66": 0.65,
    "elastane": 0.05
}
```

### Lyocell/Tencel (Sustainable Cellulose)
```python
# Lyocell dress
{
    "lyocell_tencel": 0.92,
    "elastane": 0.08
}

# Lyocell blouse
{
    "lyocell_tencel": 0.70,
    "silk": 0.30
}
```

### Hemp (Natural Fiber)
```python
# Hemp jeans
{
    "hemp": 0.60,
    "cotton_organic": 0.38,
    "elastane": 0.02
}

# Hemp shirt
{
    "hemp": 0.80,
    "cotton_conventional": 0.20
}
```

### Leather Ovine (Lamb Leather)
```python
# Lamb leather jacket
{
    "leather_ovine": 0.85,
    "viscose": 0.10,          # Lining
    "metal_brass": 0.05       # Hardware
}

# Lamb leather boots
{
    "leather_ovine": 0.70,
    "natural_rubber": 0.25,   # Sole
    "eva": 0.05               # Insole
}
```

### Suede
```python
# Suede boots
{
    "suede": 0.75,
    "synthetic_rubber_sbr": 0.20,
    "eva": 0.05
}

# Suede jacket
{
    "suede": 0.85,
    "viscose": 0.10,
    "metal_brass": 0.05
}
```

### Nubuck
```python
# Nubuck boots
{
    "nubuck": 0.75,
    "natural_rubber": 0.20,
    "eva": 0.05
}
```

### Patent Leather
```python
# Patent leather heels
{
    "patent_leather": 0.85,
    "tpu": 0.10,
    "eva": 0.05
}
```

---

## 5. Timeline and Resource Estimates

### Development Time

| Phase | Task | Time | Who |
|-------|------|------|-----|
| 1 | Update vocabularies.py | 30 min | Developer |
| 2 | Create extra_generation_config.py | 1 hour | Developer |
| 3 | Create generate_extra.py | 2 hours | Developer |
| 4 | Create merge script | 1 hour | Developer |
| 5 | Testing & validation | 2 hours | Developer |
| **TOTAL** | **Development** | **6.5 hours** | |

### Execution Time

| Task | Count | Time per Batch | Total Time |
|------|-------|----------------|------------|
| API calls (50 products/call) | ~7,800 calls | ~5 sec/call | ~11 hours |
| Rate limiting delays | - | - | ~2 hours |
| Checkpoint saves | ~39 saves | ~30 sec/save | ~20 min |
| Final merge & validation | 1 | 30 min | 30 min |
| **TOTAL** | | | **~14 hours** |

**Cost Estimate** (Gemini API):
- ~390,000 products × ~200 tokens/product = ~78M tokens
- At $0.15/1M tokens = ~$11.70

---

## 6. Validation Criteria

After generation, verify:

### Material Distribution
- [ ] All 30+ materials have ≥ 35,000 products (±2%)
- [ ] Material percentages sum to 1.0 for each product
- [ ] No orphaned or invalid material names

### Category Distribution
- [ ] Parent category distribution maintains ~25/25/24/16/10% split
- [ ] Gender distribution maintains ~62/38% split
- [ ] No category has zero products

### Data Quality
- [ ] All products have 8 fields
- [ ] No missing or null values
- [ ] Weight ranges realistic (0.01-9.0 kg)
- [ ] Distance ranges realistic (0-40,000 km)
- [ ] Country codes valid (ISO 3166-1 alpha-2)

### Ecoinvent Compatibility
- [ ] All materials map to Ecoinvent v3.8+ processes
- [ ] Material combinations are LCA-compatible

---

## 7. Risk Mitigation

### Potential Issues

1. **API Rate Limits**
   - Mitigation: Built-in rate limiter (60 requests/min)
   - Fallback: Automatic retry with exponential backoff

2. **Checkpoint Corruption**
   - Mitigation: Multiple backup checkpoints
   - Fallback: Resume from last valid checkpoint

3. **Memory Issues (large CSV)**
   - Mitigation: Stream processing, batch writes
   - Fallback: Chunk processing with pandas

4. **Distribution Skew**
   - Mitigation: Pre-calculated category splits in config
   - Fallback: Post-generation rebalancing

5. **Material Combination Errors**
   - Mitigation: Validation before API call
   - Fallback: Default to safe combinations

---

## 8. Success Metrics

**Phase 1 Success**:
- ✅ vocabularies.py updated with 8 new materials
- ✅ All 74 categories have updated material combinations
- ✅ No syntax errors, imports work

**Phase 2 Success**:
- ✅ extra_generation_config.py calculates correct totals
- ✅ Gender splits sum to 1.0
- ✅ Category counts sum to material targets

**Phase 3 Success**:
- ✅ generate_extra.py runs without errors
- ✅ Generates ~390,000 products
- ✅ Checkpoints save correctly

**Phase 4 Success**:
- ✅ Merge completes successfully
- ✅ Final dataset has ~1,291,573 products
- ✅ All materials ≥ 35,000 products
- ✅ Distribution uniformity maintained

---

## 9. Next Steps

1. **Review and Approve Plan** ← YOU ARE HERE
2. Phase 1: Update vocabularies (30 min)
3. Phase 2: Create config (1 hour)
4. Phase 3: Create generator (2 hours)
5. Phase 4: Create merger (1 hour)
6. Testing (2 hours)
7. Execute generation (~14 hours runtime)
8. Validate results (1 hour)
9. Update comprehensive analysis
10. Archive old datasets

---

## 10. File Structure After Implementation

```
bulk_product_generator/
├── data_creation/
│   ├── config/
│   │   ├── config.py                      # Existing
│   │   └── extra_generation_config.py     # NEW - Material boost config
│   │
│   ├── src/
│   │   ├── vocabularies.py                # UPDATED - Add 8 materials
│   │   ├── generator.py                   # REUSED
│   │   ├── prompts.py                     # REUSED
│   │   ├── csv_writer.py                  # REUSED
│   │   ├── checkpoint.py                  # REUSED
│   │   └── rate_limiter.py                # REUSED
│   │
│   ├── scripts/
│   │   ├── generate_extra.py              # NEW - Extra generation
│   │   └── merge_extra_with_final.py      # NEW - Merge datasets
│   │
│   ├── output/
│   │   ├── Product_data_extra.csv         # NEW - Extra products
│   │   └── checkpoints/extra/             # NEW - Checkpoints
│   │
│   └── docs/
│       └── EXTRA_GENERATION_PLAN.md       # THIS FILE
│
└── data_correction/
    └── output/
        ├── Product_data_final.csv          # EXISTING - 901k products
        └── Product_data_complete_v2.csv    # NEW - 1.29M products
```

---

**Plan Status**: ✅ Ready for Implementation
**Estimated Total Time**: 6.5 hours development + 14 hours execution
**Estimated Cost**: ~$12 (Gemini API)
**Expected Output**: 1,291,573 products with uniform material distribution
