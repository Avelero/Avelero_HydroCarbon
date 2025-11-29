"""
Extra Generation Configuration

This file defines exactly which categories to generate for each material
to reach the 35,000 target while maintaining distribution uniformity.

Materials are divided into:
1. NEW materials (starting from 0)
2. BOOST materials (increasing from current count to 35,000)

For each material, we specify:
- target: Target product count (35,000)
- current: Current product count in dataset
- categories: Dict of category_name -> config
  - count: Number of products to generate
  - gender_split: Dict of gender -> percentage (must sum to 1.0)
"""

# Target products per material
TARGET_PER_MATERIAL = 35000

# ============================================================================
# NEW MATERIALS (8 materials, 0 current)
# ============================================================================

NEW_MATERIALS = {
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
            "Maxi": {"count": 8750, "gender_split": {"Female": 1.0}},
            "Midi": {"count": 8750, "gender_split": {"Female": 1.0}},
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
}

# ============================================================================
# BOOST MATERIALS (15 materials, various current counts)
# ============================================================================

BOOST_MATERIALS = {
    "shearling_faux": {
        "target": 35000,
        "current": 2,
        "categories": {
            "Boots": {"count": 17500, "gender_split": {"Female": 0.70, "Male": 0.30}},
            "Heavy Coats": {"count": 10500, "gender_split": {"Female": 0.60, "Male": 0.40}},
            "Jackets": {"count": 6998, "gender_split": {"Female": 0.65, "Male": 0.35}},
        }
    },

    "cotton_recycled": {
        "target": 35000,
        "current": 4,
        "categories": {
            "Hoodies": {"count": 8749, "gender_split": {"Female": 0.55, "Male": 0.45}},
            "Sweatshirts": {"count": 8749, "gender_split": {"Female": 0.60, "Male": 0.40}},
            "Jeans": {"count": 6999, "gender_split": {"Female": 0.65, "Male": 0.35}},
            "Casual Pants": {"count": 5250, "gender_split": {"Male": 0.75, "Female": 0.25}},
            "Denim": {"count": 5249, "gender_split": {"Male": 0.70, "Female": 0.30}},
        }
    },

    "polyamide_recycled": {
        "target": 35000,
        "current": 28,
        "categories": {
            "Sweaters & Knitwear": {"count": 24986, "gender_split": {"Male": 1.0}},
            "Parkas": {"count": 6993, "gender_split": {"Male": 1.0}},
            "Bombers": {"count": 2993, "gender_split": {"Male": 1.0}},
        }
    },

    "jute": {
        "target": 35000,
        "current": 58,
        "categories": {
            "Sandals": {"count": 34942, "gender_split": {"Female": 1.0}},
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

    "cashmere": {
        "target": 35000,
        "current": 4730,
        "categories": {
            "Sweaters & Knitwear": {"count": 12108, "gender_split": {"Male": 1.0}},
            "Sweaters": {"count": 9081, "gender_split": {"Female": 1.0}},
            "Cloaks & Capes": {"count": 4541, "gender_split": {"Male": 1.0}},
            "Coats": {"count": 3027, "gender_split": {"Female": 1.0}},
            "Heavy Coats": {"count": 1513, "gender_split": {"Male": 1.0}},
        }
    },

    "wool_merino": {
        "target": 35000,
        "current": 7030,
        "categories": {
            "Sweaters & Knitwear": {"count": 15788, "gender_split": {"Male": 1.0}},
            "Sweaters": {"count": 5596, "gender_split": {"Female": 1.0}},
            "Cloaks & Capes": {"count": 2798, "gender_split": {"Male": 1.0}},
            "Long Sleeve Shirts": {"count": 2798, "gender_split": {"Female": 1.0}},
            "Boots": {"count": 990, "gender_split": {"Female": 0.90, "Male": 0.10}},
        }
    },

    "metal_steel": {
        "target": 35000,
        "current": 8573,
        "categories": {
            "Leather Jackets": {"count": 7927, "gender_split": {"Female": 0.73, "Male": 0.27}},
            "Boots": {"count": 7927, "gender_split": {"Female": 0.85, "Male": 0.15}},
            "Heels": {"count": 5285, "gender_split": {"Female": 1.0}},
            "Denim Jackets": {"count": 2642, "gender_split": {"Female": 0.55, "Male": 0.45}},
            "Dress Shoes": {"count": 2643, "gender_split": {"Male": 1.0}},
            "Casual Shoes": {"count": 3, "gender_split": {"Male": 1.0}},
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
            "Gowns": {"count": 7780, "gender_split": {"Female": 1.0}},
            "Blouses": {"count": 4668, "gender_split": {"Female": 1.0}},
            "Maxi": {"count": 1556, "gender_split": {"Female": 1.0}},
            "Button-Ups": {"count": 778, "gender_split": {"Female": 1.0}},
            "Long Sleeve Shirts": {"count": 777, "gender_split": {"Female": 1.0}},
        }
    },

    "coated_fabric_pu": {
        "target": 35000,
        "current": 19626,
        "categories": {
            "Raincoats": {"count": 6150, "gender_split": {"Male": 1.0}},
            "Rain Jackets": {"count": 3075, "gender_split": {"Female": 1.0}},
            "Flats": {"count": 1537, "gender_split": {"Female": 1.0}},
            "Boots": {"count": 1537, "gender_split": {"Female": 0.97, "Male": 0.03}},
            "Heels": {"count": 1538, "gender_split": {"Female": 1.0}},
            "Bombers": {"count": 1537, "gender_split": {"Female": 0.58, "Male": 0.42}},
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

# ============================================================================
# COMBINED CONFIGURATION
# ============================================================================

# Combine all materials
ALL_MATERIALS = {**NEW_MATERIALS, **BOOST_MATERIALS}

# Calculate totals
TOTAL_NEW_PRODUCTS = sum(
    data["target"] - data["current"]
    for data in NEW_MATERIALS.values()
)

TOTAL_BOOST_PRODUCTS = sum(
    data["target"] - data["current"]
    for data in BOOST_MATERIALS.values()
)

TOTAL_PRODUCTS_TO_GENERATE = TOTAL_NEW_PRODUCTS + TOTAL_BOOST_PRODUCTS

# Summary
SUMMARY = {
    "new_materials_count": len(NEW_MATERIALS),
    "boost_materials_count": len(BOOST_MATERIALS),
    "total_materials": len(ALL_MATERIALS),
    "new_products": TOTAL_NEW_PRODUCTS,
    "boost_products": TOTAL_BOOST_PRODUCTS,
    "total_products": TOTAL_PRODUCTS_TO_GENERATE,
    "target_per_material": TARGET_PER_MATERIAL,
}

# ============================================================================
# VALIDATION
# ============================================================================

def validate_config():
    """Validate the configuration for consistency"""
    errors = []

    for mat_name, mat_config in ALL_MATERIALS.items():
        # Check target
        if mat_config["target"] != TARGET_PER_MATERIAL:
            errors.append(f"{mat_name}: target {mat_config['target']} != {TARGET_PER_MATERIAL}")

        # Calculate total products to generate
        total_needed = mat_config["target"] - mat_config["current"]
        total_configured = sum(cat["count"] for cat in mat_config["categories"].values())

        if abs(total_needed - total_configured) > 1:  # Allow 1 product tolerance for rounding
            errors.append(
                f"{mat_name}: mismatch - need {total_needed}, configured {total_configured}"
            )

        # Check gender splits sum to 1.0
        for cat_name, cat_config in mat_config["categories"].items():
            gender_sum = sum(cat_config["gender_split"].values())
            if abs(gender_sum - 1.0) > 0.01:  # Allow small floating point error
                errors.append(
                    f"{mat_name} -> {cat_name}: gender split sums to {gender_sum}, not 1.0"
                )

    return errors


if __name__ == "__main__":
    print("=" * 90)
    print("EXTRA GENERATION CONFIGURATION")
    print("=" * 90)
    print()

    print("Summary:")
    for key, value in SUMMARY.items():
        print(f"  {key}: {value:,}" if isinstance(value, int) else f"  {key}: {value}")
    print()

    # Validate
    print("Validating configuration...")
    errors = validate_config()

    if errors:
        print(f"\n❌ Found {len(errors)} validation errors:")
        for error in errors:
            print(f"  • {error}")
    else:
        print("✅ Configuration is valid!")
        print()
        print(f"Ready to generate {TOTAL_PRODUCTS_TO_GENERATE:,} products")
