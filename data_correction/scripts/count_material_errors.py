#!/usr/bin/env python3
"""
Count products with material errors to determine if they should be removed
"""

import pandas as pd
import json
from collections import Counter

# Define clearly erroneous material names
ERRONEOUS_MATERIALS = {
    # Typos and misspellings
    'poliamide_6',           # should be polyamide_6
    'polporter_virgin',      # major typo
    'polyster_virgin',       # should be polyester_virgin
    'polyester_6',           # wrong naming convention
    'polyamid_6',            # typo

    # Japanese characters
    'polリエステル_recycled',
    'polリエステル_virgin',

    # Space errors
    'pol polyester_virgin',
    'pol polyester_recycled',
    'pol polyamide_6',

    # Confused material names
    'elastamide_66',         # should be polyamide_66 or elastane
    'elastester_virgin',     # confused
    'elastical',             # should be elastane

    # Wrong material combinations
    'polyester_bovine',      # polyester is not bovine
    'coated_bovine',         # unclear
    'leather_bvine',         # typo of bovine
    'leather_b_ovine',       # typo of bovine

    # More typos
    'visvester_virgin',      # typo
    'visyester_virgin',      # typo
    'polluster_recycled',    # typo
    'viscane',               # should be viscose
    'visise',                # should be viscose
    'viscluse',              # should be viscose

    # Corrupted data
    'pol<seg_125>_virgin',   # corrupted
}

# Questionable but might be intentional
QUESTIONABLE_MATERIALS = {
    'polyester_organic',     # polyester can't really be organic
    'polyester_generic',     # inconsistent naming
    'polyester_66',          # wrong number (that's for polyamide)
    'shearling_faux',        # might be valid
    'rayon',                 # valid but should use viscose
    'nylon',                 # valid but should use polyamide
    'recycled_polyester',    # valid but inconsistent (should be polyester_recycled)
    'cotton_recycled',       # valid but might be inconsistent
    'cotton_virgin',         # valid but inconsistent
    'polyamide_virgin',      # valid but inconsistent
    'nylon_recycled',        # valid but should be polyamide_recycled
    'polyester_conventional',# valid but inconsistent
    'rubber_synthetic',      # valid but inconsistent
}

print("=" * 80)
print("MATERIAL ERROR ANALYSIS")
print("=" * 80)
print()

# Load data
INPUT_CSV = "data_correction/output/Product_data_complete.csv"
print(f"📂 Loading data from: {INPUT_CSV}")
df = pd.read_csv(INPUT_CSV, low_memory=False)
print(f"✓ Loaded {len(df):,} rows\n")

# Parse materials and identify errors
erroneous_products = set()
questionable_products = set()
material_counter = Counter()

print("🔍 Scanning materials...")
for idx, mat_json in enumerate(df['materials']):
    try:
        if isinstance(mat_json, str):
            try:
                materials = json.loads(mat_json)
            except json.JSONDecodeError:
                materials = json.loads(mat_json.replace("'", '"'))
        else:
            materials = mat_json

        if isinstance(materials, dict):
            for mat in materials.keys():
                material_counter[mat] += 1

                # Check if material is erroneous
                if mat in ERRONEOUS_MATERIALS:
                    erroneous_products.add(idx)

                # Check if material is questionable
                if mat in QUESTIONABLE_MATERIALS:
                    questionable_products.add(idx)
    except:
        pass

print("✓ Scan complete\n")

# Results
print("=" * 80)
print("RESULTS")
print("=" * 80)
print()

print(f"Total products in dataset: {len(df):,}")
print()

print("📊 CLEARLY ERRONEOUS MATERIALS:")
print("-" * 80)
erroneous_count = 0
erroneous_list = []
for mat in sorted(ERRONEOUS_MATERIALS):
    count = material_counter.get(mat, 0)
    if count > 0:
        erroneous_count += count
        erroneous_list.append((mat, count))
        print(f"  ✗ {mat:35s}: {count:6,} occurrences")

print()
print(f"Total erroneous material occurrences: {erroneous_count:,}")
print(f"Products with erroneous materials:    {len(erroneous_products):,}")
print(f"Percentage of dataset:                {len(erroneous_products)/len(df)*100:.4f}%")
print()

if len(questionable_products) > 0:
    print("⚠️  QUESTIONABLE MATERIALS (might be intentional):")
    print("-" * 80)
    questionable_count = 0
    for mat in sorted(QUESTIONABLE_MATERIALS):
        count = material_counter.get(mat, 0)
        if count > 0:
            questionable_count += count
            print(f"  ? {mat:35s}: {count:6,} occurrences")

    print()
    print(f"Total questionable material occurrences: {questionable_count:,}")
    print(f"Products with questionable materials:    {len(questionable_products):,}")
    print(f"Percentage of dataset:                   {len(questionable_products)/len(df)*100:.4f}%")
    print()

# Combined stats
combined_products = erroneous_products | questionable_products
print("📊 COMBINED (Erroneous + Questionable):")
print("-" * 80)
print(f"Total affected products:  {len(combined_products):,}")
print(f"Percentage of dataset:    {len(combined_products)/len(df)*100:.4f}%")
print()

# Recommendation
print("=" * 80)
print("RECOMMENDATION")
print("=" * 80)
print()

if len(erroneous_products) / len(df) < 0.001:  # Less than 0.1%
    print("✓ The erroneous materials affect a very small portion of the dataset")
    print("  (<0.1%). It is SAFE to remove these products.")
    print()
    print("  Options:")
    print("  1. REMOVE affected products (recommended for clean dataset)")
    print("  2. FIX material names with mapping (more work, keeps data)")
    print()
    print(f"  After removal: {len(df) - len(erroneous_products):,} products remaining")
    print(f"  Data loss:     {len(erroneous_products):,} products ({len(erroneous_products)/len(df)*100:.4f}%)")
elif len(erroneous_products) / len(df) < 0.01:  # Less than 1%
    print("⚠️  The erroneous materials affect a small portion of the dataset")
    print("  (<1%). Consider fixing rather than removing.")
else:
    print("❌ The erroneous materials affect a significant portion of the dataset")
    print("  (>1%). Should fix rather than remove.")

print()
print("=" * 80)
