#!/usr/bin/env python3
"""
Comprehensive Dataset Cleanup
Addresses all issues found in comprehensive analysis:
1. Remove critical errors (material errors + orphaned categories)
2. Fix category naming inconsistencies
3. Clean up and standardize material names
4. Remove numerical outliers
"""

import pandas as pd
import json
import numpy as np
from datetime import datetime
from pathlib import Path

# Configuration
INPUT_CSV = "data_correction/output/Product_data_complete.csv"
OUTPUT_CSV = "data_correction/output/Product_data_cleaned_comprehensive.csv"
BACKUP_CSV = "data_correction/output/archives/Product_data_pre_comprehensive_cleanup.csv"
REPORT_FILE = "data_correction/output/comprehensive_analysis/cleanup_report.json"

# Create archives directory
Path("data_correction/output/archives").mkdir(exist_ok=True, parents=True)

print("=" * 80)
print("COMPREHENSIVE DATASET CLEANUP")
print("=" * 80)
print()

# Initialize cleanup report
cleanup_report = {
    "timestamp": datetime.now().isoformat(),
    "input_file": INPUT_CSV,
    "output_file": OUTPUT_CSV,
    "steps": []
}

# Load data
print(f"Loading Loading data from: {INPUT_CSV}")
df = pd.read_csv(INPUT_CSV, low_memory=False)
initial_count = len(df)
print(f"OK Loaded {initial_count:,} rows\n")

# Create backup
print(f"Saving Creating backup: {BACKUP_CSV}")
df.to_csv(BACKUP_CSV, index=False, quoting=1, escapechar='\\')
print(f"OK Backup created\n")

# ============================================================================
# STEP 1: REMOVE CRITICAL MATERIAL ERRORS (60 products)
# ============================================================================

print("=" * 80)
print("STEP 1: REMOVING CRITICAL MATERIAL ERRORS")
print("=" * 80)
print()

ERRONEOUS_MATERIALS = {
    # Typos and misspellings
    'poliamide_6', 'polporter_virgin', 'polyster_virgin', 'polyester_6', 'polyamid_6',
    # Japanese characters
    'polリエステル_recycled', 'polリエステル_virgin',
    # Space errors
    'pol polyester_virgin', 'pol polyester_recycled', 'pol polyamide_6',
    # Confused material names
    'elastamide_66', 'elastester_virgin', 'elastical',
    # Wrong material combinations
    'polyester_bovine', 'coated_bovine', 'leather_bvine', 'leather_b_ovine',
    # More typos
    'visvester_virgin', 'visyester_virgin', 'polluster_recycled',
    'viscane', 'visise', 'viscluse',
    # Corrupted data
    'pol<seg_125>_virgin',
}

print("Scanning for erroneous materials...")
rows_to_remove = []

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
                if mat in ERRONEOUS_MATERIALS:
                    rows_to_remove.append(idx)
                    break
    except:
        pass

rows_to_remove = list(set(rows_to_remove))
removed_count_step1 = len(rows_to_remove)

print(f"Found {removed_count_step1} products with erroneous materials")
print(f"Removing...")

df = df.drop(rows_to_remove).reset_index(drop=True)

print(f"OK Removed {removed_count_step1:,} products")
print(f"  Remaining: {len(df):,} products\n")

cleanup_report["steps"].append({
    "step": 1,
    "name": "Remove critical material errors",
    "products_removed": removed_count_step1,
    "remaining_products": len(df)
})

# ============================================================================
# STEP 2: FIX ORPHANED CATEGORIES
# ============================================================================

print("=" * 80)
print("STEP 2: FIXING ORPHANED CATEGORIES")
print("=" * 80)
print()

# Fix "Dresses" category (1 product) - reclassify as "Gowns"
orphan_dresses = df[df['category'] == 'Dresses']
print(f"Found {len(orphan_dresses)} product(s) in orphaned 'Dresses' category")
if len(orphan_dresses) > 0:
    print(f"  Reclassifying to 'Gowns'...")
    df.loc[df['category'] == 'Dresses', 'category'] = 'Gowns'
    print(f"OK Reclassified {len(orphan_dresses)} product(s)")

# Fix "Maxi" parent category (1 product) - move to "Dresses" parent
orphan_maxi_parent = df[df['parent_category'] == 'Maxi']
print(f"Found {len(orphan_maxi_parent)} product(s) with orphaned 'Maxi' parent category")
if len(orphan_maxi_parent) > 0:
    print(f"  Moving to 'Dresses' parent category...")
    df.loc[df['parent_category'] == 'Maxi', 'parent_category'] = 'Dresses'
    # Also ensure category is appropriate
    df.loc[df['parent_category'] == 'Dresses', 'category'] = df.loc[df['parent_category'] == 'Dresses', 'category'].replace('Maxi', 'Maxi Dresses')
    print(f"OK Fixed {len(orphan_maxi_parent)} product(s)")

print()

cleanup_report["steps"].append({
    "step": 2,
    "name": "Fix orphaned categories",
    "categories_fixed": len(orphan_dresses) + len(orphan_maxi_parent),
    "remaining_products": len(df)
})

# ============================================================================
# STEP 3: STANDARDIZE CATEGORY NAMING
# ============================================================================

print("=" * 80)
print("STEP 3: STANDARDIZING CATEGORY NAMING")
print("=" * 80)
print()

category_mappings = {
    # Standardize dress categories to "X Dresses" format
    'Maxi': 'Maxi Dresses',
    'Midi': 'Midi Dresses',
    'Mini': 'Mini Dresses',

    # Consolidate sweatwear
    'Sweatpants & Joggers': 'Joggers',
    'Sweatpants': 'Joggers',

    # Consolidate sweaters
    'Sweaters & Knitwear': 'Sweaters',

    # Consolidate hoodies/sweatshirts
    'Sweatshirts & Hoodies': 'Hoodies',
}

print("Applying category mappings:")
for old_name, new_name in category_mappings.items():
    count = (df['category'] == old_name).sum()
    if count > 0:
        print(f"  * {old_name:30s} -> {new_name:30s} ({count:,} products)")
        df.loc[df['category'] == old_name, 'category'] = new_name

print(f"\nOK Standardized {sum(1 for k, v in category_mappings.items() if (df['category'] == k).sum() > 0)} category names\n")

cleanup_report["steps"].append({
    "step": 3,
    "name": "Standardize category naming",
    "mappings_applied": len(category_mappings),
    "remaining_products": len(df)
})

# ============================================================================
# STEP 4: CLEAN UP AND STANDARDIZE MATERIAL NAMES
# ============================================================================

print("=" * 80)
print("STEP 4: CLEANING UP AND STANDARDIZING MATERIAL NAMES")
print("=" * 80)
print()

material_mappings = {
    # Inconsistent naming
    'rubber_synthetic': 'synthetic_rubber_sbr',
    'nylon_recycled': 'polyamide_recycled',
    'nylon': 'polyamide_6',
    'rayon': 'viscose',

    # Inconsistent virgin/conventional
    'cotton_virgin': 'cotton_conventional',
    'polyamide_virgin': 'polyamide_6',
    'polyester_conventional': 'polyester_virgin',
    'polyester_generic': 'polyester_virgin',
    'recycled_polyester': 'polyester_recycled',
    'cotton_recycled': 'cotton_recycled',  # Keep this one

    # Fix wrong numbers
    'polyester_66': 'polyamide_66',
}

# Remove impossible materials
IMPOSSIBLE_MATERIALS = {
    'polyester_organic',  # Polyester can't be organic
}

print("Cleaning material names in dataset...")
materials_updated = 0
materials_removed = 0
rows_to_remove_impossible = []

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
            updated = False
            new_materials = {}

            for mat, percentage in materials.items():
                # Check if material is impossible
                if mat in IMPOSSIBLE_MATERIALS:
                    rows_to_remove_impossible.append(idx)
                    break

                # Apply mapping
                if mat in material_mappings:
                    new_mat = material_mappings[mat]
                    new_materials[new_mat] = percentage
                    updated = True
                else:
                    new_materials[mat] = percentage

            if idx not in rows_to_remove_impossible and updated:
                # Save updated materials back to dataframe
                df.at[idx, 'materials'] = json.dumps(new_materials)
                materials_updated += 1
    except:
        pass

# Remove products with impossible materials
rows_to_remove_impossible = list(set(rows_to_remove_impossible))
removed_count_step4 = len(rows_to_remove_impossible)

if removed_count_step4 > 0:
    print(f"Found {removed_count_step4} products with impossible materials (e.g., polyester_organic)")
    print(f"Removing...")
    df = df.drop(rows_to_remove_impossible).reset_index(drop=True)

print(f"OK Updated materials in {materials_updated:,} products")
print(f"OK Removed {removed_count_step4:,} products with impossible materials")
print(f"  Remaining: {len(df):,} products\n")

cleanup_report["steps"].append({
    "step": 4,
    "name": "Clean up and standardize material names",
    "materials_updated": materials_updated,
    "products_removed": removed_count_step4,
    "remaining_products": len(df)
})

# ============================================================================
# STEP 5: REMOVE NUMERICAL OUTLIERS
# ============================================================================

print("=" * 80)
print("STEP 5: REMOVING NUMERICAL OUTLIERS")
print("=" * 80)
print()

# Convert to numeric
df['weight_kg'] = pd.to_numeric(df['weight_kg'], errors='coerce')
df['total_distance_km'] = pd.to_numeric(df['total_distance_km'], errors='coerce')

print("Before outlier removal:")
print(f"  Weight range:   {df['weight_kg'].min():.4f} - {df['weight_kg'].max():.2f} kg")
print(f"  Distance range: {df['total_distance_km'].min():.2f} - {df['total_distance_km'].max():.2f} km")
print()

# Define outlier criteria
WEIGHT_MIN = 0.01    # 10 grams minimum
WEIGHT_MAX = 50.0    # 50 kg maximum
DISTANCE_MIN = 1.0   # 1 km minimum
DISTANCE_MAX = 40000.0  # Earth's circumference

print(f"Outlier criteria:")
print(f"  Weight:   {WEIGHT_MIN} kg - {WEIGHT_MAX} kg")
print(f"  Distance: {DISTANCE_MIN} km - {DISTANCE_MAX} km")
print()

# Count outliers
weight_outliers_low = (df['weight_kg'] < WEIGHT_MIN).sum()
weight_outliers_high = (df['weight_kg'] > WEIGHT_MAX).sum()
distance_outliers_low = (df['total_distance_km'] < DISTANCE_MIN).sum()
distance_outliers_high = (df['total_distance_km'] > DISTANCE_MAX).sum()

print(f"Outliers found:")
print(f"  Weight too low  (< {WEIGHT_MIN} kg):      {weight_outliers_low:,}")
print(f"  Weight too high (> {WEIGHT_MAX} kg):      {weight_outliers_high:,}")
print(f"  Distance too low  (< {DISTANCE_MIN} km):  {distance_outliers_low:,}")
print(f"  Distance too high (> {DISTANCE_MAX} km):  {distance_outliers_high:,}")
print()

# Remove outliers
before_outlier_removal = len(df)

df = df[
    (df['weight_kg'] >= WEIGHT_MIN) &
    (df['weight_kg'] <= WEIGHT_MAX) &
    (df['total_distance_km'] >= DISTANCE_MIN) &
    (df['total_distance_km'] <= DISTANCE_MAX)
].reset_index(drop=True)

removed_count_step5 = before_outlier_removal - len(df)

print(f"After outlier removal:")
print(f"  Weight range:   {df['weight_kg'].min():.4f} - {df['weight_kg'].max():.2f} kg")
print(f"  Distance range: {df['total_distance_km'].min():.2f} - {df['total_distance_km'].max():.2f} km")
print()

print(f"OK Removed {removed_count_step5:,} outlier products")
print(f"  Remaining: {len(df):,} products\n")

cleanup_report["steps"].append({
    "step": 5,
    "name": "Remove numerical outliers",
    "weight_outliers_low": int(weight_outliers_low),
    "weight_outliers_high": int(weight_outliers_high),
    "distance_outliers_low": int(distance_outliers_low),
    "distance_outliers_high": int(distance_outliers_high),
    "products_removed": removed_count_step5,
    "remaining_products": len(df)
})

# ============================================================================
# FINAL SUMMARY
# ============================================================================

print("=" * 80)
print("CLEANUP SUMMARY")
print("=" * 80)
print()

final_count = len(df)
total_removed = initial_count - final_count
removal_percentage = (total_removed / initial_count) * 100

cleanup_report["summary"] = {
    "initial_products": initial_count,
    "final_products": final_count,
    "total_removed": total_removed,
    "removal_percentage": round(removal_percentage, 4)
}

print(f"Initial products:        {initial_count:,}")
print(f"Final products:          {final_count:,}")
print(f"Total removed:           {total_removed:,}")
print(f"Removal percentage:      {removal_percentage:.4f}%")
print()

print("Breakdown by step:")
print(f"  1. Material errors:       {removed_count_step1:,} products")
print(f"  2. Orphaned categories:   0 products (fixed, not removed)")
print(f"  3. Category naming:       0 products (renamed, not removed)")
print(f"  4. Material cleanup:      {removed_count_step4:,} products")
print(f"  5. Numerical outliers:    {removed_count_step5:,} products")
print()

# Save cleaned dataset
print(f"Saving Saving cleaned dataset to: {OUTPUT_CSV}")
df.to_csv(OUTPUT_CSV, index=False, quoting=1, escapechar='\\')
print(f"OK Saved!\n")

# Save cleanup report
print(f"Data Saving cleanup report to: {REPORT_FILE}")
with open(REPORT_FILE, 'w') as f:
    json.dump(cleanup_report, f, indent=2)
print(f"OK Saved!\n")

print("=" * 80)
print("OK COMPREHENSIVE CLEANUP COMPLETE")
print("=" * 80)
print()
print(f"Cleaned dataset: {OUTPUT_CSV}")
print(f"Backup created:  {BACKUP_CSV}")
print(f"Cleanup report:  {REPORT_FILE}")
print()
print("Next step: Run comprehensive analysis on cleaned data")
print("Command: python3 data_correction/scripts/comprehensive_analysis.py")
print()
