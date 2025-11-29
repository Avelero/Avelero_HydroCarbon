#!/usr/bin/env python3
"""
Targeted Cleanup Script
Removes ONLY:
1. Products with material errors (corrupted/impossible materials)
2. Products with category-parent mismatch

Does NOT remove numerical outliers (weight/distance)
"""

import pandas as pd
import json
from datetime import datetime
from pathlib import Path

# Configuration
INPUT_CSV = "data_correction/output/Product_data_complete.csv"
OUTPUT_CSV = "data_correction/output/Product_data_cleaned.csv"
BACKUP_DIR = "data_correction/output/archives"
REPORT_FILE = "data_correction/output/comprehensive_analysis/targeted_cleanup_report.json"

# Create archives directory
Path(BACKUP_DIR).mkdir(exist_ok=True, parents=True)

print("=" * 80)
print("TARGETED CLEANUP - MATERIAL ERRORS & CATEGORY MISMATCH ONLY")
print("=" * 80)
print()

# Initialize report
cleanup_report = {
    "timestamp": datetime.now().isoformat(),
    "input_file": INPUT_CSV,
    "output_file": OUTPUT_CSV,
    "actions": []
}

# Load data
print(f"Loading data from: {INPUT_CSV}")
df = pd.read_csv(INPUT_CSV, low_memory=False)
initial_count = len(df)
print(f"Loaded {initial_count:,} rows\n")

# Create backup
backup_file = f"{BACKUP_DIR}/Product_data_pre_targeted_cleanup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
print(f"Creating backup: {backup_file}")
df.to_csv(backup_file, index=False, quoting=1, escapechar='\\')
print(f"Backup created\n")

# ============================================================================
# STEP 1: REMOVE PRODUCTS WITH MATERIAL ERRORS
# ============================================================================

print("=" * 80)
print("STEP 1: REMOVING PRODUCTS WITH MATERIAL ERRORS")
print("=" * 80)
print()

# Define erroneous materials
ERRONEOUS_MATERIALS = {
    # Typos
    'poliamide_6', 'polporter_virgin', 'polyster_virgin', 'polyamid_6', 'polluster_recycled',
    'visvester_virgin', 'visyester_virgin', 'viscane', 'visise', 'viscluse',

    # Japanese characters
    'polリエステル_recycled', 'polリエステル_virgin',

    # Spacing errors
    'pol polyester_virgin', 'pol polyester_recycled', 'pol polyamide_6',

    # Impossible materials
    'polyester_organic',    # Polyester cannot be organic
    'polyester_bovine',     # Polyester is not bovine
    'coated_bovine',        # Unclear/wrong material

    # Confused materials
    'elastamide_66', 'elastester_virgin', 'elastical',
    'leather_bvine', 'leather_b_ovine',  # Typos of leather_bovine

    # Corrupted
    'pol<seg_125>_virgin',

    # Wrong numbers
    'polyester_6',  # Should be polyamide_6 or polyester_virgin
}

print(f"Scanning for {len(ERRONEOUS_MATERIALS)} types of erroneous materials...")
rows_to_remove_materials = []

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
                    rows_to_remove_materials.append(idx)
                    break
    except:
        pass

rows_to_remove_materials = list(set(rows_to_remove_materials))
removed_materials_count = len(rows_to_remove_materials)

print(f"Found {removed_materials_count} products with erroneous materials")

# Show breakdown by material type
material_error_breakdown = {}
for idx in rows_to_remove_materials:
    try:
        mat_json = df.iloc[idx]['materials']
        materials = json.loads(mat_json.replace("'", '"'))
        for mat in materials.keys():
            if mat in ERRONEOUS_MATERIALS:
                material_error_breakdown[mat] = material_error_breakdown.get(mat, 0) + 1
    except:
        pass

print("\nBreakdown by error type:")
for mat, count in sorted(material_error_breakdown.items(), key=lambda x: x[1], reverse=True):
    print(f"  * {mat:35s}: {count} products")

print(f"\nRemoving {removed_materials_count:,} products...")
df = df.drop(rows_to_remove_materials).reset_index(drop=True)
print(f"Removed {removed_materials_count:,} products")
print(f"  Remaining: {len(df):,} products\n")

cleanup_report["actions"].append({
    "step": 1,
    "action": "Remove products with material errors",
    "products_removed": removed_materials_count,
    "error_types": material_error_breakdown,
    "remaining_products": len(df)
})

# ============================================================================
# STEP 2: REMOVE PRODUCTS WITH CATEGORY-PARENT MISMATCH
# ============================================================================

print("=" * 80)
print("STEP 2: REMOVING PRODUCTS WITH CATEGORY-PARENT MISMATCH")
print("=" * 80)
print()

# Find the specific mismatch: Category "Maxi" with Parent "Maxi"
mismatched = df[(df['category'] == 'Maxi') & (df['parent_category'] == 'Maxi')]
removed_mismatch_count = len(mismatched)

print(f"Found {removed_mismatch_count} product(s) with category-parent mismatch:")
print(f"  * Category 'Maxi' with Parent 'Maxi' (should be Parent 'Dresses')")

if removed_mismatch_count > 0:
    print(f"\nRemoving {removed_mismatch_count} product(s)...")
    df = df.drop(mismatched.index).reset_index(drop=True)
    print(f"Removed {removed_mismatch_count} product(s)")
    print(f"  Remaining: {len(df):,} products\n")

cleanup_report["actions"].append({
    "step": 2,
    "action": "Remove products with category-parent mismatch",
    "products_removed": removed_mismatch_count,
    "remaining_products": len(df)
})

# ============================================================================
# SUMMARY
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

print("Breakdown by action:")
print(f"  1. Material errors:          {removed_materials_count:,} products")
print(f"  2. Category-parent mismatch: {removed_mismatch_count:,} products")
print()

print("NOTE: Weight and distance outliers were NOT removed")
print("   They will be analyzed and explained separately.")
print()

# Save cleaned dataset
print(f"Saving cleaned dataset to: {OUTPUT_CSV}")
df.to_csv(OUTPUT_CSV, index=False, quoting=1, escapechar='\\')
print(f"Saved!\n")

# Save cleanup report
print(f"Saving cleanup report to: {REPORT_FILE}")
with open(REPORT_FILE, 'w') as f:
    json.dump(cleanup_report, f, indent=2)
print(f"Saved!\n")

print("=" * 80)
print("TARGETED CLEANUP COMPLETE")
print("=" * 80)
print()
print(f"Cleaned dataset: {OUTPUT_CSV}")
print(f"Backup:          {backup_file}")
print(f"Report:          {REPORT_FILE}")
print()
print("Next step: Run outlier explanation analysis")
print("Command: python3 data_correction/scripts/explain_outliers.py")
print()
