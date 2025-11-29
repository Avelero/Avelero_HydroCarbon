#!/usr/bin/env python3
"""
Remove Hierarchy Anomalies
Removes rows where parent_category is invalid or category is a parent name.
"""

import pandas as pd
import os

INPUT_CSV = "output/Product_data_cleaned.csv"
OUTPUT_CSV = "output/Product_data_cleaned.csv"
BACKUP_CSV = "output/Product_data_cleaned_pre_anomaly_removal.csv"

# Valid Parent Categories
VALID_PARENTS = {'Tops', 'Bottoms', 'Footwear', 'Outerwear', 'Dresses'}

print("=" * 80)
print("REMOVING HIERARCHY ANOMALIES")
print("=" * 80)
print()

# Create backup
print(f"Creating backup: {BACKUP_CSV}")
os.system(f"cp {INPUT_CSV} {BACKUP_CSV}")

# Load data
print(f"Loading data from: {INPUT_CSV}")
df = pd.read_csv(INPUT_CSV, low_memory=False)
print(f"Loaded {len(df):,} rows")
print()

# Identify anomalies
print("Identifying anomalies...")

# 1. Invalid Parent Categories
invalid_parents_mask = ~df['parent_category'].isin(VALID_PARENTS)
invalid_parents_count = invalid_parents_mask.sum()
print(f"   Found {invalid_parents_count} rows with invalid parent categories")
if invalid_parents_count > 0:
    print(f"   Examples: {df[invalid_parents_mask]['parent_category'].unique()}")

# 2. Category is a Parent Name
# (Rows where the 'category' column contains a value that should only be a parent)
category_is_parent_mask = df['category'].isin(VALID_PARENTS)
category_is_parent_count = category_is_parent_mask.sum()
print(f"   Found {category_is_parent_count} rows where Category is a Parent name")
if category_is_parent_count > 0:
    print(f"   Examples: {df[category_is_parent_mask]['category'].unique()}")

# Remove anomalies
print("\nRemoving anomalies...")
df_clean = df[~invalid_parents_mask & ~category_is_parent_mask].copy()
removed_count = len(df) - len(df_clean)

print(f"   Removed {removed_count} total rows")
print(f"   Remaining: {len(df_clean):,} rows")
print()

# Save cleaned data
print(f"Saving cleaned data to: {OUTPUT_CSV}")
df_clean.to_csv(OUTPUT_CSV, index=False)
print("Saved")

print()
print("=" * 80)
print("ANOMALY REMOVAL COMPLETE")
print("=" * 80)
print()
