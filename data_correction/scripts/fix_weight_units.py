#!/usr/bin/env python3
"""
Weight Unit Correction Script

Corrects weight values that are in grams but labeled as kilograms.

Based on weight_unit_research.md findings:
1. 281 products with weight < 0.01 kg (< 10 grams) - 90.7% correction rate
   → These values are in GRAMS, multiply by 1000 to convert to KG

2. 1 product with weight > 10 kg - 100.0% correction rate
   → This value is in GRAMS, divide by 1000 to convert to KG

All weight values will be in KILOGRAMS after correction.
"""

import pandas as pd
from datetime import datetime
from pathlib import Path

# Configuration
INPUT_CSV = "data_correction/output/Product_data_cleaned.csv"
OUTPUT_CSV = "data_correction/output/Product_data_weight_fixed.csv"
BACKUP_DIR = "data_correction/output/archives"
REPORT_FILE = "data_correction/output/comprehensive_analysis/weight_fix_report.txt"

# Thresholds based on research
WEIGHT_LOW_THRESHOLD = 0.01    # < 10 grams (multiply by 1000)
WEIGHT_HIGH_THRESHOLD = 10.0   # > 10 kg (divide by 1000)

# Create directories
Path(BACKUP_DIR).mkdir(exist_ok=True, parents=True)
Path("data_correction/output/comprehensive_analysis").mkdir(exist_ok=True, parents=True)

print("=" * 80)
print("WEIGHT UNIT CORRECTION")
print("=" * 80)
print()

# Load data
print(f"📂 Loading data from: {INPUT_CSV}")
df = pd.read_csv(INPUT_CSV, low_memory=False)
initial_count = len(df)
print(f"✓ Loaded {initial_count:,} rows\n")

# Create backup
backup_file = f"{BACKUP_DIR}/Product_data_pre_weight_fix_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
print(f"💾 Creating backup: {backup_file}")
df.to_csv(backup_file, index=False, quoting=1, escapechar='\\')
print(f"✓ Backup created\n")

# ============================================================================
# ANALYSIS BEFORE CORRECTION
# ============================================================================

print("=" * 80)
print("BEFORE CORRECTION - IDENTIFYING UNIT ERRORS")
print("=" * 80)
print()

# Find problematic weights
very_low = df[df['weight_kg'] < WEIGHT_LOW_THRESHOLD].copy()
very_high = df[df['weight_kg'] > WEIGHT_HIGH_THRESHOLD].copy()

low_count = len(very_low)
high_count = len(very_high)
total_to_fix = low_count + high_count

print(f"Low weights (< {WEIGHT_LOW_THRESHOLD} kg):  {low_count:,} products")
print(f"High weights (> {WEIGHT_HIGH_THRESHOLD} kg): {high_count:,} products")
print(f"Total to correct:              {total_to_fix:,} products ({(total_to_fix/initial_count)*100:.4f}%)")
print()

# Show examples before correction
print("=" * 80)
print("EXAMPLES BEFORE CORRECTION")
print("=" * 80)
print()

if low_count > 0:
    print("LOW WEIGHT EXAMPLES (likely in grams, will multiply by 1000):")
    print(f"{'Category':<30} {'Current (kg)':>15} {'Current (g)':>15} → {'Corrected (kg)':>15}")
    print("-" * 80)
    for idx, row in very_low.head(10).iterrows():
        current_kg = row['weight_kg']
        current_g = current_kg * 1000
        corrected_kg = current_kg * 1000
        print(f"{row['category']:<30} {current_kg:>15.6f} {current_g:>15.2f} → {corrected_kg:>15.3f}")
    print()

if high_count > 0:
    print("HIGH WEIGHT EXAMPLES (in grams labeled as kg, will divide by 1000):")
    print(f"{'Category':<30} {'Current (kg)':>15} {'Current (g)':>15} → {'Corrected (kg)':>15}")
    print("-" * 80)
    for idx, row in very_high.iterrows():
        current_kg = row['weight_kg']
        current_g = current_kg  # It's actually in grams already
        corrected_kg = current_kg / 1000
        print(f"{row['category']:<30} {current_kg:>15.2f} {current_g:>15.2f} → {corrected_kg:>15.3f}")
    print()

# ============================================================================
# APPLY CORRECTIONS
# ============================================================================

print("=" * 80)
print("APPLYING CORRECTIONS")
print("=" * 80)
print()

# Make a copy for correction
df_corrected = df.copy()

# Fix low weights (multiply by 1000)
if low_count > 0:
    print(f"Fixing {low_count:,} low weight products (multiply by 1000)...")
    mask_low = df_corrected['weight_kg'] < WEIGHT_LOW_THRESHOLD
    df_corrected.loc[mask_low, 'weight_kg'] = df_corrected.loc[mask_low, 'weight_kg'] * 1000
    print(f"✓ Fixed {low_count:,} low weight products")

# Fix high weights (divide by 1000)
if high_count > 0:
    print(f"Fixing {high_count:,} high weight products (divide by 1000)...")
    mask_high = df_corrected['weight_kg'] > WEIGHT_HIGH_THRESHOLD
    df_corrected.loc[mask_high, 'weight_kg'] = df_corrected.loc[mask_high, 'weight_kg'] / 1000
    print(f"✓ Fixed {high_count:,} high weight products")

print()

# ============================================================================
# VERIFICATION AFTER CORRECTION
# ============================================================================

print("=" * 80)
print("VERIFICATION AFTER CORRECTION")
print("=" * 80)
print()

# Check distribution after correction
after_low = len(df_corrected[df_corrected['weight_kg'] < WEIGHT_LOW_THRESHOLD])
after_high = len(df_corrected[df_corrected['weight_kg'] > WEIGHT_HIGH_THRESHOLD])

print("Distribution after correction:")
print(f"  Weights < {WEIGHT_LOW_THRESHOLD} kg:  {after_low:,} products (before: {low_count:,})")
print(f"  Weights > {WEIGHT_HIGH_THRESHOLD} kg: {after_high:,} products (before: {high_count:,})")
print()

# Statistics
weight_stats = df_corrected['weight_kg'].describe()
print("Weight statistics after correction:")
print(f"  Min:    {weight_stats['min']:.6f} kg")
print(f"  Q1:     {weight_stats['25%']:.6f} kg")
print(f"  Median: {weight_stats['50%']:.6f} kg")
print(f"  Mean:   {weight_stats['mean']:.6f} kg")
print(f"  Q3:     {weight_stats['75%']:.6f} kg")
print(f"  Max:    {weight_stats['max']:.6f} kg")
print()

# Distribution by magnitude
nano = len(df_corrected[df_corrected['weight_kg'] < 0.001])
micro = len(df_corrected[(df_corrected['weight_kg'] >= 0.001) & (df_corrected['weight_kg'] < 0.01)])
very_light = len(df_corrected[(df_corrected['weight_kg'] >= 0.01) & (df_corrected['weight_kg'] < 0.1)])
light = len(df_corrected[(df_corrected['weight_kg'] >= 0.1) & (df_corrected['weight_kg'] < 1)])
normal = len(df_corrected[(df_corrected['weight_kg'] >= 1) & (df_corrected['weight_kg'] <= 5)])
heavy = len(df_corrected[(df_corrected['weight_kg'] > 5) & (df_corrected['weight_kg'] <= 10)])
very_heavy = len(df_corrected[df_corrected['weight_kg'] > 10])

print("Distribution by magnitude:")
print(f"  nano (< 1g):         {nano:,} ({(nano/len(df_corrected))*100:.2f}%)")
print(f"  micro (1-10g):       {micro:,} ({(micro/len(df_corrected))*100:.2f}%)")
print(f"  very_light (10-100g):{very_light:,} ({(very_light/len(df_corrected))*100:.2f}%)")
print(f"  light (100-1000g):   {light:,} ({(light/len(df_corrected))*100:.2f}%)")
print(f"  normal (1-5 kg):     {normal:,} ({(normal/len(df_corrected))*100:.2f}%)")
print(f"  heavy (5-10 kg):     {heavy:,} ({(heavy/len(df_corrected))*100:.2f}%)")
print(f"  very_heavy (> 10 kg):{very_heavy:,} ({(very_heavy/len(df_corrected))*100:.2f}%)")
print()

# Show corrected examples
print("=" * 80)
print("EXAMPLES AFTER CORRECTION")
print("=" * 80)
print()

if low_count > 0:
    print("Previously LOW weights (now corrected):")
    print(f"{'Category':<30} {'Before (kg)':>15} → {'After (kg)':>15} {'After (g)':>15}")
    print("-" * 80)
    # Get the indices that were corrected
    low_indices = very_low.head(10).index
    for idx in low_indices:
        before_kg = very_low.loc[idx, 'weight_kg']
        after_kg = df_corrected.loc[idx, 'weight_kg']
        after_g = after_kg * 1000
        category = df_corrected.loc[idx, 'category']
        print(f"{category:<30} {before_kg:>15.6f} → {after_kg:>15.3f} {after_g:>15.2f}")
    print()

if high_count > 0:
    print("Previously HIGH weights (now corrected):")
    print(f"{'Category':<30} {'Before (kg)':>15} → {'After (kg)':>15} {'After (g)':>15}")
    print("-" * 80)
    high_indices = very_high.index
    for idx in high_indices:
        before_kg = very_high.loc[idx, 'weight_kg']
        after_kg = df_corrected.loc[idx, 'weight_kg']
        after_g = after_kg * 1000
        category = df_corrected.loc[idx, 'category']
        print(f"{category:<30} {before_kg:>15.2f} → {after_kg:>15.3f} {after_g:>15.2f}")
    print()

# ============================================================================
# SAVE CORRECTED DATA
# ============================================================================

print("=" * 80)
print("SAVING CORRECTED DATA")
print("=" * 80)
print()

print(f"💾 Saving corrected dataset to: {OUTPUT_CSV}")
df_corrected.to_csv(OUTPUT_CSV, index=False, quoting=1, escapechar='\\')
print(f"✓ Saved!\n")

# Save report
print(f"📊 Saving correction report to: {REPORT_FILE}")
with open(REPORT_FILE, 'w') as f:
    f.write("=" * 80 + "\n")
    f.write("WEIGHT UNIT CORRECTION REPORT\n")
    f.write("=" * 80 + "\n\n")
    f.write(f"Timestamp: {datetime.now().isoformat()}\n")
    f.write(f"Input file: {INPUT_CSV}\n")
    f.write(f"Output file: {OUTPUT_CSV}\n")
    f.write(f"Backup file: {backup_file}\n\n")

    f.write(f"Total products: {initial_count:,}\n")
    f.write(f"Products corrected: {total_to_fix:,} ({(total_to_fix/initial_count)*100:.4f}%)\n\n")

    f.write("Corrections applied:\n")
    f.write(f"  Low weights (< {WEIGHT_LOW_THRESHOLD} kg): {low_count:,} products (multiplied by 1000)\n")
    f.write(f"  High weights (> {WEIGHT_HIGH_THRESHOLD} kg): {high_count:,} products (divided by 1000)\n\n")

    f.write("Weight distribution BEFORE correction:\n")
    before_stats = df['weight_kg'].describe()
    f.write(f"  Min:    {before_stats['min']:.6f} kg\n")
    f.write(f"  Median: {before_stats['50%']:.6f} kg\n")
    f.write(f"  Mean:   {before_stats['mean']:.6f} kg\n")
    f.write(f"  Max:    {before_stats['max']:.6f} kg\n\n")

    f.write("Weight distribution AFTER correction:\n")
    f.write(f"  Min:    {weight_stats['min']:.6f} kg\n")
    f.write(f"  Median: {weight_stats['50%']:.6f} kg\n")
    f.write(f"  Mean:   {weight_stats['mean']:.6f} kg\n")
    f.write(f"  Max:    {weight_stats['max']:.6f} kg\n\n")

    f.write("Distribution by magnitude AFTER correction:\n")
    f.write(f"  nano (< 1g):         {nano:,} ({(nano/len(df_corrected))*100:.2f}%)\n")
    f.write(f"  micro (1-10g):       {micro:,} ({(micro/len(df_corrected))*100:.2f}%)\n")
    f.write(f"  very_light (10-100g):{very_light:,} ({(very_light/len(df_corrected))*100:.2f}%)\n")
    f.write(f"  light (100-1000g):   {light:,} ({(light/len(df_corrected))*100:.2f}%)\n")
    f.write(f"  normal (1-5 kg):     {normal:,} ({(normal/len(df_corrected))*100:.2f}%)\n")
    f.write(f"  heavy (5-10 kg):     {heavy:,} ({(heavy/len(df_corrected))*100:.2f}%)\n")
    f.write(f"  very_heavy (> 10 kg):{very_heavy:,} ({(very_heavy/len(df_corrected))*100:.2f}%)\n\n")

    f.write("=" * 80 + "\n")
    f.write("✓ WEIGHT UNIT CORRECTION COMPLETE\n")
    f.write("=" * 80 + "\n")
    f.write("\nAll weights are now in KILOGRAMS.\n")

print(f"✓ Saved!\n")

print("=" * 80)
print("✓ WEIGHT UNIT CORRECTION COMPLETE")
print("=" * 80)
print()
print(f"Corrected dataset: {OUTPUT_CSV}")
print(f"Backup:            {backup_file}")
print(f"Report:            {REPORT_FILE}")
print()
print(f"✅ All weights are now in KILOGRAMS")
print(f"   {total_to_fix:,} products corrected ({(total_to_fix/initial_count)*100:.4f}% of dataset)")
print()
