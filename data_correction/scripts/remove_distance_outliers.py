#!/usr/bin/env python3
"""
Remove Distance Outliers

Removes products with impossible distances (> Earth's circumference).
This represents products with total_distance_km > 40,000 km.
"""

import pandas as pd
from datetime import datetime
from pathlib import Path

# Configuration
INPUT_CSV = "data_correction/output/Product_data_final.csv"
OUTPUT_CSV = "data_correction/output/Product_data_final.csv"
BACKUP_DIR = "data_correction/output/archives"
REPORT_FILE = "data_correction/output/comprehensive_analysis/distance_outlier_removal_report.md"

# Distance threshold (Earth's circumference)
EARTH_CIRCUMFERENCE = 40075  # km
DISTANCE_THRESHOLD = 40000   # km (slightly below circumference for safety)

# Create directories
Path(BACKUP_DIR).mkdir(exist_ok=True, parents=True)
Path("data_correction/output/comprehensive_analysis").mkdir(exist_ok=True, parents=True)

print("=" * 80)
print("REMOVING DISTANCE OUTLIERS")
print("=" * 80)
print()

# ============================================================================
# STEP 1: LOAD DATA
# ============================================================================

print("=" * 80)
print("STEP 1: LOADING DATA")
print("=" * 80)
print()

print(f"📂 Loading data from: {INPUT_CSV}")
df = pd.read_csv(INPUT_CSV, low_memory=False)
initial_count = len(df)
print(f"✓ Loaded {initial_count:,} products\n")

# Convert distance to numeric
df['total_distance_km'] = pd.to_numeric(df['total_distance_km'], errors='coerce')

# ============================================================================
# STEP 2: CREATE BACKUP
# ============================================================================

print("=" * 80)
print("STEP 2: CREATING BACKUP")
print("=" * 80)
print()

backup_file = f"{BACKUP_DIR}/Product_data_pre_distance_cleanup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
print(f"💾 Creating backup: {backup_file}")
df.to_csv(backup_file, index=False, quoting=1, escapechar='\\')
print(f"✓ Backup created\n")

# ============================================================================
# STEP 3: ANALYZE DISTANCE OUTLIERS
# ============================================================================

print("=" * 80)
print("STEP 3: ANALYZING DISTANCE OUTLIERS")
print("=" * 80)
print()

print(f"📍 Reference: Earth circumference = {EARTH_CIRCUMFERENCE:,} km")
print(f"🎯 Threshold: Removing products with distance > {DISTANCE_THRESHOLD:,} km")
print()

# Get statistics before removal
distance_stats_before = df['total_distance_km'].describe()
print("Distance statistics BEFORE removal:")
print(f"  Min:    {distance_stats_before['min']:.2f} km")
print(f"  Q1:     {distance_stats_before['25%']:.2f} km")
print(f"  Median: {distance_stats_before['50%']:.2f} km")
print(f"  Mean:   {distance_stats_before['mean']:.2f} km")
print(f"  Q3:     {distance_stats_before['75%']:.2f} km")
print(f"  Max:    {distance_stats_before['max']:.2f} km")
print()

# Find outliers
outliers = df[df['total_distance_km'] > DISTANCE_THRESHOLD].copy()
outlier_count = len(outliers)

print(f"Found {outlier_count:,} products with distance > {DISTANCE_THRESHOLD:,} km")
print(f"Percentage of dataset: {(outlier_count/initial_count)*100:.4f}%")
print()

# Show top outliers
if outlier_count > 0:
    print("Top 20 distance outliers:")
    print(f"{'Category':<30} {'Country':<10} {'Distance (km)':>15} {'Times Earth':>15}")
    print("-" * 75)

    top_outliers = outliers.nlargest(20, 'total_distance_km')
    for idx, row in top_outliers.iterrows():
        category = str(row['category'])[:28]
        country = str(row['manufacturer_country'])[:8] if pd.notna(row['manufacturer_country']) else 'Unknown'
        distance = row['total_distance_km']
        times_earth = distance / EARTH_CIRCUMFERENCE
        print(f"{category:<30} {country:<10} {distance:>15,.2f} {times_earth:>15.2f}x")
    print()

# Category breakdown
print("Outliers by category:")
category_counts = outliers['category'].value_counts().head(10)
for cat, count in category_counts.items():
    print(f"  {cat:<30}: {count:,} products")
print()

# Country breakdown
print("Outliers by country:")
country_counts = outliers['manufacturer_country'].value_counts().head(10)
for country, count in country_counts.items():
    print(f"  {country:<30}: {count:,} products")
print()

# ============================================================================
# STEP 4: REMOVE OUTLIERS
# ============================================================================

print("=" * 80)
print("STEP 4: REMOVING DISTANCE OUTLIERS")
print("=" * 80)
print()

print(f"Removing {outlier_count:,} products with distance > {DISTANCE_THRESHOLD:,} km...")
df_cleaned = df[df['total_distance_km'] <= DISTANCE_THRESHOLD].copy()
final_count = len(df_cleaned)

print(f"✓ Removed {outlier_count:,} products")
print(f"  Remaining: {final_count:,} products\n")

# ============================================================================
# STEP 5: VERIFICATION
# ============================================================================

print("=" * 80)
print("STEP 5: VERIFICATION")
print("=" * 80)
print()

# Get statistics after removal
distance_stats_after = df_cleaned['total_distance_km'].describe()
print("Distance statistics AFTER removal:")
print(f"  Min:    {distance_stats_after['min']:.2f} km")
print(f"  Q1:     {distance_stats_after['25%']:.2f} km")
print(f"  Median: {distance_stats_after['50%']:.2f} km")
print(f"  Mean:   {distance_stats_after['mean']:.2f} km")
print(f"  Q3:     {distance_stats_after['75%']:.2f} km")
print(f"  Max:    {distance_stats_after['max']:.2f} km")
print()

# Verify no outliers remain
remaining_outliers = len(df_cleaned[df_cleaned['total_distance_km'] > DISTANCE_THRESHOLD])
print(f"Verification: Products with distance > {DISTANCE_THRESHOLD:,} km: {remaining_outliers:,}")

if remaining_outliers == 0:
    print("✅ All distance outliers removed successfully!")
else:
    print("⚠️  Some outliers may remain")
print()

# Distribution after cleanup
print("Distance distribution after cleanup:")
dist_ranges = [
    ("< 1,000 km", 0, 1000),
    ("1,000 - 10,000 km", 1000, 10000),
    ("10,000 - 20,000 km", 10000, 20000),
    ("20,000 - 30,000 km", 20000, 30000),
    ("30,000 - 40,000 km", 30000, 40000),
]

for label, low, high in dist_ranges:
    count = len(df_cleaned[(df_cleaned['total_distance_km'] >= low) & (df_cleaned['total_distance_km'] < high)])
    pct = (count / final_count) * 100
    print(f"  {label:<20}: {count:>8,} products ({pct:>6.2f}%)")
print()

# ============================================================================
# STEP 6: SAVE CLEANED DATA
# ============================================================================

print("=" * 80)
print("STEP 6: SAVING CLEANED DATA")
print("=" * 80)
print()

print(f"💾 Saving cleaned dataset to: {OUTPUT_CSV}")
df_cleaned.to_csv(OUTPUT_CSV, index=False, quoting=1, escapechar='\\')
print(f"✓ Saved {final_count:,} products!\n")

# ============================================================================
# STEP 7: CREATE REPORT
# ============================================================================

print("=" * 80)
print("STEP 7: CREATING REPORT")
print("=" * 80)
print()

print(f"📊 Generating report: {REPORT_FILE}")

with open(REPORT_FILE, 'w') as f:
    f.write("# Distance Outlier Removal Report\n\n")
    f.write(f"**Report Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    f.write("---\n\n")

    f.write("## Summary\n\n")
    f.write(f"**Input File**: {INPUT_CSV}\n\n")
    f.write(f"**Output File**: {OUTPUT_CSV}\n\n")
    f.write(f"**Backup File**: {backup_file}\n\n")

    f.write("| Metric | Count |\n")
    f.write("|--------|-------|\n")
    f.write(f"| Initial Products | {initial_count:,} |\n")
    f.write(f"| Products Removed | {outlier_count:,} |\n")
    f.write(f"| Final Products | {final_count:,} |\n")
    f.write(f"| Data Loss | {(outlier_count/initial_count)*100:.4f}% |\n\n")

    f.write("---\n\n")
    f.write("## Removal Criteria\n\n")
    f.write(f"**Threshold**: `total_distance_km > {DISTANCE_THRESHOLD:,} km`\n\n")
    f.write(f"**Reason**: Distance exceeds Earth's circumference ({EARTH_CIRCUMFERENCE:,} km)\n\n")
    f.write("**Rationale**: Such distances are physically impossible for direct shipping.\n\n")

    f.write("---\n\n")
    f.write("## Distance Statistics\n\n")

    f.write("### Before Removal\n\n")
    f.write("| Statistic | Value (km) |\n")
    f.write("|-----------|------------|\n")
    f.write(f"| Minimum | {distance_stats_before['min']:.2f} |\n")
    f.write(f"| 25th Percentile | {distance_stats_before['25%']:.2f} |\n")
    f.write(f"| Median | {distance_stats_before['50%']:.2f} |\n")
    f.write(f"| Mean | {distance_stats_before['mean']:.2f} |\n")
    f.write(f"| 75th Percentile | {distance_stats_before['75%']:.2f} |\n")
    f.write(f"| Maximum | {distance_stats_before['max']:.2f} |\n\n")

    f.write("### After Removal\n\n")
    f.write("| Statistic | Value (km) |\n")
    f.write("|-----------|------------|\n")
    f.write(f"| Minimum | {distance_stats_after['min']:.2f} |\n")
    f.write(f"| 25th Percentile | {distance_stats_after['25%']:.2f} |\n")
    f.write(f"| Median | {distance_stats_after['50%']:.2f} |\n")
    f.write(f"| Mean | {distance_stats_after['mean']:.2f} |\n")
    f.write(f"| 75th Percentile | {distance_stats_after['75%']:.2f} |\n")
    f.write(f"| Maximum | {distance_stats_after['max']:.2f} |\n\n")

    f.write("---\n\n")
    f.write("## Outliers Removed by Category\n\n")
    f.write("| Category | Products Removed |\n")
    f.write("|----------|------------------|\n")
    for cat, count in category_counts.head(20).items():
        f.write(f"| {cat} | {count:,} |\n")
    f.write("\n")

    f.write("---\n\n")
    f.write("## Outliers Removed by Country\n\n")
    f.write("| Country Code | Products Removed |\n")
    f.write("|--------------|------------------|\n")
    for country, count in country_counts.head(20).items():
        f.write(f"| {country} | {count:,} |\n")
    f.write("\n")

    f.write("---\n\n")
    f.write("## Final Distance Distribution\n\n")
    f.write("| Range | Count | Percentage |\n")
    f.write("|-------|-------|------------|\n")
    for label, low, high in dist_ranges:
        count = len(df_cleaned[(df_cleaned['total_distance_km'] >= low) & (df_cleaned['total_distance_km'] < high)])
        pct = (count / final_count) * 100
        f.write(f"| {label} | {count:,} | {pct:.2f}% |\n")
    f.write("\n")

    f.write("---\n\n")
    f.write("## Quality Assurance\n\n")
    f.write(f"✅ **All distances ≤ {DISTANCE_THRESHOLD:,} km** (physically plausible)\n\n")
    f.write(f"✅ **Maximum distance**: {distance_stats_after['max']:.2f} km (within Earth's circumference)\n\n")
    f.write(f"✅ **Mean distance**: {distance_stats_after['mean']:.2f} km (reasonable for global supply chains)\n\n")

print(f"✓ Report saved!\n")

# ============================================================================
# FINAL SUMMARY
# ============================================================================

print("=" * 80)
print("✓ DISTANCE OUTLIER REMOVAL COMPLETE")
print("=" * 80)
print()
print(f"📁 Updated dataset: {OUTPUT_CSV}")
print(f"💾 Backup:          {backup_file}")
print(f"📊 Report:          {REPORT_FILE}")
print()
print(f"Summary:")
print(f"  Initial products:    {initial_count:,}")
print(f"  Products removed:    {outlier_count:,} ({(outlier_count/initial_count)*100:.4f}%)")
print(f"  Final products:      {final_count:,}")
print()
print(f"✅ All distances now ≤ {DISTANCE_THRESHOLD:,} km")
print(f"✅ Maximum distance: {distance_stats_after['max']:.2f} km")
print()
