#!/usr/bin/env python3
"""
Outlier Explanation Analysis
Explains weight and distance outliers instead of removing them.
Provides insights into patterns, distributions, and potential reasons.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from datetime import datetime

# Configuration
INPUT_CSV = "data_correction/output/Product_data_cleaned.csv"
OUTPUT_DIR = "data_correction/output/comprehensive_analysis"
REPORT_FILE = f"{OUTPUT_DIR}/outlier_explanation_report.md"
DPI = 300

print("=" * 80)
print("OUTLIER EXPLANATION ANALYSIS")
print("=" * 80)
print()

# Load data
print(f"Loading Loading data from: {INPUT_CSV}")
df = pd.read_csv(INPUT_CSV, low_memory=False)
print(f"OK Loaded {len(df):,} rows\n")

# Convert numeric columns
df['weight_kg'] = pd.to_numeric(df['weight_kg'], errors='coerce')
df['total_distance_km'] = pd.to_numeric(df['total_distance_km'], errors='coerce')

# Initialize report
report_lines = []
report_lines.append("# Weight and Distance Outliers - Detailed Explanation")
report_lines.append(f"\n**Analysis Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
report_lines.append(f"**Total Products**: {len(df):,}")
report_lines.append("\n---\n")

# ============================================================================
# WEIGHT OUTLIER ANALYSIS
# ============================================================================

print("=" * 80)
print("1. WEIGHT OUTLIER ANALYSIS")
print("=" * 80)
print()

report_lines.append("## 1. Weight Outliers\n")

# Overall statistics
weight_stats = {
    'min': df['weight_kg'].min(),
    'max': df['weight_kg'].max(),
    'mean': df['weight_kg'].mean(),
    'median': df['weight_kg'].median(),
    'std': df['weight_kg'].std(),
    'q1': df['weight_kg'].quantile(0.25),
    'q3': df['weight_kg'].quantile(0.75)
}

print(f"Weight Statistics:")
print(f"  Min:    {weight_stats['min']:.6f} kg")
print(f"  Q1:     {weight_stats['q1']:.4f} kg")
print(f"  Median: {weight_stats['median']:.4f} kg")
print(f"  Mean:   {weight_stats['mean']:.4f} kg")
print(f"  Q3:     {weight_stats['q3']:.4f} kg")
print(f"  Max:    {weight_stats['max']:.2f} kg")
print(f"  Std:    {weight_stats['std']:.4f} kg\n")

report_lines.append("### Overall Statistics\n")
report_lines.append("| Metric | Value |")
report_lines.append("|--------|-------|")
report_lines.append(f"| Minimum | {weight_stats['min']:.6f} kg |")
report_lines.append(f"| 25th Percentile (Q1) | {weight_stats['q1']:.4f} kg |")
report_lines.append(f"| Median | {weight_stats['median']:.4f} kg |")
report_lines.append(f"| Mean | {weight_stats['mean']:.4f} kg |")
report_lines.append(f"| 75th Percentile (Q3) | {weight_stats['q3']:.4f} kg |")
report_lines.append(f"| Maximum | {weight_stats['max']:.2f} kg |")
report_lines.append(f"| Standard Deviation | {weight_stats['std']:.4f} kg |")
report_lines.append("\n")

# Define outlier thresholds
WEIGHT_EXTREME_LOW = 0.01   # < 10 grams
WEIGHT_LOW = 0.05           # < 50 grams
WEIGHT_NORMAL_MIN = 0.05
WEIGHT_NORMAL_MAX = 5.0
WEIGHT_HIGH = 5.0           # > 5 kg
WEIGHT_EXTREME_HIGH = 10.0  # > 10 kg

# Categorize weights
weight_categories = {
    'extreme_low': df[df['weight_kg'] < WEIGHT_EXTREME_LOW],
    'low': df[(df['weight_kg'] >= WEIGHT_EXTREME_LOW) & (df['weight_kg'] < WEIGHT_LOW)],
    'normal': df[(df['weight_kg'] >= WEIGHT_NORMAL_MIN) & (df['weight_kg'] <= WEIGHT_NORMAL_MAX)],
    'high': df[(df['weight_kg'] > WEIGHT_HIGH) & (df['weight_kg'] <= WEIGHT_EXTREME_HIGH)],
    'extreme_high': df[df['weight_kg'] > WEIGHT_EXTREME_HIGH]
}

print("Weight Distribution by Category:")
for cat, data in weight_categories.items():
    pct = (len(data) / len(df)) * 100
    print(f"  {cat:15s}: {len(data):7,} products ({pct:5.2f}%)")
print()

report_lines.append("### Weight Distribution by Category\n")
report_lines.append("| Category | Count | Percentage | Range |")
report_lines.append("|----------|-------|------------|-------|")
for cat, data in weight_categories.items():
    pct = (len(data) / len(df)) * 100
    if cat == 'extreme_low':
        range_str = f"< {WEIGHT_EXTREME_LOW} kg (< 10g)"
    elif cat == 'low':
        range_str = f"{WEIGHT_EXTREME_LOW}-{WEIGHT_LOW} kg (10-50g)"
    elif cat == 'normal':
        range_str = f"{WEIGHT_NORMAL_MIN}-{WEIGHT_NORMAL_MAX} kg"
    elif cat == 'high':
        range_str = f"{WEIGHT_HIGH}-{WEIGHT_EXTREME_HIGH} kg"
    else:
        range_str = f"> {WEIGHT_EXTREME_HIGH} kg"
    report_lines.append(f"| {cat.replace('_', ' ').title()} | {len(data):,} | {pct:.2f}% | {range_str} |")
report_lines.append("\n")

# Analyze LOW weight outliers
print("Analyzing LOW weight outliers (< 10 grams):")
low_outliers = weight_categories['extreme_low']
if len(low_outliers) > 0:
    report_lines.append("### Low Weight Outliers (< 10 grams)\n")
    report_lines.append(f"**Count**: {len(low_outliers):,} products\n")
    report_lines.append("\n**Possible Explanations**:\n")
    report_lines.append("1. **Unit Conversion Errors**: Weight might have been recorded in different units (e.g., entered as 100 instead of 0.1 kg)\n")
    report_lines.append("2. **Accessories**: Small items like scarves, ties, or handkerchiefs\n")
    report_lines.append("3. **Decorative Items**: Fashion accessories with minimal weight\n")
    report_lines.append("4. **Data Entry Errors**: Missing decimal points or wrong values\n")
    report_lines.append("\n**Category Breakdown**:\n")

    # Category breakdown for low weights
    low_cats = low_outliers['category'].value_counts().head(10)
    print(f"  Top categories with very low weight:")
    for cat, count in low_cats.items():
        print(f"    * {cat:30s}: {count:4,} products")

    report_lines.append("| Category | Count |")
    report_lines.append("|----------|-------|")
    for cat, count in low_cats.items():
        report_lines.append(f"| {cat} | {count:,} |")
    report_lines.append("\n")

    # Show specific examples
    examples = low_outliers.nsmallest(5, 'weight_kg')[['category', 'parent_category', 'weight_kg']]
    report_lines.append("**Examples (lightest products)**:\n")
    report_lines.append("| Category | Parent | Weight |")
    report_lines.append("|----------|--------|--------|")
    for _, row in examples.iterrows():
        report_lines.append(f"| {row['category']} | {row['parent_category']} | {row['weight_kg']:.6f} kg |")
    report_lines.append("\n")

# Analyze HIGH weight outliers
print("\nAnalyzing HIGH weight outliers (> 5 kg):")
high_outliers = pd.concat([weight_categories['high'], weight_categories['extreme_high']])
if len(high_outliers) > 0:
    report_lines.append("### High Weight Outliers (> 5 kg)\n")
    report_lines.append(f"**Count**: {len(high_outliers):,} products\n")
    report_lines.append("\n**Possible Explanations**:\n")
    report_lines.append("1. **Heavy Outerwear**: Winter coats, parkas with thick insulation\n")
    report_lines.append("2. **Footwear**: Heavy boots, work boots with steel reinforcement\n")
    report_lines.append("3. **Bundled Items**: Products sold in sets or bundles\n")
    report_lines.append("4. **Packaging Weight**: Weight including packaging materials\n")
    report_lines.append("5. **Unit Errors**: Weight might be in grams but labeled as kg (e.g., 734 g entered as 734 kg)\n")
    report_lines.append("\n**Category Breakdown**:\n")

    # Category breakdown for high weights
    high_cats = high_outliers['category'].value_counts().head(10)
    print(f"  Top categories with high weight:")
    for cat, count in high_cats.items():
        print(f"    * {cat:30s}: {count:4,} products")

    report_lines.append("| Category | Count |")
    report_lines.append("|----------|-------|")
    for cat, count in high_cats.items():
        report_lines.append(f"| {cat} | {count:,} |")
    report_lines.append("\n")

    # Show specific examples
    examples = high_outliers.nlargest(5, 'weight_kg')[['category', 'parent_category', 'weight_kg']]
    report_lines.append("**Examples (heaviest products)**:\n")
    report_lines.append("| Category | Parent | Weight |")
    report_lines.append("|----------|--------|--------|")
    for _, row in examples.iterrows():
        report_lines.append(f"| {row['category']} | {row['parent_category']} | {row['weight_kg']:.2f} kg |")
    report_lines.append("\n")

# ============================================================================
# DISTANCE OUTLIER ANALYSIS
# ============================================================================

print("\n" + "=" * 80)
print("2. DISTANCE OUTLIER ANALYSIS")
print("=" * 80)
print()

report_lines.append("## 2. Distance Outliers\n")

# Overall statistics
distance_stats = {
    'min': df['total_distance_km'].min(),
    'max': df['total_distance_km'].max(),
    'mean': df['total_distance_km'].mean(),
    'median': df['total_distance_km'].median(),
    'std': df['total_distance_km'].std(),
    'q1': df['total_distance_km'].quantile(0.25),
    'q3': df['total_distance_km'].quantile(0.75)
}

print(f"Distance Statistics:")
print(f"  Min:    {distance_stats['min']:.2f} km")
print(f"  Q1:     {distance_stats['q1']:.2f} km")
print(f"  Median: {distance_stats['median']:.2f} km")
print(f"  Mean:   {distance_stats['mean']:.2f} km")
print(f"  Q3:     {distance_stats['q3']:.2f} km")
print(f"  Max:    {distance_stats['max']:.2f} km")
print(f"  Std:    {distance_stats['std']:.2f} km\n")

report_lines.append("### Overall Statistics\n")
report_lines.append("| Metric | Value |")
report_lines.append("|--------|-------|")
report_lines.append(f"| Minimum | {distance_stats['min']:.2f} km |")
report_lines.append(f"| 25th Percentile (Q1) | {distance_stats['q1']:.2f} km |")
report_lines.append(f"| Median | {distance_stats['median']:.2f} km |")
report_lines.append(f"| Mean | {distance_stats['mean']:.2f} km |")
report_lines.append(f"| 75th Percentile (Q3) | {distance_stats['q3']:.2f} km |")
report_lines.append(f"| Maximum | {distance_stats['max']:.2f} km |")
report_lines.append(f"| Standard Deviation | {distance_stats['std']:.2f} km |")
report_lines.append("\n")

# Define distance thresholds
DISTANCE_EXTREME_LOW = 1        # < 1 km
DISTANCE_LOW = 100              # < 100 km
DISTANCE_SHORT = 1000           # 100-1000 km
DISTANCE_MEDIUM = 5000          # 1000-5000 km
DISTANCE_LONG = 15000           # 5000-15000 km
DISTANCE_VERY_LONG = 25000      # 15000-25000 km
DISTANCE_EXTREME = 40000        # > 40000 km (Earth's circumference)

# Categorize distances
distance_categories = {
    'extreme_low': df[df['total_distance_km'] < DISTANCE_EXTREME_LOW],
    'local': df[(df['total_distance_km'] >= DISTANCE_EXTREME_LOW) & (df['total_distance_km'] < DISTANCE_LOW)],
    'regional': df[(df['total_distance_km'] >= DISTANCE_LOW) & (df['total_distance_km'] < DISTANCE_SHORT)],
    'continental': df[(df['total_distance_km'] >= DISTANCE_SHORT) & (df['total_distance_km'] < DISTANCE_MEDIUM)],
    'intercontinental': df[(df['total_distance_km'] >= DISTANCE_MEDIUM) & (df['total_distance_km'] < DISTANCE_LONG)],
    'trans_pacific': df[(df['total_distance_km'] >= DISTANCE_LONG) & (df['total_distance_km'] < DISTANCE_VERY_LONG)],
    'around_world': df[(df['total_distance_km'] >= DISTANCE_VERY_LONG) & (df['total_distance_km'] < DISTANCE_EXTREME)],
    'impossible': df[df['total_distance_km'] >= DISTANCE_EXTREME]
}

print("Distance Distribution by Category:")
for cat, data in distance_categories.items():
    pct = (len(data) / len(df)) * 100
    print(f"  {cat:20s}: {len(data):7,} products ({pct:5.2f}%)")
print()

report_lines.append("### Distance Distribution by Category\n")
report_lines.append("| Category | Count | Percentage | Range | Interpretation |")
report_lines.append("|----------|-------|------------|-------|----------------|")

interpretations = {
    'extreme_low': 'Too short for international supply chain',
    'local': 'Local/domestic production',
    'regional': 'Regional supply chain',
    'continental': 'Cross-continental shipping',
    'intercontinental': 'Typical international supply chain',
    'trans_pacific': 'Trans-Pacific shipping (China -> US/EU)',
    'around_world': 'Near Earth\'s circumference',
    'impossible': 'Impossible (> Earth\'s circumference!)'
}

for cat, data in distance_categories.items():
    pct = (len(data) / len(df)) * 100
    if cat == 'extreme_low':
        range_str = f"< {DISTANCE_EXTREME_LOW} km"
    elif cat == 'local':
        range_str = f"{DISTANCE_EXTREME_LOW}-{DISTANCE_LOW} km"
    elif cat == 'regional':
        range_str = f"{DISTANCE_LOW}-{DISTANCE_SHORT} km"
    elif cat == 'continental':
        range_str = f"{DISTANCE_SHORT:,}-{DISTANCE_MEDIUM:,} km"
    elif cat == 'intercontinental':
        range_str = f"{DISTANCE_MEDIUM:,}-{DISTANCE_LONG:,} km"
    elif cat == 'trans_pacific':
        range_str = f"{DISTANCE_LONG:,}-{DISTANCE_VERY_LONG:,} km"
    elif cat == 'around_world':
        range_str = f"{DISTANCE_VERY_LONG:,}-{DISTANCE_EXTREME:,} km"
    else:
        range_str = f"> {DISTANCE_EXTREME:,} km"

    report_lines.append(f"| {cat.replace('_', ' ').title()} | {len(data):,} | {pct:.2f}% | {range_str} | {interpretations[cat]} |")
report_lines.append("\n")

# Analyze IMPOSSIBLE distances (> 40,000 km)
print("Analyzing IMPOSSIBLE distances (> 40,000 km):")
impossible = distance_categories['impossible']
if len(impossible) > 0:
    report_lines.append("### Impossible Distances (> 40,000 km)\n")
    report_lines.append(f"**Count**: {len(impossible):,} products ({len(impossible)/len(df)*100:.2f}%)\n")
    report_lines.append("\n**Why This is Impossible**:\n")
    report_lines.append(f"- Earth's circumference: ~40,075 km\n")
    report_lines.append(f"- Maximum possible distance: ~20,038 km (half circumference, antipodal points)\n")
    report_lines.append(f"- These products claim distances of {impossible['total_distance_km'].min():.0f} to {impossible['total_distance_km'].max():.0f} km\n")
    report_lines.append("\n**Possible Explanations**:\n")
    report_lines.append("1. **Cumulative Supply Chain**: Sum of all legs (raw materials -> factory -> warehouse -> retail) instead of direct distance\n")
    report_lines.append("2. **Round-Trip Calculation**: Distance was doubled or tripled (including return trips)\n")
    report_lines.append("3. **Multiple Sourcing**: Materials from multiple distant locations summed together\n")
    report_lines.append("4. **Calculation Error**: Wrong formula or unit conversion in distance calculation\n")
    report_lines.append("5. **Data Generation Artifact**: Algorithm error during synthetic data generation\n")
    report_lines.append("\n")

    # Country distribution for impossible distances
    impossible_countries = impossible['manufacturer_country'].value_counts().head(10)
    report_lines.append("**Top Countries with Impossible Distances**:\n")
    report_lines.append("| Country | Products | Avg Distance |")
    report_lines.append("|---------|----------|--------------|")
    print(f"  Top countries with impossible distances:")
    for country, count in impossible_countries.items():
        avg_dist = impossible[impossible['manufacturer_country'] == country]['total_distance_km'].mean()
        print(f"    * {country:3s}: {count:6,} products (avg: {avg_dist:,.0f} km)")
        report_lines.append(f"| {country} | {count:,} | {avg_dist:,.0f} km |")
    report_lines.append("\n")

    # Category distribution
    impossible_cats = impossible['category'].value_counts().head(10)
    report_lines.append("**Top Categories with Impossible Distances**:\n")
    report_lines.append("| Category | Count |")
    report_lines.append("|----------|-------|")
    for cat, count in impossible_cats.items():
        report_lines.append(f"| {cat} | {count:,} |")
    report_lines.append("\n")

# Analyze LOW distances (< 1 km)
print("\nAnalyzing LOW distances (< 1 km):")
low_dist = distance_categories['extreme_low']
if len(low_dist) > 0:
    report_lines.append("### Very Low Distances (< 1 km)\n")
    report_lines.append(f"**Count**: {len(low_dist):,} products\n")
    report_lines.append("\n**Possible Explanations**:\n")
    report_lines.append("1. **Unit Conversion**: Distance might be in meters but labeled as km (e.g., 500m entered as 500 km)\n")
    report_lines.append("2. **Local Production**: Truly local/domestic manufacturing\n")
    report_lines.append("3. **Placeholder Values**: Test or default values\n")
    report_lines.append("4. **Partial Distance**: Only one leg of supply chain recorded\n")
    report_lines.append("\n")

    # Show examples
    examples = low_dist.nsmallest(10, 'total_distance_km')[['category', 'manufacturer_country', 'total_distance_km']]
    report_lines.append("**Examples (shortest distances)**:\n")
    report_lines.append("| Category | Country | Distance |")
    report_lines.append("|----------|---------|----------|")
    for _, row in examples.iterrows():
        report_lines.append(f"| {row['category']} | {row['manufacturer_country']} | {row['total_distance_km']:.2f} km |")
    report_lines.append("\n")

# ============================================================================
# RECOMMENDATIONS
# ============================================================================

report_lines.append("## Recommendations\n")
report_lines.append("### For Weight Outliers:\n")
report_lines.append("1. **Review Low Weights**: Check products < 10g for unit conversion errors\n")
report_lines.append("2. **Review High Weights**: Check products > 10 kg for data entry errors\n")
report_lines.append("3. **Keep Moderate Outliers**: Products between 0.01-10 kg are likely valid\n")
report_lines.append("\n### For Distance Outliers:\n")
report_lines.append("1. **Understand the Metric**: Clarify if distance is direct, cumulative, or round-trip\n")
report_lines.append("2. **Cap Impossible Values**: Consider capping at 40,000 km (Earth's circumference)\n")
report_lines.append("3. **Investigate Patterns**: The systematic nature suggests calculation formula issue\n")
report_lines.append("4. **Keep if Intentional**: If cumulative supply chain is the intended metric, document it\n")
report_lines.append("\n")

print("\nSaving Saving explanation report...")
with open(REPORT_FILE, 'w') as f:
    f.write('\n'.join(report_lines))
print(f"OK Saved: {REPORT_FILE}\n")

print("=" * 80)
print("OK OUTLIER EXPLANATION COMPLETE")
print("=" * 80)
print(f"\nDetailed explanation: {REPORT_FILE}")
