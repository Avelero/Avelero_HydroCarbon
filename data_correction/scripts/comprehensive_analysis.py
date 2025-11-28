#!/usr/bin/env python3
"""
Comprehensive Dataset Analysis - Shows EVERY option (no "top N" limiting)
This allows for better mistake detection by showing all categories, materials, and countries.
"""

import os
import sys
import json
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from datetime import datetime
from pathlib import Path

# Configuration
INPUT_CSV = "data_correction/output/Product_data_complete.csv"
OUTPUT_DIR = "data_correction/output/comprehensive_analysis"
REPORT_FILE = "data_correction/output/comprehensive_analysis/full_analysis_report.json"
DPI = 300

# Style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = DPI
plt.rcParams['savefig.dpi'] = DPI

print("=" * 80)
print("COMPREHENSIVE DATASET ANALYSIS - ALL OPTIONS DISPLAYED")
print("=" * 80)
print()

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load data
print(f"📂 Loading data from: {INPUT_CSV}")
if not os.path.exists(INPUT_CSV):
    print(f"❌ Error: File not found: {INPUT_CSV}")
    sys.exit(1)

df = pd.read_csv(INPUT_CSV, low_memory=False)
print(f"✓ Loaded {len(df):,} rows")
print()

# Convert numeric columns
print("🔧 Converting numeric columns...")
df['weight_kg'] = pd.to_numeric(df['weight_kg'], errors='coerce')
df['total_distance_km'] = pd.to_numeric(df['total_distance_km'], errors='coerce')

# Report on any missing values
missing_weight = df['weight_kg'].isna().sum()
missing_distance = df['total_distance_km'].isna().sum()
if missing_weight > 0 or missing_distance > 0:
    print(f"⚠️  Found missing values:")
    print(f"   Weight: {missing_weight:,} missing")
    print(f"   Distance: {missing_distance:,} missing")
    df = df.dropna(subset=['weight_kg', 'total_distance_km'])
    print(f"✓ After removing rows with missing numeric values: {len(df):,} rows")
print()

# ============================================================================
# COMPREHENSIVE STATISTICAL ANALYSIS
# ============================================================================

print("=" * 80)
print("COMPREHENSIVE STATISTICAL ANALYSIS")
print("=" * 80)
print()

# Initialize report
report = {
    "timestamp": datetime.now().isoformat(),
    "total_products": int(len(df)),
    "analysis_type": "comprehensive_no_limits"
}

# ============================================================================
# 1. CATEGORY ANALYSIS - ALL CATEGORIES
# ============================================================================

print("1. CATEGORY ANALYSIS (All Categories)")
print("-" * 80)

category_counts = df['category'].value_counts().sort_values(ascending=False)
report['total_categories'] = int(len(category_counts))
report['all_categories'] = category_counts.to_dict()

print(f"Total unique categories: {len(category_counts)}")
print(f"Category range: {category_counts.min():,} to {category_counts.max():,} products")
print(f"Mean products per category: {category_counts.mean():.1f}")
print(f"Median products per category: {category_counts.median():.1f}")
print()

# Display ALL categories
print("ALL CATEGORIES (sorted by count):")
for i, (cat, count) in enumerate(category_counts.items(), 1):
    pct = (count / len(df)) * 100
    print(f"  {i:3d}. {cat:40s}: {count:8,} ({pct:5.2f}%)")
print()

# Visualize ALL categories
print("Creating visualization: all_categories.png")
fig, ax = plt.subplots(figsize=(20, max(12, len(category_counts) * 0.3)))
category_counts_sorted = category_counts.sort_values(ascending=True)
colors = plt.cm.viridis(np.linspace(0, 1, len(category_counts_sorted)))
category_counts_sorted.plot(kind='barh', ax=ax, color=colors, edgecolor='black', linewidth=0.5)
ax.set_xlabel('Product Count', fontsize=14, fontweight='bold')
ax.set_ylabel('Category', fontsize=14, fontweight='bold')
ax.set_title(f'ALL Categories Distribution ({len(category_counts)} categories)',
             fontsize=16, fontweight='bold')
ax.grid(axis='x', linestyle='--', alpha=0.3)

# Add counts on bars
for i, v in enumerate(category_counts_sorted):
    ax.text(v + (category_counts_sorted.max() * 0.01), i, f'{v:,}',
            va='center', fontsize=8)

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/01_all_categories.png", dpi=DPI, bbox_inches='tight')
plt.close()
print("✓ Saved\n")

# ============================================================================
# 2. PARENT CATEGORY ANALYSIS - ALL PARENT CATEGORIES
# ============================================================================

print("2. PARENT CATEGORY ANALYSIS (All Parent Categories)")
print("-" * 80)

parent_counts = df['parent_category'].value_counts().sort_values(ascending=False)
report['total_parent_categories'] = int(len(parent_counts))
report['all_parent_categories'] = parent_counts.to_dict()

print(f"Total unique parent categories: {len(parent_counts)}")
print()
print("ALL PARENT CATEGORIES:")
for i, (parent, count) in enumerate(parent_counts.items(), 1):
    pct = (count / len(df)) * 100
    print(f"  {i}. {parent:30s}: {count:8,} ({pct:5.2f}%)")
print()

# Visualize
print("Creating visualization: all_parent_categories.png")
fig, ax = plt.subplots(figsize=(14, 8))
parent_counts.plot(kind='bar', ax=ax, color='coral', edgecolor='black', linewidth=1.5)
ax.set_xlabel('Parent Category', fontsize=12, fontweight='bold')
ax.set_ylabel('Product Count', fontsize=12, fontweight='bold')
ax.set_title(f'ALL Parent Categories Distribution ({len(parent_counts)} parent categories)',
             fontsize=14, fontweight='bold')
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
ax.grid(axis='y', linestyle='--', alpha=0.3)

for i, v in enumerate(parent_counts):
    ax.text(i, v + (parent_counts.max() * 0.01), f'{v:,}',
            ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/02_all_parent_categories.png", dpi=DPI, bbox_inches='tight')
plt.close()
print("✓ Saved\n")

# ============================================================================
# 3. GENDER ANALYSIS - ALL GENDERS
# ============================================================================

print("3. GENDER ANALYSIS (All Genders)")
print("-" * 80)

gender_counts = df['gender'].value_counts().sort_values(ascending=False)
report['all_genders'] = gender_counts.to_dict()

print("ALL GENDERS:")
for gender, count in gender_counts.items():
    pct = (count / len(df)) * 100
    print(f"  {gender:10s}: {count:8,} ({pct:5.2f}%)")
print()

# Visualize
print("Creating visualization: all_genders.png")
fig, ax = plt.subplots(figsize=(10, 6))
colors = ['#FF69B4', '#4169E1', '#32CD32'][:len(gender_counts)]
gender_counts.plot(kind='bar', ax=ax, color=colors, edgecolor='black', linewidth=2)
ax.set_xlabel('Gender', fontsize=12, fontweight='bold')
ax.set_ylabel('Product Count', fontsize=12, fontweight='bold')
ax.set_title('ALL Gender Distribution', fontsize=14, fontweight='bold')
ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
ax.grid(axis='y', linestyle='--', alpha=0.3)

for i, v in enumerate(gender_counts):
    ax.text(i, v + (gender_counts.max() * 0.01), f'{v:,}',
            ha='center', va='bottom', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/03_all_genders.png", dpi=DPI, bbox_inches='tight')
plt.close()
print("✓ Saved\n")

# ============================================================================
# 4. COUNTRY ANALYSIS - ALL COUNTRIES
# ============================================================================

print("4. MANUFACTURING COUNTRY ANALYSIS (All Countries)")
print("-" * 80)

country_counts = df['manufacturer_country'].value_counts().sort_values(ascending=False)
report['total_countries'] = int(len(country_counts))
report['all_countries'] = country_counts.to_dict()

print(f"Total unique countries: {len(country_counts)}")
print(f"Country range: {country_counts.min():,} to {country_counts.max():,} products")
print()

print("ALL COUNTRIES (sorted by count):")
for i, (country, count) in enumerate(country_counts.items(), 1):
    pct = (count / len(df)) * 100
    print(f"  {i:3d}. {country:3s}: {count:8,} ({pct:5.2f}%)")
print()

# Visualize ALL countries
print("Creating visualization: all_countries.png")
fig, ax = plt.subplots(figsize=(18, max(12, len(country_counts) * 0.25)))
country_counts_sorted = country_counts.sort_values(ascending=True)
colors = plt.cm.plasma(np.linspace(0, 1, len(country_counts_sorted)))
country_counts_sorted.plot(kind='barh', ax=ax, color=colors, edgecolor='black', linewidth=0.5)
ax.set_xlabel('Product Count', fontsize=14, fontweight='bold')
ax.set_ylabel('Country Code', fontsize=14, fontweight='bold')
ax.set_title(f'ALL Manufacturing Countries Distribution ({len(country_counts)} countries)',
             fontsize=16, fontweight='bold')
ax.grid(axis='x', linestyle='--', alpha=0.3)

# Add counts on bars
for i, v in enumerate(country_counts_sorted):
    ax.text(v + (country_counts_sorted.max() * 0.01), i, f'{v:,}',
            va='center', fontsize=7)

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/04_all_countries.png", dpi=DPI, bbox_inches='tight')
plt.close()
print("✓ Saved\n")

# ============================================================================
# 5. MATERIAL ANALYSIS - ALL MATERIALS
# ============================================================================

print("5. MATERIAL ANALYSIS (All Materials)")
print("-" * 80)

print("Parsing all materials from dataset...")
material_counts = Counter()
materials_parsed = 0
materials_failed = 0

for mat_json in df['materials']:
    try:
        # Handle both JSON and string representations
        if isinstance(mat_json, str):
            # Try to parse as JSON
            try:
                materials = json.loads(mat_json)
            except json.JSONDecodeError:
                # Try replacing single quotes with double quotes
                materials = json.loads(mat_json.replace("'", '"'))
        else:
            materials = mat_json

        # Count materials
        if isinstance(materials, dict):
            for mat in materials.keys():
                material_counts[mat] += 1
            materials_parsed += 1
        elif isinstance(materials, list):
            for mat in materials:
                material_counts[mat] += 1
            materials_parsed += 1
    except Exception as e:
        materials_failed += 1

print(f"Successfully parsed: {materials_parsed:,} rows")
print(f"Failed to parse: {materials_failed:,} rows")
print()

# Convert to sorted series
material_series = pd.Series(dict(material_counts)).sort_values(ascending=False)
report['total_materials'] = int(len(material_series))
report['all_materials'] = material_series.to_dict()

print(f"Total unique materials: {len(material_series)}")
print(f"Material usage range: {material_series.min():,} to {material_series.max():,} products")
print()

print("ALL MATERIALS (sorted by frequency):")
for i, (mat, count) in enumerate(material_series.items(), 1):
    pct = (count / materials_parsed) * 100
    print(f"  {i:3d}. {mat:40s}: {count:8,} ({pct:5.2f}%)")
print()

# Visualize ALL materials
print("Creating visualization: all_materials.png")
fig, ax = plt.subplots(figsize=(18, max(14, len(material_series) * 0.3)))
material_series_sorted = material_series.sort_values(ascending=True)
colors = plt.cm.coolwarm(np.linspace(0, 1, len(material_series_sorted)))
material_series_sorted.plot(kind='barh', ax=ax, color=colors, edgecolor='black', linewidth=0.5)
ax.set_xlabel('Usage Count', fontsize=14, fontweight='bold')
ax.set_ylabel('Material', fontsize=14, fontweight='bold')
ax.set_title(f'ALL Materials Distribution ({len(material_series)} materials)',
             fontsize=16, fontweight='bold')
ax.grid(axis='x', linestyle='--', alpha=0.3)

# Add counts on bars
for i, v in enumerate(material_series_sorted):
    ax.text(v + (material_series_sorted.max() * 0.01), i, f'{v:,}',
            va='center', fontsize=7)

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/05_all_materials.png", dpi=DPI, bbox_inches='tight')
plt.close()
print("✓ Saved\n")

# ============================================================================
# 6. NUMERICAL DISTRIBUTIONS
# ============================================================================

print("6. NUMERICAL DISTRIBUTIONS")
print("-" * 80)

# Weight statistics
weight_stats = {
    "min": float(df['weight_kg'].min()),
    "max": float(df['weight_kg'].max()),
    "mean": float(df['weight_kg'].mean()),
    "median": float(df['weight_kg'].median()),
    "std": float(df['weight_kg'].std())
}
report['weight_statistics'] = weight_stats

print("Weight (kg) Statistics:")
print(f"  Min:    {weight_stats['min']:.4f} kg")
print(f"  Max:    {weight_stats['max']:.4f} kg")
print(f"  Mean:   {weight_stats['mean']:.4f} kg")
print(f"  Median: {weight_stats['median']:.4f} kg")
print(f"  Std:    {weight_stats['std']:.4f} kg")
print()

# Distance statistics
distance_stats = {
    "min": float(df['total_distance_km'].min()),
    "max": float(df['total_distance_km'].max()),
    "mean": float(df['total_distance_km'].mean()),
    "median": float(df['total_distance_km'].median()),
    "std": float(df['total_distance_km'].std())
}
report['distance_statistics'] = distance_stats

print("Distance (km) Statistics:")
print(f"  Min:    {distance_stats['min']:.2f} km")
print(f"  Max:    {distance_stats['max']:.2f} km")
print(f"  Mean:   {distance_stats['mean']:.2f} km")
print(f"  Median: {distance_stats['median']:.2f} km")
print(f"  Std:    {distance_stats['std']:.2f} km")
print()

# Visualize distributions
print("Creating visualization: numerical_distributions.png")
fig, axes = plt.subplots(2, 2, figsize=(18, 12))

# Weight histogram
axes[0, 0].hist(df['weight_kg'], bins=100, color='skyblue', edgecolor='black', alpha=0.7)
axes[0, 0].axvline(weight_stats['mean'], color='red', linestyle='--', linewidth=2,
                   label=f"Mean: {weight_stats['mean']:.4f} kg")
axes[0, 0].axvline(weight_stats['median'], color='green', linestyle='--', linewidth=2,
                   label=f"Median: {weight_stats['median']:.4f} kg")
axes[0, 0].set_xlabel('Weight (kg)', fontsize=12, fontweight='bold')
axes[0, 0].set_ylabel('Frequency', fontsize=12, fontweight='bold')
axes[0, 0].set_title('Weight Distribution', fontsize=14, fontweight='bold')
axes[0, 0].legend()
axes[0, 0].grid(axis='y', linestyle='--', alpha=0.3)

# Weight box plot
axes[0, 1].boxplot(df['weight_kg'], vert=True)
axes[0, 1].set_ylabel('Weight (kg)', fontsize=12, fontweight='bold')
axes[0, 1].set_title('Weight Box Plot', fontsize=14, fontweight='bold')
axes[0, 1].grid(axis='y', linestyle='--', alpha=0.3)

# Distance histogram
axes[1, 0].hist(df['total_distance_km'], bins=100, color='orange', edgecolor='black', alpha=0.7)
axes[1, 0].axvline(distance_stats['mean'], color='red', linestyle='--', linewidth=2,
                   label=f"Mean: {distance_stats['mean']:.0f} km")
axes[1, 0].axvline(distance_stats['median'], color='green', linestyle='--', linewidth=2,
                   label=f"Median: {distance_stats['median']:.0f} km")
axes[1, 0].set_xlabel('Distance (km)', fontsize=12, fontweight='bold')
axes[1, 0].set_ylabel('Frequency', fontsize=12, fontweight='bold')
axes[1, 0].set_title('Distance Distribution', fontsize=14, fontweight='bold')
axes[1, 0].legend()
axes[1, 0].grid(axis='y', linestyle='--', alpha=0.3)

# Distance box plot
axes[1, 1].boxplot(df['total_distance_km'], vert=True)
axes[1, 1].set_ylabel('Distance (km)', fontsize=12, fontweight='bold')
axes[1, 1].set_title('Distance Box Plot', fontsize=14, fontweight='bold')
axes[1, 1].grid(axis='y', linestyle='--', alpha=0.3)

plt.suptitle('Numerical Feature Distributions', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/06_numerical_distributions.png", dpi=DPI, bbox_inches='tight')
plt.close()
print("✓ Saved\n")

# ============================================================================
# 7. CATEGORY HIERARCHY ANALYSIS
# ============================================================================

print("7. CATEGORY HIERARCHY (Parent -> Category)")
print("-" * 80)

hierarchy_counts = df.groupby(['parent_category', 'category']).size().reset_index(name='count')
hierarchy_counts = hierarchy_counts.sort_values(['parent_category', 'count'], ascending=[True, False])
report['hierarchy'] = hierarchy_counts.to_dict('records')

print("Full Hierarchy:")
current_parent = None
for _, row in hierarchy_counts.iterrows():
    if row['parent_category'] != current_parent:
        current_parent = row['parent_category']
        print(f"\n{current_parent}:")
    pct = (row['count'] / len(df)) * 100
    print(f"  ├─ {row['category']:40s}: {row['count']:8,} ({pct:5.2f}%)")
print()

# Visualize hierarchy
print("Creating visualization: category_hierarchy.png")
fig, ax = plt.subplots(figsize=(20, max(14, len(hierarchy_counts) * 0.25)))

parents = hierarchy_counts['parent_category'].unique()
colors = plt.cm.Set3(np.linspace(0, 1, len(parents)))
parent_color_map = dict(zip(parents, colors))

x_pos = 0
for parent in parents:
    parent_data = hierarchy_counts[hierarchy_counts['parent_category'] == parent]
    total_width = len(parent_data)

    # Plot bars
    bars = ax.bar(range(x_pos, x_pos + total_width),
                  parent_data['count'],
                  color=parent_color_map[parent],
                  alpha=0.8,
                  edgecolor='black',
                  linewidth=0.5,
                  width=1.0)

    # Label parent
    center = x_pos + total_width / 2
    ax.text(center, parent_data['count'].max() * 1.02, parent,
            ha='center', va='bottom', fontweight='bold', fontsize=11,
            bbox=dict(boxstyle='round', facecolor=parent_color_map[parent], alpha=0.5))

    # Add category labels
    for i, (idx, row) in enumerate(parent_data.iterrows()):
        ax.text(x_pos + i, row['count'] / 2, row['category'],
                rotation=90, ha='center', va='center', fontsize=7, fontweight='bold')

    x_pos += total_width + 1

ax.set_title('Complete Category Hierarchy', fontsize=16, fontweight='bold')
ax.set_ylabel('Product Count', fontsize=12, fontweight='bold')
ax.set_xticks([])
ax.grid(axis='y', linestyle='--', alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/07_category_hierarchy.png", dpi=DPI, bbox_inches='tight')
plt.close()
print("✓ Saved\n")

# ============================================================================
# 8. SAVE JSON REPORT
# ============================================================================

print("Saving comprehensive analysis report...")
with open(REPORT_FILE, 'w') as f:
    json.dump(report, f, indent=2)
print(f"✓ Saved: {REPORT_FILE}\n")

# ============================================================================
# SUMMARY
# ============================================================================

print("=" * 80)
print("✓ COMPREHENSIVE ANALYSIS COMPLETE")
print("=" * 80)
print()
print(f"📊 Analyzed {len(df):,} products")
print(f"📁 Output directory: {OUTPUT_DIR}/")
print()
print("Generated files:")
print(f"  1. 01_all_categories.png          - All {len(category_counts)} categories")
print(f"  2. 02_all_parent_categories.png   - All {len(parent_counts)} parent categories")
print(f"  3. 03_all_genders.png              - All {len(gender_counts)} genders")
print(f"  4. 04_all_countries.png            - All {len(country_counts)} countries")
print(f"  5. 05_all_materials.png            - All {len(material_series)} materials")
print(f"  6. 06_numerical_distributions.png  - Weight & distance distributions")
print(f"  7. 07_category_hierarchy.png       - Complete parent->category hierarchy")
print(f"  8. full_analysis_report.json       - Complete analysis data")
print()
print("=" * 80)
print("NOTE: This analysis shows EVERY option, not just 'top N'")
print("This allows for better mistake detection across the entire dataset")
print("=" * 80)
