#!/usr/bin/env python3
"""
Combine extra products with cleaned dataset and analyze with visualizations
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)

# Paths
CLEANED_DATA = Path("data_correction/output/Product_data_cleaned.csv")
EXTRA_DATA = Path("data_correction/extra_generation/extra_products.csv")
OUTPUT_DATA = Path("data_correction/output/Product_data_complete.csv")
PLOTS_DIR = Path("data_correction/output/analysis_plots")

# Create plots directory
PLOTS_DIR.mkdir(exist_ok=True, parents=True)

print("=" * 80)
print("COMBINING DATASETS AND ANALYZING")
print("=" * 80)

# 1. Load datasets
print(f"\n📂 Loading cleaned dataset: {CLEANED_DATA}")
df_cleaned = pd.read_csv(CLEANED_DATA, low_memory=False)
print(f"✓ Loaded {len(df_cleaned):,} rows")

print(f"\n📂 Loading extra products: {EXTRA_DATA}")
df_extra = pd.read_csv(EXTRA_DATA, low_memory=False)
print(f"✓ Loaded {len(df_extra):,} rows")

# 2. Combine datasets
print(f"\n🔗 Combining datasets...")
df_complete = pd.concat([df_cleaned, df_extra], ignore_index=True)
print(f"✓ Combined dataset: {len(df_complete):,} total rows")

# Clean numeric columns
print(f"\n🧹 Cleaning numeric columns...")
df_complete['weight_kg'] = pd.to_numeric(df_complete['weight_kg'], errors='coerce')
df_complete['total_distance_km'] = pd.to_numeric(df_complete['total_distance_km'], errors='coerce')
df_complete = df_complete.dropna(subset=['weight_kg', 'total_distance_km'])
print(f"✓ After cleaning: {len(df_complete):,} rows")

# 3. Save combined dataset
print(f"\n💾 Saving combined dataset to: {OUTPUT_DATA}")
df_complete.to_csv(OUTPUT_DATA, index=False, quoting=1, escapechar='\\')
print(f"✓ Saved!")

# ============================================================================
# ANALYSIS & VISUALIZATIONS
# ============================================================================

print(f"\n📊 Creating analysis plots...")

# ----------------------------------------------------------------------------
# 1. Category Distribution
# ----------------------------------------------------------------------------
print("\n1. Category distribution...")
fig, ax = plt.subplots(figsize=(16, 10))
category_counts = df_complete['category'].value_counts().sort_values(ascending=True)
category_counts.plot(kind='barh', ax=ax, color='steelblue', edgecolor='black')
ax.set_xlabel('Count', fontsize=12, fontweight='bold')
ax.set_ylabel('Category', fontsize=12, fontweight='bold')
ax.set_title('Product Distribution by Category (Complete Dataset)', fontsize=14, fontweight='bold')
ax.axvline(x=25000, color='red', linestyle='--', linewidth=2, label='Target: 25,000')
ax.legend()
for i, v in enumerate(category_counts):
    ax.text(v + 500, i, f'{v:,}', va='center', fontsize=9)
plt.tight_layout()
plt.savefig(PLOTS_DIR / "01_category_distribution.png", dpi=300, bbox_inches='tight')
plt.close()
print(f"   ✓ Saved: {PLOTS_DIR / '01_category_distribution.png'}")

# ----------------------------------------------------------------------------
# 2. Dress Categories Focus
# ----------------------------------------------------------------------------
print("\n2. Dress categories comparison (before/after)...")
dress_categories = ['Gowns', 'Maxi', 'Midi', 'Mini', 'Maxi Dresses', 'Midi Dresses', 'Mini Dresses']

# Get counts from both datasets
cleaned_counts = df_cleaned[df_cleaned['category'].isin(dress_categories)]['category'].value_counts()
complete_counts = df_complete[df_complete['category'].isin(dress_categories)]['category'].value_counts()

# Combine for comparison
dress_df = pd.DataFrame({
    'Before': cleaned_counts,
    'After': complete_counts
}).fillna(0)

fig, ax = plt.subplots(figsize=(12, 6))
dress_df.plot(kind='bar', ax=ax, color=['lightcoral', 'lightgreen'], edgecolor='black')
ax.set_xlabel('Category', fontsize=12, fontweight='bold')
ax.set_ylabel('Count', fontsize=12, fontweight='bold')
ax.set_title('Dress Categories: Before vs After Extra Generation', fontsize=14, fontweight='bold')
ax.axhline(y=25000, color='red', linestyle='--', linewidth=2, label='Target: 25,000')
ax.legend()
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
for container in ax.containers:
    ax.bar_label(container, fmt='%,.0f', fontsize=9)
plt.tight_layout()
plt.savefig(PLOTS_DIR / "02_dress_categories_before_after.png", dpi=300, bbox_inches='tight')
plt.close()
print(f"   ✓ Saved: {PLOTS_DIR / '02_dress_categories_before_after.png'}")

# ----------------------------------------------------------------------------
# 3. Gender Distribution
# ----------------------------------------------------------------------------
print("\n3. Gender distribution...")
fig, ax = plt.subplots(figsize=(10, 6))
gender_counts = df_complete['gender'].value_counts()
colors = ['#FF69B4', '#4169E1']
gender_counts.plot(kind='bar', ax=ax, color=colors, edgecolor='black')
ax.set_xlabel('Gender', fontsize=12, fontweight='bold')
ax.set_ylabel('Count', fontsize=12, fontweight='bold')
ax.set_title('Product Distribution by Gender', fontsize=14, fontweight='bold')
ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
for i, v in enumerate(gender_counts):
    ax.text(i, v + 5000, f'{v:,}', ha='center', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig(PLOTS_DIR / "03_gender_distribution.png", dpi=300, bbox_inches='tight')
plt.close()
print(f"   ✓ Saved: {PLOTS_DIR / '03_gender_distribution.png'}")

# ----------------------------------------------------------------------------
# 4. Parent Category Distribution
# ----------------------------------------------------------------------------
print("\n4. Parent category distribution...")
fig, ax = plt.subplots(figsize=(12, 6))
parent_counts = df_complete['parent_category'].value_counts()
parent_counts.plot(kind='bar', ax=ax, color='coral', edgecolor='black')
ax.set_xlabel('Parent Category', fontsize=12, fontweight='bold')
ax.set_ylabel('Count', fontsize=12, fontweight='bold')
ax.set_title('Product Distribution by Parent Category', fontsize=14, fontweight='bold')
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
for i, v in enumerate(parent_counts):
    ax.text(i, v + 5000, f'{v:,}', ha='center', fontsize=10)
plt.tight_layout()
plt.savefig(PLOTS_DIR / "04_parent_category_distribution.png", dpi=300, bbox_inches='tight')
plt.close()
print(f"   ✓ Saved: {PLOTS_DIR / '04_parent_category_distribution.png'}")

# ----------------------------------------------------------------------------
# 5. Manufacturing Country Distribution (Top 20)
# ----------------------------------------------------------------------------
print("\n5. Top 20 manufacturing countries...")
fig, ax = plt.subplots(figsize=(12, 8))
country_counts = df_complete['manufacturer_country'].value_counts().head(20)
country_counts.plot(kind='barh', ax=ax, color='teal', edgecolor='black')
ax.set_xlabel('Count', fontsize=12, fontweight='bold')
ax.set_ylabel('Country Code', fontsize=12, fontweight='bold')
ax.set_title('Top 20 Manufacturing Countries', fontsize=14, fontweight='bold')
for i, v in enumerate(country_counts):
    ax.text(v + 500, i, f'{v:,}', va='center', fontsize=9)
plt.tight_layout()
plt.savefig(PLOTS_DIR / "05_top20_countries.png", dpi=300, bbox_inches='tight')
plt.close()
print(f"   ✓ Saved: {PLOTS_DIR / '05_top20_countries.png'}")

# ----------------------------------------------------------------------------
# 6. Weight Distribution
# ----------------------------------------------------------------------------
print("\n6. Product weight distribution...")
fig, ax = plt.subplots(figsize=(12, 6))
ax.hist(df_complete['weight_kg'], bins=50, color='skyblue', edgecolor='black', alpha=0.7)
ax.set_xlabel('Weight (kg)', fontsize=12, fontweight='bold')
ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
ax.set_title('Product Weight Distribution', fontsize=14, fontweight='bold')
ax.axvline(df_complete['weight_kg'].mean(), color='red', linestyle='--', linewidth=2,
           label=f'Mean: {df_complete["weight_kg"].mean():.3f} kg')
ax.axvline(df_complete['weight_kg'].median(), color='green', linestyle='--', linewidth=2,
           label=f'Median: {df_complete["weight_kg"].median():.3f} kg')
ax.legend()
plt.tight_layout()
plt.savefig(PLOTS_DIR / "06_weight_distribution.png", dpi=300, bbox_inches='tight')
plt.close()
print(f"   ✓ Saved: {PLOTS_DIR / '06_weight_distribution.png'}")

# ----------------------------------------------------------------------------
# 7. Distance Distribution
# ----------------------------------------------------------------------------
print("\n7. Supply chain distance distribution...")
fig, ax = plt.subplots(figsize=(12, 6))
ax.hist(df_complete['total_distance_km'], bins=50, color='orange', edgecolor='black', alpha=0.7)
ax.set_xlabel('Distance (km)', fontsize=12, fontweight='bold')
ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
ax.set_title('Supply Chain Distance Distribution', fontsize=14, fontweight='bold')
ax.axvline(df_complete['total_distance_km'].mean(), color='red', linestyle='--', linewidth=2,
           label=f'Mean: {df_complete["total_distance_km"].mean():.0f} km')
ax.axvline(df_complete['total_distance_km'].median(), color='green', linestyle='--', linewidth=2,
           label=f'Median: {df_complete["total_distance_km"].median():.0f} km')
ax.legend()
plt.tight_layout()
plt.savefig(PLOTS_DIR / "07_distance_distribution.png", dpi=300, bbox_inches='tight')
plt.close()
print(f"   ✓ Saved: {PLOTS_DIR / '07_distance_distribution.png'}")

# ----------------------------------------------------------------------------
# 8. Material Distribution (Top 15)
# ----------------------------------------------------------------------------
print("\n8. Top 15 materials used...")

# Parse materials and count
material_counts = {}
for materials_json in df_complete['materials']:
    try:
        materials = json.loads(materials_json.replace("'", '"'))
        for material in materials.keys():
            material_counts[material] = material_counts.get(material, 0) + 1
    except:
        pass

material_series = pd.Series(material_counts).sort_values(ascending=False).head(15)

fig, ax = plt.subplots(figsize=(12, 8))
material_series.plot(kind='barh', ax=ax, color='purple', edgecolor='black', alpha=0.7)
ax.set_xlabel('Count', fontsize=12, fontweight='bold')
ax.set_ylabel('Material', fontsize=12, fontweight='bold')
ax.set_title('Top 15 Materials Used', fontsize=14, fontweight='bold')
for i, v in enumerate(material_series):
    ax.text(v + 500, i, f'{v:,}', va='center', fontsize=9)
plt.tight_layout()
plt.savefig(PLOTS_DIR / "08_top15_materials.png", dpi=300, bbox_inches='tight')
plt.close()
print(f"   ✓ Saved: {PLOTS_DIR / '08_top15_materials.png'}")

# ============================================================================
# SUMMARY STATISTICS
# ============================================================================

print("\n" + "=" * 80)
print("SUMMARY STATISTICS")
print("=" * 80)

print(f"\n📊 Dataset Overview:")
print(f"   Total Products: {len(df_complete):,}")
print(f"   Unique Categories: {df_complete['category'].nunique()}")
print(f"   Unique Parent Categories: {df_complete['parent_category'].nunique()}")
print(f"   Unique Countries: {df_complete['manufacturer_country'].nunique()}")

print(f"\n📊 Gender Distribution:")
for gender, count in df_complete['gender'].value_counts().items():
    pct = (count / len(df_complete)) * 100
    print(f"   {gender}: {count:,} ({pct:.1f}%)")

print(f"\n📊 Top 5 Categories:")
for cat, count in df_complete['category'].value_counts().head(5).items():
    print(f"   {cat}: {count:,}")

print(f"\n📊 Dress Categories Status:")
for cat in ['Gowns', 'Maxi', 'Midi', 'Mini']:
    # Handle both naming conventions
    count = len(df_complete[df_complete['category'].isin([cat, f'{cat} Dresses'])])
    target = 25000
    status = "✓" if count >= target else "✗"
    print(f"   {status} {cat}: {count:,} (Target: {target:,})")

print(f"\n📊 Numeric Statistics:")
print(f"   Weight (kg):")
print(f"      Mean: {df_complete['weight_kg'].mean():.3f}")
print(f"      Median: {df_complete['weight_kg'].median():.3f}")
print(f"      Std Dev: {df_complete['weight_kg'].std():.3f}")
print(f"   Distance (km):")
print(f"      Mean: {df_complete['total_distance_km'].mean():.0f}")
print(f"      Median: {df_complete['total_distance_km'].median():.0f}")
print(f"      Std Dev: {df_complete['total_distance_km'].std():.0f}")

print("\n" + "=" * 80)
print("✓ ANALYSIS COMPLETE!")
print(f"✓ Combined dataset saved to: {OUTPUT_DATA}")
print(f"✓ Plots saved to: {PLOTS_DIR}/")
print("=" * 80)
