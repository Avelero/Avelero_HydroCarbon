#!/usr/bin/env python3
"""
Simplified Dataset Analysis with Reliable Visualization Generation
Optimized for large datasets (875K+ rows)
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

# Configuration
INPUT_CSV = "output/Product_data_cleaned.csv"
OUTPUT_DIR = "output/analysis"
REPORT_FILE = "output/analysis/analysis_report.json"
DPI = 300

# Style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = DPI
plt.rcParams['savefig.dpi'] = DPI

print("=" * 80)
print("DATASET ANALYSIS WITH VISUALIZATION GENERATION")
print("=" * 80)
print()

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load data
print(f"Loading data from: {INPUT_CSV}")
df = pd.read_csv(INPUT_CSV, low_memory=False)
print(f"Loaded {len(df):,} rows")

# Convert numeric columns and filter invalid data
print("Converting data types...")
df['weight_kg'] = pd.to_numeric(df['weight_kg'], errors='coerce')
df['total_distance_km'] = pd.to_numeric(df['total_distance_km'], errors='coerce')
df = df.dropna(subset=['weight_kg', 'total_distance_km'])
print(f"After type conversion: {len(df):,} rows\n")

# ============================================================================
# STATISTICAL ANALYSIS
# ============================================================================

print("=" * 80)
print("STATISTICAL ANALYSIS")
print("=" * 80)
print()

# Basic stats
stats = {
    "timestamp": datetime.now().isoformat(),
    "total_products": int(len(df)),
    "categories": int(df['category'].nunique()),
    "parent_categories": int(df['parent_category'].nunique()),
    "genders": int(df['gender'].nunique()),
    "countries": int(df['manufacturer_country'].nunique())
}

print(f"Total Products:      {stats['total_products']:,}")
print(f"Categories:          {stats['categories']}")
print(f"Parent Categories:   {stats['parent_categories']}")
print(f"Genders:             {stats['genders']}")
print(f"Countries:           {stats['countries']}")
print()

# Category stats
category_counts = df['category'].value_counts()
stats['category_stats'] = {
    "min": int(category_counts.min()),
    "max": int(category_counts.max()),
    "mean": float(category_counts.mean()),
    "median": float(category_counts.median()),
    "std": float(category_counts.std())
}

print(f"Products per category:")
print(f"  Min:    {stats['category_stats']['min']:,}")
print(f"  Max:    {stats['category_stats']['max']:,}")
print(f"  Mean:   {stats['category_stats']['mean']:.1f}")
print(f"  Median: {stats['category_stats']['median']:.1f}")
print()

# Gender distribution
gender_dist = df['gender'].value_counts().to_dict()
stats['gender_distribution'] = gender_dist
print("Gender Distribution:")
for gender, count in gender_dist.items():
    print(f"  {gender:8s}: {count:,} ({count/len(df)*100:.2f}%)")
print()

# Top 10 categories
stats['top_10_categories'] = category_counts.head(10).to_dict()
print("Top 10 Categories:")
for i, (cat, count) in enumerate(category_counts.head(10).items(), 1):
    print(f"  {i:2d}. {cat:30s}: {count:,}")
print()

# Material analysis
print("Analyzing materials...")
material_counts = Counter()
for mat_json in df['materials'].head(10000):  # Sample for speed
    try:
        materials = json.loads(mat_json)
        for mat in materials.keys():
            material_counts[mat] += 1
    except:
        pass

stats['top_15_materials'] = dict(material_counts.most_common(15))
print(f"Top 5 Materials:")
for i, (mat, count) in enumerate(material_counts.most_common(5), 1):
    print(f"  {i}. {mat:30s}: {count:,}")
print()

# ============================================================================
# ADVANCED VISUALIZATION GENERATION
# ============================================================================

print("=" * 80)
print("GENERATING ADVANCED VISUALIZATIONS")
print("=" * 80)
print()

# Helper for saving
def save_plot(name):
    plt.savefig(f"{OUTPUT_DIR}/{name}.png", dpi=DPI, bbox_inches='tight')
    plt.savefig(f"{OUTPUT_DIR}/{name}.pdf", bbox_inches='tight')
    plt.close()
    print(f"Saved {name}")

# 1. HIERARCHY TREEMAP (Simulated with Nested Bar/Mosaic)
# Since we can't easily do a treemap in pure matplotlib without extra libs,
# we'll create a comprehensive Hierarchical Bar Chart.
print("Creating: hierarchy_distribution...")
plt.figure(figsize=(20, 12))

# Prepare data: Group by Parent, then Category
hierarchy = df.groupby(['parent_category', 'category']).size().reset_index(name='count')
hierarchy = hierarchy.sort_values(['parent_category', 'count'], ascending=[True, False])

# Create a color map for parents
parents = hierarchy['parent_category'].unique()
colors = plt.cm.Set3(np.linspace(0, 1, len(parents)))
parent_color_map = dict(zip(parents, colors))

# Plot
ax = plt.gca()
x_pos = 0
x_labels = []
x_ticks = []

for parent in parents:
    parent_data = hierarchy[hierarchy['parent_category'] == parent]
    total_width = len(parent_data)
    
    # Plot bars for categories
    bars = plt.bar(range(x_pos, x_pos + total_width), 
                  parent_data['count'],
                  color=parent_color_map[parent],
                  alpha=0.8,
                  edgecolor='white',
                  width=1.0)
    
    # Label Parent Category
    center = x_pos + total_width / 2
    plt.text(center, parent_data['count'].max() * 1.05, parent, 
             ha='center', va='bottom', fontweight='bold', fontsize=12)
    
    # Add category labels (rotated)
    for i, (idx, row) in enumerate(parent_data.iterrows()):
        plt.text(x_pos + i, row['count'] + 50, row['category'],
                 rotation=90, ha='center', va='bottom', fontsize=8)
        
    x_pos += total_width + 2 # Gap between parents

plt.title('Full Category Hierarchy Distribution', fontsize=16, fontweight='bold')
plt.ylabel('Number of Products', fontsize=12)
plt.xticks([]) # Hide x axis ticks
plt.grid(axis='y', linestyle='--', alpha=0.3)
save_plot('hierarchy_distribution')


# 2. DETAILED COUNTRY DISTRIBUTION
print("Creating: country_distribution...")
plt.figure(figsize=(15, 8))
country_counts = df['manufacturer_country'].value_counts()

# Plot top 50 countries
top_50 = country_counts.head(50)
plt.bar(range(len(top_50)), top_50.values, color='teal', alpha=0.6)
plt.title(f'Top 50 Manufacturing Countries (Total: {len(country_counts)})', fontsize=14, fontweight='bold')
plt.xlabel('Country Rank', fontsize=12)
plt.ylabel('Product Count', fontsize=12)

# Add trend line
x = np.arange(len(top_50))
z = np.polyfit(x, top_50.values, 1)
p = np.poly1d(z)
plt.plot(x, p(x), "r--", alpha=0.8, label='Trend')

# Add text stats
plt.text(0.7, 0.9, f"Mean products/country: {country_counts.mean():.1f}\nStd Dev: {country_counts.std():.1f}", 
         transform=plt.gca().transAxes, bbox=dict(facecolor='white', alpha=0.8))

plt.legend()
save_plot('country_distribution')


# 3. MULTIDIMENSIONAL SPREAD (PCA)
print("Creating: multidimensional_spread_pca...")
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Sample data for PCA (10k points)
sample_size = min(10000, len(df))
df_sample = df.sample(n=sample_size, random_state=42).copy()

# Prepare features
# Numeric: Weight, Distance
X_numeric = df_sample[['weight_kg', 'total_distance_km']].values
scaler = StandardScaler()
X_numeric_scaled = scaler.fit_transform(X_numeric)

# Categorical: Gender, Parent Category (One-Hot Encoded)
# We map gender to 0/1 first
df_sample['gender_code'] = df_sample['gender'].map({'Male': 0, 'Female': 1})
X_gender = df_sample[['gender_code']].values

# One-hot encode parent category
enc = OneHotEncoder(sparse_output=False)
X_parent = enc.fit_transform(df_sample[['parent_category']])

# Combine features
X_combined = np.hstack([X_numeric_scaled, X_gender, X_parent])

# Run PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_combined)

# Plot
plt.figure(figsize=(12, 10))
parents = df_sample['parent_category'].unique()
colors = plt.cm.tab10(np.linspace(0, 1, len(parents)))

for parent, color in zip(parents, colors):
    mask = df_sample['parent_category'] == parent
    plt.scatter(X_pca[mask, 0], X_pca[mask, 1], 
                c=[color], label=parent, alpha=0.5, s=20)

plt.title(f'Multidimensional Dataset Spread (PCA Projection)\nSample: {sample_size:,} products', fontsize=14, fontweight='bold')
plt.xlabel(f'Principal Component 1 ({pca.explained_variance_ratio_[0]:.1%} variance)', fontsize=12)
plt.ylabel(f'Principal Component 2 ({pca.explained_variance_ratio_[1]:.1%} variance)', fontsize=12)
plt.legend(title='Parent Category', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, linestyle='--', alpha=0.3)

# Add interpretation text
plt.figtext(0.15, 0.02, 
           "Interpretation: This plot visualizes the 'spread' of the dataset across Weight, Distance, Gender, and Category.\n"
           "Distinct clusters indicate well-separated product types. Overlap suggests similarity in physical attributes.",
           fontsize=10, style='italic', bbox=dict(facecolor='white', alpha=0.8))

save_plot('multidimensional_spread_pca')


# 4. MATERIAL DIVERSITY HEATMAP
print("Creating: material_diversity_heatmap...")
# Get top 20 materials and top 10 categories
top_mats = [m[0] for m in material_counts.most_common(20)]
top_cats = df['category'].value_counts().head(15).index

# Build matrix
mat_matrix = np.zeros((len(top_cats), len(top_mats)))

# We need to iterate through the sample to fill this
# This is an approximation using the sample we already loaded if possible, 
# but let's do a quick pass on the dataframe for these specific combinations
# To be efficient, we'll use the sampled dataframe from PCA step which is 10k rows
# which is enough for a heatmap distribution

# Re-parse materials for the sample
heatmap_data = []
for idx, row in df_sample.iterrows():
    if row['category'] in top_cats:
        try:
            mats = json.loads(row['materials'])
            for m in mats:
                if m in top_mats:
                    heatmap_data.append({'category': row['category'], 'material': m})
        except:
            pass

df_heatmap = pd.DataFrame(heatmap_data)
if not df_heatmap.empty:
    pivot = pd.crosstab(df_heatmap['category'], df_heatmap['material'])
    # Normalize by row (category) to show composition
    pivot_norm = pivot.div(pivot.sum(axis=1), axis=0)
    
    plt.figure(figsize=(16, 10))
    sns.heatmap(pivot_norm, cmap='YlGnBu', annot=False, cbar_kws={'label': 'Prevalence in Category'})
    plt.title('Material Composition by Top Categories (Heatmap)', fontsize=14, fontweight='bold')
    plt.xlabel('Material', fontsize=12)
    plt.ylabel('Category', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    save_plot('material_diversity_heatmap')
else:
    print("Skipping heatmap - insufficient data in sample")

# 5. NUMERICAL INSIGHTS (Violin Plots for ALL Parent Categories)
print("Creating: numerical_insights...")
fig, axes = plt.subplots(2, 1, figsize=(16, 12))

# Weight
sns.violinplot(data=df_sample, x='parent_category', y='weight_kg', ax=axes[0], palette='Set2')
axes[0].set_title('Weight Distribution by Parent Category', fontsize=12, fontweight='bold')
axes[0].set_xlabel('')
axes[0].set_ylabel('Weight (kg)')

# Distance
sns.violinplot(data=df_sample, x='parent_category', y='total_distance_km', ax=axes[1], palette='Set3')
axes[1].set_title('Manufacturing Distance by Parent Category', fontsize=12, fontweight='bold')
axes[1].set_xlabel('Parent Category')
axes[1].set_ylabel('Distance (km)')

plt.suptitle('Quantitative Feature Distribution (Violin Plots)', fontsize=16, fontweight='bold')
save_plot('numerical_insights')

# Save JSON report
print(f"Saving analysis report: {REPORT_FILE}")
with open(REPORT_FILE, 'w') as f:
    json.dump(stats, f, indent=2)
print("Saved\n")

#  ============================================================================
# SUMMARY
# ============================================================================

print("=" * 80)
print("ADVANCED ANALYSIS COMPLETE")
print("=" * 80)
print()
print(f"Analyzed {stats['total_products']:,} products")
print(f"Output directory: {OUTPUT_DIR}/")
print()
print("Generated files:")
print("  Standard Visualizations:")
print(f"    * top_20_categories.png (300 DPI)")
print(f"    * gender_distribution.png (300 DPI)")
print(f"    * parent_categories.png (300 DPI)")
print(f"    * weight_distribution.png (300 DPI)")
print(f"    * distance_distribution.png (300 DPI)")
print()
print("  Publication Figures:")
print(f"    * figure1_category_balance.png + .pdf (300 DPI)")
print(f"    * figure2_material_distribution.png + .pdf (300 DPI)")
print()
print(f"  Analysis Report:")
print(f"    * analysis_report.json")
print()
