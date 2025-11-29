#!/usr/bin/env python3
"""
Analyze current material distribution to plan extra generation.

Identifies:
1. Which materials are underrepresented (< 35,000 products)
2. How many additional products needed per material
3. Which categories should receive the new materials
"""

import pandas as pd
import json
from collections import Counter

# Configuration
INPUT_CSV = "data_correction/output/Product_data_final.csv"
TARGET_PER_MATERIAL = 35000  # Target products per material

print("=" * 90)
print("MATERIAL DISTRIBUTION ANALYSIS FOR EXTRA GENERATION")
print("=" * 90)
print()

# Load data
print(f"📂 Loading data from: {INPUT_CSV}")
df = pd.read_csv(INPUT_CSV, low_memory=False)
print(f"✓ Loaded {len(df):,} products\n")

# Parse all materials
print("Parsing materials from all products...")
material_counts = Counter()
material_by_category = {}
material_by_parent = {}
material_by_gender = {}

for idx, row in df.iterrows():
    try:
        mat_json = row['materials']
        if isinstance(mat_json, str):
            materials = json.loads(mat_json.replace("'", '"'))
        else:
            materials = mat_json

        if isinstance(materials, dict):
            for mat in materials.keys():
                material_counts[mat] += 1

                # Track by category
                category = row['category']
                if mat not in material_by_category:
                    material_by_category[mat] = Counter()
                material_by_category[mat][category] += 1

                # Track by parent
                parent = row['parent_category']
                if mat not in material_by_parent:
                    material_by_parent[mat] = Counter()
                material_by_parent[mat][parent] += 1

                # Track by gender
                gender = row['gender']
                if mat not in material_by_gender:
                    material_by_gender[mat] = Counter()
                material_by_gender[mat][gender] += 1
    except:
        pass

print(f"✓ Parsed materials from {len(df):,} products\n")

# Identify materials needing boost
print("=" * 90)
print("MATERIALS NEEDING BOOST (< 35,000 products)")
print("=" * 90)
print()

underrepresented = []
for mat, count in sorted(material_counts.items(), key=lambda x: x[1]):
    if count < TARGET_PER_MATERIAL:
        needed = TARGET_PER_MATERIAL - count
        underrepresented.append((mat, count, needed))

print(f"{'Material':<30} {'Current':<12} {'Needed':<12} {'Target':<12}")
print("-" * 90)

total_needed = 0
for mat, current, needed in underrepresented:
    print(f"{mat:<30} {current:>11,} {needed:>11,} {TARGET_PER_MATERIAL:>11,}")
    total_needed += needed

print("-" * 90)
print(f"{'TOTAL PRODUCTS TO GENERATE:':<30} {'':<12} {total_needed:>11,}")
print()

# Show top categories for each underrepresented material
print("=" * 90)
print("TOP CATEGORIES FOR EACH UNDERREPRESENTED MATERIAL")
print("=" * 90)
print()

for mat, current, needed in underrepresented[:15]:  # Show first 15
    print(f"\n{mat} (current: {current:,}, need: {needed:,})")
    print(f"{'Category':<30} {'Count':<12} {'%':<8} {'Parent':<20} {'Gender Split'}")
    print("-" * 90)

    top_categories = material_by_category[mat].most_common(5)
    for cat, count in top_categories:
        pct = (count / current) * 100

        # Find parent for this category
        parent_counts = df[df['category'] == cat]['parent_category'].value_counts()
        parent = parent_counts.index[0] if len(parent_counts) > 0 else 'Unknown'

        # Find gender split for this material in this category
        gender_counts = df[(df['category'] == cat) &
                          (df['materials'].str.contains(mat, na=False))]['gender'].value_counts()
        gender_str = ', '.join([f"{g}: {c}" for g, c in gender_counts.items()])

        print(f"{cat:<30} {count:>11,} {pct:>6.1f}% {parent:<20} {gender_str}")

# Analyze parent category distribution
print("\n" + "=" * 90)
print("PARENT CATEGORY DISTRIBUTION FOR UNDERREPRESENTED MATERIALS")
print("=" * 90)
print()

print(f"{'Material':<30} {'Tops':<10} {'Bottoms':<10} {'Dresses':<10} {'Footwear':<10} {'Outerwear':<10}")
print("-" * 90)

for mat, current, needed in underrepresented[:15]:
    parent_dist = material_by_parent.get(mat, {})
    tops = parent_dist.get('Tops', 0)
    bottoms = parent_dist.get('Bottoms', 0)
    dresses = parent_dist.get('Dresses', 0)
    footwear = parent_dist.get('Footwear', 0)
    outerwear = parent_dist.get('Outerwear', 0)

    print(f"{mat:<30} {tops:>9,} {bottoms:>9,} {dresses:>9,} {footwear:>9,} {outerwear:>9,}")

# Gender distribution
print("\n" + "=" * 90)
print("GENDER DISTRIBUTION FOR UNDERREPRESENTED MATERIALS")
print("=" * 90)
print()

print(f"{'Material':<30} {'Male':<12} {'Female':<12} {'Male %':<10}")
print("-" * 90)

for mat, current, needed in underrepresented[:15]:
    gender_dist = material_by_gender.get(mat, {})
    male = gender_dist.get('Male', 0)
    female = gender_dist.get('Female', 0)
    total = male + female
    male_pct = (male / total * 100) if total > 0 else 0

    print(f"{mat:<30} {male:>11,} {female:>11,} {male_pct:>8.1f}%")

print("\n" + "=" * 90)
print("✓ ANALYSIS COMPLETE")
print("=" * 90)
print()
print(f"Summary:")
print(f"  Materials needing boost: {len(underrepresented)}")
print(f"  Total products to generate: {total_needed:,}")
print(f"  Target per material: {TARGET_PER_MATERIAL:,}")
print()
