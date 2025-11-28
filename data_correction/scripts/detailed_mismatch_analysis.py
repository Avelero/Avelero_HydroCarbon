#!/usr/bin/env python3
"""
Detailed Mismatch and Incorrections Analysis

Analyzes:
1. Category-Parent mismatches
2. Gender-Category mismatches
3. Material-Category logic errors
4. Naming inconsistencies
5. Data integrity issues
6. Numerical anomalies
7. Supply chain logic errors
"""

import pandas as pd
import json
import numpy as np
from collections import defaultdict, Counter
from datetime import datetime

# Configuration
INPUT_CSV = "data_correction/output/Product_data_complete.csv"
OUTPUT_DIR = "data_correction/output/comprehensive_analysis"
REPORT_FILE = f"{OUTPUT_DIR}/detailed_mismatch_report.md"

print("=" * 80)
print("DETAILED MISMATCH AND INCORRECTIONS ANALYSIS")
print("=" * 80)
print()

# Load data
print(f"📂 Loading data from: {INPUT_CSV}")
df = pd.read_csv(INPUT_CSV, low_memory=False)
print(f"✓ Loaded {len(df):,} rows\n")

# Convert numeric columns
df['weight_kg'] = pd.to_numeric(df['weight_kg'], errors='coerce')
df['total_distance_km'] = pd.to_numeric(df['total_distance_km'], errors='coerce')

# Initialize report
report_lines = []
report_lines.append("# Detailed Mismatch and Incorrections Analysis")
report_lines.append(f"\n**Analysis Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
report_lines.append(f"**Total Products**: {len(df):,}")
report_lines.append("\n---\n")

# ============================================================================
# 1. CATEGORY-PARENT MISMATCHES
# ============================================================================

print("=" * 80)
print("1. ANALYZING CATEGORY-PARENT MISMATCHES")
print("=" * 80)
print()

report_lines.append("## 1. Category-Parent Mismatches\n")

# Define expected parent categories for each category
EXPECTED_PARENT_MAP = {
    # Dresses
    'Gowns': 'Dresses', 'Maxi': 'Dresses', 'Midi': 'Dresses', 'Mini': 'Dresses',
    'Maxi Dresses': 'Dresses', 'Midi Dresses': 'Dresses', 'Mini Dresses': 'Dresses',
    'Dresses': 'Dresses',

    # Tops
    'Polos': 'Tops', 'Short Sleeve Shirts': 'Tops', 'Long Sleeve Shirts': 'Tops',
    'Button-Ups': 'Tops', 'Sleeveless': 'Tops', 'Tank Tops': 'Tops',
    'Blouses': 'Tops', 'Crop Tops': 'Tops', 'Bodysuits': 'Tops',
    'Sweaters': 'Tops', 'Sweaters & Knitwear': 'Tops', 'Hoodies': 'Tops',
    'Sweatshirts': 'Tops', 'Sweatshirts & Hoodies': 'Tops', 'Jerseys': 'Tops',

    # Bottoms
    'Jumpsuits': 'Bottoms', 'Shorts': 'Bottoms', 'Leggings': 'Bottoms',
    'Maxi Skirts': 'Bottoms', 'Midi Skirts': 'Bottoms', 'Mini Skirts': 'Bottoms',
    'Jeans': 'Bottoms', 'Sweatpants': 'Bottoms', 'Sweatpants & Joggers': 'Bottoms',
    'Casual Pants': 'Bottoms', 'Swimwear': 'Bottoms', 'Cropped Pants': 'Bottoms',
    'Pants': 'Bottoms', 'Joggers': 'Bottoms', 'Denim': 'Bottoms',

    # Outerwear
    'Bombers': 'Outerwear', 'Denim Jackets': 'Outerwear', 'Leather Jackets': 'Outerwear',
    'Vests': 'Outerwear', 'Heavy Coats': 'Outerwear', 'Blazers': 'Outerwear',
    'Coats': 'Outerwear', 'Light Jackets': 'Outerwear', 'Fur & Faux Fur': 'Outerwear',
    'Raincoats': 'Outerwear', 'Jackets': 'Outerwear', 'Down Jackets': 'Outerwear',
    'Cloaks & Capes': 'Outerwear', 'Rain Jackets': 'Outerwear', 'Parkas': 'Outerwear',

    # Footwear
    'Sandals': 'Footwear', 'Casual Shoes': 'Footwear', 'Sneakers': 'Footwear',
    'Boots': 'Footwear', 'Athletic Shoes': 'Footwear', 'Dress Shoes': 'Footwear',
    'Flats': 'Footwear', 'Loafers': 'Footwear', 'Heels': 'Footwear',
}

mismatches = []
orphans = []

for cat in df['category'].unique():
    parent_counts = df[df['category'] == cat]['parent_category'].value_counts()

    if cat not in EXPECTED_PARENT_MAP:
        orphans.append((cat, parent_counts.to_dict()))
    else:
        expected_parent = EXPECTED_PARENT_MAP[cat]
        for actual_parent, count in parent_counts.items():
            if actual_parent != expected_parent:
                mismatches.append({
                    'category': cat,
                    'expected_parent': expected_parent,
                    'actual_parent': actual_parent,
                    'count': count
                })

print(f"Category-Parent Mismatches Found: {len(mismatches)}")
print(f"Orphaned Categories Found: {len(orphans)}\n")

if mismatches:
    report_lines.append("### ❌ Category-Parent Mismatches\n")
    report_lines.append("| Category | Expected Parent | Actual Parent | Product Count |")
    report_lines.append("|----------|----------------|---------------|---------------|")
    for m in sorted(mismatches, key=lambda x: x['count'], reverse=True):
        print(f"  ❌ {m['category']:30s}: Expected '{m['expected_parent']}', Got '{m['actual_parent']}' ({m['count']:,} products)")
        report_lines.append(f"| {m['category']} | {m['expected_parent']} | {m['actual_parent']} | {m['count']:,} |")
    report_lines.append("\n")
else:
    report_lines.append("### ✅ No Category-Parent Mismatches Found\n")

if orphans:
    report_lines.append("### ⚠️ Orphaned/Unknown Categories\n")
    report_lines.append("| Category | Parent Category | Product Count |")
    report_lines.append("|----------|----------------|---------------|")
    for cat, parents in orphans:
        print(f"  ⚠️  Orphan: {cat:30s} - Parents: {parents}")
        for parent, count in parents.items():
            report_lines.append(f"| {cat} | {parent} | {count:,} |")
    report_lines.append("\n")

# ============================================================================
# 2. GENDER-CATEGORY LOGIC ERRORS
# ============================================================================

print("\n" + "=" * 80)
print("2. ANALYZING GENDER-CATEGORY LOGIC ERRORS")
print("=" * 80)
print()

report_lines.append("## 2. Gender-Category Logic Errors\n")

# Define gender expectations
TYPICALLY_FEMALE = {'Gowns', 'Maxi Dresses', 'Midi Dresses', 'Mini Dresses', 'Blouses', 'Heels'}
TYPICALLY_MALE = {'Button-Ups'}  # Reduced, most categories can be unisex

gender_issues = []

for cat in TYPICALLY_FEMALE:
    if cat in df['category'].values:
        male_count = len(df[(df['category'] == cat) & (df['gender'] == 'Male')])
        total_count = len(df[df['category'] == cat])
        if male_count > 0:
            pct = (male_count / total_count) * 100
            gender_issues.append({
                'category': cat,
                'expected_gender': 'Female',
                'unexpected_gender': 'Male',
                'count': male_count,
                'total': total_count,
                'percentage': pct
            })

print(f"Gender Logic Issues Found: {len(gender_issues)}\n")

if gender_issues:
    report_lines.append("### ⚠️ Potentially Unusual Gender Assignments\n")
    report_lines.append("**Note**: These may be intentional (unisex fashion), but worth reviewing.\n")
    report_lines.append("| Category | Expected Gender | Unexpected Count | Total | Percentage |")
    report_lines.append("|----------|----------------|-----------------|-------|------------|")
    for issue in sorted(gender_issues, key=lambda x: x['percentage'], reverse=True):
        print(f"  ⚠️  {issue['category']:30s}: {issue['count']:,} Male products out of {issue['total']:,} ({issue['percentage']:.2f}%)")
        report_lines.append(f"| {issue['category']} | {issue['expected_gender']} | {issue['count']:,} {issue['unexpected_gender']} | {issue['total']:,} | {issue['percentage']:.2f}% |")
    report_lines.append("\n")
else:
    report_lines.append("### ✅ No Significant Gender-Category Issues\n")

# ============================================================================
# 3. MATERIAL-CATEGORY LOGIC ERRORS
# ============================================================================

print("\n" + "=" * 80)
print("3. ANALYZING MATERIAL-CATEGORY LOGIC ERRORS")
print("=" * 80)
print()

report_lines.append("## 3. Material-Category Logic Errors\n")

# Define material expectations
FOOTWEAR_MATERIALS = {'leather_bovine', 'eva', 'natural_rubber', 'synthetic_rubber_sbr', 'rubber_synthetic'}
CLOTHING_MATERIALS = {'polyester_virgin', 'cotton_conventional', 'viscose', 'elastane', 'modal', 'silk', 'linen_flax', 'wool_generic'}

material_issues = []

print("Checking for leather in non-footwear items...")
for idx, row in df.iterrows():
    try:
        materials = json.loads(row['materials'].replace("'", '"'))
        category = row['category']
        parent = row['parent_category']

        # Check if clothing has footwear materials
        if parent != 'Footwear':
            for mat in materials.keys():
                if mat in {'eva', 'natural_rubber', 'synthetic_rubber_sbr', 'rubber_synthetic'}:
                    material_issues.append({
                        'type': 'footwear_material_in_clothing',
                        'category': category,
                        'parent': parent,
                        'material': mat,
                        'row': idx
                    })
                    break

        # Check if footwear has only clothing materials
        if parent == 'Footwear':
            has_footwear_mat = any(mat in materials for mat in FOOTWEAR_MATERIALS)
            if not has_footwear_mat:
                material_issues.append({
                    'type': 'no_footwear_material_in_footwear',
                    'category': category,
                    'parent': parent,
                    'materials': list(materials.keys()),
                    'row': idx
                })
    except:
        pass

print(f"Material-Category Logic Issues Found: {len(material_issues)}\n")

if material_issues:
    # Count by type
    issue_counts = Counter([issue['type'] for issue in material_issues])

    report_lines.append("### ❌ Material-Category Mismatches\n")

    if issue_counts.get('footwear_material_in_clothing', 0) > 0:
        count = issue_counts['footwear_material_in_clothing']
        report_lines.append(f"**Footwear materials in clothing**: {count:,} products\n")
        print(f"  Found {count:,} clothing items with footwear materials (rubber, EVA)")

        # Show examples
        examples = [i for i in material_issues if i['type'] == 'footwear_material_in_clothing'][:10]
        if examples:
            report_lines.append("Examples:\n")
            for ex in examples:
                report_lines.append(f"- Row {ex['row']}: {ex['category']} ({ex['parent']}) contains {ex['material']}\n")
            report_lines.append("\n")

    if issue_counts.get('no_footwear_material_in_footwear', 0) > 0:
        count = issue_counts['no_footwear_material_in_footwear']
        report_lines.append(f"**Footwear without typical footwear materials**: {count:,} products\n")
        print(f"  Found {count:,} footwear items without leather/rubber/EVA")

        # Show examples
        examples = [i for i in material_issues if i['type'] == 'no_footwear_material_in_footwear'][:10]
        if examples:
            report_lines.append("Examples:\n")
            for ex in examples:
                mats = ', '.join(ex['materials'][:3])
                report_lines.append(f"- Row {ex['row']}: {ex['category']} only has: {mats}\n")
            report_lines.append("\n")
else:
    report_lines.append("### ✅ No Significant Material-Category Issues\n")

# ============================================================================
# 4. NAMING INCONSISTENCIES
# ============================================================================

print("\n" + "=" * 80)
print("4. ANALYZING NAMING INCONSISTENCIES")
print("=" * 80)
print()

report_lines.append("## 4. Naming Inconsistencies\n")

naming_issues = {
    'duplicate_categories': [],
    'similar_names': [],
    'inconsistent_formats': []
}

# Find duplicate/overlapping categories
categories = df['category'].unique()

# Maxi/Midi/Mini vs Maxi/Midi/Mini Dresses
dress_variants = {
    'Maxi': df[df['category'] == 'Maxi']['category'].count() if 'Maxi' in categories else 0,
    'Maxi Dresses': df[df['category'] == 'Maxi Dresses']['category'].count() if 'Maxi Dresses' in categories else 0,
    'Midi': df[df['category'] == 'Midi']['category'].count() if 'Midi' in categories else 0,
    'Midi Dresses': df[df['category'] == 'Midi Dresses']['category'].count() if 'Midi Dresses' in categories else 0,
    'Mini': df[df['category'] == 'Mini']['category'].count() if 'Mini' in categories else 0,
    'Mini Dresses': df[df['category'] == 'Mini Dresses']['category'].count() if 'Mini Dresses' in categories else 0,
}

if any(dress_variants.values()):
    naming_issues['duplicate_categories'].append({
        'type': 'Dress naming inconsistency',
        'variants': dress_variants
    })

# Sweatpants variants
sweat_variants = {
    'Sweatpants': df[df['category'] == 'Sweatpants']['category'].count() if 'Sweatpants' in categories else 0,
    'Sweatpants & Joggers': df[df['category'] == 'Sweatpants & Joggers']['category'].count() if 'Sweatpants & Joggers' in categories else 0,
    'Joggers': df[df['category'] == 'Joggers']['category'].count() if 'Joggers' in categories else 0,
}

if sum(sweat_variants.values()) > 0:
    naming_issues['duplicate_categories'].append({
        'type': 'Sweatpants/Joggers overlap',
        'variants': sweat_variants
    })

# Sweater variants
sweater_variants = {
    'Sweaters': df[df['category'] == 'Sweaters']['category'].count() if 'Sweaters' in categories else 0,
    'Sweaters & Knitwear': df[df['category'] == 'Sweaters & Knitwear']['category'].count() if 'Sweaters & Knitwear' in categories else 0,
}

if sum(sweater_variants.values()) > 0:
    naming_issues['duplicate_categories'].append({
        'type': 'Sweaters overlap',
        'variants': sweater_variants
    })

# Hoodie/Sweatshirt variants
hoodie_variants = {
    'Hoodies': df[df['category'] == 'Hoodies']['category'].count() if 'Hoodies' in categories else 0,
    'Sweatshirts': df[df['category'] == 'Sweatshirts']['category'].count() if 'Sweatshirts' in categories else 0,
    'Sweatshirts & Hoodies': df[df['category'] == 'Sweatshirts & Hoodies']['category'].count() if 'Sweatshirts & Hoodies' in categories else 0,
}

if sum(hoodie_variants.values()) > 0:
    naming_issues['duplicate_categories'].append({
        'type': 'Hoodie/Sweatshirt overlap',
        'variants': hoodie_variants
    })

print(f"Naming Inconsistencies Found: {len(naming_issues['duplicate_categories'])}\n")

if naming_issues['duplicate_categories']:
    report_lines.append("### ⚠️ Overlapping/Duplicate Category Names\n")
    for issue in naming_issues['duplicate_categories']:
        report_lines.append(f"**{issue['type']}**:\n")
        for variant, count in issue['variants'].items():
            if count > 0:
                print(f"  • {variant:30s}: {count:,} products")
                report_lines.append(f"- {variant}: {count:,} products\n")
        report_lines.append("\n")
else:
    report_lines.append("### ✅ No Naming Inconsistencies Found\n")

# ============================================================================
# 5. NUMERICAL ANOMALIES
# ============================================================================

print("\n" + "=" * 80)
print("5. ANALYZING NUMERICAL ANOMALIES")
print("=" * 80)
print()

report_lines.append("## 5. Numerical Anomalies\n")

# Weight anomalies
weight_anomalies = {
    'extreme_low': len(df[df['weight_kg'] < 0.01]),  # < 10 grams
    'extreme_high': len(df[df['weight_kg'] > 50]),   # > 50 kg
    'zero_or_negative': len(df[df['weight_kg'] <= 0]),
    'missing': df['weight_kg'].isna().sum()
}

# Distance anomalies
distance_anomalies = {
    'extreme_low': len(df[df['total_distance_km'] < 1]),      # < 1 km
    'extreme_high': len(df[df['total_distance_km'] > 40000]), # > Earth circumference
    'zero_or_negative': len(df[df['total_distance_km'] <= 0]),
    'missing': df['total_distance_km'].isna().sum()
}

print(f"Weight Anomalies: {sum(weight_anomalies.values())}")
print(f"Distance Anomalies: {sum(distance_anomalies.values())}\n")

report_lines.append("### Weight Anomalies\n")
report_lines.append("| Anomaly Type | Count | Description |")
report_lines.append("|--------------|-------|-------------|")
for anom_type, count in weight_anomalies.items():
    if count > 0:
        desc = ""
        if anom_type == 'extreme_low':
            desc = "< 10 grams (unrealistic for clothing)"
            print(f"  ❌ {count:,} products with weight < 10 grams")
        elif anom_type == 'extreme_high':
            desc = "> 50 kg (impossibly heavy)"
            print(f"  ❌ {count:,} products with weight > 50 kg")
        elif anom_type == 'zero_or_negative':
            desc = "Zero or negative weight"
            print(f"  ❌ {count:,} products with zero/negative weight")
        elif anom_type == 'missing':
            desc = "Missing weight data"
            print(f"  ⚠️  {count:,} products missing weight")

        report_lines.append(f"| {anom_type.replace('_', ' ').title()} | {count:,} | {desc} |")
report_lines.append("\n")

# Weight extremes
if not df['weight_kg'].isna().all():
    min_weight = df['weight_kg'].min()
    max_weight = df['weight_kg'].max()
    print(f"  Weight range: {min_weight:.4f} kg - {max_weight:.2f} kg")
    report_lines.append(f"**Weight Range**: {min_weight:.4f} kg - {max_weight:.2f} kg\n")

report_lines.append("\n### Distance Anomalies\n")
report_lines.append("| Anomaly Type | Count | Description |")
report_lines.append("|--------------|-------|-------------|")
for anom_type, count in distance_anomalies.items():
    if count > 0:
        desc = ""
        if anom_type == 'extreme_low':
            desc = "< 1 km (unrealistic supply chain)"
            print(f"  ❌ {count:,} products with distance < 1 km")
        elif anom_type == 'extreme_high':
            desc = "> 40,000 km (> Earth's circumference)"
            print(f"  ❌ {count:,} products with distance > 40,000 km")
        elif anom_type == 'zero_or_negative':
            desc = "Zero or negative distance"
            print(f"  ❌ {count:,} products with zero/negative distance")
        elif anom_type == 'missing':
            desc = "Missing distance data"
            print(f"  ⚠️  {count:,} products missing distance")

        report_lines.append(f"| {anom_type.replace('_', ' ').title()} | {count:,} | {desc} |")
report_lines.append("\n")

# Distance extremes
if not df['total_distance_km'].isna().all():
    min_dist = df['total_distance_km'].min()
    max_dist = df['total_distance_km'].max()
    print(f"  Distance range: {min_dist:.2f} km - {max_dist:.2f} km")
    report_lines.append(f"**Distance Range**: {min_dist:.2f} km - {max_dist:.2f} km\n")

# ============================================================================
# 6. MATERIAL DATA ERRORS
# ============================================================================

print("\n" + "=" * 80)
print("6. ANALYZING MATERIAL DATA ERRORS")
print("=" * 80)
print()

report_lines.append("\n## 6. Material Data Errors\n")

material_errors = {
    'typos': [],
    'corrupted': [],
    'japanese': [],
    'spacing': [],
    'impossible': []
}

# Known error patterns
TYPOS = {'poliamide_6', 'polyster_virgin', 'polyamid_6', 'polporter_virgin', 'polluster_recycled',
         'visvester_virgin', 'visyester_virgin', 'viscane', 'visise', 'viscluse'}
CORRUPTED = {'pol<seg_125>_virgin'}
JAPANESE = {'polリエステル_recycled', 'polリエステル_virgin'}
SPACING = {'pol polyester_virgin', 'pol polyester_recycled', 'pol polyamide_6'}
IMPOSSIBLE = {'polyester_organic', 'polyester_bovine', 'coated_bovine'}

all_materials = Counter()
error_counts = defaultdict(int)

for mat_json in df['materials']:
    try:
        materials = json.loads(mat_json.replace("'", '"'))
        for mat in materials.keys():
            all_materials[mat] += 1

            if mat in TYPOS:
                material_errors['typos'].append(mat)
                error_counts['typos'] += 1
            if mat in CORRUPTED:
                material_errors['corrupted'].append(mat)
                error_counts['corrupted'] += 1
            if mat in JAPANESE:
                material_errors['japanese'].append(mat)
                error_counts['japanese'] += 1
            if mat in SPACING:
                material_errors['spacing'].append(mat)
                error_counts['spacing'] += 1
            if mat in IMPOSSIBLE:
                material_errors['impossible'].append(mat)
                error_counts['impossible'] += 1
    except:
        pass

total_material_errors = sum(error_counts.values())
print(f"Material Errors Found: {total_material_errors}\n")

if error_counts:
    report_lines.append("### Material Error Types\n")
    report_lines.append("| Error Type | Count | Examples |")
    report_lines.append("|------------|-------|----------|")

    for error_type, count in error_counts.items():
        if count > 0:
            examples = list(set(material_errors[error_type]))[:3]
            examples_str = ', '.join(examples)
            print(f"  {error_type:15s}: {count:,} occurrences - Examples: {examples_str}")
            report_lines.append(f"| {error_type.title()} | {count:,} | {examples_str} |")

    report_lines.append("\n")

# List all erroneous materials
report_lines.append("### Complete List of Erroneous Materials\n")
all_errors = TYPOS | CORRUPTED | JAPANESE | SPACING | IMPOSSIBLE
report_lines.append("| Material | Count | Issue |")
report_lines.append("|----------|-------|-------|")
for mat in sorted(all_errors):
    if mat in all_materials:
        count = all_materials[mat]
        issue = []
        if mat in TYPOS: issue.append("Typo")
        if mat in CORRUPTED: issue.append("Corrupted")
        if mat in JAPANESE: issue.append("Japanese chars")
        if mat in SPACING: issue.append("Spacing error")
        if mat in IMPOSSIBLE: issue.append("Impossible")
        issue_str = ", ".join(issue)
        report_lines.append(f"| `{mat}` | {count} | {issue_str} |")
report_lines.append("\n")

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "=" * 80)
print("SUMMARY OF ISSUES")
print("=" * 80)
print()

report_lines.append("\n## Summary\n")

total_issues = (
    len(mismatches) +
    len(orphans) +
    len(gender_issues) +
    len(material_issues) +
    len(naming_issues['duplicate_categories']) +
    sum(weight_anomalies.values()) +
    sum(distance_anomalies.values()) +
    total_material_errors
)

print(f"Total Issues Found: {total_issues:,}")
print(f"  - Category-Parent Mismatches: {len(mismatches)}")
print(f"  - Orphaned Categories: {len(orphans)}")
print(f"  - Gender Logic Issues: {len(gender_issues)}")
print(f"  - Material-Category Issues: {len(material_issues)}")
print(f"  - Naming Inconsistencies: {len(naming_issues['duplicate_categories'])}")
print(f"  - Weight Anomalies: {sum(weight_anomalies.values())}")
print(f"  - Distance Anomalies: {sum(distance_anomalies.values())}")
print(f"  - Material Errors: {total_material_errors}")

report_lines.append(f"**Total Issues Found**: {total_issues:,}\n")
report_lines.append("| Issue Type | Count |")
report_lines.append("|------------|-------|")
report_lines.append(f"| Category-Parent Mismatches | {len(mismatches)} |")
report_lines.append(f"| Orphaned Categories | {len(orphans)} |")
report_lines.append(f"| Gender Logic Issues | {len(gender_issues)} |")
report_lines.append(f"| Material-Category Issues | {len(material_issues)} |")
report_lines.append(f"| Naming Inconsistencies | {len(naming_issues['duplicate_categories'])} |")
report_lines.append(f"| Weight Anomalies | {sum(weight_anomalies.values())} |")
report_lines.append(f"| Distance Anomalies | {sum(distance_anomalies.values())} |")
report_lines.append(f"| Material Errors | {total_material_errors} |")
report_lines.append(f"| **TOTAL** | **{total_issues:,}** |")
report_lines.append("\n")

# Save report
print(f"\n💾 Saving detailed report to: {REPORT_FILE}")
with open(REPORT_FILE, 'w') as f:
    f.write('\n'.join(report_lines))
print("✓ Saved!\n")

print("=" * 80)
print("✓ DETAILED ANALYSIS COMPLETE")
print("=" * 80)
print(f"\nDetailed report: {REPORT_FILE}")
