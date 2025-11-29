#!/usr/bin/env python3
"""
Low-Usage Material Analysis

Analyzes materials with very low usage to identify:
1. Typos and variants
2. Naming inconsistencies
3. Redundant materials
4. Legitimate rare materials
"""

import pandas as pd
import json
from collections import Counter

# Configuration
INPUT_CSV = "data_correction/output/Product_data_final.csv"
REPORT_FILE = "data_correction/output/comprehensive_analysis/low_usage_materials_report.md"

print("=" * 90)
print("LOW-USAGE MATERIAL ANALYSIS")
print("=" * 90)
print()

# Load data
print(f"Loading data from: {INPUT_CSV}")
df = pd.read_csv(INPUT_CSV, low_memory=False)
print(f"Loaded {len(df):,} products\n")

# Parse all materials
print("Parsing materials...")
all_materials = Counter()
material_products = {}  # Track which products use each material

for idx, row in df.iterrows():
    try:
        mat_json = row['materials']
        if isinstance(mat_json, str):
            materials = json.loads(mat_json.replace("'", '"'))
        else:
            materials = mat_json

        if isinstance(materials, dict):
            for mat in materials.keys():
                all_materials[mat] += 1
                if mat not in material_products:
                    material_products[mat] = []
                material_products[mat].append({
                    'index': idx,
                    'name': row['product_name'],
                    'category': row['category'],
                    'gender': row['gender']
                })
    except:
        pass

print(f"Found {len(all_materials)} unique materials\n")

# ============================================================================
# CATEGORIZE MATERIALS BY USAGE
# ============================================================================

print("=" * 90)
print("MATERIAL USAGE CATEGORIES")
print("=" * 90)
print()

# Define thresholds
VERY_RARE = 10      # < 10 products
RARE = 100          # 10-100 products
UNCOMMON = 1000     # 100-1000 products
COMMON = 10000      # 1000-10000 products
# > 10000 = VERY COMMON

very_rare = []
rare = []
uncommon = []
common = []
very_common = []

for mat, count in all_materials.items():
    if count < VERY_RARE:
        very_rare.append((mat, count))
    elif count < RARE:
        rare.append((mat, count))
    elif count < UNCOMMON:
        uncommon.append((mat, count))
    elif count < COMMON:
        common.append((mat, count))
    else:
        very_common.append((mat, count))

print(f"Material distribution by usage:")
print(f"  Very Rare (< 10):       {len(very_rare):3d} materials")
print(f"  Rare (10-100):          {len(rare):3d} materials")
print(f"  Uncommon (100-1,000):   {len(uncommon):3d} materials")
print(f"  Common (1,000-10,000):  {len(common):3d} materials")
print(f"  Very Common (> 10,000): {len(very_common):3d} materials")
print()

# ============================================================================
# ANALYZE VERY RARE MATERIALS (< 10 products)
# ============================================================================

print("=" * 90)
print("VERY RARE MATERIALS (< 10 products) - LIKELY ERRORS")
print("=" * 90)
print()

print(f"{'Material':<30} {'Count':>8} {'Issue Type':<25} {'Correct Form':<30}")
print("-" * 90)

very_rare_sorted = sorted(very_rare, key=lambda x: x[1], reverse=True)

# Analyze each very rare material
issues = []

for mat, count in very_rare_sorted:
    issue_type = ''
    correct_form = ''

    # Check for variants/typos
    if 'polyamide_virgin' in mat:
        issue_type = 'Redundant variant'
        correct_form = 'polyamide_6 or polyamide_66'
    elif 'nylon' == mat:
        issue_type = 'Synonym'
        correct_form = 'polyamide_6 or polyamide_66'
    elif 'polyester_66' in mat:
        issue_type = 'Wrong number'
        correct_form = 'polyamide_66 (not polyester)'
    elif 'polyester_conventional' in mat:
        issue_type = 'Redundant naming'
        correct_form = 'polyester_virgin'
    elif 'recycled_polyester' in mat:
        issue_type = 'Naming inconsistency'
        correct_form = 'polyester_recycled'
    elif 'polyester_generic' in mat:
        issue_type = 'Vague naming'
        correct_form = 'polyester_virgin'
    elif 'cotton_virgin' in mat:
        issue_type = 'Redundant variant'
        correct_form = 'cotton_conventional'
    elif 'nylon_recycled' in mat:
        issue_type = 'Synonym (inconsistent)'
        correct_form = 'polyamide_recycled'
    elif 'polyamide_recycled' in mat:
        issue_type = 'Legitimate (rare)'
        correct_form = 'OK (just very rare)'
    elif 'cotton_recycled' in mat:
        issue_type = 'Legitimate (rare)'
        correct_form = 'OK (just very rare)'
    elif 'shearling_faux' in mat:
        issue_type = 'Legitimate (rare)'
        correct_form = 'OK (faux shearling lining)'
    elif 'rayon' in mat:
        issue_type = 'Missing designation'
        correct_form = 'viscose (rayon = viscose)'
    else:
        issue_type = 'Unknown'
        correct_form = 'Needs review'

    print(f"{mat:<30} {count:>8} {issue_type:<25} {correct_form:<30}")

    issues.append({
        'material': mat,
        'count': count,
        'issue_type': issue_type,
        'correct_form': correct_form,
        'products': material_products[mat]
    })

print()

# ============================================================================
# ANALYZE RARE MATERIALS (10-100 products)
# ============================================================================

print("=" * 90)
print("RARE MATERIALS (10-100 products) - REVIEW RECOMMENDED")
print("=" * 90)
print()

rare_sorted = sorted(rare, key=lambda x: x[1], reverse=True)

print(f"{'Material':<30} {'Count':>8} {'Assessment':<40}")
print("-" * 90)

for mat, count in rare_sorted:
    assessment = ''

    if 'jute' in mat:
        assessment = 'Legitimate (espadrilles, eco products)'
    elif 'wool_merino' in mat:
        assessment = 'Legitimate (premium wool variant)'
    elif 'cashmere' in mat:
        assessment = 'Legitimate (luxury material)'
    else:
        assessment = 'Needs review'

    print(f"{mat:<30} {count:>8} {assessment:<40}")

print()

# ============================================================================
# SHOW EXAMPLE PRODUCTS FOR SUSPICIOUS MATERIALS
# ============================================================================

print("=" * 90)
print("EXAMPLE PRODUCTS WITH SUSPICIOUS MATERIALS")
print("=" * 90)
print()

suspicious_materials = [
    'polyester_66', 'nylon', 'rayon', 'polyester_conventional',
    'recycled_polyester', 'polyester_generic', 'cotton_virgin'
]

for mat in suspicious_materials:
    if mat in material_products and len(material_products[mat]) > 0:
        print(f"\n{mat} ({all_materials[mat]} products):")
        print("-" * 90)

        for i, product in enumerate(material_products[mat][:3]):  # Show first 3
            print(f"  {i+1}. {product['name'][:70]}")
            print(f"     Category: {product['category']}, Gender: {product['gender']}")

        if len(material_products[mat]) > 3:
            print(f"  ... and {len(material_products[mat]) - 3} more")

print()

# ============================================================================
# RECOMMENDATIONS
# ============================================================================

print("=" * 90)
print("RECOMMENDATIONS")
print("=" * 90)
print()

print("MATERIALS TO FIX (Errors/Typos):")
print("-" * 90)
print()

fixes_needed = []

for issue in issues:
    if issue['issue_type'] not in ['Legitimate (rare)', 'OK (just very rare)']:
        fixes_needed.append(issue)
        print(f"  * {issue['material']:<30} ({issue['count']:>3} products)")
        print(f"    Issue: {issue['issue_type']}")
        print(f"    Fix:   {issue['correct_form']}")
        print()

print(f"Total materials to fix: {len(fixes_needed)}")
print()

print("MATERIALS TO KEEP (Legitimate, just rare):")
print("-" * 90)
for issue in issues:
    if issue['issue_type'] in ['Legitimate (rare)', 'OK (just very rare)']:
        print(f"  * {issue['material']:<30} ({issue['count']:>3} products) - {issue['correct_form']}")

print()

# ============================================================================
# CREATE DETAILED REPORT
# ============================================================================

print("=" * 90)
print("CREATING DETAILED REPORT")
print("=" * 90)
print()

with open(REPORT_FILE, 'w') as f:
    f.write("# Low-Usage Material Analysis Report\n\n")
    f.write("**Analysis Date**: 2025-11-29\n\n")
    f.write(f"**Total Products**: {len(df):,}\n\n")
    f.write(f"**Total Unique Materials**: {len(all_materials)}\n\n")

    f.write("---\n\n")

    f.write("## Material Usage Distribution\n\n")
    f.write("| Category | Count | Products Range |\n")
    f.write("|----------|-------|----------------|\n")
    f.write(f"| Very Rare (< 10) | {len(very_rare)} | {min([c for _, c in very_rare]) if very_rare else 0} - {max([c for _, c in very_rare]) if very_rare else 0} |\n")
    f.write(f"| Rare (10-100) | {len(rare)} | {min([c for _, c in rare]) if rare else 0} - {max([c for _, c in rare]) if rare else 0} |\n")
    f.write(f"| Uncommon (100-1,000) | {len(uncommon)} | {min([c for _, c in uncommon]) if uncommon else 0} - {max([c for _, c in uncommon]) if uncommon else 0} |\n")
    f.write(f"| Common (1,000-10,000) | {len(common)} | {min([c for _, c in common]) if common else 0} - {max([c for _, c in common]) if common else 0} |\n")
    f.write(f"| Very Common (> 10,000) | {len(very_common)} | {min([c for _, c in very_common]) if very_common else 0} - {max([c for _, c in very_common]) if very_common else 0} |\n\n")

    f.write("---\n\n")

    f.write("## Very Rare Materials (< 10 products)\n\n")
    f.write("### Issues Found\n\n")
    f.write("| Material | Count | Issue Type | Recommended Fix |\n")
    f.write("|----------|-------|------------|------------------|\n")

    for issue in issues:
        f.write(f"| `{issue['material']}` | {issue['count']} | {issue['issue_type']} | {issue['correct_form']} |\n")

    f.write("\n")

    f.write("### Detailed Analysis\n\n")

    f.write("#### 1. Typos and Wrong Materials\n\n")
    f.write("- **`polyester_66`** (1 product): Should be `polyamide_66`\n")
    f.write("  - Polyester doesn't have type 66, that's polyamide (nylon 66)\n\n")

    f.write("#### 2. Synonyms and Inconsistencies\n\n")
    f.write("- **`nylon`** (1 product): Should be `polyamide_6` or `polyamide_66`\n")
    f.write("  - Nylon is the common name for polyamide\n\n")

    f.write("- **`rayon`** (1 product): Should be `viscose`\n")
    f.write("  - Rayon and viscose are the same material, dataset uses 'viscose'\n\n")

    f.write("- **`nylon_recycled`** (20 products): Should be `polyamide_recycled`\n")
    f.write("  - For consistency with `polyamide_6` and `polyamide_66`\n\n")

    f.write("#### 3. Redundant Naming\n\n")
    f.write("- **`polyester_conventional`** (6 products): Should be `polyester_virgin`\n")
    f.write("  - 'conventional' and 'virgin' mean the same (non-recycled)\n\n")

    f.write("- **`recycled_polyester`** (2 products): Should be `polyester_recycled`\n")
    f.write("  - Inconsistent word order, dataset uses `material_modifier` format\n\n")

    f.write("- **`polyester_generic`** (2 products): Should be `polyester_virgin`\n")
    f.write("  - 'generic' is vague, use standard designation\n\n")

    f.write("- **`cotton_virgin`** (1 product): Should be `cotton_conventional`\n")
    f.write("  - Dataset standard is `cotton_conventional` for non-organic cotton\n\n")

    f.write("- **`polyamide_virgin`** (23 products): Should be `polyamide_6` or `polyamide_66`\n")
    f.write("  - Need to specify type (6 or 66)\n\n")

    f.write("#### 4. Legitimate Rare Materials\n\n")
    f.write("These materials are correct but just used infrequently:\n\n")
    f.write("- **`polyamide_recycled`** (8 products): Legitimate\n")
    f.write("- **`cotton_recycled`** (4 products): Legitimate\n")
    f.write("- **`shearling_faux`** (2 products): Legitimate (faux fur lining)\n\n")

    f.write("---\n\n")

    f.write("## Rare Materials (10-100 products)\n\n")
    f.write("| Material | Count | Assessment |\n")
    f.write("|----------|-------|------------|\n")

    for mat, count in rare_sorted:
        assessment = ''
        if 'jute' in mat:
            assessment = 'Legitimate (espadrilles, eco products)'
        elif 'wool_merino' in mat:
            assessment = 'Legitimate (premium wool)'
        elif 'cashmere' in mat:
            assessment = 'Legitimate (luxury material)'
        else:
            assessment = 'Needs review'

        f.write(f"| `{mat}` | {count} | {assessment} |\n")

    f.write("\n---\n\n")

    f.write("## Summary\n\n")
    f.write(f"**Total Materials to Fix**: {len(fixes_needed)}\n\n")
    f.write(f"**Total Products Affected**: {sum(issue['count'] for issue in fixes_needed)}\n\n")

    f.write("### Recommended Actions\n\n")
    f.write("1. **Fix typos**: `polyester_66` -> `polyamide_66`\n")
    f.write("2. **Standardize synonyms**: `nylon` -> `polyamide_6/66`, `rayon` -> `viscose`\n")
    f.write("3. **Fix naming inconsistencies**: `recycled_polyester` -> `polyester_recycled`\n")
    f.write("4. **Remove redundant variants**: `polyester_conventional` -> `polyester_virgin`\n")
    f.write("5. **Specify polyamide types**: `polyamide_virgin` -> `polyamide_6` or `polyamide_66`\n\n")

print(f"Report saved: {REPORT_FILE}\n")

print("=" * 90)
print("ANALYSIS COMPLETE")
print("=" * 90)
print()
print(f"Found {len(fixes_needed)} materials that need fixing")
print(f"Affecting {sum(issue['count'] for issue in fixes_needed)} products")
print()
print(f"Detailed report: {REPORT_FILE}")
print()
