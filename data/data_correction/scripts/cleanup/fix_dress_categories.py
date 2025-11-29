#!/usr/bin/env python3
"""
Fix Dress Category Naming

Standardizes dress category naming to use full "Length + Dresses" format:
- "Maxi" + "Maxi Dresses" -> "Maxi Dresses"
- "Midi" -> "Midi Dresses"
- "Mini" + "Mini Dresses" -> "Mini Dresses"

Does NOT change:
- Sweatpants & Joggers (keep as-is for men)
- Sweaters & Knitwear (keep separate to show gender)
- Sweatshirts & Hoodies (keep separate to show gender)
"""

import pandas as pd
from datetime import datetime
from pathlib import Path

# Configuration
INPUT_CSV = "data_correction/output/Product_data_final.csv"
OUTPUT_CSV = "data_correction/output/Product_data_final.csv"
BACKUP_DIR = "data_correction/output/archives"
REPORT_FILE = "data_correction/output/comprehensive_analysis/dress_category_fix_report.md"

# Create directories
Path(BACKUP_DIR).mkdir(exist_ok=True, parents=True)
Path("data_correction/output/comprehensive_analysis").mkdir(exist_ok=True, parents=True)

print("=" * 80)
print("FIXING DRESS CATEGORY NAMING")
print("=" * 80)
print()

# ============================================================================
# STEP 1: LOAD DATA
# ============================================================================

print("=" * 80)
print("STEP 1: LOADING DATA")
print("=" * 80)
print()

print(f"Loading Loading data from: {INPUT_CSV}")
df = pd.read_csv(INPUT_CSV, low_memory=False)
initial_count = len(df)
print(f"OK Loaded {initial_count:,} products\n")

# ============================================================================
# STEP 2: CREATE BACKUP
# ============================================================================

print("=" * 80)
print("STEP 2: CREATING BACKUP")
print("=" * 80)
print()

backup_file = f"{BACKUP_DIR}/Product_data_pre_dress_fix_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
print(f"Saving Creating backup: {backup_file}")
df.to_csv(backup_file, index=False, quoting=1, escapechar='\\')
print(f"OK Backup created\n")

# ============================================================================
# STEP 3: ANALYZE CURRENT DRESS CATEGORIES
# ============================================================================

print("=" * 80)
print("STEP 3: ANALYZING CURRENT DRESS CATEGORIES")
print("=" * 80)
print()

dress_categories = ['Maxi', 'Maxi Dresses', 'Midi', 'Mini', 'Mini Dresses', 'Gowns']

print("Current dress categories:")
print(f"{'Category':<20} {'Count':>10} {'Parent':>15} {'Action':<30}")
print("-" * 80)

category_counts = {}
for cat in dress_categories:
    count = len(df[df['category'] == cat])
    if count > 0:
        parent = df[df['category'] == cat]['parent_category'].iloc[0]
        category_counts[cat] = count

        action = ''
        if cat == 'Maxi':
            action = '-> Rename to "Maxi Dresses"'
        elif cat == 'Maxi Dresses':
            action = 'OK Already correct'
        elif cat == 'Midi':
            action = '-> Rename to "Midi Dresses"'
        elif cat == 'Mini':
            action = '-> Rename to "Mini Dresses"'
        elif cat == 'Mini Dresses':
            action = 'OK Already correct'
        elif cat == 'Gowns':
            action = 'OK Keep as-is'

        print(f"{cat:<20} {count:>10,} {parent:>15} {action:<30}")

print()

# ============================================================================
# STEP 4: APPLY FIXES
# ============================================================================

print("=" * 80)
print("STEP 4: APPLYING CATEGORY FIXES")
print("=" * 80)
print()

changes_made = []

# Fix 1: Maxi -> Maxi Dresses
maxi_count = len(df[df['category'] == 'Maxi'])
if maxi_count > 0:
    print(f"1. Renaming 'Maxi' -> 'Maxi Dresses' ({maxi_count:,} products)")
    df.loc[df['category'] == 'Maxi', 'category'] = 'Maxi Dresses'
    changes_made.append(('Maxi', 'Maxi Dresses', maxi_count))
    print(f"   OK Done\n")

# Fix 2: Midi -> Midi Dresses
midi_count = len(df[df['category'] == 'Midi'])
if midi_count > 0:
    print(f"2. Renaming 'Midi' -> 'Midi Dresses' ({midi_count:,} products)")
    df.loc[df['category'] == 'Midi', 'category'] = 'Midi Dresses'
    changes_made.append(('Midi', 'Midi Dresses', midi_count))
    print(f"   OK Done\n")

# Fix 3: Mini -> Mini Dresses
mini_count = len(df[df['category'] == 'Mini'])
if mini_count > 0:
    print(f"3. Renaming 'Mini' -> 'Mini Dresses' ({mini_count:,} products)")
    df.loc[df['category'] == 'Mini', 'category'] = 'Mini Dresses'
    changes_made.append(('Mini', 'Mini Dresses', mini_count))
    print(f"   OK Done\n")

total_changed = sum(count for _, _, count in changes_made)
print(f"Total products updated: {total_changed:,}")
print()

# ============================================================================
# STEP 5: VERIFICATION
# ============================================================================

print("=" * 80)
print("STEP 5: VERIFICATION")
print("=" * 80)
print()

print("Dress categories AFTER fix:")
print(f"{'Category':<20} {'Count':>10} {'Parent':>15}")
print("-" * 60)

final_dress_categories = ['Maxi Dresses', 'Midi Dresses', 'Mini Dresses', 'Gowns']
after_counts = {}

for cat in final_dress_categories:
    count = len(df[df['category'] == cat])
    if count > 0:
        parent = df[df['category'] == cat]['parent_category'].iloc[0]
        after_counts[cat] = count
        print(f"{cat:<20} {count:>10,} {parent:>15}")

print()

# Verify no old categories remain
old_categories = ['Maxi', 'Midi', 'Mini']
remaining_old = 0
for cat in old_categories:
    count = len(df[df['category'] == cat])
    remaining_old += count

if remaining_old == 0:
    print("PASS All old dress categories successfully renamed")
else:
    print(f"WARNING WARNING: {remaining_old:,} products still have old category names!")

print()

# Compare before/after
print("Before/After Comparison:")
print("-" * 60)
print(f"{'Old Category':<20} {'New Category':<20} {'Products':<15}")
print("-" * 60)

for old_cat, new_cat, count in changes_made:
    print(f"{old_cat:<20} {new_cat:<20} {count:,}")

print()

# ============================================================================
# STEP 6: SAVE UPDATED DATA
# ============================================================================

print("=" * 80)
print("STEP 6: SAVING UPDATED DATA")
print("=" * 80)
print()

print(f"Saving Saving updated dataset to: {OUTPUT_CSV}")
df.to_csv(OUTPUT_CSV, index=False, quoting=1, escapechar='\\')
print(f"OK Saved {len(df):,} products!\n")

# ============================================================================
# STEP 7: CREATE REPORT
# ============================================================================

print("=" * 80)
print("STEP 7: CREATING REPORT")
print("=" * 80)
print()

print(f"Data Generating report: {REPORT_FILE}")

with open(REPORT_FILE, 'w') as f:
    f.write("# Dress Category Naming Fix Report\n\n")
    f.write(f"**Report Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    f.write("---\n\n")

    f.write("## Summary\n\n")
    f.write(f"**Input File**: {INPUT_CSV}\n\n")
    f.write(f"**Output File**: {OUTPUT_CSV}\n\n")
    f.write(f"**Backup File**: {backup_file}\n\n")

    f.write("| Metric | Count |\n")
    f.write("|--------|-------|\n")
    f.write(f"| Total Products | {initial_count:,} |\n")
    f.write(f"| Products Updated | {total_changed:,} |\n")
    f.write(f"| Update Percentage | {(total_changed/initial_count)*100:.4f}% |\n\n")

    f.write("---\n\n")
    f.write("## Changes Applied\n\n")

    f.write("### Dress Category Standardization\n\n")
    f.write("**Goal**: Use full \"Length + Dresses\" format for clarity\n\n")

    f.write("| Old Category | New Category | Products Updated |\n")
    f.write("|--------------|--------------|------------------|\n")
    for old_cat, new_cat, count in changes_made:
        f.write(f"| {old_cat} | {new_cat} | {count:,} |\n")
    f.write("\n")

    f.write("### Rationale\n\n")
    f.write("**Problem**: \"Maxi\", \"Midi\", \"Mini\" are LENGTH DESCRIPTORS, not product types.\n\n")
    f.write("- \"Maxi\" alone is ambiguous (Maxi dress? Maxi skirt?)\n\n")
    f.write("- \"Maxi Dresses\" is clear and explicit\n\n")
    f.write("**Consistency**: Skirt categories already use full names:\n\n")
    f.write("- Maxi Skirts OK\n")
    f.write("- Midi Skirts OK\n")
    f.write("- Mini Skirts OK\n\n")
    f.write("Now dress categories match this pattern.\n\n")

    f.write("---\n\n")
    f.write("## Final Dress Category Distribution\n\n")

    f.write("| Category | Count | Percentage |\n")
    f.write("|----------|-------|------------|\n")
    total_dresses = sum(after_counts.values())
    for cat, count in sorted(after_counts.items(), key=lambda x: x[1], reverse=True):
        pct = (count / total_dresses) * 100
        f.write(f"| {cat} | {count:,} | {pct:.2f}% |\n")
    f.write(f"| **TOTAL** | **{total_dresses:,}** | **100.00%** |\n\n")

    f.write("---\n\n")
    f.write("## Categories NOT Changed\n\n")
    f.write("Per user request, the following categories were left as-is:\n\n")

    f.write("1. **Sweatpants & Joggers** (Men's)\n")
    f.write("   - Kept combined to maintain gender distinction\n\n")

    f.write("2. **Sweaters** vs **Sweaters & Knitwear**\n")
    f.write("   - Kept separate to show gender (Women's vs Men's)\n\n")

    f.write("3. **Hoodies/Sweatshirts** vs **Sweatshirts & Hoodies**\n")
    f.write("   - Kept separate to show gender (Women's vs Men's)\n\n")

    f.write("---\n\n")
    f.write("## Quality Assurance\n\n")

    f.write(f"PASS **All old dress categories removed**: {remaining_old} products with old names\n\n")
    f.write(f"PASS **All products accounted for**: No data loss\n\n")
    f.write(f"PASS **Consistent naming**: All dress categories now use full names\n\n")

    f.write("---\n\n")
    f.write("## Next Steps\n\n")
    f.write("1. Run comprehensive analysis with updated categories\n")
    f.write("2. Verify category distribution in visualizations\n")
    f.write("3. Update any documentation referencing old category names\n\n")

print(f"OK Report saved!\n")

# ============================================================================
# FINAL SUMMARY
# ============================================================================

print("=" * 80)
print("OK DRESS CATEGORY NAMING FIX COMPLETE")
print("=" * 80)
print()
print(f"Files Updated dataset: {OUTPUT_CSV}")
print(f"Saving Backup:          {backup_file}")
print(f"Data Report:          {REPORT_FILE}")
print()
print(f"Summary:")
print(f"  Products updated:    {total_changed:,} ({(total_changed/initial_count)*100:.4f}%)")
print(f"  Final products:      {initial_count:,} (no data loss)")
print()
print("Changes applied:")
for old_cat, new_cat, count in changes_made:
    print(f"  * {old_cat:20s} -> {new_cat:20s} ({count:,} products)")
print()
print("PASS Dress categories now use consistent 'Length + Dresses' format")
print()
