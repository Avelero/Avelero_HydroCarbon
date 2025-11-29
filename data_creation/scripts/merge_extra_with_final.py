#!/usr/bin/env python3
"""
Merge Extra Products with Final Dataset

Combines Product_data_final.csv with Product_data_extra.csv and validates
material distribution to ensure all materials reach ~35,000 products.

Steps:
1. Load Product_data_final.csv (901,573 products)
2. Load Product_data_extra.csv (~628,000 products)
3. Merge both datasets
4. Validate material distribution
5. Save as Product_data_complete_v2.csv (~1,529,000 products)
6. Run comprehensive analysis
"""

import pandas as pd
import json
from collections import Counter
from datetime import datetime
from pathlib import Path

# ============================================================================
# CONFIGURATION
# ============================================================================

FINAL_CSV = "../data_correction/output/Product_data_final.csv"
EXTRA_CSV = "output/Product_data_extra.csv"
OUTPUT_CSV = "../data_correction/output/Product_data_complete_v2.csv"
BACKUP_DIR = "../data_correction/output/archives"
REPORT_FILE = "../data_correction/output/comprehensive_analysis/merge_report.md"

TARGET_PER_MATERIAL = 35000
TOLERANCE = 0.02  # 2% tolerance (35,000 ± 700)

# Create directories
Path(BACKUP_DIR).mkdir(exist_ok=True, parents=True)
Path("../data_correction/output/comprehensive_analysis").mkdir(exist_ok=True, parents=True)

# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    print("=" * 90)
    print("MERGE EXTRA PRODUCTS WITH FINAL DATASET")
    print("=" * 90)
    print()
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # ========================================================================
    # STEP 1: LOAD DATASETS
    # ========================================================================

    print("=" * 90)
    print("STEP 1: LOADING DATASETS")
    print("=" * 90)
    print()

    print(f"📂 Loading final dataset: {FINAL_CSV}")
    df_final = pd.read_csv(FINAL_CSV, low_memory=False)
    print(f"✓ Loaded {len(df_final):,} products")
    print()

    print(f"📂 Loading extra dataset: {EXTRA_CSV}")
    df_extra = pd.read_csv(EXTRA_CSV, low_memory=False)
    print(f"✓ Loaded {len(df_extra):,} products")
    print()

    # ========================================================================
    # STEP 2: MERGE DATASETS
    # ========================================================================

    print("=" * 90)
    print("STEP 2: MERGING DATASETS")
    print("=" * 90)
    print()

    print("Concatenating datasets...")
    df_merged = pd.concat([df_final, df_extra], ignore_index=True)
    print(f"✓ Merged dataset: {len(df_merged):,} products")
    print()

    # Check for duplicates (shouldn't have any, but verify)
    print("Checking for duplicates...")
    duplicates = df_merged.duplicated().sum()
    if duplicates > 0:
        print(f"⚠️  WARNING: Found {duplicates:,} duplicate rows")
        print("Removing duplicates...")
        df_merged = df_merged.drop_duplicates()
        print(f"✓ After deduplication: {len(df_merged):,} products")
    else:
        print(f"✓ No duplicates found")
    print()

    # ========================================================================
    # STEP 3: VALIDATE MATERIAL DISTRIBUTION
    # ========================================================================

    print("=" * 90)
    print("STEP 3: VALIDATING MATERIAL DISTRIBUTION")
    print("=" * 90)
    print()

    print("Parsing materials from all products...")
    material_counts = Counter()

    for idx, row in df_merged.iterrows():
        try:
            mat_json = row['materials']
            if isinstance(mat_json, str):
                materials = json.loads(mat_json.replace("'", '"'))
            else:
                materials = mat_json

            if isinstance(materials, dict):
                for mat in materials.keys():
                    material_counts[mat] += 1
        except:
            pass

    print(f"✓ Parsed materials from {len(df_merged):,} products")
    print()

    # Check material targets
    print(f"Material Distribution (Target: {TARGET_PER_MATERIAL:,} ± {int(TARGET_PER_MATERIAL * TOLERANCE):,}):")
    print(f"{'Material':<30} {'Count':>12} {'Target':>12} {'Diff':>12} {'Status':<10}")
    print("-" * 90)

    below_target = []
    above_tolerance = []
    success_count = 0

    for mat, count in sorted(material_counts.items(), key=lambda x: x[1]):
        diff = count - TARGET_PER_MATERIAL
        diff_pct = (diff / TARGET_PER_MATERIAL) * 100
        tolerance_limit = TARGET_PER_MATERIAL * TOLERANCE

        if count < TARGET_PER_MATERIAL - tolerance_limit:
            status = f"❌ -{abs(diff_pct):.1f}%"
            below_target.append((mat, count, diff))
        elif count > TARGET_PER_MATERIAL + tolerance_limit:
            status = f"⚠️  +{diff_pct:.1f}%"
            above_tolerance.append((mat, count, diff))
        else:
            status = "✅ OK"
            success_count += 1

        print(f"{mat:<30} {count:>12,} {TARGET_PER_MATERIAL:>12,} {diff:>+12,} {status:<10}")

    print("-" * 90)
    print(f"Total materials: {len(material_counts)}")
    print(f"  ✅ Within target: {success_count}")
    print(f"  ❌ Below target: {len(below_target)}")
    print(f"  ⚠️  Above tolerance: {len(above_tolerance)}")
    print()

    # ========================================================================
    # STEP 4: VALIDATE OTHER DISTRIBUTIONS
    # ========================================================================

    print("=" * 90)
    print("STEP 4: VALIDATING OTHER DISTRIBUTIONS")
    print("=" * 90)
    print()

    # Gender distribution
    gender_dist = df_merged['gender'].value_counts()
    print("Gender Distribution:")
    for gender, count in gender_dist.items():
        pct = (count / len(df_merged)) * 100
        print(f"  {gender}: {count:,} ({pct:.2f}%)")
    print()

    # Parent category distribution
    parent_dist = df_merged['parent_category'].value_counts()
    print("Parent Category Distribution:")
    for parent, count in parent_dist.items():
        pct = (count / len(df_merged)) * 100
        print(f"  {parent}: {count:,} ({pct:.2f}%)")
    print()

    # Category count
    category_count = df_merged['category'].nunique()
    print(f"Total Categories: {category_count}")
    print()

    # ========================================================================
    # STEP 5: CREATE BACKUP
    # ========================================================================

    print("=" * 90)
    print("STEP 5: CREATING BACKUP")
    print("=" * 90)
    print()

    backup_file = f"{BACKUP_DIR}/Product_data_final_pre_merge_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    print(f"💾 Backing up original final dataset to: {backup_file}")
    df_final.to_csv(backup_file, index=False, quoting=1, escapechar='\\')
    print(f"✓ Backup created")
    print()

    # ========================================================================
    # STEP 6: SAVE MERGED DATASET
    # ========================================================================

    print("=" * 90)
    print("STEP 6: SAVING MERGED DATASET")
    print("=" * 90)
    print()

    print(f"💾 Saving merged dataset to: {OUTPUT_CSV}")
    df_merged.to_csv(OUTPUT_CSV, index=False, quoting=1, escapechar='\\')
    print(f"✓ Saved {len(df_merged):,} products")
    print()

    # ========================================================================
    # STEP 7: CREATE REPORT
    # ========================================================================

    print("=" * 90)
    print("STEP 7: CREATING MERGE REPORT")
    print("=" * 90)
    print()

    print(f"📊 Generating report: {REPORT_FILE}")

    with open(REPORT_FILE, 'w') as f:
        f.write("# Extra Products Merge Report\n\n")
        f.write(f"**Merge Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\\n\\n")
        f.write("---\\n\\n")

        f.write("## Summary\n\n")
        f.write(f"**Final Dataset**: {len(df_final):,} products\\n\\n")
        f.write(f"**Extra Dataset**: {len(df_extra):,} products\\n\\n")
        f.write(f"**Merged Dataset**: {len(df_merged):,} products\\n\\n")
        f.write(f"**Duplicates Removed**: {duplicates:,}\\n\\n")
        f.write("---\\n\\n")

        f.write("## Material Distribution\n\n")
        f.write(f"**Target per Material**: {TARGET_PER_MATERIAL:,}\\n\\n")
        f.write(f"**Tolerance**: ±{int(TARGET_PER_MATERIAL * TOLERANCE):,} ({TOLERANCE * 100:.0f}%)\\n\\n")

        f.write("| Material | Count | Target | Difference | Status |\\n")
        f.write("|----------|-------|--------|------------|--------|\\n")

        for mat, count in sorted(material_counts.items()):
            diff = count - TARGET_PER_MATERIAL
            diff_pct = (diff / TARGET_PER_MATERIAL) * 100
            tolerance_limit = TARGET_PER_MATERIAL * TOLERANCE

            if count < TARGET_PER_MATERIAL - tolerance_limit:
                status = f"❌ Below ({diff_pct:+.1f}%)"
            elif count > TARGET_PER_MATERIAL + tolerance_limit:
                status = f"⚠️ Above ({diff_pct:+.1f}%)"
            else:
                status = "✅ OK"

            f.write(f"| {mat} | {count:,} | {TARGET_PER_MATERIAL:,} | {diff:+,} | {status} |\\n")

        f.write("\\n### Statistics\\n\\n")
        f.write(f"- Total materials: {len(material_counts)}\\n")
        f.write(f"- Within target: {success_count}\\n")
        f.write(f"- Below target: {len(below_target)}\\n")
        f.write(f"- Above tolerance: {len(above_tolerance)}\\n\\n")

        if below_target:
            f.write("### Materials Below Target\\n\\n")
            for mat, count, diff in below_target:
                f.write(f"- **{mat}**: {count:,} (need {abs(diff):,} more)\\n")
            f.write("\\n")

        f.write("---\\n\\n")

        f.write("## Distribution Validation\n\n")
        f.write("### Gender Distribution\\n\\n")
        for gender, count in gender_dist.items():
            pct = (count / len(df_merged)) * 100
            f.write(f"- {gender}: {count:,} ({pct:.2f}%)\\n")

        f.write("\\n### Parent Category Distribution\\n\\n")
        for parent, count in parent_dist.items():
            pct = (count / len(df_merged)) * 100
            f.write(f"- {parent}: {count:,} ({pct:.2f}%)\\n")

        f.write(f"\\n### Categories\\n\\n")
        f.write(f"Total categories: {category_count}\\n\\n")

        f.write("---\\n\\n")
        f.write("## Next Steps\\n\\n")
        f.write("1. Run comprehensive analysis on Product_data_complete_v2.csv\\n")
        f.write("2. Generate visualizations\\n")
        f.write("3. Validate LCA compatibility\\n")

    print(f"✓ Report saved")
    print()

    # ========================================================================
    # FINAL SUMMARY
    # ========================================================================

    print("=" * 90)
    print("✓ MERGE COMPLETE")
    print("=" * 90)
    print()
    print(f"📁 Final dataset: {OUTPUT_CSV}")
    print(f"📊 Report: {REPORT_FILE}")
    print()
    print(f"Summary:")
    print(f"  Original products: {len(df_final):,}")
    print(f"  Extra products: {len(df_extra):,}")
    print(f"  Merged products: {len(df_merged):,}")
    print(f"  Total materials: {len(material_counts)}")
    print(f"  Materials at target: {success_count}/{len(material_counts)}")
    print()

    if len(below_target) > 0:
        print("⚠️  Some materials are still below target")
        print("Consider running additional generation for these materials:")
        for mat, count, diff in below_target[:5]:
            print(f"  • {mat}: need {abs(diff):,} more products")
        print()
    else:
        print("✅ All materials meet or exceed target!")
        print()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\\n\\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
