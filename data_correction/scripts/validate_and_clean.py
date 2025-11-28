#!/usr/bin/env python3
"""
Data Validation and Cleaning Script
Validates the raw product dataset and removes incomplete rows.
"""

import os
import sys
import pandas as pd
from pathlib import Path
from datetime import datetime

# Configuration
INPUT_CSV = "input/Product_data.csv"
OUTPUT_DIR = "output"
OUTPUT_CSV = "output/Product_data_cleaned.csv"
VALIDATION_REPORT = "output/validation_report.txt"

# Expected columns
EXPECTED_COLUMNS = [
    'product_name',
    'gender',
    'parent_category',
    'category',
    'manufacturer_country',
    'materials',
    'weight_kg',
    'total_distance_km'
]


def validate_and_clean():
    """Main validation and cleaning function"""
    
    print("=" * 80)
    print("DATA VALIDATION AND CLEANING")
    print("=" * 80)
    print()
    
    # Check input file exists
    if not os.path.exists(INPUT_CSV):
        print(f"❌ Error: Input file not found: {INPUT_CSV}")
        sys.exit(1)
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load data
    print(f"📂 Loading data from: {INPUT_CSV}")
    try:
        df = pd.read_csv(INPUT_CSV, on_bad_lines='warn')
        print(f"✓ Loaded {len(df):,} rows")
    except Exception as e:
        print(f"❌ Error loading CSV: {e}")
        sys.exit(1)
    
    print()
    
    # Validate columns
    print("🔍 Validating columns...")
    if list(df.columns) != EXPECTED_COLUMNS:
        print(f"❌ Column mismatch!")
        print(f"   Expected: {EXPECTED_COLUMNS}")
        print(f"   Found: {list(df.columns)}")
        sys.exit(1)
    print(f"✓ All {len(EXPECTED_COLUMNS)} columns present and correct")
    print()
    
    # Check for null values
    print("🔍 Checking for incomplete rows...")
    null_counts = df.isnull().sum()
    total_nulls = null_counts.sum()
    
    print("\nNull values per column:")
    for col in EXPECTED_COLUMNS:
        count = null_counts[col]
        if count > 0:
            print(f"   ❌ {col:25s}: {count:,} null values")
        else:
            print(f"   ✓ {col:25s}: complete")
    
    # Identify incomplete rows
    incomplete_mask = df.isnull().any(axis=1)
    incomplete_rows = df[incomplete_mask]
    num_incomplete = len(incomplete_rows)
    
    print()
    if num_incomplete > 0:
        print(f"⚠️  Found {num_incomplete:,} incomplete rows ({num_incomplete/len(df)*100:.2f}%)")
    else:
        print(f"✓ No incomplete rows found!")
    
    # Show sample of incomplete rows
    if num_incomplete > 0:
        print("\n📋 Sample of incomplete rows (first 10):")
        print("-" * 80)
        for idx, row in incomplete_rows.head(10).iterrows():
            missing_cols = [col for col in EXPECTED_COLUMNS if pd.isnull(row[col])]
            print(f"   Row {idx:,}: Missing {missing_cols}")
            print(f"      Product: {row['product_name']}")
        if num_incomplete > 10:
            print(f"   ... and {num_incomplete - 10:,} more")
    
    # Clean data - remove incomplete rows
    print()
    print("🧹 Cleaning data...")
    df_cleaned = df.dropna()
    num_cleaned = len(df_cleaned)
    num_removed = len(df) - num_cleaned
    
    print(f"   Original rows: {len(df):,}")
    print(f"   Removed rows:  {num_removed:,} ({num_removed/len(df)*100:.2f}%)")
    print(f"   Final rows:    {num_cleaned:,} ({num_cleaned/len(df)*100:.2f}%)")
    
    # Verify cleaned data has no nulls
    assert df_cleaned.isnull().sum().sum() == 0, "Cleaned data still has null values!"
    print(f"✓ Verified: No null values in cleaned data")
    
    # Save cleaned data
    print()
    print(f"💾 Saving cleaned data to: {OUTPUT_CSV}")
    df_cleaned.to_csv(OUTPUT_CSV, index=False)
    print(f"✓ Saved {num_cleaned:,} rows")
    
    # Generate validation report
    print()
    print(f"📝 Generating validation report: {VALIDATION_REPORT}")
    
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("DATA VALIDATION AND CLEANING REPORT")
    report_lines.append("=" * 80)
    report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("")
    report_lines.append(f"Input File:  {INPUT_CSV}")
    report_lines.append(f"Output File: {OUTPUT_CSV}")
    report_lines.append("")
    report_lines.append("-" * 80)
    report_lines.append("SUMMARY")
    report_lines.append("-" * 80)
    report_lines.append(f"Original rows:     {len(df):,}")
    report_lines.append(f"Incomplete rows:   {num_incomplete:,} ({num_incomplete/len(df)*100:.2f}%)")
    report_lines.append(f"Removed rows:      {num_removed:,} ({num_removed/len(df)*100:.2f}%)")
    report_lines.append(f"Final rows:        {num_cleaned:,} ({num_cleaned/len(df)*100:.2f}%)")
    report_lines.append("")
    report_lines.append("-" * 80)
    report_lines.append("NULL VALUES BY COLUMN")
    report_lines.append("-" * 80)
    for col in EXPECTED_COLUMNS:
        count = null_counts[col]
        report_lines.append(f"{col:30s}: {count:,} null values")
    report_lines.append("")
    
    if num_incomplete > 0:
        report_lines.append("-" * 80)
        report_lines.append("SAMPLE OF REMOVED ROWS (First 20)")
        report_lines.append("-" * 80)
        for idx, row in incomplete_rows.head(20).iterrows():
            missing_cols = [col for col in EXPECTED_COLUMNS if pd.isnull(row[col])]
            report_lines.append(f"Row {idx:,}:")
            report_lines.append(f"  Product: {row['product_name']}")
            report_lines.append(f"  Missing: {', '.join(missing_cols)}")
            report_lines.append("")
    
    report_lines.append("-" * 80)
    report_lines.append("VALIDATION RESULT: PASS ✓")
    report_lines.append("-" * 80)
    report_lines.append(f"Cleaned dataset saved to: {OUTPUT_CSV}")
    report_lines.append(f"Final dataset contains {num_cleaned:,} complete rows")
    report_lines.append("=" * 80)
    
    report_text = "\n".join(report_lines)
    
    with open(VALIDATION_REPORT, 'w') as f:
        f.write(report_text)
    
    print(f"✓ Report saved")
    
    # Final summary
    print()
    print("=" * 80)
    print("✓ VALIDATION AND CLEANING COMPLETE")
    print("=" * 80)
    print()
    print(f"📊 Results:")
    print(f"   Original dataset:  {len(df):,} rows")
    print(f"   Cleaned dataset:   {num_cleaned:,} rows")
    print(f"   Data quality:      {num_cleaned/len(df)*100:.2f}% complete")
    print()
    print(f"📁 Output files:")
    print(f"   Cleaned CSV:  {OUTPUT_CSV}")
    print(f"   Report:       {VALIDATION_REPORT}")
    print()
    
    return df_cleaned, num_removed, num_cleaned


if __name__ == "__main__":
    try:
        validate_and_clean()
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
