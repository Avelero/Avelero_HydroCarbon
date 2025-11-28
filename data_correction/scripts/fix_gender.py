#!/usr/bin/env python3
"""
Fix Gender Labels and Remove Header Rows
Fixes gender inconsistencies and removes corrupted header rows from the cleaned dataset.
"""

import pandas as pd
import os

INPUT_CSV = "output/Product_data_cleaned.csv"
OUTPUT_CSV = "output/Product_data_cleaned.csv"
BACKUP_CSV = "output/Product_data_cleaned_backup.csv"

print("=" * 80)
print("FIXING GENDER LABELS AND REMOVING CORRUPTED ROWS")
print("=" * 80)
print()

# Create backup
print(f"📋 Creating backup: {BACKUP_CSV}")
os.system(f"cp {INPUT_CSV} {BACKUP_CSV}")

# Load data
print(f"📂 Loading data from: {INPUT_CSV}")
df = pd.read_csv(INPUT_CSV)
print(f"✓ Loaded {len(df):,} rows")
print()

# Remove header rows (corrupted data)
print("🔍 Removing corrupted header rows...")
before_count = len(df)
df = df[df['gender'] != 'gender']
df = df[df['product_name'] != 'product_name']
removed_headers = before_count - len(df)
print(f"   Removed {removed_headers} corrupted header rows")
print()

# Fix gender labels
print("🔧 Fixing gender labels...")
gender_before = df['gender'].value_counts().to_dict()

# Map Men's -> Male, Women's -> Female
df['gender'] = df['gender'].replace({
    "Men's": "Male",
    "Women's": "Female"
})

gender_after = df['gender'].value_counts().to_dict()

print("   Before:")
for gender, count in sorted(gender_before.items(), key=lambda x: x[1], reverse=True):
    print(f"      {gender:10s}: {count:,}")

print("\n   After:")
for gender, count in sorted(gender_after.items(), key=lambda x: x[1], reverse=True):
    print(f"      {gender:10s}: {count:,}")

print()

# Verify only Male/Female remain
unique_genders = df['gender'].unique()
print(f"✓ Unique gender values after fix: {list(unique_genders)}")

if set(unique_genders) == {'Male', 'Female'}:
    print("✓ SUCCESS: Only Male and Female remain")
elif set(unique_genders) == {'Female', 'Male'}:
    print("✓ SUCCESS: Only Male and Female remain")
else:
    print(f"⚠️  WARNING: Unexpected gender values found: {unique_genders}")

print()

# Save fixed data
print(f"💾 Saving fixed data to: {OUTPUT_CSV}")
df.to_csv(OUTPUT_CSV, index=False)
print(f"✓ Saved {len(df):,} rows")

print()
print("=" * 80)
print("✓ GENDER FIX COMPLETE")
print("=" * 80)
print()
print(f"📊 Summary:")
print(f"   Removed corrupted rows: {removed_headers}")
print(f"   Final row count:        {len(df):,}")
print(f"   Gender values:          {list(df['gender'].unique())}")
print(f"   Male products:          {(df['gender']=='Male').sum():,}")
print(f"   Female products:        {(df['gender']=='Female').sum():,}")
print()
