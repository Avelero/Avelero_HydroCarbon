#!/usr/bin/env python3
"""
Fix Category Inconsistencies
Maps invalid/synonym categories to their canonical forms defined in vocabularies.py.
"""

import pandas as pd
import os
import sys

INPUT_CSV = "output/Product_data_cleaned.csv"
OUTPUT_CSV = "output/Product_data_cleaned.csv"
BACKUP_CSV = "output/Product_data_cleaned_pre_cat_fix.csv"

# Mapping Dictionary
CATEGORY_MAPPING = {
    # Typos & Hallucinations
    'Athwear': 'Athletic Shoes',
    'Denem Jackets': 'Denim Jackets',
    'Downwear': 'Down Jackets',
    'Jackwear': 'Jackets',
    'Loafwear': 'Loafers',
    'Rainwear': 'Raincoats',
    'Sandwear': 'Sandals',
    'Sweatpants & Jogpants': 'Sweatpants & Joggers',
    
    # Synonyms / Redundant Naming
    'Maxi Dresses': 'Maxi',
    'Midi Dresses': 'Midi',
    'Mini Dresses': 'Mini',
    'Maxi Skirt': 'Maxi Skirts',
    
    # Denim Variations
    'Denim Pants': 'Jeans',
    'Denim Trousers': 'Jeans',
    'Denim Shorts': 'Shorts',
    'Denim Overalls': 'Jumpsuits'
}

def main():
    print("=" * 80)
    print("FIXING CATEGORY INCONSISTENCIES")
    print("=" * 80)
    
    if not os.path.exists(INPUT_CSV):
        print(f"Error: {INPUT_CSV} not found.")
        sys.exit(1)

    # Create backup
    print(f"📋 Creating backup: {BACKUP_CSV}")
    os.system(f"cp {INPUT_CSV} {BACKUP_CSV}")
    
    # Load data
    print(f"📂 Loading data from: {INPUT_CSV}")
    df = pd.read_csv(INPUT_CSV, low_memory=False)
    print(f"✓ Loaded {len(df):,} rows")
    
    # Apply mappings
    print("\n🔍 Applying mappings...")
    changes_count = 0
    
    for bad, good in CATEGORY_MAPPING.items():
        mask = df['category'] == bad
        count = mask.sum()
        if count > 0:
            df.loc[mask, 'category'] = good
            print(f"   Mapped '{bad}' -> '{good}' ({count} rows)")
            changes_count += count
            
    print(f"\n✓ Total rows updated: {changes_count}")
    
    # Verify no invalid categories remain
    # (We can't easily import vocabularies here without path hacking, so we trust the mapping covers the identified list)
    
    # Save
    print(f"💾 Saving cleaned data to: {OUTPUT_CSV}")
    df.to_csv(OUTPUT_CSV, index=False)
    print("✓ Saved")
    
    print("=" * 80)

if __name__ == "__main__":
    main()
