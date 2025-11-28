#!/usr/bin/env python3
"""
Generate Extra Products for Dataset Balancing
Target: Balance Dress categories to ~25,000 products each.
Optimized for high-throughput generation (400-500 items/call).
"""

import os
import sys
import pandas as pd
import time
import random
from pathlib import Path

# Resolve paths relative to this script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '../../'))
DATA_CREATION_SRC = os.path.join(PROJECT_ROOT, 'data_creation/src')
DATA_CREATION_CONFIG = os.path.join(PROJECT_ROOT, 'data_creation/config')

sys.path.append(DATA_CREATION_SRC)
sys.path.append(DATA_CREATION_CONFIG)

try:
    from generator import GeminiGenerator
    from vocabularies import CATEGORY_LEAVES, COUNTRIES
    import config
    from prompts import build_generation_prompt_csv
except ImportError as e:
    print(f"Error importing generator modules: {e}")
    print(f"Checked paths:\n  {DATA_CREATION_SRC}\n  {DATA_CREATION_CONFIG}")
    sys.exit(1)

# Configuration
INPUT_CSV = os.path.join(SCRIPT_DIR, "../output/Product_data_cleaned.csv")
OUTPUT_CSV = os.path.join(SCRIPT_DIR, "extra_products.csv")
TARGET_COUNT = 25000
BATCH_SIZE = 1000  # Optimized for max output tokens

# Target Categories (Canonical Names)
TARGET_CATEGORIES = [
    "Gowns",
    "Maxi",  # Will also cover "Maxi Dresses"
    "Midi",  # Will also cover "Midi Dresses"
    "Mini"   # Will also cover "Mini Dresses"
]

def get_current_counts(df):
    """Calculate current counts for target categories, handling synonyms."""
    counts = {}
    
    # Gowns
    counts["Gowns"] = len(df[df['category'] == 'Gowns'])
    
    # Maxi (combine "Maxi" and "Maxi Dresses")
    counts["Maxi"] = len(df[df['category'].isin(['Maxi', 'Maxi Dresses'])])
    
    # Midi (combine "Midi" and "Midi Dresses")
    counts["Midi"] = len(df[df['category'].isin(['Midi', 'Midi Dresses'])])
    
    # Mini (combine "Mini" and "Mini Dresses")
    counts["Mini"] = len(df[df['category'].isin(['Mini', 'Mini Dresses'])])
    
    return counts

def generate_batch(generator, category_name, count):
    """Generate a batch of products for a specific category."""
    # Convert category name to category object
    category_obj = None
    for cat in CATEGORY_LEAVES:
        if cat['name'] == category_name:
            category_obj = cat
            break

    if category_obj is None:
        print(f"  Error: Category '{category_name}' not found in CATEGORY_LEAVES")
        return pd.DataFrame()

    # Randomly select countries for diversity
    countries_subset = random.sample(COUNTRIES, min(15, len(COUNTRIES)))

    prompt = build_generation_prompt_csv(
        categories_to_generate=[category_obj],
        countries_subset=countries_subset,
        n_per_category=count
    )
    
    try:
        # Generate text
        # High temperature for diversity, but not too high to break formatting
        text = generator.generate_text(prompt, temperature=0.95) 
        
        # Parse CSV
        lines = text.strip().split('\n')
        data_lines = [line for line in lines if ',' in line and not line.lower().startswith('product_name')]
        
        # Create DataFrame
        from io import StringIO
        csv_data = config.CSV_HEADER + '\n' + '\n'.join(data_lines)
        batch_df = pd.read_csv(StringIO(csv_data), quoting=1, escapechar='\\', on_bad_lines='skip')
        
        # ---------------------------------------------------------------------
        # APPLY "GOLD STANDARD" VALIDATION (My Ideology)
        # ---------------------------------------------------------------------
        if not batch_df.empty:
            valid_rows = []
            
            # Get valid materials set for fast lookup
            from vocabularies import MATERIAL_VOCAB
            valid_materials = set(MATERIAL_VOCAB)
            
            for _, row in batch_df.iterrows():
                is_valid = True
                
                # 1. Validate Materials (No Typos, No Hallucinations)
                try:
                    import json
                    mats = json.loads(row['materials'].replace("'", '"'))
                    for m in mats:
                        if m not in valid_materials:
                            # Auto-fix common typos if possible, otherwise reject
                            if m == 'polyster_virgin': m = 'polyester_virgin'
                            elif m == 'poliamide_6': m = 'polyamide_6'
                            
                            if m not in valid_materials:
                                is_valid = False # Reject row with unknown material
                                break
                except:
                    is_valid = False
                
                if not is_valid: continue

                # 2. Validate Numeric Variance (No Round Numbers)
                # We want to avoid "1.0", "0.5", "5000" which look synthetic
                w = float(row['weight_kg'])
                d = float(row['total_distance_km'])
                
                if w.is_integer() or (w * 10).is_integer(): # e.g. 1.0 or 1.5
                    is_valid = False
                
                if d % 100 == 0: # e.g. 5000, 1200
                    is_valid = False
                    
                # 3. Validate Ranges (Sanity Check)
                if w < 0.05 or w > 3.0: is_valid = False
                if d < 100 or d > 40000: is_valid = False
                
                if is_valid:
                    valid_rows.append(row)
            
            # Reconstruct DataFrame with only valid rows
            if valid_rows:
                batch_df = pd.DataFrame(valid_rows)
            else:
                batch_df = pd.DataFrame() # All rejected
                
        return batch_df
    except Exception as e:
        print(f"  Error generating batch: {e}")
        return pd.DataFrame()

def main():
    print("=" * 80)
    print("EXTRA PRODUCT GENERATION FOR BALANCING (OPTIMIZED)")
    print("=" * 80)
    print(f"Target per category: {TARGET_COUNT:,}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Output file: {OUTPUT_CSV}")
    print("-" * 80)
    
    # 1. Load current data
    print(f"📂 Loading current dataset: {INPUT_CSV}")
    if os.path.exists(INPUT_CSV):
        df = pd.read_csv(INPUT_CSV, low_memory=False)
        print(f"✓ Loaded {len(df):,} rows")
    else:
        print(f"Error: Input file not found at {INPUT_CSV}")
        sys.exit(1)
        
    # 2. Calculate deficits
    current_counts = get_current_counts(df)
    print("\nCurrent Status:")
    total_needed = 0
    deficits = {}
    
    for cat in TARGET_CATEGORIES:
        count = current_counts.get(cat, 0)
        deficit = max(0, TARGET_COUNT - count)
        deficits[cat] = deficit
        total_needed += deficit
        print(f"  {cat:10s}: {count:6,} (Target: {TARGET_COUNT:,}) -> Need: {deficit:,}")
        
    if total_needed == 0:
        print("\n✓ All categories met target! No generation needed.")
        return

    print(f"\nTotal products to generate: {total_needed:,}")
    print(f"Estimated batches: {total_needed / BATCH_SIZE:.1f}")
    
    # 3. Initialize Generator
    print("\nInitializing Gemini Generator...")
    try:
        generator = GeminiGenerator(implementation="sdk")
    except Exception as e:
        print(f"Failed to initialize generator: {e}")
        sys.exit(1)
        
    # 4. Parallel Generation Loop
    total_generated = 0
    
    # Check if output file exists to resume
    if os.path.exists(OUTPUT_CSV):
        try:
            existing_extra = pd.read_csv(OUTPUT_CSV)
            print(f"Resuming... Found {len(existing_extra):,} already generated products.")
            total_generated = len(existing_extra)
        except:
            print("Output file exists but is empty or corrupt. Starting fresh.")
            with open(OUTPUT_CSV, 'w') as f:
                f.write(config.CSV_HEADER + '\n')
    else:
        # Write header
        with open(OUTPUT_CSV, 'w') as f:
            f.write(config.CSV_HEADER + '\n')
            
    start_time = time.time()
    
    # Prepare tasks
    tasks = []
    for category, deficit in deficits.items():
        if deficit <= 0: continue
        
        # Split deficit into batches
        num_batches = (deficit + BATCH_SIZE - 1) // BATCH_SIZE
        for _ in range(num_batches):
            # Last batch might be smaller
            current_batch_size = min(BATCH_SIZE, deficit)
            tasks.append((category, current_batch_size))
            deficit -= current_batch_size
            
    print(f"\n🚀 Starting parallel generation with {config.PARALLEL_WORKERS} workers...")
    print(f"Total batches to process: {len(tasks)}")
    
    import concurrent.futures
    import threading
    
    csv_lock = threading.Lock()
    completed_batches = 0
    
    def process_batch(task):
        cat, size = task
        # Retry logic inside worker
        for attempt in range(3):
            batch_df = generate_batch(generator, cat, size)
            if not batch_df.empty:
                with csv_lock:
                    batch_df.to_csv(OUTPUT_CSV, mode='a', header=False, index=False, quoting=1, escapechar='\\')
                return len(batch_df)
            time.sleep(2) # Backoff
        return 0

    with concurrent.futures.ThreadPoolExecutor(max_workers=config.PARALLEL_WORKERS) as executor:
        futures = [executor.submit(process_batch, task) for task in tasks]
        
        for future in concurrent.futures.as_completed(futures):
            count = future.result()
            total_generated += count
            completed_batches += 1
            
            elapsed = time.time() - start_time
            rate = total_generated / elapsed if elapsed > 0 else 0
            
            if completed_batches % 5 == 0 or completed_batches == len(tasks):
                print(f"\rProgress: {completed_batches}/{len(tasks)} batches | Total: {total_generated:,} | Rate: {rate:.1f} prods/s", end="")

    total_duration = time.time() - start_time
    print("\n" + "=" * 80)
    print(f"✓ GENERATION COMPLETE")
    print(f"Total generated: {total_generated:,}")
    print(f"Total time: {total_duration:.1f}s")
    print(f"Saved to: {OUTPUT_CSV}")
    print("=" * 80)

if __name__ == "__main__":
    main()
