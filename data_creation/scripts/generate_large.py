#!/usr/bin/env python3
"""
Script to generate large datasets by running the generator multiple times.
This accumulates results across multiple runs to reach target product counts.
"""

import argparse
import os
import sys
import pandas as pd
from datetime import datetime

def run_batch_generation(products_per_category, iteration):
    """Run the generation script and return the output file path."""
    print(f"\n{'='*60}")
    print(f"ITERATION {iteration}: Generating {products_per_category} products/category")
    print(f"{'='*60}\n")
    
    # Run the main.py script
    result = os.system(f"python main.py --products-per-category {products_per_category}")
    
    if result != 0:
        raise RuntimeError(f"Generation failed with exit code: {result}")
    
    # Find the most recent output file
    output_files = sorted(
        [f for f in os.listdir("output") if f.startswith("fashion_products_") and f.endswith(".csv")],
        key=lambda x: os.path.getmtime(os.path.join("output", x)),
        reverse=True
    )
    
    if not output_files:
        raise RuntimeError("No output file found!")
    
    return os.path.join("output", output_files[0])

def main():
    parser = argparse.ArgumentParser(
        description="Generate large fashion product datasets by accumulating multiple runs"
    )
    
    parser.add_argument(
        "--target-per-category",
        type=int,
        required=True,
        help="Target number of products per category (total)"
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=50,
        help="Number of products to generate per category in each iteration (default: 50)"
    )
    
    parser.add_argument(
        "--output-file",
        type=str,
        help="Final output filename (default: fashion_products_large_TIMESTAMP.csv)"
    )
    
    args = parser.parse_args()
    
    # Calculate number of iterations needed
    iterations = (args.target_per_category + args.batch_size - 1) // args.batch_size
    
    print("="*60)
    print("LARGE DATASET GENERATION")
    print("="*60)
    print(f"Target products per category: {args.target_per_category}")
    print(f"Batch size per iteration: {args.batch_size}")
    print(f"Number of iterations needed: {iterations}")
    print("="*60)
    
    all_dataframes = []
    
    # Run multiple iterations
    for i in range(1, iterations + 1):
        # For the last iteration, adjust the batch size to hit the target exactly
        if i == iterations:
            remaining = args.target_per_category - (args.batch_size * (i - 1))
            batch_size = min(remaining, args.batch_size)
        else:
            batch_size = args.batch_size
        
        try:
            output_file = run_batch_generation(batch_size, i)
            print(f"\n✓ Iteration {i} complete: {output_file}")
            
            # Load the generated data
            df = pd.read_csv(output_file)
            all_dataframes.append(df)
            print(f"  Loaded {len(df)} products")
            
        except Exception as e:
            print(f"\n✗ Iteration {i} failed: {e}")
            print("\nPartial results have been saved. You can resume later.")
            break
    
    # Combine all dataframes
    if all_dataframes:
        combined_df = pd.concat(all_dataframes, ignore_index=True)
        
        # Generate output filename
        if args.output_file:
            final_output = os.path.join("output", args.output_file)
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            final_output = os.path.join("output", f"fashion_products_large_{timestamp}.csv")
        
        # Save combined dataset
        combined_df.to_csv(final_output, index=False)
        
        print(f"\n{'='*60}")
        print("GENERATION COMPLETE!")
        print(f"{'='*60}")
        print(f"Total products generated: {len(combined_df)}")
        print(f"Categories covered: {combined_df['category'].nunique()}")
        print(f"Products per category (average): {len(combined_df) / combined_df['category'].nunique():.1f}")
        print(f"Final output file: {final_output}")
        print(f"{'='*60}\n")
        
    else:
        print("\n✗ No data was generated")
        sys.exit(1)

if __name__ == "__main__":
    main()
