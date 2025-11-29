#!/usr/bin/env python3

import argparse
import datetime
import os
import sys

# Add paths for imports
sys.path.append(os.path.dirname(__file__))  # Add scripts directory
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))  # Add data_creation directory
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'config'))

import pandas as pd

import config
from fashion_generator.core.generator import GeminiGenerator, generate_full_dataset, generate_batch_csv_stratified
from fashion_generator.core.analyzer import analyze_dataset, print_analysis, export_analysis
from fashion_generator.data.vocabularies import CATEGORY_LEAVES


def main():
    parser = argparse.ArgumentParser(
        description="Generate fashion product dataset using Google Gemini"
    )

    parser.add_argument(
        "--products-per-category",
        type=int,
        default=config.PRODUCTS_PER_CATEGORY,
        help=f"Number of products to generate per category (default: {config.PRODUCTS_PER_CATEGORY})",
    )

    parser.add_argument(
        "--test",
        action="store_true",
        help="Run in test mode (generate small batch only)",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=config.OUTPUT_DIR,
        help=f"Output directory for generated files (default: {config.OUTPUT_DIR})",
    )

    parser.add_argument(
        "--implementation",
        type=str,
        choices=["sdk"],
        default=config.IMPLEMENTATION,
        help=f"Gemini implementation to use (default: {config.IMPLEMENTATION})",
    )

    parser.add_argument(
        "--no-analysis",
        action="store_true",
        help="Skip dataset analysis after generation",
    )

    parser.add_argument(
        "--resume",
        action="store_true",
        default=True,
        help="Resume from checkpoint if exists (default: True)",
    )

    parser.add_argument(
        "--no-checkpoint",
        action="store_true",
        help="Disable checkpoints (not recommended for large generations)",
    )

    parser.add_argument(
        "--output-filename",
        type=str,
        default=None,
        help="Custom output filename (e.g., 'product_data.csv'). If not specified, uses timestamp.",
    )

    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Start fresh (delete old checkpoints and output file)",
    )

    args = parser.parse_args()

    # Handle checkpoint configuration
    if args.no_checkpoint:
        config.ENABLE_CHECKPOINTS = False

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Handle fresh start
    if args.fresh:
        import glob
        print(" FRESH START MODE: Cleaning up old files\n")

        # Remove all checkpoints
        checkpoint_pattern = os.path.join(config.CHECKPOINT_DIR, "checkpoint_*.json")
        for checkpoint_file in glob.glob(checkpoint_pattern):
            try:
                os.remove(checkpoint_file)
                print(f" Removed checkpoint: {checkpoint_file}")
            except OSError:
                pass

        # Remove specific output file if provided
        if args.output_filename:
            output_path = os.path.join(args.output_dir, args.output_filename)
            if os.path.exists(output_path):
                try:
                    os.remove(output_path)
                    print(f" Removed output file: {output_path}")
                except OSError:
                    pass

        # Disable resume when starting fresh
        args.resume = False
        print()

    print("Fashion Product Dataset Generator")
    print()

    # Initialize generator
    try:
        generator = GeminiGenerator(implementation=args.implementation)
    except Exception as e:
        print(f"Failed to initialize generator: {e}")
        sys.exit(1)

    print()

    # Generate dataset
    try:
        if args.test:
            print("TEST MODE: Generating small batch\n")
            # Use actual category objects from CATEGORY_LEAVES
            test_categories = [
                next(cat for cat in CATEGORY_LEAVES if cat['name'] == "Jeans"),
                next(cat for cat in CATEGORY_LEAVES if cat['name'] == "Sneakers"),
                next(cat for cat in CATEGORY_LEAVES if cat['name'] == "Tank Tops"),
            ]
            products_df = generate_batch_csv_stratified(
                generator=generator,
                categories=test_categories,
                n_per_category=2,
            )
        else:
            print(" FULL GENERATION MODE\n")
            products_df = generate_full_dataset(
                generator=generator,
                n_products_per_category=args.products_per_category,
                resume=args.resume,
                output_filename=args.output_filename,
            )

    except Exception as e:
        print(f"Generation failed: {e}")
        sys.exit(1)

    # Save dataset
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"fashion_products_{timestamp}.csv"
    filepath = os.path.join(args.output_dir, filename)

    products_df.to_csv(filepath, index=False)
    print(f"\n Saved {len(products_df)} products to: {filepath}")

    # Perform analysis
    if not args.no_analysis:
        print("\n")
        analysis = analyze_dataset(products_df)
        print_analysis(analysis)

        # Export analysis
        analysis_filename = f"analysis_{timestamp}.json"
        analysis_filepath = os.path.join(args.output_dir, analysis_filename)
        export_analysis(analysis, analysis_filepath)

    print("\nDone!")


if __name__ == "__main__":
    main()
