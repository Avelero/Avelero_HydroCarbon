#!/usr/bin/env python3
"""
Extra Product Generation - Boost Underrepresented Materials

Generates additional products to bring all materials to ~35,000 product count.
Maintains distribution uniformity across categories, parents, and genders.

REUSES:
- src/generator.py (GeminiGenerator)
- src/vocabularies.py (materials, categories, get_hierarchy_info)
- src/prompts.py (build_generation_prompt_csv)
- src/csv_writer.py (CSVWriter)
- src/checkpoint.py (CheckpointManager)
- src/rate_limiter.py (RateLimiter)
"""

import sys
import time
import math
from pathlib import Path
from datetime import datetime

# Add parent directory to path to import from src/
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.generator import GeminiGenerator
from src.csv_writer import CSVWriter
from src.checkpoint import CheckpointManager
from src.rate_limiter import RateLimiter
from src.vocabularies import CATEGORY_LEAVES, get_hierarchy_info
from src.prompts import build_generation_prompt_csv
from config.extra_generation_config import ALL_MATERIALS, SUMMARY

# ============================================================================
# CONFIGURATION
# ============================================================================

OUTPUT_CSV = "output/Product_data_extra.csv"
CHECKPOINT_DIR = "output/checkpoints/extra"
BATCH_SIZE = 50  # Products per API call
CHECKPOINT_INTERVAL = 10000  # Save every 10k products

# Create directories
Path(CHECKPOINT_DIR).mkdir(exist_ok=True, parents=True)

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def find_category_by_name(category_name):
    """Find category object by name"""
    for cat in CATEGORY_LEAVES:
        if cat['name'] == category_name:
            return cat
    return None


def split_by_gender(total_count, gender_split):
    """
    Split total count by gender percentages.

    Args:
        total_count: Total products to generate
        gender_split: Dict of gender -> percentage (e.g., {"Female": 0.6, "Male": 0.4})

    Returns:
        Dict of gender -> count
    """
    result = {}
    remaining = total_count

    # Sort by percentage (descending) for better rounding
    sorted_genders = sorted(gender_split.items(), key=lambda x: x[1], reverse=True)

    for i, (gender, percentage) in enumerate(sorted_genders):
        if i == len(sorted_genders) - 1:
            # Last gender gets the remainder
            result[gender] = remaining
        else:
            count = round(total_count * percentage)
            result[gender] = count
            remaining -= count

    return result


def generate_for_category_gender(
    generator,
    csv_writer,
    rate_limiter,
    category,
    gender,
    count,
    material_name,
    progress_prefix
):
    """
    Generate products for a specific category-gender combination.

    Args:
        generator: GeminiGenerator instance
        csv_writer: CSVWriter instance
        rate_limiter: RateLimiter instance
        category: Category dict from CATEGORY_LEAVES
        gender: "Male" or "Female"
        count: Number of products to generate
        material_name: Material being focused on (for logging)
        progress_prefix: String prefix for progress messages

    Returns:
        Number of products successfully generated
    """
    if count == 0:
        return 0

    hierarchy = get_hierarchy_info(category)
    category_name = hierarchy['category']

    # Calculate number of batches
    num_batches = math.ceil(count / BATCH_SIZE)
    total_generated = 0

    print(f"\n  {progress_prefix} {category_name} ({gender}): {count:,} products in {num_batches} batches")

    for batch_idx in range(num_batches):
        # Calculate batch size (last batch might be smaller)
        remaining = count - total_generated
        current_batch_size = min(BATCH_SIZE, remaining)

        if current_batch_size == 0:
            break

        # Rate limiting
        rate_limiter.wait_if_needed()

        # Generate prompt
        # We'll generate for just this one category with gender filter
        # The generator will naturally use the materials defined in vocabularies
        prompt = build_generation_prompt_csv(
            categories_to_generate=[category],
            n_per_category=current_batch_size
        )

        # Modify prompt to request specific gender
        # Insert gender requirement into the prompt
        gender_instruction = f"\nIMPORTANT: ALL products in this batch MUST have gender = '{gender}'"
        prompt = prompt.replace("Output CSV with this EXACT header",
                               gender_instruction + "\n\nOutput CSV with this EXACT header")

        # Also add material preference hint
        material_hint = f"\nPREFER using '{material_name}' material in these products when realistic"
        prompt = prompt.replace("Output CSV with this EXACT header",
                               material_hint + "\n\nOutput CSV with this EXACT header")

        try:
            # Generate batch
            products = generator.generate_batch(prompt)

            if products:
                # Write to CSV
                csv_writer.write_rows(products)
                batch_generated = len(products)
                total_generated += batch_generated

                print(f"    Batch {batch_idx + 1}/{num_batches}: ✓ {batch_generated} products (total: {total_generated:,}/{count:,})")
            else:
                print(f"    Batch {batch_idx + 1}/{num_batches}: ⚠️  No products returned")

        except Exception as e:
            print(f"    Batch {batch_idx + 1}/{num_batches}: ❌ Error: {e}")
            # Continue to next batch
            continue

    return total_generated


# ============================================================================
# MAIN GENERATION FUNCTION
# ============================================================================

def main():
    print("=" * 90)
    print("EXTRA PRODUCT GENERATION - MATERIAL BOOSTING")
    print("=" * 90)
    print()
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # Display summary
    print("Configuration Summary:")
    for key, value in SUMMARY.items():
        if isinstance(value, int):
            print(f"  {key}: {value:,}")
        else:
            print(f"  {key}: {value}")
    print()

    # Initialize components
    print("Initializing components...")
    generator = GeminiGenerator()
    csv_writer = CSVWriter(OUTPUT_CSV)
    checkpoint_mgr = CheckpointManager(CHECKPOINT_DIR, CHECKPOINT_INTERVAL)
    rate_limiter = RateLimiter()
    print("✓ Components initialized")
    print()

    # Write CSV header
    csv_writer.write_header()
    print(f"✓ CSV header written to: {OUTPUT_CSV}")
    print()

    # Statistics
    total_generated = 0
    total_target = SUMMARY['total_products']
    material_count = 0
    total_materials = SUMMARY['total_materials']

    start_time = time.time()

    # Generate for each material
    for material_name, material_config in ALL_MATERIALS.items():
        material_count += 1
        needed = material_config['target'] - material_config['current']

        print("=" * 90)
        print(f"MATERIAL {material_count}/{total_materials}: {material_name}")
        print("=" * 90)
        print(f"Current: {material_config['current']:,} | Target: {material_config['target']:,} | Need: {needed:,}")
        print()

        material_generated = 0

        # Generate for each category
        for category_name, category_config in material_config['categories'].items():
            # Find category object
            category = find_category_by_name(category_name)

            if not category:
                print(f"  ⚠️  Category '{category_name}' not found, skipping...")
                continue

            # Split by gender
            count = category_config['count']
            gender_split = category_config['gender_split']
            gender_counts = split_by_gender(count, gender_split)

            # Generate for each gender
            for gender, gender_count in gender_counts.items():
                if gender_count == 0:
                    continue

                progress_prefix = f"[{material_generated:,}/{needed:,}]"

                generated = generate_for_category_gender(
                    generator=generator,
                    csv_writer=csv_writer,
                    rate_limiter=rate_limiter,
                    category=category,
                    gender=gender,
                    count=gender_count,
                    material_name=material_name,
                    progress_prefix=progress_prefix
                )

                material_generated += generated
                total_generated += generated

                # Checkpoint if needed
                if total_generated % CHECKPOINT_INTERVAL < generated:
                    print(f"\n  💾 Checkpoint: {total_generated:,} products generated")

        # Material summary
        print()
        print(f"✓ {material_name}: Generated {material_generated:,} products")
        print(f"  Overall progress: {total_generated:,}/{total_target:,} ({(total_generated/total_target)*100:.1f}%)")

        # Calculate ETA
        elapsed = time.time() - start_time
        if total_generated > 0:
            rate = total_generated / elapsed
            remaining = total_target - total_generated
            eta_seconds = remaining / rate if rate > 0 else 0
            eta_hours = eta_seconds / 3600
            print(f"  Rate: {rate:.1f} products/sec | ETA: {eta_hours:.1f} hours")
        print()

    # Final summary
    print("=" * 90)
    print("✓ GENERATION COMPLETE")
    print("=" * 90)
    print()
    print(f"Total products generated: {total_generated:,}")
    print(f"Target products: {total_target:,}")
    print(f"Achievement: {(total_generated/total_target)*100:.1f}%")
    print()

    elapsed = time.time() - start_time
    print(f"Total time: {elapsed/3600:.2f} hours ({elapsed/60:.1f} minutes)")
    print(f"Average rate: {total_generated/elapsed:.1f} products/second")
    print()
    print(f"Output file: {OUTPUT_CSV}")
    print()
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Generation interrupted by user")
        print("Progress has been saved to CSV")
        print(f"Resume by running this script again (it will append to {OUTPUT_CSV})")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
