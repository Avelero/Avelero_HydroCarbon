"""
Fashion Product Dataset Generator using Google Gemini
"""

import os
import sys
import pandas as pd
import io
import csv
import random
from typing import List
from datetime import datetime

# Add path for config
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'config'))

import config
from ..data.vocabularies import CATEGORY_LEAVES, COUNTRIES
from ..data.prompts import build_generation_prompt_csv
from ..utils.checkpoint import CheckpointManager
from ..utils.csv_writer import IncrementalCSVWriter


class GeminiGenerator:
    """Generator class that handles Gemini API calls"""

    def __init__(self, implementation: str = None):
        """
        Initialize the generator.

        Args:
            implementation: 'sdk'. If None, uses config.IMPLEMENTATION
        """
        self.implementation = implementation or config.IMPLEMENTATION

        if self.implementation == "sdk":
            self._setup_sdk()
        else:
            raise ValueError(f"Unknown implementation: {self.implementation}")

    def _setup_sdk(self):
        """Setup google-generativeai SDK"""
        try:
            import google.generativeai as genai

            if not config.GOOGLE_API_KEY:
                raise ValueError(
                    "GOOGLE_API_KEY not found in environment variables.\n"
                    "Please set it: export GOOGLE_API_KEY='your-api-key-here'\n"
                    "Get your key from: https://aistudio.google.com/apikey"
                )

            genai.configure(api_key=config.GOOGLE_API_KEY)
            self.genai = genai
            print("Google Generative AI SDK configured successfully!")
            print(f"Using model: {config.GEMINI_MODEL}")

        except ImportError:
            raise ImportError(
                "google-generativeai not installed. Install with: pip install google-generativeai"
            )

    def generate_text(self, prompt: str, temperature: float = None) -> str:
        """
        Generate text using Gemini.

        Args:
            prompt: The input prompt
            temperature: Sampling temperature (0.0-2.0)

        Returns:
            Generated text string

        Raises:
            RateLimitError: If rate limit is hit (429 error)
        """
        from ..utils.rate_limiter import RateLimitError

        temperature = temperature or config.TEMPERATURE
        print("Generating with Gemini... ", end="", flush=True)

        try:
            if self.implementation == "sdk":
                model = self.genai.GenerativeModel(config.GEMINI_MODEL)

                generation_config = self.genai.types.GenerationConfig(
                    temperature=temperature,
                    max_output_tokens=config.MAX_OUTPUT_TOKENS,
                )

                # Configure safety settings to be permissive for fashion product generation
                safety_settings = {
                    self.genai.types.HarmCategory.HARM_CATEGORY_HATE_SPEECH: self.genai.types.HarmBlockThreshold.BLOCK_NONE,
                    self.genai.types.HarmCategory.HARM_CATEGORY_HARASSMENT: self.genai.types.HarmBlockThreshold.BLOCK_NONE,
                    self.genai.types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: self.genai.types.HarmBlockThreshold.BLOCK_NONE,
                    self.genai.types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: self.genai.types.HarmBlockThreshold.BLOCK_NONE,
                }

                result = model.generate_content(
                    prompt,
                    generation_config=generation_config,
                    safety_settings=safety_settings
                )

                # Try to get the text - this will raise an exception if blocked
                try:
                    response = result.text
                except ValueError as e:
                    # Content was blocked or no valid response
                    error_msg = str(e)
                    if "finish_reason" in error_msg.lower():
                        # Extract finish reason from error message
                        if "finish_reason" in error_msg and "2" in error_msg:
                            raise ValueError(
                                "Content blocked by safety filters. This might be due to:\n"
                                "  1. Request being too large (reduce --products-per-category)\n"
                                "  2. Unexpected content triggering safety policies\n"
                                "  3. Token limit exceeded (reduce products or increase MAX_OUTPUT_TOKENS)"
                            )
                    raise ValueError(f"Failed to generate content: {error_msg}")

            print("Done!")
            return response.strip()

        except Exception as e:
            error_msg = str(e).lower()

            # Check for rate limit errors (429)
            if "429" in error_msg or "rate limit" in error_msg or "quota" in error_msg:
                print(f"RATE LIMIT HIT!")
                raise RateLimitError(f"Rate limit exceeded: {e}")

            print(f"Error: {e}")
            raise


def generate_batch_csv_stratified(
    generator: GeminiGenerator,
    categories: List[str],
    n_per_category: int = 2,
    max_retries: int = None,
) -> pd.DataFrame:
    """
    Generate products for specific categories with balanced distribution.

    Args:
        generator: GeminiGenerator instance
        categories: List of categories to generate
        n_per_category: Number of products per category
        max_retries: Maximum retry attempts

    Returns:
        Pandas DataFrame with product data
    """
    max_retries = max_retries or config.MAX_RETRIES

    # Vary countries per batch for diversity
    countries_subset = random.sample(COUNTRIES, min(config.COUNTRY_SUBSET_SIZE, len(COUNTRIES)))
    
    # Vary temperature for increased diversity (if enabled)
    if config.TEMPERATURE_VARIANCE:
        temperature = random.uniform(config.TEMPERATURE_MIN, config.TEMPERATURE_MAX)
    else:
        temperature = config.TEMPERATURE

    prompt = build_generation_prompt_csv(
        categories_to_generate=categories,
        countries_subset=countries_subset,
        n_per_category=n_per_category,
    )

    for attempt in range(max_retries):
        print(f"  Attempt {attempt + 1}/{max_retries}...")

        # Generate text with varied temperature
        text = generator.generate_text(prompt, temperature=temperature)

        # Clean up the response
        lines = text.strip().split("\n")
        clean_lines = []

        # Add header
        clean_lines.append(config.CSV_HEADER)

        # Expected field count
        expected_field_count = len(config.CSV_HEADER.split(','))

        # Filter and validate CSV data lines
        for line_num, line in enumerate(lines, 1):
            line = line.strip()
            # Skip empty lines, explanations, and header repetitions
            if not line or not "," in line:
                continue
            if line.lower().startswith(("here", "note", "csv", "product_name", "output", "example")):
                continue

            # Quick validation: count commas (accounting for quoted sections)
            # This is a rough check - actual parsing will validate properly
            try:
                # Try to parse this single line to validate field count
                reader = csv.reader([line], quoting=csv.QUOTE_ALL, escapechar='\\')
                fields = next(reader)
                if len(fields) != expected_field_count:
                    print(f"  ! Warning: Line {line_num} has {len(fields)} fields (expected {expected_field_count})")
                    print(f"    Line: {line[:100]}...")
                    # Skip malformed lines
                    continue
                clean_lines.append(line)
            except Exception as e:
                print(f"  ! Warning: Could not parse line {line_num}: {e}")
                print(f"    Line: {line[:100]}...")
                continue

        csv_text = "\n".join(clean_lines)

        # Try to parse CSV
        try:
            # Use proper CSV quoting to handle JSON fields with commas
            df = pd.read_csv(
                io.StringIO(csv_text),
                quoting=1,  # QUOTE_ALL
                escapechar='\\',
                on_bad_lines='warn'
            )

            # Validate we have the expected columns
            expected_cols = config.CSV_HEADER.split(',')
            if len(df.columns) != len(expected_cols):
                raise ValueError(
                    f"Column count mismatch: expected {len(expected_cols)} columns "
                    f"({expected_cols}), got {len(df.columns)} ({list(df.columns)})"
                )

            print(f"  Generated {len(df)} products")
            return df

        except Exception as e:
            print(f"  CSV parse error: {e}")

            if attempt == max_retries - 1:
                print("\nFull generated text:")
                print(text)
                raise e

    raise RuntimeError(f"Failed after {max_retries} attempts")


def generate_full_dataset(
    generator: GeminiGenerator,
    n_products_per_category: int = None,
    resume: bool = True,
    parallel: bool = True,
    output_filename: str = None
) -> pd.DataFrame:
    """
    Generate complete balanced dataset with parallel chunked generation and checkpoint support.

    Args:
        generator: GeminiGenerator instance
        n_products_per_category: Number of products to generate per category
        resume: If True, resume from checkpoint if exists
        parallel: If True, use parallel workers (default: True)
        output_filename: Custom output filename (e.g., 'product_data.csv')

    Returns:
        Pandas DataFrame with all products
    """
    import math
    import time
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from ..utils.rate_limiter import TokenBucketRateLimiter, RateLimitError

    n_products_per_category = n_products_per_category or config.PRODUCTS_PER_CATEGORY
    chunk_size = config.CHUNK_SIZE

    # Initialize rate limiter
    effective_rpm = int(config.TIER_1_RPM_LIMIT * config.RATE_LIMIT_BUFFER)
    rate_limiter = TokenBucketRateLimiter(max_rpm=effective_rpm)

    # Create session ID and output file
    session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    if output_filename:
        output_file = f"{config.OUTPUT_DIR}/{output_filename}"
    else:
        output_file = f"{config.OUTPUT_DIR}/fashion_products_{session_id}.csv"

    # Calculate total chunks across all categories
    chunks_per_category = math.ceil(n_products_per_category / chunk_size)
    total_chunks = len(CATEGORY_LEAVES) * chunks_per_category

    # Try to resume from checkpoint
    checkpoint_manager = None
    csv_writer = None

    if resume and config.ENABLE_CHECKPOINTS:
        # If custom output filename is specified, only resume if checkpoint matches
        if output_filename:
            # Look for checkpoint matching this specific output file
            latest_checkpoint = CheckpointManager.find_checkpoint_by_output(
                config.CHECKPOINT_DIR, output_file
            )
            if latest_checkpoint:
                print(f" Found matching checkpoint: {latest_checkpoint}")
                checkpoint_manager = CheckpointManager.load_existing(latest_checkpoint)
                if checkpoint_manager:
                    csv_writer = IncrementalCSVWriter(output_file)
                    csv_writer.resume_from_existing()
                    print(f" Resuming with existing CSV: {output_file}")
                    print(f" {checkpoint_manager.get_progress_summary()}")
            else:
                print(f" No checkpoint found for {output_file} - starting fresh")
        else:
            # No custom filename - resume from latest checkpoint
            latest_checkpoint = CheckpointManager.find_latest_checkpoint(config.CHECKPOINT_DIR)
            if latest_checkpoint:
                print(f" Found checkpoint: {latest_checkpoint}")
                checkpoint_manager = CheckpointManager.load_existing(latest_checkpoint)
                if checkpoint_manager:
                    output_file = checkpoint_manager.output_file
                    csv_writer = IncrementalCSVWriter(output_file)
                    csv_writer.resume_from_existing()

                    print(f" Resuming with existing CSV: {output_file}")
                    print(f" {checkpoint_manager.get_progress_summary()}")

    # Initialize new session if no checkpoint
    if checkpoint_manager is None:
        checkpoint_manager = CheckpointManager(
            session_id=session_id,
            output_file=output_file,
            generation_config={
                "products_per_category": n_products_per_category,
                "total_categories": len(CATEGORY_LEAVES),
                "chunk_size": chunk_size,
                "chunks_per_category": chunks_per_category,
                "total_chunks": total_chunks,
                "checkpoint_dir": config.CHECKPOINT_DIR
            }
        )
        csv_writer = IncrementalCSVWriter(output_file)
        csv_writer.write_header()

    print(f"\n Generating {n_products_per_category} products per category")
    print(f" Total categories: {len(CATEGORY_LEAVES)}")
    print(f" Chunk size: {chunk_size} products per API call")
    print(f" Total chunks: {total_chunks} ({chunks_per_category} per category)")
    if parallel:
        print(f" Parallel workers: {config.PARALLEL_WORKERS}")
        print(f" Rate limit: {effective_rpm} RPM (buffered)")
    print(f" Output file: {output_file}\n")

    # Build work queue (all chunks to generate)
    work_queue = []
    for category_index, category in enumerate(CATEGORY_LEAVES):
        for chunk_index in range(chunks_per_category):
            # Skip if already completed
            if checkpoint_manager.is_chunk_completed(category_index, chunk_index):
                continue

            # Calculate products for this chunk
            remaining_in_category = n_products_per_category - (chunk_index * chunk_size)
            products_in_chunk = min(chunk_size, remaining_in_category)

            work_queue.append({
                "category_index": category_index,
                "category": category,
                "chunk_index": chunk_index,
                "products_in_chunk": products_in_chunk,
            })

    print(f" Work queue: {len(work_queue)} chunks to generate")
    print(f" {checkpoint_manager.get_progress_summary()}\n")

    # Worker function
    def generate_chunk_worker(work_item):
        """Worker function to generate a single chunk"""
        category_index = work_item["category_index"]
        category = work_item["category"]
        chunk_index = work_item["chunk_index"]
        products_in_chunk = work_item["products_in_chunk"]

        try:
            # Acquire rate limit token
            wait_time = rate_limiter.acquire(tokens=1)
            if wait_time > 0:
                print(f"  [Worker] Rate limit: waited {wait_time:.2f}s")

            # Generate chunk
            df = generate_batch_csv_stratified(
                generator=generator,
                categories=[category],
                n_per_category=products_in_chunk,
            )

            # Write and checkpoint (thread-safe)
            csv_writer.append_batch(df)
            checkpoint_manager.mark_chunk_complete(category_index, chunk_index, len(df))

            return {
                "status": "success",
                "category_index": category_index,
                "chunk_index": chunk_index,
                "products": len(df),
            }

        except RateLimitError as e:
            # Rate limit hit - save checkpoint immediately
            checkpoint_manager.mark_chunk_failed(category_index, chunk_index, str(e))
            return {
                "status": "rate_limit",
                "category_index": category_index,
                "chunk_index": chunk_index,
                "error": str(e),
            }

        except Exception as e:
            # Other error
            checkpoint_manager.mark_chunk_failed(category_index, chunk_index, str(e))
            return {
                "status": "failed",
                "category_index": category_index,
                "chunk_index": chunk_index,
                "error": str(e),
            }

    # Process chunks
    if parallel and len(work_queue) > 1:
        # Parallel execution
        with ThreadPoolExecutor(max_workers=config.PARALLEL_WORKERS) as executor:
            futures = {executor.submit(generate_chunk_worker, work): work for work in work_queue}

            for future in as_completed(futures):
                work = futures[future]
                result = future.result()

                category_name = CATEGORY_LEAVES[result["category_index"]]["name"]

                if result["status"] == "success":
                    print(f"[{category_name}] Chunk {result['chunk_index'] + 1} completed ({result['products']} products)")
                    print(f"  {checkpoint_manager.get_progress_summary()}")

                elif result["status"] == "rate_limit":
                    print(f"\nRATE LIMIT HIT - Pausing for {config.RATE_LIMIT_PAUSE_SECONDS}s")
                    print(f"  Checkpoint saved. Safe to resume later.")
                    executor.shutdown(wait=False, cancel_futures=True)
                    time.sleep(config.RATE_LIMIT_PAUSE_SECONDS)
                    raise RateLimitError("Rate limit exceeded - checkpoint saved")

                else:
                    print(f"[{category_name}] Chunk {result['chunk_index'] + 1} failed: {result['error']}")
                    executor.shutdown(wait=False, cancel_futures=True)
                    raise RuntimeError(result["error"])

    else:
        # Sequential execution (fallback)
        for work in work_queue:
            result = generate_chunk_worker(work)
            category_name = CATEGORY_LEAVES[result["category_index"]]["name"]

            if result["status"] == "success":
                print(f"[{category_name}] Chunk {result['chunk_index'] + 1} completed ({result['products']} products)")
            elif result["status"] == "rate_limit":
                print(f"\nRATE LIMIT HIT - Pausing for {config.RATE_LIMIT_PAUSE_SECONDS}s")
                time.sleep(config.RATE_LIMIT_PAUSE_SECONDS)
                raise RateLimitError("Rate limit exceeded")
            else:
                print(f"[{category_name}] Chunk {result['chunk_index'] + 1} failed: {result['error']}")
                raise RuntimeError(result["error"])

    # Mark complete and cleanup
    checkpoint_manager.mark_complete()
    print(f"\n Generation complete!")

    # Optionally cleanup checkpoint
    if config.ENABLE_CHECKPOINTS:
        checkpoint_manager.cleanup()

    # Load final CSV
    final_df = pd.read_csv(output_file)

    print(f"Generated {len(final_df)} total products")
    print(f"  Categories covered: {final_df['category'].nunique()}")
    print(f"  Output: {output_file}")

    return final_df
