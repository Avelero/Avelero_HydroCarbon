"""
Configuration settings for dataset generation
"""

import os

# Generation settings
PRODUCTS_PER_CATEGORY = 10  # Number of products to generate per category (per batch iteration)
CHUNK_SIZE = 1000  # Number of products to generate per API call (to avoid rate limits)
BATCH_SIZE = 5  # Number of categories to process at once (DEPRECATED - now using chunks)
MAX_RETRIES = 3  # Maximum retry attempts for failed generations

# Gemini API settings
GEMINI_MODEL = "gemini-2.5-flash"  # Stable, fast, 2000 RPM on Tier 1
# Alternative: "gemini-2.0-flash" (even faster, newer)
# Alternative: "gemini-1.5-pro-002" (better quality, but 1000 RPM limit)

TEMPERATURE = 0.7  # Sampling temperature (0.0-2.0)
TEMPERATURE_MIN = 0.6  # Minimum temperature for variety
TEMPERATURE_MAX = 0.9  # Maximum temperature for variety
TEMPERATURE_VARIANCE = True  # Enable temperature variation
MAX_RETRIES = 3  # Maximum retry attempts

# Tier 1 API Rate Limiting
TIER_1_RPM_LIMIT = 2000  # Requests per minute (Flash models)
TIER_1_TPM_LIMIT = 1000000  # Tokens per minute
REQUEST_DELAY_SECONDS = 2  # Conservative delay between batches
RETRY_DELAY_SECONDS = 5  # Delay before retry on rate limit error
MAX_OUTPUT_TOKENS = 1000000  # Maximum tokens to generate (increased for larger batches)

# Parallel generation settings
PARALLEL_WORKERS = 5  # Number of concurrent workers (conservative: 5 workers * 60s = 300 RPM max)
RATE_LIMIT_BUFFER = 0.8  # Use 80% of rate limit capacity for safety (1600 RPM effective)
RATE_LIMIT_PAUSE_SECONDS = 65  # Pause duration when rate limit is hit (wait for RPM window to reset)

# Weight ranges by category type (kg)
WEIGHT_RANGES = {
    "lightweight_apparel": (0.1, 0.3),      # T-shirts, tank tops, underwear
    "medium_apparel": (0.3, 0.8),           # Shirts, pants, dresses
    "heavy_apparel": (0.8, 2.0),            # Coats, jackets, jeans
    "footwear_light": (0.3, 0.6),           # Sandals, slippers
    "footwear_medium": (0.6, 1.2),          # Sneakers, casual shoes
    "footwear_heavy": (1.2, 2.5),           # Boots, heavy shoes
    "accessories_small": (0.05, 0.2),       # Scarves, gloves, hats
    "accessories_medium": (0.2, 0.8),       # Bags, small backpacks
    "accessories_large": (0.8, 2.0),        # Large bags, luggage
}

# Distance ranges by manufacturing region (km)
DISTANCE_RANGES = {
    "local": (500, 2000),                   # Domestic production
    "regional": (2000, 5000),               # Nearby countries
    "continental": (5000, 10000),           # Same continent
    "intercontinental": (10000, 25000),     # Cross-ocean
}

# Country subset size for batch variety
COUNTRY_SUBSET_SIZE = 10  # Number of countries to include per batch

# Output settings
OUTPUT_DIR = "output"
CSV_HEADER = "product_name,gender,parent_category,category,manufacturer_country,materials,weight_kg,total_distance_km"

# API key (will be loaded from environment variable)
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", None)

# Which implementation to use
# - 'sdk': Uses google-generativeai SDK (requires API key)
IMPLEMENTATION = "sdk"

# Checkpoint settings for resumable generation
CHECKPOINT_DIR = "output/checkpoints"
ENABLE_CHECKPOINTS = True
CHECKPOINT_SAVE_INTERVAL = 1  # Save after every N batches (1 = every batch)

