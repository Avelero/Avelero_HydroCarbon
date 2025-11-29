# Fashion Product Dataset Generator

## Project Structure

```
dataset_creation/
├── config.py           # Configuration settings
├── vocabularies.py     # Materials, categories, countries
├── prompts.py          # Prompt building
├── generator.py        # Gemini integration and generation logic
├── analyzer.py         # Dataset analysis and statistics
├── main.py             # Main entry point
├── requirements.txt    # Dependencies (NO PYTORCH!)
├── .env.example        # Example environment variables
├── output/             # Generated datasets (created automatically)
└── README.md           # This file
```

## Installation

1. **Clone or navigate to the project:**
   ```bash
   cd dataset_creation
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up your API key:**
   ```bash
   # Copy the example env file
   cp .env.example .env

   # Edit .env and add your Google Gemini API key
   # Get a free key from: https://aistudio.google.com/apikey
   ```

   Or export it directly:
   ```bash
   export GOOGLE_API_KEY='your-api-key-here'
   ```

## Usage

### Basic Usage

Generate a full dataset with default settings (2 products per category):

```bash
python main.py
```

### Test Mode

Generate a small test batch first:

```bash
python main.py --test
```

### Custom Settings

Generate more products per category:

```bash
python main.py --products-per-category 5
```

### All Options

```bash
python main.py --help
```

Available options:
- `--products-per-category N` - Number of products per category (default: 2)
- `--test` - Run in test mode (small batch only)
- `--output-dir DIR` - Output directory (default: output/)
- `--no-analysis` - Skip dataset analysis

### Examples

```bash
# Generate 10 products per category
python main.py --products-per-category 10

# Test with custom output directory
python main.py --test --output-dir my_datasets/

# Generate without analysis
python main.py --no-analysis
```

## Output

The script generates:

1. **CSV Dataset**: `fashion_products_YYYYMMDD_HHMMSS.csv`
   - Product name
   - Category
   - Manufacturing country
   - Materials (1-4) with composition shares
   - Weight (kg)
   - Total distance (km)

2. **Analysis Report**: `analysis_YYYYMMDD_HHMMSS.json`
   - Total products and categories
   - Category distribution
   - Country distribution
   - Weight and distance statistics
   - Material usage analysis

## Configuration

Edit `config.py` to customize:

```python
PRODUCTS_PER_CATEGORY = 2   # Products per category
BATCH_SIZE = 5              # Categories per batch
MAX_RETRIES = 3             # Retry attempts
TEMPERATURE = 0.7           # Gemini temperature
GEMINI_MODEL = "gemini-2.0-flash-exp"  # Model to use
```

## Dataset Structure

Generated CSV contains:

| Column | Description |
|--------|-------------|
| `product_name` | Fashion product name |
| `category_name` | Product category (Jeans, Sneakers, etc.) |
| `manufacturer_country` | ISO country code |
| `material_1` to `material_4` | Material types |
| `share_1` to `share_4` | Material composition (sum to 1.0) |
| `weight_kg` | Product weight in kilograms |
| `total_distance_km` | Supply chain distance |

## Vocabulary

### Categories (30 total)
- **Bottoms**: Jeans, Pants, Casual Pants, Cropped Pants, Shorts, Leggings, Sweatpants & Joggers
- **Tops**: Tank Tops, Long/Short Sleeve Shirts, Sweatshirts, Hoodies, Sweaters, Blouses, Polos
- **Dresses**: Mini, Midi, Maxi, Gowns
- **Footwear**: Sneakers, Boots, Sandals, Casual Shoes, Athletic Shoes
- **Outerwear**: Heavy Coats, Light Jackets, Down Jackets, Rain Jackets, Leather Jackets, Parkas

### Materials (60+ types)
- Natural fibers (cotton, linen, wool, silk)
- Synthetics (polyester, nylon, acrylic)
- Leather types
- Rubber and foam materials
- Metals and plastics for trims

## Troubleshooting

**API Key Error:**
```
ValueError: GOOGLE_API_KEY not found in environment variables
```
Solution: Set your API key as shown in Installation step 3.

**Import Error:**
```
ImportError: google-generativeai not installed
```
Solution: Run `pip install -r requirements.txt`

**Rate Limiting:**
If you hit API rate limits, the script will automatically retry with exponential backoff.

## License

MIT License - Feel free to use for research and commercial purposes.
