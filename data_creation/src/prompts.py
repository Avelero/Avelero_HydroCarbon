"""
Prompt building functions for Gemini generation
"""

from vocabularies import MATERIAL_VOCAB, MATERIAL_COMBINATIONS, COUNTRIES, get_hierarchy_info


def build_generation_prompt_csv(
    categories_to_generate: list,
    countries_subset: list = None,
    n_per_category: int = 2
) -> str:
    """
    Build a prompt for generating CSV fashion product data.

    Args:
        categories_to_generate: List of category dicts from CATEGORY_LEAVES
        countries_subset: Optional subset of countries to use
        n_per_category: Number of products per category

    Returns:
        Formatted prompt string
    """
    materials_str = ", ".join(MATERIAL_VOCAB)
    countries_str = ", ".join(countries_subset if countries_subset else COUNTRIES)
    total_items = len(categories_to_generate) * n_per_category
    
    # Build category info with hierarchy
    categories_info = []
    for cat in categories_to_generate:
        info = get_hierarchy_info(cat)
        # Get material suggestions for this category
        mat_suggestions = MATERIAL_COMBINATIONS.get(info['category'], [])
        mat_str = ", ".join(mat_suggestions[:3])  # Show first 3 materials
        categories_info.append(
            f"{info['category']} (Gender: {info['gender']}, Parent: {info['parent_category']}, Suggested materials: {mat_str})"
        )
    categories_str = "\n  ".join(categories_info)

    prompt = f"""Generate {total_items} realistic fashion products as CSV data with natural, varied numeric values.

CRITICAL FORMATTING REQUIREMENTS:
1. Output MUST be valid CSV format with proper quoting
2. Each row MUST have EXACTLY 8 fields (columns)
3. Fields containing commas or quotes MUST be wrapped in double quotes
4. All numeric values (weights, distances) MUST have natural variance and precision
5. NEVER use round numbers - use realistic decimals like 0.743, 1.267, 8456, 11834

Generate EXACTLY {n_per_category} product(s) for EACH of these categories:
  {categories_str}

Output CSV with this EXACT header (do NOT repeat this header in your output):
product_name,gender,parent_category,category,manufacturer_country,materials,weight_kg,total_distance_km

EXAMPLE OUTPUT (3 rows):
"Classic Denim Jeans",Female,Bottoms,Jeans,BD,"{{""cotton_conventional"": 0.72, ""elastane"": 0.28}}",0.847,12456.73
"Leather Ankle Boots",Male,Footwear,Boots,CN,"{{""leather_bovine"": 0.85, ""rubber_synthetic"": 0.15}}",1.234,9876.21
"Floral Maxi Dress",Female,Dresses,Maxi Dresses,IN,"{{""viscose"": 0.65, ""polyester_virgin"": 0.35}}",0.512,11234.89

RULES:
- product_name: Generate realistic product names (e.g., "Classic Denim Jeans", "Leather Ankle Boots")
- gender: "Male" or "Female" (based on category hierarchy gender)
- parent_category: Parent category name (Bottoms, Tops, Dresses, Footwear, Outerwear)
- category: Exact category name from the list above
- manufacturer_country: ISO-2 country code from: {countries_str}
- materials: JSON object with material names as keys and composition shares (0.0-1.0) as values
  * Format: {{"material_name": share, "material_name2": share2}}
  * Example: {{"cotton_conventional": 0.7, "polyester_virgin": 0.25, "elastane": 0.05}}
  * Can have 1-4 materials per product
  * Material shares MUST sum to exactly 1.0
  * Use suggested materials for each category when possible
  * Available materials: {materials_str}
- weight_kg: Realistic product weight in kilograms with natural variance (NOT round numbers)
  * Lightweight apparel (t-shirts, tank tops, underwear): 0.12-0.28 kg (e.g., 0.167, 0.213, 0.248)
  * Medium apparel (shirts, pants, dresses): 0.32-0.76 kg (e.g., 0.437, 0.568, 0.691)
  * Heavy apparel (coats, jackets, jeans): 0.83-1.87 kg (e.g., 0.947, 1.234, 1.672)
  * Light footwear (sandals, slippers): 0.31-0.58 kg (e.g., 0.374, 0.429, 0.513)
  * Medium footwear (sneakers, casual shoes): 0.62-1.17 kg (e.g., 0.743, 0.891, 1.064)
  * Heavy footwear (boots): 1.23-2.38 kg (e.g., 1.367, 1.829, 2.145)
  * Small accessories (scarves, hats, gloves): 0.06-0.19 kg (e.g., 0.087, 0.134, 0.176)
  * Medium accessories (bags, small backpacks): 0.23-0.77 kg (e.g., 0.328, 0.519, 0.684)
  * Large accessories (large bags, luggage): 0.84-1.93 kg (e.g., 1.127, 1.458, 1.763)
  * USE VARIED DECIMAL VALUES - avoid round numbers like 0.5, 1.0, 1.5
- total_distance_km: Realistic supply chain distance in kilometers with natural variance (NOT round numbers)
  * Consider manufacturing location realistically: Asia to Europe/US typically 8000-15000 km, local production 600-1800 km
  * Local/domestic production: 520-1940 km (e.g., 743, 1267, 1538)
  * Regional (nearby countries): 2130-4870 km (e.g., 2847, 3392, 4156)
  * Continental (same continent): 5240-9730 km (e.g., 6184, 7529, 8916)
  * Intercontinental (cross-ocean): 10300-23800 km (e.g., 11764, 15428, 19237)
  * Consider realistic shipping routes: China to Europe ≈ 10000-12000 km, Bangladesh to US ≈ 12000-14000 km, Vietnam to Europe ≈ 9000-11000 km
  * USE VARIED VALUES WITH 2-3 DECIMAL PLACES - avoid round numbers like 5000, 10000, 15000, 20000, 25000
- NO header repetition
- NO explanations or notes
- ONLY {total_items} data rows

Output {total_items} CSV rows:"""

    return prompt
