"""
Material, Category, and Country Vocabularies
"""

import json
from pathlib import Path

# ============================================================================
# CATEGORIES - Loaded from categories_rows.json
# ============================================================================

def load_categories():
    """Load hierarchical categories from categories_rows.json"""
    json_path = Path(__file__).parent.parent.parent / "data" / "categories_rows.json"
    with open(json_path, 'r') as f:
        return json.load(f)

CATEGORIES_DATA = load_categories()
CATEGORIES_BY_ID = {cat['id']: cat for cat in CATEGORIES_DATA}

def get_hierarchy_info(category):
    """
    Get gender, parent_category, and category name from hierarchy.
    
    Args:
        category: Category dict from CATEGORIES_DATA
        
    Returns:
        dict with keys: gender, parent_category, category, full_path
    """
    path = []
    current = category
    while current:
        path.insert(0, current)
        parent_id = current['parent_id']
        current = CATEGORIES_BY_ID.get(parent_id) if parent_id else None
    
    return {
        'gender': path[0]['name'] if len(path) > 0 else None,        # Men's/Women's
        'parent_category': path[1]['name'] if len(path) > 1 else None,  # Bottoms/Tops/etc
        'category': path[-1]['name'],                                 # Leaf category
        'full_path': ' > '.join([p['name'] for p in path])
    }

# Get leaf categories (those with no children)
def get_leaf_categories():
    """Get all leaf categories (final product types with no children)"""
    children_map = {}
    for cat in CATEGORIES_DATA:
        pid = cat['parent_id']
        if pid not in children_map:
            children_map[pid] = []
        children_map[pid].append(cat)
    
    leaf_ids = set(cat['id'] for cat in CATEGORIES_DATA) - \
               set(p for p in children_map.keys() if p is not None)
    return [cat for cat in CATEGORIES_DATA if cat['id'] in leaf_ids]

CATEGORY_LEAVES = get_leaf_categories()  # 74 leaf categories
CATEGORY_NAMES = [cat['name'] for cat in CATEGORY_LEAVES]  # For compatibility

# ============================================================================
# MATERIALS - Ecoinvent-verified combinations by category
# ============================================================================

# Material combinations for all 74 leaf categories
# All materials are ecoinvent-verified (v3.8+)
MATERIAL_COMBINATIONS = {
    # BOTTOMS
    "Casual Pants": ["cotton_conventional", "linen_flax", "hemp", "lyocell_tencel", "elastane"],
    "Cropped Pants": ["cotton_conventional", "linen_flax", "elastane"],
    "Denim": ["cotton_conventional", "cotton_organic", "cotton_recycled", "hemp", "polyester_virgin", "elastane"],
    "Jeans": ["cotton_conventional", "cotton_organic", "cotton_recycled", "hemp", "polyester_virgin", "elastane"],
    "Joggers": ["cotton_conventional", "polyester_recycled", "elastane"],
    "Jumpsuits": ["viscose", "cotton_conventional", "elastane"],
    "Leggings": ["polyester_virgin", "elastane", "polyamide_6"],
    "Maxi Skirts": ["viscose", "cotton_conventional", "polyester_virgin", "elastane"],
    "Midi Skirts": ["viscose", "cotton_conventional", "polyester_virgin"],
    "Mini Skirts": ["cotton_conventional", "polyester_virgin", "elastane"],
    "Pants": ["cotton_conventional", "wool_generic", "polyester_virgin", "elastane"],
    "Shorts": ["cotton_conventional", "polyester_virgin"],
    "Sweatpants": ["cotton_conventional", "polyester_virgin", "elastane"],
    "Sweatpants & Joggers": ["cotton_conventional", "polyester_recycled", "elastane"],
    "Swimwear": ["polyamide_66", "elastane", "polyester_virgin"],
    
    # DRESSES
    "Gowns": ["silk", "polyester_virgin", "viscose"],
    "Maxi": ["cotton_conventional", "viscose", "modal", "lyocell_tencel"],
    "Midi": ["viscose", "polyester_virgin", "lyocell_tencel", "elastane"],
    "Mini": ["cotton_conventional", "viscose", "lyocell_tencel", "elastane"],
    
    # FOOTWEAR
    "Athletic Shoes": ["polyester_virgin", "eva", "natural_rubber", "tpu"],
    "Boots": ["leather_bovine", "leather_ovine", "suede", "nubuck", "shearling_faux", "synthetic_rubber_sbr", "eva", "metal_steel"],
    "Casual Shoes": ["cotton_conventional", "leather_bovine", "leather_ovine", "suede", "nubuck", "natural_rubber", "eva"],
    "Dress Shoes": ["leather_bovine", "leather_ovine", "patent_leather", "natural_rubber", "eva"],
    "Flats": ["leather_bovine", "patent_leather", "jute_sole", "eva"],
    "Heels": ["leather_bovine", "patent_leather", "suede", "eva", "tpu"],
    "Loafers": ["leather_bovine", "suede", "nubuck", "natural_rubber"],
    "Sandals": ["leather_bovine", "jute", "jute_sole", "eva", "natural_rubber"],
    "Sneakers": ["leather_bovine", "suede", "nubuck", "polyester_virgin", "eva", "natural_rubber"],
    
    # OUTERWEAR
    "Blazers": ["wool_generic", "polyester_virgin", "viscose"],
    "Bombers": ["polyamide_6", "polyamide_recycled", "polyester_virgin"],
    "Cloaks & Capes": ["wool_generic", "cashmere", "viscose"],
    "Coats": ["wool_generic", "cashmere", "polyester_virgin"],
    "Denim Jackets": ["cotton_conventional", "polyester_virgin", "metal_brass"],
    "Down Jackets": ["down_feather", "down_synthetic", "polyamide_66", "polyester_virgin", "membrane_tpu"],
    "Fur & Faux Fur": ["acrylic", "polyester_virgin"],
    "Heavy Coats": ["wool_generic", "wool_merino", "cashmere", "down_feather", "polyester_virgin", "viscose"],
    "Jackets": ["polyester_virgin", "suede", "shearling_faux", "polyamide_6"],
    "Leather Jackets": ["leather_bovine", "leather_ovine", "suede", "viscose", "metal_brass"],
    "Light Jackets": ["polyester_virgin", "cotton_conventional", "polyamide_6"],
    "Parkas": ["down_feather", "down_synthetic", "polyamide_6", "polyester_virgin"],
    "Rain Jackets": ["polyester_virgin", "polyamide_6", "membrane_tpu", "coated_fabric_pu"],
    "Raincoats": ["polyester_virgin", "coated_fabric_pu", "membrane_tpu"],
    "Vests": ["down_synthetic", "polyamide_6", "polyester_virgin", "wool_generic"],
    
    # TOPS
    "Blouses": ["viscose", "silk", "lyocell_tencel", "polyester_virgin"],
    "Bodysuits": ["cotton_conventional", "elastane"],
    "Button-Ups": ["cotton_conventional", "linen_flax", "hemp", "lyocell_tencel"],
    "Crop Tops": ["cotton_conventional", "elastane"],
    "Hoodies": ["cotton_conventional", "cotton_recycled", "polyester_recycled", "elastane"],
    "Jerseys": ["polyester_virgin"],
    "Long Sleeve Shirts": ["cotton_conventional", "linen_flax", "wool_merino", "lyocell_tencel", "polyester_virgin"],
    "Polos": ["cotton_conventional", "polyester_virgin", "elastane"],
    "Short Sleeve Shirts": ["cotton_conventional", "linen_flax", "hemp", "lyocell_tencel"],
    "Sleeveless": ["cotton_conventional", "modal", "elastane"],
    "Sweaters": ["wool_generic", "wool_merino", "cashmere", "cotton_conventional", "acrylic"],
    "Sweaters & Knitwear": ["wool_merino", "wool_generic", "acrylic", "cashmere", "polyamide_recycled"],
    "Sweatshirts": ["cotton_conventional", "cotton_recycled", "polyester_virgin"],
    "Sweatshirts & Hoodies": ["cotton_conventional", "polyester_virgin"],
    "Tank Tops": ["cotton_conventional", "modal", "elastane"],
}

# Extract MATERIAL_VOCAB from actual combinations
# This ensures we only include materials actually used
MATERIAL_VOCAB = sorted(list(set(
    material 
    for materials in MATERIAL_COMBINATIONS.values() 
    for material in materials
)))

# ============================================================================
# COUNTRIES - Manufacturing locations
# ============================================================================

# ISO 3166-1 alpha-2 country codes
COUNTRIES = [
    "AD", "AE", "AF", "AG", "AI", "AL", "AM", "AO", "AQ", "AR",
    "AS", "AT", "AU", "AW", "AX", "AZ", "BA", "BB", "BD", "BE",
    "BF", "BG", "BH", "BI", "BJ", "BL", "BM", "BN", "BO", "BQ",
    "BR", "BS", "BT", "BV", "BW", "BY", "BZ", "CA", "CC", "CD",
    "CF", "CG", "CH", "CI", "CK", "CL", "CM", "CN", "CO", "CR",
    "CU", "CV", "CW", "CX", "CY", "CZ", "DE", "DJ", "DK", "DM",
    "DO", "DZ", "EC", "EE", "EG", "EH", "ER", "ES", "ET", "FI",
    "FJ", "FK", "FM", "FO", "FR", "GA", "GB", "GD", "GE", "GF",
    "GG", "GH", "GI", "GL", "GM", "GN", "GP", "GQ", "GR", "GS",
    "GT", "GU", "GW", "GY", "HK", "HM", "HN", "HR", "HT", "HU",
    "ID", "IE", "IL", "IM", "IN", "IO", "IQ", "IR", "IS", "IT",
    "JE", "JM", "JO", "JP", "KE", "KG", "KH", "KI", "KM", "KN",
    "KP", "KR", "KW", "KY", "KZ", "LA", "LB", "LC", "LI", "LK",
    "LR", "LS", "LT", "LU", "LV", "LY", "MA", "MC", "MD", "ME",
    "MF", "MG", "MH", "MK", "ML", "MM", "MN", "MO", "MP", "MQ",
    "MR", "MS", "MT", "MU", "MV", "MW", "MX", "MY", "MZ", "NA",
    "NC", "NE", "NF", "NG", "NI", "NL", "NO", "NP", "NR", "NU",
    "NZ", "OM", "PA", "PE", "PF", "PG", "PH", "PK", "PL", "PM",
    "PN", "PR", "PS", "PT", "PW", "PY", "QA", "RE", "RO", "RS",
    "RU", "RW", "SA", "SB", "SC", "SD", "SE", "SG", "SH", "SI",
    "SJ", "SK", "SL", "SM", "SN", "SO", "SR", "SS", "ST", "SV",
    "SX", "SY", "SZ", "TC", "TD", "TF", "TG", "TH", "TJ", "TK",
    "TL", "TM", "TN", "TO", "TR", "TT", "TV", "TW", "TZ", "UA",
    "UG", "UM", "US", "UY", "UZ", "VA", "VC", "VE", "VG", "VI",
    "VN", "VU", "WF", "WS", "YE", "YT", "ZA", "ZM", "ZW"
]
