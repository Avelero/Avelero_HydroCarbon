# Water Footprint Calculation

## Overview

Water footprint measures the total freshwater used in the production of materials. This calculation focuses on the material water footprint - the water consumed in producing the raw materials that make up a product.

## Formula

```
water_total = Σ (W × P_i × WF_i)
```

Where:
- `W` = Product weight (kg)
- `P_i` = Percentage of material i in product (0-1)
- `WF_i` = Water footprint factor for material i (liters/kg)

## Example Calculation

For a 0.5kg product with 80% cotton and 20% polyester:

| Material | Percentage | Water Footprint (L/kg) |
|----------|------------|------------------------|
| Cotton | 80% | 9,113 |
| Polyester | 20% | 60 |

```
water_total = 0.5 × (0.8 × 9113 + 0.2 × 60)
            = 0.5 × (7290.4 + 12)
            = 3651.2 liters
```

## Material Water Footprints

Water footprint values vary significantly by material:

### High Water Footprint Materials
- **Cotton (conventional)**: ~9,113 L/kg
  - Requires significant irrigation
  - Varies by growing region and method
- **Leather**: ~17,000+ L/kg
  - Includes water for animal husbandry
- **Wool**: ~4,000+ L/kg

### Low Water Footprint Materials
- **Polyester (virgin)**: ~60 L/kg
  - Synthetic, petroleum-based
- **Polyester (recycled)**: ~30-40 L/kg
- **Nylon**: ~70 L/kg
- **Elastane**: ~50-100 L/kg

### Natural vs Synthetic

Natural fibers generally have higher water footprints due to:
- Agricultural water needs
- Processing and washing requirements
- Growing season irrigation

Synthetic fibers have lower direct water footprints but:
- Rely on fossil fuels (indirect impacts)
- May cause microplastic pollution

## Data Sources

Water footprint values are derived from:
- Water Footprint Network data
- Life Cycle Assessment (LCA) databases
- Industry-specific studies

Values in `material_dataset_final.csv` column 10: `water_footprint_liters`

## Units

- Input: Weight in kilograms (kg), percentages as decimals (0-1)
- Output: Water footprint in liters (L)

## Limitations

This calculation considers:
- ✅ Blue water (surface and groundwater)
- ✅ Green water (rainwater stored in soil)
- ❌ Grey water (water needed to dilute pollutants) - not fully accounted

Transport water footprint is not included as it's typically negligible compared to material production.
