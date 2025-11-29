# Low-Usage Material Analysis Report

**Analysis Date**: 2025-11-29

**Total Products**: 901,573

**Total Unique Materials**: 39

---

## Material Usage Distribution

| Category | Count | Products Range |
|----------|-------|----------------|
| Very Rare (< 10) | 10 | 1 - 8 |
| Rare (10-100) | 4 | 20 - 83 |
| Uncommon (100-1,000) | 0 | 0 - 0 |
| Common (1,000-10,000) | 4 | 1668 - 8573 |
| Very Common (> 10,000) | 21 | 13309 - 397916 |

---

## Very Rare Materials (< 10 products)

### Issues Found

| Material | Count | Issue Type | Recommended Fix |
|----------|-------|------------|------------------|
| `polyamide_recycled` | 8 | Legitimate (rare) | OK (just very rare) |
| `polyester_conventional` | 6 | Redundant naming | polyester_virgin |
| `cotton_recycled` | 4 | Legitimate (rare) | OK (just very rare) |
| `shearling_faux` | 2 | Legitimate (rare) | OK (faux shearling lining) |
| `polyester_generic` | 2 | Vague naming | polyester_virgin |
| `recycled_polyester` | 2 | Naming inconsistency | polyester_recycled |
| `rayon` | 1 | Missing designation | viscose (rayon = viscose) |
| `cotton_virgin` | 1 | Redundant variant | cotton_conventional |
| `polyester_66` | 1 | Wrong number | polyamide_66 (not polyester) |
| `nylon` | 1 | Synonym | polyamide_6 or polyamide_66 |

### Detailed Analysis

#### 1. Typos and Wrong Materials

- **`polyester_66`** (1 product): Should be `polyamide_66`
  - Polyester doesn't have type 66, that's polyamide (nylon 66)

#### 2. Synonyms and Inconsistencies

- **`nylon`** (1 product): Should be `polyamide_6` or `polyamide_66`
  - Nylon is the common name for polyamide

- **`rayon`** (1 product): Should be `viscose`
  - Rayon and viscose are the same material, dataset uses 'viscose'

- **`nylon_recycled`** (20 products): Should be `polyamide_recycled`
  - For consistency with `polyamide_6` and `polyamide_66`

#### 3. Redundant Naming

- **`polyester_conventional`** (6 products): Should be `polyester_virgin`
  - 'conventional' and 'virgin' mean the same (non-recycled)

- **`recycled_polyester`** (2 products): Should be `polyester_recycled`
  - Inconsistent word order, dataset uses `material_modifier` format

- **`polyester_generic`** (2 products): Should be `polyester_virgin`
  - 'generic' is vague, use standard designation

- **`cotton_virgin`** (1 product): Should be `cotton_conventional`
  - Dataset standard is `cotton_conventional` for non-organic cotton

- **`polyamide_virgin`** (23 products): Should be `polyamide_6` or `polyamide_66`
  - Need to specify type (6 or 66)

#### 4. Legitimate Rare Materials

These materials are correct but just used infrequently:

- **`polyamide_recycled`** (8 products): ✅ Legitimate
- **`cotton_recycled`** (4 products): ✅ Legitimate
- **`shearling_faux`** (2 products): ✅ Legitimate (faux fur lining)

---

## Rare Materials (10-100 products)

| Material | Count | Assessment |
|----------|-------|------------|
| `jute_sole` | 83 | ✅ Legitimate (espadrilles, eco products) |
| `jute` | 58 | ✅ Legitimate (espadrilles, eco products) |
| `polyamide_virgin` | 23 | ⚠️ Needs review |
| `nylon_recycled` | 20 | ⚠️ Needs review |

---

## Summary

**Total Materials to Fix**: 7

**Total Products Affected**: 14

### Recommended Actions

1. **Fix typos**: `polyester_66` → `polyamide_66`
2. **Standardize synonyms**: `nylon` → `polyamide_6/66`, `rayon` → `viscose`
3. **Fix naming inconsistencies**: `recycled_polyester` → `polyester_recycled`
4. **Remove redundant variants**: `polyester_conventional` → `polyester_virgin`
5. **Specify polyamide types**: `polyamide_virgin` → `polyamide_6` or `polyamide_66`

