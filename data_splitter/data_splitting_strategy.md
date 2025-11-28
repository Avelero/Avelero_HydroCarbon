# Data Splitting Strategy for Machine Learning

## Overview
This document outlines the strategy for splitting the `Product_data_cleaned.csv` dataset into Training, Validation, and Test sets. 

**Goal:** Create robust splits that ensure the model learns to generalize well, specifically addressing the "long tail" of rare materials identified in the analysis.

## The Challenge: Class Imbalance
Our analysis revealed a significant imbalance in materials:
- **Head:** Polyester, Cotton, Elastane (>300k samples)
- **Tail:** Jute, Cashmere, Silk (<100 samples)

**Risk:** A standard random split might put *all* 58 examples of "Jute" into the Training set, leaving 0 for Testing. The model would learn "Jute" but we would never know if it generalizes, or vice versa.

## Recommended Strategy: Stratified Sampling by Rare Feature

Instead of splitting randomly or just by Category, we must stratify by the **Rarest Material** present in each product.

### 1. Define "Rarity"
Calculate the frequency of every material in the dataset.
- `High Frequency`: >10,000 samples
- `Medium Frequency`: 1,000 - 10,000 samples
- `Low Frequency`: <1,000 samples

### 2. Assign Stratification Label
For each product:
1.  Look at its material composition (e.g., `{'cotton': 0.9, 'jute': 0.1}`).
2.  Identify the material with the *lowest* global frequency (in this case, `jute`).
3.  Assign this product to the stratum: `Stratum = Jute`.

If a product only has common materials (e.g., `cotton`, `polyester`), assign it to `Stratum = Common`.

### 3. Perform Stratified Split
Use `sklearn.model_selection.train_test_split` with the `stratify` parameter set to this new `Stratum` label.

**Recommended Ratios:**
- **Train:** 80%
- **Validation:** 10%
- **Test:** 10%

### 4. Handling "Too Rare" Classes
For materials with <10 samples (e.g., `polyester_organic` with 15 samples):
- **Option A:** Ensure at least 1 sample is in Test and 1 in Val (requires custom splitter).
- **Option B (Recommended):** Group them into an "Other/Rare" stratum for splitting purposes, but keep the raw labels for training.

## Implementation Steps (Future `split_data.py`)

1.  **Load Data:** Read `Product_data_cleaned.csv`.
2.  **Count Materials:** Create a frequency map of all materials.
3.  **Create Stratum Column:**
    ```python
    def get_rarest_material(materials_json):
        mats = json.loads(materials_json)
        # Return material with lowest global count
        return min(mats, key=lambda m: global_counts[m])
    
    df['stratum'] = df['materials'].apply(get_rarest_material)
    ```
4.  **Split:**
    ```python
    from sklearn.model_selection import train_test_split
    
    train, temp = train_test_split(df, test_size=0.2, stratify=df['stratum'], random_state=42)
    val, test = train_test_split(temp, test_size=0.5, stratify=temp['stratum'], random_state=42)
    ```
5.  **Verify:** Check that rare materials exist in all three sets.
6.  **Save:** `train.csv`, `val.csv`, `test.csv`.

## Benefits
- **Guaranteed Representation:** Rare materials are forced into the Test set, ensuring we can evaluate model performance on the "long tail".
- **Robustness:** Prevents the model from overfitting to common classes while ignoring rare ones.
