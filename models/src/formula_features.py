"""
Formula Features Module

Calculate physics-based formula features to blend with ML predictions.
Uses actual formulas from data_calculations module.
"""

import pandas as pd
import numpy as np
from typing import Dict

# Material emission factors (loaded from material_dataset_final.csv)
MATERIAL_CARBON_FACTORS = {}
MATERIAL_WATER_FACTORS = {}

# Average weighted emission factor for transport (gCO2e/tkm)
# Simplified approximation based on typical modal split
AVG_WEIGHTED_EF = 50.0  # gCO2e/tkm


def load_material_factors(material_dataset_path: str) -> None:
    """
    Load material carbon and water emission factors from CSV.
    
    Args:
        material_dataset_path: Path to material_dataset_final.csv
    """
    global MATERIAL_CARBON_FACTORS, MATERIAL_WATER_FACTORS
    
    print(f"Loading material emission factors from {material_dataset_path}...")
    df = pd.read_csv(material_dataset_path)
    
    # Column indices based on data_calculations README:
    # Column 0: material name
    # Column 7: carbon_footprint_kgCO2e  
    # Column 10: water_footprint_liters
    
    material_col = df.columns[0]
    carbon_col = df.columns[7]
    water_col = df.columns[10]
    
    for _, row in df.iterrows():
        material = row[material_col]
        carbon_factor = row[carbon_col]
        water_factor = row[water_col]
        
        MATERIAL_CARBON_FACTORS[material] = carbon_factor
        MATERIAL_WATER_FACTORS[material] = water_factor
    
    print(f"  Loaded factors for {len(MATERIAL_CARBON_FACTORS)} materials")


def calculate_formula_carbon_material(
    weight_kg: float,
    materials_dict: Dict[str, float],
    carbon_factors: Dict[str, float] = None
) -> float:
    """
    Calculate carbon_material using physics formula:
    carbon_material = Σ(weight × material_i% × carbon_factor_i)
    
    Args:
        weight_kg: Product weight in kg
        materials_dict: Dict of {material_name: percentage (0-1)}
        carbon_factors: Material carbon factors (uses global if None)
        
    Returns:
        Estimated carbon_material in kgCO2e, or NaN if inputs missing
    """
    if pd.isna(weight_kg) or not materials_dict:
        return np.nan
    
    if carbon_factors is None:
        carbon_factors = MATERIAL_CARBON_FACTORS
    
    total_carbon = 0.0
    for material, percentage in materials_dict.items():
        if material in carbon_factors and percentage > 0:
            total_carbon += weight_kg * percentage * carbon_factors[material]
    
    return total_carbon


def calculate_formula_carbon_transport(
    weight_kg: float,
    distance_km: float,
    weighted_ef: float = AVG_WEIGHTED_EF
) -> float:
    """
    Calculate carbon_transport using physics formula:
    carbon_transport = (weight/1000) × distance × (weighted_EF/1000)
    
    Args:
        weight_kg: Product weight in kg
        distance_km: Total distance in km
        weighted_ef: Weighted emission factor in gCO2e/tkm
        
    Returns:
        Estimated carbon_transport in kgCO2e, or NaN if inputs missing
    """
    if pd.isna(weight_kg) or pd.isna(distance_km):
        return np.nan
    
    # Formula from data_calculations/docs/carbon_footprint.md
    carbon_transport = (weight_kg / 1000) * distance_km * (weighted_ef / 1000)
    
    return carbon_transport


def calculate_formula_water_total(
    weight_kg: float,
    materials_dict: Dict[str, float],
    water_factors: Dict[str, float] = None
) -> float:
    """
    Calculate water_total using physics formula:
    water_total = Σ(weight × material_i% × water_factor_i)
    
    Args:
        weight_kg: Product weight in kg
        materials_dict: Dict of {material_name: percentage (0-1)}
        water_factors: Material water factors (uses global if None)
        
    Returns:
        Estimated water_total in liters, or NaN if inputs missing
    """
    if pd.isna(weight_kg) or not materials_dict:
        return np.nan
    
    if water_factors is None:
        water_factors = MATERIAL_WATER_FACTORS
    
    total_water = 0.0
    for material, percentage in materials_dict.items():
        if material in water_factors and percentage > 0:
            total_water += weight_kg * percentage * water_factors[material]
    
    return total_water


def add_formula_features(
    df: pd.DataFrame,
    material_columns: list,
    material_dataset_path: str = None
) -> pd.DataFrame:
    """
    Add formula-based features to DataFrame.
    Adds 3 new columns: formula_carbon_material, formula_carbon_transport, formula_water_total
    
    Args:
        df: DataFrame with weight_kg, total_distance_km, and material columns
        material_columns: List of material column names (one-hot encoded)
        material_dataset_path: Path to load material factors (if not already loaded)
        
    Returns:
        DataFrame with 3 additional formula feature columns
    """
    # Load material factors if not already loaded
    if not MATERIAL_CARBON_FACTORS and material_dataset_path:
        load_material_factors(material_dataset_path)
    
    print("Calculating formula-based features...")
    
    # VECTORIZED CALCULATION (much faster than row-by-row apply)
    # Calculate weighted carbon and water intensity per kg of material mix
    carbon_intensity = np.zeros(len(df))
    water_intensity = np.zeros(len(df))
    
    for mat_col in material_columns:
        if mat_col in df.columns:
            mat_pct = df[mat_col].fillna(0).values
            carbon_factor = MATERIAL_CARBON_FACTORS.get(mat_col, 0)
            water_factor = MATERIAL_WATER_FACTORS.get(mat_col, 0)
            carbon_intensity += mat_pct * carbon_factor
            water_intensity += mat_pct * water_factor
    
    # formula_carbon_material = weight_kg * carbon_intensity
    weight = df['weight_kg'].fillna(0).values
    df['formula_carbon_material'] = weight * carbon_intensity
    
    # formula_carbon_transport = (weight/1000) * distance * (weighted_EF/1000)
    distance = df['total_distance_km'].fillna(0).values
    df['formula_carbon_transport'] = (weight / 1000) * distance * (AVG_WEIGHTED_EF / 1000)
    
    # formula_water_total = weight_kg * water_intensity
    df['formula_water_total'] = weight * water_intensity
    
    # Set to NaN where inputs are missing (to match original behavior)
    weight_missing = df['weight_kg'].isna()
    distance_missing = df['total_distance_km'].isna()
    materials_missing = df[material_columns].sum(axis=1) == 0
    
    df.loc[weight_missing | materials_missing, 'formula_carbon_material'] = np.nan
    df.loc[weight_missing | distance_missing, 'formula_carbon_transport'] = np.nan
    df.loc[weight_missing | materials_missing, 'formula_water_total'] = np.nan
    
    # Report statistics
    n_total = len(df)
    n_carbon_mat = df['formula_carbon_material'].notna().sum()
    n_carbon_trans = df['formula_carbon_transport'].notna().sum()
    n_water = df['formula_water_total'].notna().sum()
    
    print(f"  Formula features calculated:")
    print(f"    formula_carbon_material: {n_carbon_mat}/{n_total} ({100*n_carbon_mat/n_total:.1f}%) available")
    print(f"    formula_carbon_transport: {n_carbon_trans}/{n_total} ({100*n_carbon_trans/n_total:.1f}%) available")
    print(f"    formula_water_total: {n_water}/{n_total} ({100*n_water/n_total:.1f}%) available")
    
    return df


if __name__ == '__main__':
    # Test formula features
    from .data_loader import load_data, get_material_dataset_path, MATERIAL_COLUMNS
    
    X_train, y_train, _, _ = load_data(sample_size=100)
    
    # Add formula features
    X_train = add_formula_features(
        X_train,
        MATERIAL_COLUMNS,
        get_material_dataset_path()
    )
    
    print("\nSample with formula features:")
    print(X_train[['weight_kg', 'total_distance_km', 'formula_carbon_material', 
                    'formula_carbon_transport', 'formula_water_total']].head())
    
    print("\nCompare formula vs actual targets:")
    comparison = pd.DataFrame({
        'actual_carbon_material': y_train['carbon_material'].head(),
        'formula_carbon_material': X_train['formula_carbon_material'].head(),
        'actual_water': y_train['water_total'].head(),
        'formula_water': X_train['formula_water_total'].head()
    })
    print(comparison)
