"""
Dataset analysis and statistics
"""

import pandas as pd
from collections import Counter
from typing import Dict, List


def analyze_dataset(df: pd.DataFrame) -> Dict:
    """
    Perform comprehensive analysis on the dataset.

    Args:
        df: DataFrame with product data

    Returns:
        Dictionary with analysis results
    """
    results = {}

    # Basic statistics
    results["total_products"] = len(df)
    results["categories_covered"] = df["category"].nunique()

    # Category distribution
    results["category_distribution"] = df["category"].value_counts().to_dict()

    # Country distribution
    results["country_distribution"] = (
        df["manufacturer_country"].value_counts().head(10).to_dict()
    )

    # Weight statistics
    if "weight_kg" in df.columns:
        results["weight_stats"] = {
            "mean": df["weight_kg"].mean(),
            "std": df["weight_kg"].std(),
            "min": df["weight_kg"].min(),
            "max": df["weight_kg"].max(),
            "median": df["weight_kg"].median(),
        }

    # Distance statistics
    if "total_distance_km" in df.columns:
        results["distance_stats"] = {
            "mean": df["total_distance_km"].mean(),
            "std": df["total_distance_km"].std(),
            "min": df["total_distance_km"].min(),
            "max": df["total_distance_km"].max(),
            "median": df["total_distance_km"].median(),
        }

    # Material analysis
    all_materials = []
    if "materials" in df.columns:
        import json
        for materials_str in df["materials"].dropna():
            try:
                materials_dict = json.loads(materials_str)
                all_materials.extend(materials_dict.keys())
            except (json.JSONDecodeError, AttributeError):
                # Skip invalid JSON
                continue

    material_counts = Counter(all_materials)
    results["total_material_entries"] = len(all_materials)
    results["unique_materials"] = len(material_counts)
    results["top_materials"] = dict(material_counts.most_common(10))

    return results


def print_analysis(analysis: Dict):
    """
    Print analysis results in a formatted way.

    Args:
        analysis: Dictionary from analyze_dataset()
    """
    print("DATASET ANALYSIS")

    print(f"Overview:")
    print(f"  Total products: {analysis['total_products']}")
    print(f"  Categories covered: {analysis['categories_covered']}")

    print(f"\nCategory Distribution:")
    for category, count in sorted(
        analysis["category_distribution"].items(), key=lambda x: x[1], reverse=True
    )[:10]:
        print(f"  {category}: {count}")

    print(f"\nTop 10 Manufacturing Countries:")
    for country, count in sorted(
        analysis["country_distribution"].items(), key=lambda x: x[1], reverse=True
    ):
        print(f"  {country}: {count}")

    if "weight_stats" in analysis:
        print(f"\nWeight Statistics (kg):")
        for stat, value in analysis["weight_stats"].items():
            print(f"  {stat}: {value:.3f}")

    if "distance_stats" in analysis:
        print(f"\nDistance Statistics (km):")
        for stat, value in analysis["distance_stats"].items():
            print(f"  {stat}: {value:.1f}")

    print(f"\nMaterial Analysis:")
    print(f"  Total material entries: {analysis['total_material_entries']}")
    print(f"  Unique materials used: {analysis['unique_materials']}")
    print(f"\nTop 10 Materials:")
    for material, count in analysis["top_materials"].items():
        print(f"  {material}: {count}")

    print("\n" + "=" * 80)


def export_analysis(analysis: Dict, filepath: str):
    """
    Export analysis results to a text file.

    Args:
        analysis: Dictionary from analyze_dataset()
        filepath: Path to save the analysis
    """
    import json
    import numpy as np

    # Custom encoder to handle numpy types
    def default_converter(obj):
        if isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")

    with open(filepath, "w") as f:
        json.dump(analysis, f, indent=2, default=default_converter)

    print(f" Analysis exported to: {filepath}")
