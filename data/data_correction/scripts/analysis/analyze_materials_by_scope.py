#!/usr/bin/env python3
"""
Analyze Materials by Parent Category (Scope)
Generates a plot showing material usage distribution for each parent category.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
from collections import Counter

INPUT_CSV = "output/Product_data_cleaned.csv"
OUTPUT_PLOT = "output/analysis/materials_by_scope.png"
DPI = 300

def main():
    print("=" * 80)
    print("ANALYZING MATERIALS BY SCOPE")
    print("=" * 80)
    
    # Load data
    print(f"Loading Loading data from: {INPUT_CSV}")
    df = pd.read_csv(INPUT_CSV, low_memory=False)
    
    # Aggregate materials by parent category
    print("Searching Aggregating material counts...")
    
    parent_materials = {}
    
    for idx, row in df.iterrows():
        parent = row['parent_category']
        if pd.isna(parent):
            continue
            
        if parent not in parent_materials:
            parent_materials[parent] = Counter()
            
        try:
            # Handle potential single quotes or formatting issues
            mat_str = row['materials'].replace("'", '"')
            materials = json.loads(mat_str)
            
            # Count each material occurrence (not weighting by composition share for this count, just usage frequency)
            # User asked for "amount they are used", frequency is usually what's meant in this context
            for m in materials.keys():
                parent_materials[parent][m] += 1
        except:
            continue
            
    # Convert to DataFrame for plotting
    data = []
    for parent, counts in parent_materials.items():
        total = sum(counts.values())
        for mat, count in counts.items():
            data.append({
                'Parent Category': parent,
                'Material': mat,
                'Count': count,
                'Percentage': (count / total) * 100
            })
            
    df_mats = pd.DataFrame(data)
    
    # Filter for top materials to keep plot readable
    # Keep top 10 materials overall
    top_materials = df_mats.groupby('Material')['Count'].sum().nlargest(12).index
    df_plot = df_mats[df_mats['Material'].isin(top_materials)].copy()
    
    # Plot
    print(f"Data Generating plot: {OUTPUT_PLOT}")
    
    # Pivot for stacked bar chart
    pivot = df_plot.pivot(index='Parent Category', columns='Material', values='Count').fillna(0)
    
    # Sort parents by total count
    pivot['total'] = pivot.sum(axis=1)
    pivot = pivot.sort_values('total', ascending=True)
    pivot = pivot.drop('total', axis=1)
    
    # Plot
    plt.figure(figsize=(14, 8))
    pivot.plot(kind='barh', stacked=True, colormap='tab20', figsize=(14, 8), width=0.8)
    
    plt.title('Material Usage by Sector (Top 12 Materials)', fontsize=16, fontweight='bold')
    plt.xlabel('Number of Products Using Material', fontsize=12)
    plt.ylabel('Sector (Parent Category)', fontsize=12)
    plt.legend(title='Material', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(axis='x', linestyle='--', alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(OUTPUT_PLOT, dpi=DPI, bbox_inches='tight')
    print("OK Saved")
    
    # Print summary table
    print("\nSummary (Top 5 Materials per Sector):")
    for parent in parent_materials:
        print(f"\n{parent}:")
        for m, c in parent_materials[parent].most_common(5):
            print(f"  - {m}: {c}")

if __name__ == "__main__":
    main()
