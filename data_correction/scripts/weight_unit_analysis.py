#!/usr/bin/env python3
"""
Weight Unit Consistency Analysis
Detailed research to detect if weights are mixed between grams and kilograms
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from collections import Counter
from datetime import datetime

# Configuration
INPUT_CSV = "data_correction/output/Product_data_cleaned.csv"
OUTPUT_DIR = "data_correction/output/comprehensive_analysis"
REPORT_FILE = f"{OUTPUT_DIR}/weight_unit_research.md"
DPI = 300

print("=" * 80)
print("WEIGHT UNIT CONSISTENCY ANALYSIS")
print("=" * 80)
print()

# Load data
print(f"📂 Loading data from: {INPUT_CSV}")
df = pd.read_csv(INPUT_CSV, low_memory=False)
print(f"✓ Loaded {len(df):,} rows\n")

# Convert to numeric
df['weight_kg'] = pd.to_numeric(df['weight_kg'], errors='coerce')
df = df.dropna(subset=['weight_kg'])
print(f"After removing NaN weights: {len(df):,} rows\n")

# Initialize report
report = []
report.append("# Weight Unit Consistency Analysis - Detailed Research\n")
report.append(f"**Analysis Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
report.append(f"**Total Products**: {len(df):,}")
report.append(f"**Field Name**: `weight_kg` (supposedly in kilograms)\n")
report.append("---\n")

# ============================================================================
# 1. BASIC DISTRIBUTION ANALYSIS
# ============================================================================

print("=" * 80)
print("1. BASIC DISTRIBUTION ANALYSIS")
print("=" * 80)
print()

report.append("## 1. Basic Distribution Statistics\n")

# Calculate statistics
stats_data = {
    'count': len(df),
    'min': df['weight_kg'].min(),
    'max': df['weight_kg'].max(),
    'mean': df['weight_kg'].mean(),
    'median': df['weight_kg'].median(),
    'std': df['weight_kg'].std(),
    'q1': df['weight_kg'].quantile(0.25),
    'q3': df['weight_kg'].quantile(0.75),
    'iqr': df['weight_kg'].quantile(0.75) - df['weight_kg'].quantile(0.25)
}

print("Weight Statistics (as labeled - 'kg'):")
for key, val in stats_data.items():
    if key == 'count':
        print(f"  {key:10s}: {val:,}")
    else:
        print(f"  {key:10s}: {val:.6f} kg")
print()

report.append("| Statistic | Value (kg) | Value (grams) |")
report.append("|-----------|------------|---------------|")
for key, val in stats_data.items():
    if key == 'count':
        report.append(f"| Count | {val:,} | {val:,} |")
    else:
        report.append(f"| {key.upper()} | {val:.6f} | {val*1000:.3f} |")
report.append("\n")

# ============================================================================
# 2. HISTOGRAM ANALYSIS - LOOKING FOR MULTIMODAL DISTRIBUTION
# ============================================================================

print("=" * 80)
print("2. HISTOGRAM ANALYSIS - DETECTING MULTIPLE CLUSTERS")
print("=" * 80)
print()

report.append("## 2. Distribution Analysis\n")

# Create logarithmic bins to see the full range
log_weights = np.log10(df['weight_kg'])

# Count values in different magnitude ranges
magnitude_ranges = {
    'nano (< 0.001 kg = < 1g)': len(df[df['weight_kg'] < 0.001]),
    'micro (0.001-0.01 kg = 1-10g)': len(df[(df['weight_kg'] >= 0.001) & (df['weight_kg'] < 0.01)]),
    'very_light (0.01-0.1 kg = 10-100g)': len(df[(df['weight_kg'] >= 0.01) & (df['weight_kg'] < 0.1)]),
    'light (0.1-1 kg = 100-1000g)': len(df[(df['weight_kg'] >= 0.1) & (df['weight_kg'] < 1)]),
    'normal (1-5 kg)': len(df[(df['weight_kg'] >= 1) & (df['weight_kg'] < 5)]),
    'heavy (5-10 kg)': len(df[(df['weight_kg'] >= 5) & (df['weight_kg'] < 10)]),
    'very_heavy (> 10 kg)': len(df[df['weight_kg'] >= 10])
}

print("Weight Distribution by Magnitude:")
report.append("### Weight Distribution by Magnitude\n")
report.append("| Range | Count | Percentage | Interpretation |")
report.append("|-------|-------|------------|----------------|")

for range_name, count in magnitude_ranges.items():
    pct = (count / len(df)) * 100
    print(f"  {range_name:40s}: {count:8,} ({pct:6.2f}%)")

    # Interpretation
    if '< 1g' in range_name or '< 0.001 kg' in range_name:
        interp = "❌ Too light - likely unit error"
    elif '1-10g' in range_name or '0.001-0.01 kg' in range_name:
        interp = "⚠️ Very suspicious - unlikely for clothing"
    elif '10-100g' in range_name:
        interp = "⚠️ Possible - accessories or unit error"
    elif '100-1000g' in range_name or '0.1-1 kg' in range_name:
        interp = "✅ Normal for clothing"
    elif '1-5 kg' in range_name:
        interp = "✅ Normal for heavy items"
    elif range_name == 'heavy (5-10 kg)':
        interp = "⚠️ Unusual but possible"
    else:
        interp = "❌ Impossible for single garment"

    report.append(f"| {range_name} | {count:,} | {pct:.2f}% | {interp} |")

report.append("\n")
print()

# ============================================================================
# 3. PATTERN DETECTION - FACTOR OF 1000 ANALYSIS
# ============================================================================

print("=" * 80)
print("3. PATTERN DETECTION - TESTING FOR 1000x UNIT ERRORS")
print("=" * 80)
print()

report.append("## 3. Unit Conversion Error Detection\n")
report.append("**Hypothesis**: Some weights might be in grams but labeled as kg (off by 1000x)\n\n")

# Get outliers
very_low = df[df['weight_kg'] < 0.01].copy()
very_high = df[df['weight_kg'] > 10].copy()
normal = df[(df['weight_kg'] >= 0.1) & (df['weight_kg'] <= 5)].copy()

print(f"Very low weights (< 0.01 kg): {len(very_low):,}")
print(f"Normal weights (0.1-5 kg): {len(normal):,}")
print(f"Very high weights (> 10 kg): {len(very_high):,}\n")

# Test: If we multiply low weights by 1000, do they fall into normal range?
if len(very_low) > 0:
    very_low['weight_if_grams'] = very_low['weight_kg'] * 1000  # Convert assuming they're in grams
    falls_into_normal = len(very_low[(very_low['weight_if_grams'] >= 0.1) & (very_low['weight_if_grams'] <= 5)])
    pct_corrected = (falls_into_normal / len(very_low)) * 100

    print(f"Testing: If low weights are actually in GRAMS (not kg):")
    print(f"  • {falls_into_normal:,} out of {len(very_low):,} ({pct_corrected:.1f}%) would fall into normal range")
    print()

    report.append(f"### Low Weight Analysis (< 0.01 kg)\n")
    report.append(f"**Found**: {len(very_low):,} products with weight < 0.01 kg (< 10 grams)\n")
    report.append(f"\n**Test**: If these values are actually in GRAMS (mislabeled as kg):\n")
    report.append(f"- {falls_into_normal:,} products ({pct_corrected:.1f}%) would fall into normal range (0.1-5 kg)\n")

    if pct_corrected > 80:
        report.append(f"\n🔴 **STRONG EVIDENCE**: {pct_corrected:.1f}% correction rate suggests these ARE in grams!\n")
        print(f"  🔴 STRONG EVIDENCE: {pct_corrected:.1f}% would be corrected!")
    elif pct_corrected > 50:
        report.append(f"\n⚠️ **MODERATE EVIDENCE**: {pct_corrected:.1f}% correction rate suggests possible unit mixing\n")
        print(f"  ⚠️ MODERATE EVIDENCE: {pct_corrected:.1f}% would be corrected")
    else:
        report.append(f"\n✅ **UNLIKELY**: Only {pct_corrected:.1f}% would be corrected - probably not unit error\n")
        print(f"  ✅ Only {pct_corrected:.1f}% would be corrected - unlikely to be unit error")
    print()

# Test: If we divide high weights by 1000, do they fall into normal range?
if len(very_high) > 0:
    very_high['weight_if_kg'] = very_high['weight_kg'] / 1000  # Convert assuming they're in grams
    falls_into_normal_high = len(very_high[(very_high['weight_if_kg'] >= 0.1) & (very_high['weight_if_kg'] <= 5)])
    pct_corrected_high = (falls_into_normal_high / len(very_high)) * 100

    print(f"Testing: If high weights are actually in GRAMS (labeled as kg):")
    print(f"  • {falls_into_normal_high:,} out of {len(very_high):,} ({pct_corrected_high:.1f}%) would fall into normal range")
    print()

    report.append(f"### High Weight Analysis (> 10 kg)\n")
    report.append(f"**Found**: {len(very_high):,} products with weight > 10 kg\n")
    report.append(f"\n**Test**: If these values are in GRAMS (should be divided by 1000):\n")
    report.append(f"- {falls_into_normal_high:,} products ({pct_corrected_high:.1f}%) would fall into normal range (0.1-5 kg)\n")

    if pct_corrected_high > 80:
        report.append(f"\n🔴 **STRONG EVIDENCE**: {pct_corrected_high:.1f}% correction rate - these ARE in grams!\n")
        print(f"  🔴 STRONG EVIDENCE: {pct_corrected_high:.1f}% would be corrected!")
    elif pct_corrected_high > 50:
        report.append(f"\n⚠️ **MODERATE EVIDENCE**: {pct_corrected_high:.1f}% correction rate suggests possible unit mixing\n")
        print(f"  ⚠️ MODERATE EVIDENCE: {pct_corrected_high:.1f}% would be corrected")
    else:
        report.append(f"\n✅ **UNLIKELY**: Only {pct_corrected_high:.1f}% would be corrected\n")
        print(f"  ✅ Only {pct_corrected_high:.1f}% would be corrected")
    print()

# ============================================================================
# 4. CATEGORY-SPECIFIC ANALYSIS
# ============================================================================

print("=" * 80)
print("4. CATEGORY-SPECIFIC WEIGHT ANALYSIS")
print("=" * 80)
print()

report.append("## 4. Category-Specific Analysis\n")
report.append("Analyzing if certain categories have systematic weight issues.\n\n")

# Get category weight stats
category_stats = df.groupby('category')['weight_kg'].agg(['count', 'mean', 'median', 'min', 'max', 'std']).reset_index()
category_stats = category_stats.sort_values('mean')

# Find categories with suspiciously low or high weights
suspicious_low = category_stats[category_stats['mean'] < 0.1]
suspicious_high = category_stats[category_stats['mean'] > 3]

if len(suspicious_low) > 0:
    print(f"Categories with suspiciously LOW average weight:")
    report.append("### Categories with Suspiciously Low Average Weight\n")
    report.append("| Category | Count | Mean | Median | Min | Max |")
    report.append("|----------|-------|------|--------|-----|-----|")

    for _, row in suspicious_low.iterrows():
        print(f"  • {row['category']:30s}: {row['count']:6,} products, avg: {row['mean']:.4f} kg")
        report.append(f"| {row['category']} | {int(row['count']):,} | {row['mean']:.4f} kg | {row['median']:.4f} kg | {row['min']:.6f} kg | {row['max']:.4f} kg |")
    print()
    report.append("\n")

if len(suspicious_high) > 0:
    print(f"Categories with suspiciously HIGH average weight:")
    report.append("### Categories with Suspiciously High Average Weight\n")
    report.append("| Category | Count | Mean | Median | Min | Max |")
    report.append("|----------|-------|------|--------|-----|-----|")

    for _, row in suspicious_high.iterrows():
        print(f"  • {row['category']:30s}: {row['count']:6,} products, avg: {row['mean']:.4f} kg")
        report.append(f"| {row['category']} | {int(row['count']):,} | {row['mean']:.4f} kg | {row['median']:.4f} kg | {row['min']:.4f} kg | {row['max']:.2f} kg |")
    print()
    report.append("\n")

# ============================================================================
# 5. DETAILED EXAMPLES OF OUTLIERS
# ============================================================================

print("=" * 80)
print("5. DETAILED EXAMPLES")
print("=" * 80)
print()

report.append("## 5. Detailed Examples\n")

# Lightest products
lightest = df.nsmallest(20, 'weight_kg')[['category', 'parent_category', 'weight_kg']]
report.append("### 20 Lightest Products\n")
report.append("| Rank | Category | Parent | Weight (kg) | Weight (g) | If Actually Grams (kg) |")
report.append("|------|----------|--------|-------------|------------|------------------------|")
print("20 Lightest Products:")
for i, (_, row) in enumerate(lightest.iterrows(), 1):
    weight_kg = row['weight_kg']
    weight_g = weight_kg * 1000
    if_grams = weight_kg * 1000  # If value is actually in grams
    print(f"  {i:2d}. {row['category']:30s} {weight_kg:.6f} kg ({weight_g:.3f} g) - If grams: {if_grams:.3f} kg")
    report.append(f"| {i} | {row['category']} | {row['parent_category']} | {weight_kg:.6f} | {weight_g:.3f} | {if_grams:.3f} |")
print()
report.append("\n")

# Heaviest products
heaviest = df.nlargest(20, 'weight_kg')[['category', 'parent_category', 'weight_kg']]
report.append("### 20 Heaviest Products\n")
report.append("| Rank | Category | Parent | Weight (kg) | Weight (g) | If Actually Grams (kg) |")
report.append("|------|----------|--------|-------------|------------|------------------------|")
print("20 Heaviest Products:")
for i, (_, row) in enumerate(heaviest.iterrows(), 1):
    weight_kg = row['weight_kg']
    weight_g = weight_kg * 1000
    if_grams = weight_kg / 1000  # If value is actually in grams
    print(f"  {i:2d}. {row['category']:30s} {weight_kg:.2f} kg ({weight_g:.0f} g) - If grams: {if_grams:.3f} kg")
    report.append(f"| {i} | {row['category']} | {row['parent_category']} | {weight_kg:.2f} | {weight_g:.0f} | {if_grams:.3f} |")
print()
report.append("\n")

# ============================================================================
# 6. STATISTICAL TESTS
# ============================================================================

print("=" * 80)
print("6. STATISTICAL ANALYSIS")
print("=" * 80)
print()

report.append("## 6. Statistical Analysis\n")

# Test for multimodality using kernel density estimation
from scipy.stats import gaussian_kde

# Sample data for KDE (too large otherwise)
sample_size = min(10000, len(df))
weight_sample = df['weight_kg'].sample(n=sample_size, random_state=42).values

# Remove extreme outliers for visualization
weight_sample_filtered = weight_sample[(weight_sample > 0.001) & (weight_sample < 10)]

if len(weight_sample_filtered) > 100:
    # Calculate log-scale for better visualization
    log_weights_sample = np.log10(weight_sample_filtered)

    # Shapiro-Wilk test for normality
    shapiro_stat, shapiro_p = stats.shapiro(weight_sample_filtered[:5000])  # Max 5000 samples

    print(f"Shapiro-Wilk Normality Test:")
    print(f"  Statistic: {shapiro_stat:.6f}")
    print(f"  P-value: {shapiro_p:.6e}")
    if shapiro_p < 0.05:
        print(f"  ✗ Data is NOT normally distributed (p < 0.05)")
        report.append(f"**Normality Test (Shapiro-Wilk)**:\n")
        report.append(f"- Statistic: {shapiro_stat:.6f}\n")
        report.append(f"- P-value: {shapiro_p:.6e}\n")
        report.append(f"- **Result**: ✗ Data is NOT normally distributed (suggests multiple distributions mixed)\n\n")
    else:
        print(f"  ✓ Data appears normally distributed")
        report.append(f"**Normality Test**: ✓ Data appears normally distributed\n\n")
    print()

# Calculate skewness and kurtosis
skewness = stats.skew(weight_sample_filtered)
kurt = stats.kurtosis(weight_sample_filtered)

print(f"Distribution Shape:")
print(f"  Skewness: {skewness:.4f} ({'right-skewed' if skewness > 0 else 'left-skewed'})")
print(f"  Kurtosis: {kurt:.4f} ({'heavy-tailed' if kurt > 0 else 'light-tailed'})")
print()

report.append(f"**Distribution Shape**:\n")
report.append(f"- Skewness: {skewness:.4f} ({'right-skewed (long tail to right)' if skewness > 0 else 'left-skewed'})\n")
report.append(f"- Kurtosis: {kurt:.4f} ({'heavy-tailed (more outliers)' if kurt > 0 else 'light-tailed'})\n\n")

# ============================================================================
# 7. VISUALIZATION
# ============================================================================

print("Creating visualizations...")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 1. Histogram (linear scale)
axes[0, 0].hist(weight_sample_filtered, bins=100, edgecolor='black', alpha=0.7)
axes[0, 0].set_xlabel('Weight (kg)', fontweight='bold')
axes[0, 0].set_ylabel('Frequency', fontweight='bold')
axes[0, 0].set_title('Weight Distribution (Linear Scale)', fontweight='bold')
axes[0, 0].axvline(df['weight_kg'].median(), color='red', linestyle='--', label=f'Median: {df["weight_kg"].median():.3f} kg')
axes[0, 0].legend()
axes[0, 0].grid(axis='y', alpha=0.3)

# 2. Histogram (log scale)
axes[0, 1].hist(weight_sample_filtered, bins=100, edgecolor='black', alpha=0.7)
axes[0, 1].set_xlabel('Weight (kg)', fontweight='bold')
axes[0, 1].set_ylabel('Frequency', fontweight='bold')
axes[0, 1].set_title('Weight Distribution (Log Scale)', fontweight='bold')
axes[0, 1].set_xscale('log')
axes[0, 1].grid(True, which='both', alpha=0.3)

# 3. Box plot by magnitude
weight_categories_plot = []
weight_labels_plot = []
for range_name, _ in magnitude_ranges.items():
    if '< 1g' in range_name or '< 0.001' in range_name:
        subset = df[df['weight_kg'] < 0.001]['weight_kg'].values
        label = '< 1g'
    elif '1-10g' in range_name or '0.001-0.01' in range_name:
        subset = df[(df['weight_kg'] >= 0.001) & (df['weight_kg'] < 0.01)]['weight_kg'].values
        label = '1-10g'
    elif '10-100g' in range_name:
        subset = df[(df['weight_kg'] >= 0.01) & (df['weight_kg'] < 0.1)]['weight_kg'].values
        label = '10-100g'
    elif '100-1000g' in range_name or '0.1-1 kg' in range_name:
        subset = df[(df['weight_kg'] >= 0.1) & (df['weight_kg'] < 1)]['weight_kg'].values
        label = '0.1-1kg'
    elif '1-5 kg' in range_name:
        subset = df[(df['weight_kg'] >= 1) & (df['weight_kg'] < 5)]['weight_kg'].values
        label = '1-5kg'
    elif '5-10 kg' in range_name:
        subset = df[(df['weight_kg'] >= 5) & (df['weight_kg'] < 10)]['weight_kg'].values
        label = '5-10kg'
    elif '> 10 kg' in range_name:
        subset = df[df['weight_kg'] >= 10]['weight_kg'].values
        label = '>10kg'
    else:
        continue

    if len(subset) > 0:
        weight_categories_plot.append(subset)
        weight_labels_plot.append(label)

axes[1, 0].boxplot(weight_categories_plot, labels=weight_labels_plot)
axes[1, 0].set_ylabel('Weight (kg)', fontweight='bold')
axes[1, 0].set_xlabel('Weight Range', fontweight='bold')
axes[1, 0].set_title('Weight Distribution by Category', fontweight='bold')
axes[1, 0].set_yscale('log')
axes[1, 0].grid(axis='y', alpha=0.3)
plt.setp(axes[1, 0].xaxis.get_majorticklabels(), rotation=45, ha='right')

# 4. Scatter plot showing pattern
sample_for_scatter = df.sample(n=min(5000, len(df)), random_state=42)
axes[1, 1].scatter(range(len(sample_for_scatter)), sample_for_scatter['weight_kg'], alpha=0.3, s=1)
axes[1, 1].set_xlabel('Sample Index', fontweight='bold')
axes[1, 1].set_ylabel('Weight (kg)', fontweight='bold')
axes[1, 1].set_title('Weight Values Scatter (Random Sample)', fontweight='bold')
axes[1, 1].set_yscale('log')
axes[1, 1].axhline(0.001, color='red', linestyle='--', alpha=0.5, label='1 gram')
axes[1, 1].axhline(0.01, color='orange', linestyle='--', alpha=0.5, label='10 grams')
axes[1, 1].axhline(10, color='purple', linestyle='--', alpha=0.5, label='10 kg')
axes[1, 1].legend()
axes[1, 1].grid(True, which='both', alpha=0.3)

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/weight_distribution_analysis.png", dpi=DPI, bbox_inches='tight')
plt.close()
print(f"✓ Saved visualization: weight_distribution_analysis.png\n")

# ============================================================================
# 8. CONCLUSION AND RECOMMENDATIONS
# ============================================================================

report.append("## 7. Conclusions and Recommendations\n")

# Analyze results
issues_found = []
if len(very_low) > 0 and pct_corrected > 70:
    issues_found.append("low_weight_unit_error")
if len(very_high) > 0 and pct_corrected_high > 70:
    issues_found.append("high_weight_unit_error")

if len(issues_found) > 0:
    report.append("### 🔴 UNIT CONSISTENCY ISSUES DETECTED\n")
    report.append("\n**Evidence Summary**:\n")

    if "low_weight_unit_error" in issues_found:
        report.append(f"1. **Low weights (< 0.01 kg)**: {len(very_low):,} products\n")
        report.append(f"   - {pct_corrected:.1f}% would be corrected by multiplying by 1000\n")
        report.append(f"   - **Conclusion**: These values are likely in GRAMS but labeled as KG\n\n")

    if "high_weight_unit_error" in issues_found:
        report.append(f"2. **High weights (> 10 kg)**: {len(very_high):,} products\n")
        report.append(f"   - {pct_corrected_high:.1f}% would be corrected by dividing by 1000\n")
        report.append(f"   - **Conclusion**: These values are in GRAMS but labeled as KG\n\n")

    report.append("### Recommended Actions\n")
    report.append("1. ✅ **Fix low weights**: Multiply values < 0.01 kg by 1000\n")
    report.append("2. ✅ **Fix high weights**: Divide values > 10 kg by 1000\n")
    report.append("3. ✅ **Verify**: Check corrected values fall into expected ranges\n")
    report.append("4. ✅ **Document**: Update data dictionary to confirm all weights in KG\n")
else:
    report.append("### ✅ UNIT CONSISTENCY APPEARS ACCEPTABLE\n")
    report.append("\n**However**, there are still outliers that should be reviewed:\n")
    report.append(f"- {len(very_low):,} products with weight < 0.01 kg (< 10 grams)\n")
    report.append(f"- {len(very_high):,} products with weight > 10 kg\n")
    report.append("\nThese may be:\n")
    report.append("- Data entry errors\n")
    report.append("- Accessories or special items\n")
    report.append("- Bundled products\n")

report.append("\n---\n")
report.append(f"\n**Visualization**: [weight_distribution_analysis.png](weight_distribution_analysis.png)")

# Save report
print("💾 Saving detailed research report...")
with open(REPORT_FILE, 'w') as f:
    f.write('\n'.join(report))
print(f"✓ Saved: {REPORT_FILE}\n")

print("=" * 80)
print("✓ WEIGHT UNIT ANALYSIS COMPLETE")
print("=" * 80)
print(f"\nDetailed research report: {REPORT_FILE}")
print(f"Visualization: {OUTPUT_DIR}/weight_distribution_analysis.png")
