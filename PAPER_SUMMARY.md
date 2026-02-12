# HydroCarbon Model - Research Paper Summary

## Overview

This document provides a comprehensive summary of the **HydroCarbon** research paper, which describes a state-of-the-art physics-informed machine learning model for predicting environmental footprints in fashion products.

## Model at a Glance

| Aspect | Details |
|--------|---------|
| **Model Name** | HydroCarbon |
| **Type** | XGBoost Multi-Output Regressor |
| **Inputs** | 129 features (product attributes) |
| **Outputs** | 4 predictions (carbon material, carbon transport, carbon total, water total) |
| **Training Data** | 676,178 synthetic fashion products |
| **Validation Data** | 225,393 products |
| **Key Innovation** | Hybrid physics-ML architecture with robustness to missing data |
| **Core Performance** | $R^2 = 0.9999$ on complete data, $R^2 = 0.936$ with 40% missing data |

## Research Paper Structure

The research paper is organized into 9 main sections:

### 1. Introduction
- **Motivation**: Environmental footprint calculation requires complete data that's rarely available
- **Problem**: Traditional LCA calculators fail with missing inputs
- **Solution**: ML model that learns physics and estimates intelligently when data is incomplete
- **Impact**: Enables real-world sustainability assessments with partial product information

### 2. Mathematical Foundations

Detailed mathematical formulas for environmental footprint calculations:

#### Carbon Footprint Formula
$$C_{total} = C_{material} + C_{transport}$$

**Material Carbon:**
$$C_{material} = \sum_{i=1}^{n} W \times P_i \times CF_i$$

**Transport Carbon:**
$$C_{transport} = \frac{W}{1000} \times D \times \frac{\sum_{m=1}^{5} s_m(D) \times EF_m}{1000}$$

**Key Innovation - Multinomial Logit Modal Split:**
$$P_m(D) = \frac{\exp(U_m(D))}{\sum_{k=1}^{5} \exp(U_k(D))}$$
$$U_m(D) = \beta_{0,m} + \beta_{1,m} \times \ln(D)$$

Where:
- $W$ = Product weight (kg)
- $P_i$ = Material percentage (0-1)
- $CF_i$ = Carbon factor for material $i$
- $D$ = Distance (km)
- $s_m(D)$ = Modal share for transport mode $m$
- $EF_m$ = Emission factor for mode $m$

#### Water Footprint Formula
$$W_{total} = \sum_{i=1}^{n} W \times P_i \times WF_i$$

### 3. Model Architecture

**Hybrid Physics-ML Design:**

```
Input Features (129)
    ├─ Contextual Features (93): Gender, Category, Parent Category
    └─ Physics Features (36): Weight, Distance, Material %
            ↓
Formula Feature Calculation (Physics Path)
    └─ Direct computation using available data
            ↓
XGBoost Multi-Output Regressor
    ├─ Physics-informed features as input
    ├─ Custom objective with physics constraints
    └─ Multi-target prediction (4 outputs)
            ↓
Predictions
    ├─ carbon_material (kgCO2e)
    ├─ carbon_transport (kgCO2e)
    ├─ carbon_total (kgCO2e)
    └─ water_total (liters)
```

**Key Components:**
- **Formula Features**: Physics-based calculations injected as high-importance features
- **Physics Constraint**: Custom loss function enforces $C_{total} \approx C_{material} + C_{transport}$
- **Dual Path**: Statistical imputation when physics data is missing, exact calculation when available

### 4. Robustness Training

**Problem**: Real-world data has missing values (weight, exact material composition, manufacturing origin)

**Solution**: Feature dropout augmentation during training
- Randomly mask 20% of features per batch
- Forces model to learn from contextual cues
- Maintains $R^2 > 0.93$ even with 40% missing data

**Two-Model Strategy:**
1. **Baseline Model**: Maximum accuracy on complete data
2. **Robustness Model**: Feature dropout augmentation for production use

**Performance Comparison:**

| Missing Data | Model | Carbon Total $R^2$ | MAE |
|--------------|-------|-------------------|-----|
| 0% | Baseline | 0.9999 | 0.044 kgCO2e |
| 40% | Baseline | -0.380 | 4.12 kgCO2e |
| 40% | Robustness | **0.936** | **0.29 kgCO2e** |

### 5. Implementation Details

**C-Based Calculator (High Performance):**
- Processes 900,000 products in ~45 seconds
- Standalone binary, no dependencies
- Streaming architecture for memory efficiency
- Docker-ready for production deployment

**Python ML Pipeline (Flexibility):**
- XGBoost 2.0+ with GPU acceleration (~2 minute training)
- Preprocessing pipeline with one-hot encoding
- Model serialization for easy deployment
- API wrapper for production inference

**File Structure:**
```
trained_model/
├── baseline/
│   ├── xgb_model.json          # Model weights
│   ├── preprocessor.pkl        # Fitted preprocessing
│   └── trainer_config.pkl      # Scaling parameters
└── robustness/
    └── ... (same structure)
```

### 6. Data Sources

**Material Emission Factors:**
- TU Delft Idemat 2026 database
- 34 materials with carbon and water factors
- Peer-reviewed and industry-validated

**Transport Parameters:**
- CE Delft STREAM 2020 emission factors
- Multinomial logit model calibrated on EU freight data
- 5 modes: Road, Rail, Inland Waterway, Sea, Air

**Example Material Factors:**

| Material | Carbon (kgCO2e/kg) | Water (L/kg) |
|----------|-------------------|--------------|
| Cotton (conventional) | 0.94 | 9,113 |
| Wool (merino) | 13.89 | 170,000 |
| Polyester (virgin) | 2.13 | 60 |
| Leather (bovine) | 8.45 | 17,100 |
| Hemp | 0.85 | 2,719 |

### 7. Evaluation and Results

**Performance on Complete Data:**

| Target | $R^2$ | MAE | Interpretation |
|--------|------|-----|----------------|
| Carbon Material | 0.9999 | 0.041 kgCO2e | Off by ~41g CO2e |
| Carbon Transport | 0.9998 | 0.001 kgCO2e | Off by ~1g CO2e |
| Carbon Total | 0.9999 | 0.044 kgCO2e | Off by ~44g CO2e |
| Water Total | 0.9998 | 115.3 L | Off by ~115L |

**Why Accuracy is So High:**
- Model learns deterministic formulas, not discovering hidden patterns
- Analogy: Training a model to predict rectangle area from length × width
- Near-perfect correlation expected, not data leakage

### 8. Use Cases

**Suitable Applications:**
1. **E-commerce Integration**: Real-time footprint on product pages with minimal data
2. **Sustainable Design**: Optimize material selection during product development
3. **Supply Chain Analysis**: Assess manufacturing location impacts
4. **Corporate Reporting**: Batch process thousands of products
5. **Consumer Education**: Transparent environmental impact communication

**Example API Usage:**
```python
predictor = FootprintPredictor("trained_model/robustness")

results = predictor.predict(
    gender="Male",
    category="Jeans",
    weight_kg=0.934,
    materials={"cotton_conventional": 0.92, "elastane": 0.08},
    total_distance_km=12847
)

print(f"Carbon: {results['carbon_total']:.2f} kgCO2e")
print(f"Water: {results['water_total']:.0f} liters")
# Output: Carbon: 2.26 kgCO2e, Water: 7,888 liters
```

**Limitations:**
- Proof-of-concept status (not ISO 14040/14044 compliant yet)
- Synthetic data (pattern-based, not brand-specific)
- Cradle-to-gate scope only
- Primarily calibrated for EU-bound products

### 9. Future Work

**Production Enhancements:**
- ISO 14040/14044 compliance audit
- PEF methodology alignment
- Scope 1-3 emissions and end-of-life modeling
- Real-world validation studies

**Model Improvements:**
- Uncertainty quantification with Bayesian methods
- Multi-modal architecture (vision + tabular)
- Federated learning for proprietary data
- Continual learning from new LCA studies

## Key Innovations

### 1. Synthetic Data Generation at Scale
- Generated 900,000+ products using Google Gemini 2.5 Flash
- Controlled vocabulary ensures plausibility and calculability
- Overcomes data scarcity in LCA research

### 2. Physics-Informed ML Architecture
- **Formula Features**: Physics calculations as model inputs
- **Physics Constraints**: Custom loss function enforces physical laws
- **Dual Path**: Seamless transition between calculation and estimation

### 3. Robustness to Missing Data
- Feature dropout augmentation during training
- Context-aware imputation (learns category-specific expectations)
- Maintains high accuracy even with 40% missing features

### 4. Performance Optimization
- C-based calculator (900k products in 45 seconds)
- GPU-accelerated training (~2 minutes on Tesla T4)
- Standalone production deployment

## Model Performance Summary

| Metric | Score | Context |
|--------|-------|---------|
| **Complete Data $R^2$** | 0.9999 | Near-perfect accuracy |
| **40% Missing $R^2$** | 0.936 | Production-ready robustness |
| **Carbon Total MAE** | 0.044 kgCO2e | ~44g average error |
| **Water Total MAE** | 115.3 L | ~115L average error |
| **Inference Speed** | <1ms (CPU) | Real-time capable |
| **Training Time** | ~2 min (GPU) | Efficient iteration |

## Research Impact

### Academic Contributions

1. **Methodological Innovation**: Demonstrates effectiveness of physics-informed ML for environmental modeling
2. **Data Generation**: Shows how LLMs can generate training data for scientific ML
3. **Robustness Techniques**: Feature dropout augmentation for missing data scenarios
4. **Hybrid Architecture**: Combining deterministic formulas with statistical learning

### Practical Impact

1. **Sustainability Reporting**: Enables companies to assess products with incomplete data
2. **Design Optimization**: Material selection based on environmental impact
3. **Supply Chain Decisions**: Compare manufacturing locations
4. **Consumer Transparency**: Provide environmental impact information at scale

### Open Science

- Fully open-source implementation
- Reproducible methodology
- Publicly available synthetic dataset
- Transparent calculation formulas

## Paper Compilation

### Quick Start
```bash
# Make script executable
chmod +x compile_paper.sh

# Compile the paper
./compile_paper.sh

# Output: paper_hydrocarbon_model.pdf
```

### Manual Compilation
```bash
pdflatex paper_hydrocarbon_model.tex
bibtex paper_hydrocarbon_model
pdflatex paper_hydrocarbon_model.tex
pdflatex paper_hydrocarbon_model.tex
```

### Requirements
- pdflatex or xelatex
- bibtex or biber
- TikZ and PGF packages for diagrams
- full TeX Live distribution recommended

## Using Supplemntary Materials

Generate supplementary data and charts:
```bash
python generate_paper_supplementary.py
```

This creates:
- `supplementary_material_factors.csv` - All material emission factors
- `supplementary_transport_params.csv` - Transport model parameters
- `supplementary_model_performance.csv` - Complete performance metrics
- `supplementary_feature_importance.csv` - Feature importance scores
- `model_summary.json` - Complete model configuration
- `model_performance_chart.png` - Performance visualization
- `feature_importance_plot.png` - Feature importance chart

## Citation

If you use this model or paper in your research, please cite:

```bibtex
@techreport{hydrocarbon2025,
  title={{HydroCarbon: A Physics-Informed Machine Learning Model 
         for Environmental Footprint Prediction in Fashion Products}},
  author={{Avelero Project}},
  year={2025},
  institution={Open Source},
  url={https://github.com/Avelero/Avelero_HydroCarbon}
}
```

## Contact and Support

- **GitHub Repository**: https://github.com/Avelero/Avelero_HydroCarbon
- **Issues**: Open an issue for bugs or questions
- **Documentation**: See README.md and PAPER_README.md
- **Examples**: See Trained-Implementation/preview.py for usage examples

## Conclusion

HydroCarbon represents a significant step forward in applying machine learning to environmental sustainability. By combining synthetic data generation, physics-based calculations, and robust machine learning, it achieves accuracy that exceeds traditional approaches while handling the messy reality of incomplete product data.

The model's hybrid architecture—seamlessly transitioning between exact calculation and intelligent estimation—demonstrates how domain knowledge can be integrated with modern ML to create practical, deployable solutions for sustainability challenges.

With $R^2 > 0.999$ on complete data and $R^2 > 0.93$ with 40% missing features, HydroCarbon enables accurate environmental impact assessment across the fashion industry value chain, potentially accelerating the transition to more sustainable production and consumption patterns.

---

**Research Paper Generated**: $(date +'%Y-%m-%d')
**Model Version**: 2.0
**Status**: Proof of Concept (Production version in development)
