# HydroCarbon Model - Quick Reference Guide

## ⚡ Fast Facts

| Metric | Value |
|--------|-------|
| **Model Type** | XGBoost Multi-Output Regressor |
| **Accuracy** | $R^2 = 0.9999$ (complete data) |
| **Robustness** | $R^2 = 0.936$ (40% missing) |
| **Training Time** | ~2 minutes (GPU) |
| **Inference Speed** | <1ms per prediction |
| **Dataset Size** | 900,000 products |
| **Features** | 129 (93 contextual + 36 physics) |
| **Targets** | 4 (2 carbon + 1 water + 1 total) |

## 🎯 Key Innovation

**Hybrid Physics-ML Architecture**
- Physics-based formulas + Machine learning
- Handles missing data gracefully
- Seamless calculation ↔ estimation transition

## 🧮 Core Formulas

### Carbon Material
```
C_material = Σ(weight × material_% × carbon_factor)
```

### Carbon Transport
```
C_transport = (weight/1000) × distance × (weighted_EF/1000)
```

### Water Footprint
```
W_total = Σ(weight × material_% × water_factor)
```

## 🏗️ Architecture

```
Inputs (129 features)
    ├─ Contextual (93): Gender, Category, Parent Category
    └─ Physics (36): Weight, Distance, Materials
            ↓
Formula Features (Injected)
    ├─ formula_carbon_material
    ├─ formula_carbon_transport
    └─ formula_water_total
            ↓
XGBoost Model
    ├─ 1000 estimators
    ├─ Max depth: 8
    └─ Learning rate: 0.05
            ↓
Outputs (4 predictions)
    ├─ carbon_material (kgCO2e)
    ├─ carbon_transport (kgCO2e)
    ├─ carbon_total (kgCO2e)
    └─ water_total (liters)
```

## 📊 Performance

### Complete Data
| Target | $R^2$ | MAE |
|--------|------|-----|
| Carbon Material | 0.9999 | 0.041 kgCO2e |
| Carbon Transport | 0.9998 | 0.001 kgCO2e |
| Carbon Total | 0.9999 | 0.044 kgCO2e |
| Water Total | 0.9998 | 115.3 L |

### 40% Missing Data
| Model | $R^2$ | MAE |
|-------|------|-----|
| Baseline | -0.380 | 4.12 kgCO2e |
| Robustness | **0.936** | **0.29 kgCO2e** |

## 🔧 Usage

### Quick Start
```python
from hydrocarbon import FootprintPredictor

# Load model
predictor = FootprintPredictor("trained_model/robustness")

# Predict
results = predictor.predict(
    gender="Male",
    category="Jeans",
    weight_kg=0.934,
    materials={"cotton_conventional": 0.92, "elastane": 0.08},
    total_distance_km=12847
)

print(f"Carbon: {results['carbon_total']:.2f} kgCO2e")
print(f"Water: {results['water_total']:.0f} liters")
# → Carbon: 2.26 kgCO2e
# → Water: 7,888 liters
```

### Terminal Usage
```bash
# Interactive mode
python Trained-Implementation/preview.py

# Single prediction
python Trained-Implementation/preview.py --csv \
  'Jeans,Male,Bottoms,Jeans,BD,"{""cotton_conventional"":0.92,""elastane"":0.08}",0.934,12847'
```

## 🚗 Transport Modal Split

For distance $D$, probability of mode $m$:
$$P_m(D) = \frac{\exp(\beta_{0,m} + \beta_{1,m} \ln D)}{\sum_k \exp(\beta_{0,k} + \beta_{1,k} \ln D)}$$

| Mode | EF (gCO2e/tkm) | β₀ | β₁ | Example (12,847 km) |
|------|----------------|----|----|---------------------|
| Road | 72.9 | 0.000 | 0.000 | 5% |
| Rail | 22.0 | -10.537 | 1.372 | 19% |
| Inland Waterway | 31.0 | -5.770 | 0.762 | 7% |
| Sea | 10.3 | -17.108 | 2.364 | 60% |
| Air | 782.0 | -17.345 | 1.881 | 9% |

## 📈 Model Comparison

| Characteristic | Baseline | Robustness |
|----------------|----------|------------|
| **Training Method** | Standard | 20% dropout |
| **Complete Data $R^2$** | 0.9999 | 0.9999 |
| **40% Missing $R^2$** | -0.380 | **0.936** |
| **Use Case** | Maximum accuracy | Production deployment |

## 📁 File Structure

```
Trained-Implementation/
└── trained_model/
    ├── baseline/
    │   ├── xgb_model.json
    │   ├── preprocessor.pkl
    │   ├── trainer_config.pkl
    │   └── evaluation/
    │       └── evaluation_report.json
    └── robustness/
        └── ... (same structure)

data/data_calculations/
├── src/
│   ├── carbon/          # Carbon calculators
│   ├── water/           # Water calculators
│   └── utils/           # Parsers
├── include/             # Header files
└── build/               # Compiled binaries

models/
├── main.py
├── src/
│   ├── trainer.py       # XGBoost training
│   ├── preprocessor.py  # Feature preprocessing
│   ├── formula_features.py  # Physics calculations
│   └── data_loader.py   # Data loading
└── data_input/          # Training data
```

## 👷 Training Process

1. **Synthetic Data Generation** (900,000 products)
   ```bash
   cd data/data_creation
   python scripts/main.py
   ```

2. **Physics Calculation** (C-based, fast)
   ```bash
   cd data/data_calculations
   make && make run
   ```

3. **Model Training** (GPU recommended)
   ```bash
   cd models
   python main.py --mode robustness
   # ~2 minutes on Tesla T4
   ```

4. **Model Evaluation**
   ```bash
   python src/evaluator.py
   ```

## 🎓 Technical Details

### XGBoost Configuration
```yaml
n_estimators: 1000
max_depth: 8
learning_rate: 0.05  
subsample: 0.8
colsample_bytree: 0.8
early_stopping_rounds: 50
device: cuda  # GPU acceleration
tree_method: histogram
```

### Feature Engineering
- **Formula Features**: Physics calculations injected as features
- **Target Scaling**: Log transformation for numerical stability
- **Custom Objective**: MSE + physics constraint penalty
- **Physics Constraint**: $C_{total} \approx C_{material} + C_{transport}$

### Performance
- **Training**: 676,178 samples, ~2 minutes on GPU
- **Inference**: <1ms per prediction
- **Scalability**: Batch processing of 10,000 products in <10 seconds

## 📊 Data Sources

**Material Factors:**
- TU Delft Idemat 2026
- 34 materials with carbon/water footprints

**Transport Factors:**
- CE Delft STREAM 2020
- 5 transport modes with modal split model

**Water Factors:**
- Water Footprint Network
- Blue + green water consumption

## ⚠️ Important Notes

**Proof of Concept Status:**
- Not yet ISO 14040/14044 compliant
- Production version in development
- Currently cradle-to-gate scope only

**Synthetic Data:**
- Pattern-based, not brand-specific
- Validated formulas, not measured emissions
- Represents fashion industry averages

**Accuracy Explanation:**
- High $R^2$ is expected (model learns formulas)
- Not data leakage - deterministic relationship
- Similar to learning area = length × width

## 🚀 Use Cases

✅ **Recommended:**
- E-commerce footprint estimates with incomplete data
- Conceptual product sustainability assessment
- Educational and awareness applications
- Internal research and development

❌ **Not Recommended:**
- Public product labeling (not certified)
- Regulatory compliance reporting
- Legal or contractual obligations
- Final LCA studies

## 🔍 Feature Importance (Top 5)

1. **formula_carbon_material** (1000.0) - Physics calculation
2. **formula_carbon_transport** (850.0) - Physics calculation  
3. **weight_kg** (500.0) - Direct product weight
4. **total_distance_km** (450.0) - Transport distance
5. **category_Jeans** (200.0) - Product category

## 📚 Paper Compilation

### Quick Compile
```bash
chmod +x compile_paper.sh
./compile_paper.sh
```

### Manual Compile
```bash
pdflatex paper_hydrocarbon_model.tex
bibtex paper_hydrocarbon_model
pdflatex paper_hydrocarbon_model.tex
pdflatex paper_hydrocarbon_model.tex
```

### Generate Supplements
```bash
python generate_paper_supplementary.py
```

## 📞 Support

- **GitHub**: https://github.com/Avelero/Avelero_HydroCarbon
- **Paper**: paper_hydrocarbon_model.pdf
- **Guide**: PAPER_README.md
- **Examples**: Trained-Implementation/preview.py

## 🎯 Research Highlight

**Problem Solved:** Traditional LCA fails with missing data
**Solution:** Hybrid physics-ML with 20% dropout during training
**Result:** $R^2 = 0.936$ with 40% missing features (17× better than baseline)

---

**Last Updated**: 2025-01-22
**Model Version**: 2.0
**Status**: Proof of Concept → Production Ready
