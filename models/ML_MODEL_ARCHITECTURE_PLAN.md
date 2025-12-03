# Machine Learning Model Architecture Plan
## Footprint Prediction with Missing Value Handling

**Project:** Bulk Product Generator - Footprint Prediction
**Date:** 2025-12-02
**Version:** 1.0

---

## 1. Problem Overview

### 1.1 Objective
Build a flexible machine learning model that predicts environmental footprints for fashion products, capable of handling missing input features.

### 1.2 Prediction Targets (Multi-Output Regression)
- `carbon_material` - Carbon footprint from materials (kg CO2e)
- `carbon_transport` - Carbon footprint from transportation (kg CO2e)
- `carbon_total` - Total carbon footprint (kg CO2e)
- `water_total` - Total water footprint (liters)

**Target Relationships and Calculations:**
- **Carbon Total Calculation**: `carbon_total = carbon_material + carbon_transport`
  - The total carbon footprint is the sum of material and transportation components
  - This relationship can be used as a consistency check during training
  - Consider adding a custom loss term to enforce this constraint: `L_consistency = |carbon_total - (carbon_material + carbon_transport)|`
  
- **Independent Predictions**:
  - `carbon_material` - Depends primarily on: materials composition, weight, manufacturing processes
  - `carbon_transport` - Depends primarily on: total_distance_km, weight, manufacturer_country
  - `water_total` - Depends primarily on: materials composition (e.g., cotton has high water footprint), weight
  
- **Prediction Strategy**:
  - **Option A (Independent)**: Train separate models for each target independently
  - **Option B (Hierarchical)**: First predict `carbon_material` and `carbon_transport`, then calculate `carbon_total` as their sum (ensures consistency)
  - **Option C (Constrained)**: Train with physics-based constraints to enforce `carbon_total = carbon_material + carbon_transport` relationship

### 1.3 Input Features
- `gender` - Categorical (Male, Female, Unisex)
- `parent_category` - Categorical (Bottoms, Tops, Dresses, etc.)
- `category` - Categorical (Maxi Skirts, T-Shirts, etc.)
- `manufacturer_country` - Categorical (ISO country codes)
- `materials` - Complex JSON format with material composition percentages
- `weight_kg` - Numerical (product weight)
- `total_distance_km` - Numerical (shipping distance)

### 1.4 Key Challenge: Missing Value Flexibility
The model must maintain accuracy even when critical features are missing:
- Missing `category` → Use parent_category + materials
- Missing `total_distance_km` → Predict from manufacturer_country
- Missing `weight_kg` → Infer from category + materials
- Missing `materials` → Use category averages

---

## 2. Data Preprocessing Pipeline

### 2.1 Material Feature Engineering

```python
"""
Materials are stored as JSON: {"cotton_conventional":0.63,"polyester_recycled":0.37}

Strategy: Create one-hot encoded features for each material type with percentage values
"""

# Material types identified from dataset:
MATERIAL_TYPES = [
    'cotton_conventional',
    'cotton_organic',
    'polyester_virgin',
    'polyester_recycled',
    'viscose',
    'elastane',
    'modal',
    'linen_flax',
    'nylon',
    'wool',
    'silk',
    'acrylic',
    # ... expand as needed
]

# Feature representation:
# material_cotton_conventional: 0.63
# material_polyester_recycled: 0.37
# material_count: 2
# material_diversity_score: Shannon entropy
```

### 2.2 Categorical Encoding Strategies

**Option A: Target Encoding (Recommended)**
- Encode categories by their mean target values
- Reduces dimensionality compared to one-hot
- Captures relationship with target variable
- Include cross-validation to prevent overfitting

**Option B: One-Hot Encoding**
- Standard approach for tree-based models
- May create high-dimensional sparse features
- Good for categories with <50 unique values

**Option C: Embedding Layers (Deep Learning)**
- Learn dense representations during training
- Effective for high-cardinality features
- Captures semantic relationships

### 2.3 Missing Value Indicators

Create binary indicator features for missingness:
```python
# Add indicator columns
is_category_missing
is_distance_missing
is_weight_missing
is_materials_missing
```

These help the model learn patterns in missing data.

### 2.4 Feature Imputation Strategies

**Strategy 1: Multiple Imputation**
```python
# For numerical features:
- weight_kg: Impute with median by category
- total_distance_km: Impute with median by manufacturer_country

# For categorical features:
- category: Use 'Unknown' category or parent_category mode
- manufacturer_country: Use global mode or 'Unknown'
```

**Strategy 2: Model-Based Imputation**
```python
# Train separate models to predict missing values:
- XGBoost to predict weight_kg from category + materials
- KNN to predict total_distance_km from manufacturer_country
```

**Strategy 3: End-to-End Learning (Deep Learning)**
```python
# Use masking layers that learn to handle missing values
# Neural network learns imputation implicitly
```

---

## 3. Model Architectures

### 3.1 Architecture 1: Gradient Boosting Ensemble

**Model: XGBoost Multi-Output Regressor**

**Why XGBoost:**
- Excellent handling of mixed feature types
- Built-in missing value handling (learns optimal split directions)
- Feature importance for interpretability
- Fast training and inference
- Proven performance on tabular data

**Architecture:**
```python
from xgboost import XGBRegressor
from sklearn.multioutput import MultiOutputRegressor

# Train separate XGBoost model for each target
models = {
    'carbon_material': XGBRegressor(
        n_estimators=500,
        max_depth=8,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        missing=np.nan,  # Handle missing values natively
        tree_method='hist',
        enable_categorical=True  # Native categorical support
    ),
    'carbon_transport': XGBRegressor(...),
    'carbon_total': XGBRegressor(...),
    'water_total': XGBRegressor(...)
}
```

**Missing Value Handling:**
- XGBoost learns optimal direction for missing values during training
- Add missing indicators as explicit features
- No explicit imputation needed

**Pros:**
- Superior accuracy on tabular data
- Built-in missing value handling
- Feature importance analysis
- Fast inference

**Cons:**
- Requires careful hyperparameter tuning
- May overfit with insufficient data
- Less flexible than neural networks

**Expected Performance:** ★★★★★ (Best for tabular data)

---

### 3.2 Architecture 2: LightGBM with Custom Loss

**Model: LightGBM with Multi-Task Learning**

**Why LightGBM:**
- Faster training than XGBoost
- Better handling of categorical features
- Lower memory usage
- Excellent for large datasets

**Architecture:**
```python
import lightgbm as lgb

# Single model with multi-target output
params = {
    'objective': 'regression',
    'metric': 'rmse',
    'boosting_type': 'gbdt',
    'num_leaves': 64,
    'learning_rate': 0.05,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': -1,
    'categorical_feature': ['gender', 'parent_category', 'category', 'manufacturer_country']
}

# Train separate models or single multi-output
model = lgb.train(params, train_data, num_boost_round=1000)
```

**Missing Value Handling:**
- Native support for NA values
- Categorical feature optimization
- Add missing indicators

**Pros:**
- Fastest training time
- Excellent categorical handling
- Memory efficient
- Good with missing data

**Cons:**
- May require more hyperparameter tuning
- Slightly less accurate than XGBoost in some cases

**Expected Performance:** ★★★★☆

---

### 3.3 Architecture 3: Deep Neural Network with Masking

**Model: Multi-Task Neural Network with Missing Value Embeddings**

**Why Deep Learning:**
- Can learn complex non-linear relationships
- Shared representations across targets
- Flexible architecture for missing data
- Can incorporate domain knowledge

**Architecture:**
```python
import tensorflow as tf
from tensorflow import keras

class MissingValueAwareNN(keras.Model):
    def __init__(self, categorical_dims, material_features, numerical_features):
        super().__init__()

        # Embedding layers for categorical features
        self.gender_embedding = keras.layers.Embedding(
            input_dim=categorical_dims['gender'],
            output_dim=8,
            mask_zero=True  # Handle missing values
        )
        self.category_embedding = keras.layers.Embedding(
            input_dim=categorical_dims['category'],
            output_dim=32,
            mask_zero=True
        )
        self.country_embedding = keras.layers.Embedding(
            input_dim=categorical_dims['manufacturer_country'],
            output_dim=16,
            mask_zero=True
        )

        # Material processing subnet
        self.material_dense1 = keras.layers.Dense(64, activation='relu')
        self.material_dense2 = keras.layers.Dense(32, activation='relu')

        # Numerical features processing
        self.numerical_dense = keras.layers.Dense(16, activation='relu')

        # Missing value indicator processing
        self.missing_indicator_dense = keras.layers.Dense(8, activation='relu')

        # Shared hidden layers
        self.shared_dense1 = keras.layers.Dense(256, activation='relu')
        self.shared_dropout1 = keras.layers.Dropout(0.3)
        self.shared_dense2 = keras.layers.Dense(128, activation='relu')
        self.shared_dropout2 = keras.layers.Dropout(0.2)
        self.shared_dense3 = keras.layers.Dense(64, activation='relu')

        # Task-specific heads (one per target)
        self.carbon_material_head = keras.Sequential([
            keras.layers.Dense(32, activation='relu'),
            keras.layers.Dense(16, activation='relu'),
            keras.layers.Dense(1, name='carbon_material')
        ])

        self.carbon_transport_head = keras.Sequential([
            keras.layers.Dense(32, activation='relu'),
            keras.layers.Dense(16, activation='relu'),
            keras.layers.Dense(1, name='carbon_transport')
        ])

        self.carbon_total_head = keras.Sequential([
            keras.layers.Dense(32, activation='relu'),
            keras.layers.Dense(16, activation='relu'),
            keras.layers.Dense(1, name='carbon_total')
        ])

        self.water_total_head = keras.Sequential([
            keras.layers.Dense(32, activation='relu'),
            keras.layers.Dense(16, activation='relu'),
            keras.layers.Dense(1, name='water_total')
        ])

    def call(self, inputs):
        # Process categorical features
        gender_emb = self.gender_embedding(inputs['gender'])
        category_emb = self.category_embedding(inputs['category'])
        country_emb = self.country_embedding(inputs['manufacturer_country'])

        # Flatten embeddings
        gender_flat = keras.layers.Flatten()(gender_emb)
        category_flat = keras.layers.Flatten()(category_emb)
        country_flat = keras.layers.Flatten()(country_emb)

        # Process material features
        material_processed = self.material_dense1(inputs['materials'])
        material_processed = self.material_dense2(material_processed)

        # Process numerical features
        numerical_processed = self.numerical_dense(inputs['numerical'])

        # Process missing indicators
        missing_processed = self.missing_indicator_dense(inputs['missing_indicators'])

        # Concatenate all features
        combined = keras.layers.Concatenate()([
            gender_flat,
            category_flat,
            country_flat,
            material_processed,
            numerical_processed,
            missing_processed
        ])

        # Shared representation learning
        shared = self.shared_dense1(combined)
        shared = self.shared_dropout1(shared)
        shared = self.shared_dense2(shared)
        shared = self.shared_dropout2(shared)
        shared = self.shared_dense3(shared)

        # Task-specific predictions
        carbon_material = self.carbon_material_head(shared)
        carbon_transport = self.carbon_transport_head(shared)
        carbon_total = self.carbon_total_head(shared)
        water_total = self.water_total_head(shared)

        return {
            'carbon_material': carbon_material,
            'carbon_transport': carbon_transport,
            'carbon_total': carbon_total,
            'water_total': water_total
        }

# Custom loss function with task weighting
def multi_task_loss(y_true, y_pred, task_weights):
    """
    Weighted MSE loss for multi-task learning

    Args:
        y_true: Dictionary of true values for each task
        y_pred: Dictionary of predictions for each task
        task_weights: Dictionary of weights for each task
    """
    losses = {}
    for task in ['carbon_material', 'carbon_transport', 'carbon_total', 'water_total']:
        mse = tf.reduce_mean(tf.square(y_true[task] - y_pred[task]))
        losses[task] = task_weights[task] * mse

    return sum(losses.values())

# Compile model
model = MissingValueAwareNN(
    categorical_dims={
        'gender': 4,  # Male, Female, Unisex, Unknown
        'category': 100,  # Approximate number of categories
        'manufacturer_country': 200,  # Number of countries
    },
    material_features=20,  # Number of material types
    numerical_features=2  # weight_kg, total_distance_km
)

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss=multi_task_loss,
    metrics=['mae', 'mse']
)
```

**Missing Value Handling:**
- Embedding layers with `mask_zero=True`
- Explicit missing indicator features
- Learn optimal representation for missing data during training

**Training Strategy:**
```python
# Use learning rate scheduling
lr_scheduler = keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=5,
    min_lr=1e-6
)

# Early stopping to prevent overfitting
early_stopping = keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=15,
    restore_best_weights=True
)

# Train with validation split
history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=200,
    batch_size=128,
    callbacks=[lr_scheduler, early_stopping]
)
```

**Pros:**
- Learns complex feature interactions
- Shared representations across tasks
- Flexible missing value handling
- Can incorporate physics-based constraints

**Cons:**
- Requires more data than tree-based models
- Longer training time
- More hyperparameters to tune
- Risk of overfitting

**Expected Performance:** ★★★★☆ (Good if sufficient data available)

---

### 3.4 Architecture 4: Stacked Ensemble with Hierarchical Imputation

**Model: Multi-Stage Ensemble Pipeline**

**Concept:**
Combine multiple model types in a stacked ensemble, with hierarchical imputation strategy.

**Stage 1: Imputation Models**
```python
# Train specialized models to impute missing values
imputation_models = {
    'weight_kg': XGBRegressor().fit(
        X_train[~X_train['weight_kg'].isna()][['category', 'materials']],
        y_train[~X_train['weight_kg'].isna()]['weight_kg']
    ),
    'total_distance_km': XGBRegressor().fit(
        X_train[~X_train['total_distance_km'].isna()][['manufacturer_country']],
        y_train[~X_train['total_distance_km'].isna()]['total_distance_km']
    )
}
```

**Stage 2: Base Models**
```python
# Train diverse base models
base_models = {
    'xgboost': XGBRegressor(),
    'lightgbm': lgb.LGBMRegressor(),
    'random_forest': RandomForestRegressor(n_estimators=200),
    'neural_network': MissingValueAwareNN(),
    'linear_regression': Ridge(alpha=1.0)  # For baseline
}

# Generate predictions from each base model
base_predictions = {}
for name, model in base_models.items():
    base_predictions[name] = model.predict(X_train)
```

**Stage 3: Meta-Learner**
```python
# Stack base model predictions
X_meta = np.hstack([pred.reshape(-1, 1) for pred in base_predictions.values()])

# Train meta-model to combine base predictions
meta_model = XGBRegressor(
    n_estimators=100,
    max_depth=3,  # Shallow to prevent overfitting
    learning_rate=0.1
)
meta_model.fit(X_meta, y_train)
```

**Pros:**
- Combines strengths of multiple algorithms
- Robust to individual model weaknesses
- Often achieves best overall accuracy

**Cons:**
- Complex training pipeline
- Longer inference time
- Requires careful cross-validation

**Expected Performance:** ★★★★★ (Best overall, but most complex)

---

### 3.5 Architecture 5: AutoML with Missing Value Optimization

**Model: AutoGluon Tabular**

**Why AutoML:**
- Automatically tries multiple algorithms
- Optimizes hyperparameters
- Handles missing values intelligently
- Minimal manual tuning

**Architecture:**
```python
from autogluon.tabular import TabularPredictor

# Separate predictor for each target
predictors = {}

for target in ['carbon_material', 'carbon_transport', 'carbon_total', 'water_total']:
    predictor = TabularPredictor(
        label=target,
        problem_type='regression',
        eval_metric='mean_absolute_error',
        path=f'./models/autogluon_{target}'
    )

    predictor.fit(
        train_data=train_df,
        time_limit=3600,  # 1 hour per target
        presets='best_quality',  # or 'high_quality', 'good_quality', 'medium_quality'
        infer_limit=0.001,  # Fast inference requirement
        excluded_model_types=['KNN'],  # Exclude slow models if needed
        hyperparameters={
            'GBM': {},  # XGBoost, LightGBM, CatBoost
            'NN_TORCH': {},  # Neural networks
            'RF': {},  # Random Forest
            'XT': {},  # Extra Trees
        }
    )

    predictors[target] = predictor
```

**Pros:**
- Minimal code and effort
- State-of-the-art performance
- Automatic ensemble creation
- Built-in feature engineering

**Cons:**
- Less control over architecture
- Longer training time
- Larger model size
- May be overkill for simple problems

**Expected Performance:** ★★★★★ (Excellent, minimal effort)

---

## 4. Advanced Missing Value Strategies

### 4.1 Probabilistic Imputation with Uncertainty

```python
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

# Use iterative imputation (similar to MICE algorithm)
imputer = IterativeImputer(
    estimator=XGBRegressor(),
    max_iter=10,
    random_state=42,
    add_indicator=True  # Add missing indicators
)

X_imputed = imputer.fit_transform(X_train)
```

### 4.2 Conditional Imputation Based on Feature Relationships

```python
def conditional_imputation(df):
    """
    Smart imputation based on domain knowledge
    """
    # If distance is missing but country is known
    if pd.isna(df['total_distance_km']) and not pd.isna(df['manufacturer_country']):
        # Use average distance for that country
        df['total_distance_km'] = country_distance_mapping[df['manufacturer_country']]

    # If weight is missing but category and materials are known
    if pd.isna(df['weight_kg']) and not pd.isna(df['category']):
        # Use category-specific weight estimation
        df['weight_kg'] = predict_weight_from_category_materials(
            df['category'],
            df['materials']
        )

    return df
```

### 4.3 Missing Value Encoding as Feature

```python
# Create categorical encoding for missing patterns
def create_missing_pattern_feature(df):
    """
    Encode which features are missing as a categorical feature
    """
    pattern = []
    for idx, row in df.iterrows():
        missing = []
        if pd.isna(row['category']): missing.append('C')
        if pd.isna(row['weight_kg']): missing.append('W')
        if pd.isna(row['total_distance_km']): missing.append('D')
        if pd.isna(row['materials']): missing.append('M')

        pattern.append('_'.join(missing) if missing else 'COMPLETE')

    df['missing_pattern'] = pattern
    return df
```

---

## 5. Model Evaluation Strategy

### 5.1 Cross-Validation Scheme

**Stratified K-Fold with Missing Data Simulation**

```python
from sklearn.model_selection import KFold
import numpy as np

# 5-fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Evaluation metrics
metrics = {
    'mae': [],
    'rmse': [],
    'mape': [],
    'r2': []
}

for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
    # Train model
    model.fit(X[train_idx], y[train_idx])

    # Evaluate on validation set
    y_pred = model.predict(X[val_idx])

    # Calculate metrics for each target
    for target in ['carbon_material', 'carbon_transport', 'carbon_total', 'water_total']:
        mae = mean_absolute_error(y_val[target], y_pred[target])
        rmse = np.sqrt(mean_squared_error(y_val[target], y_pred[target]))
        mape = mean_absolute_percentage_error(y_val[target], y_pred[target])
        r2 = r2_score(y_val[target], y_pred[target])

        metrics[target][fold] = {
            'mae': mae,
            'rmse': rmse,
            'mape': mape,
            'r2': r2
        }
```

### 5.2 Missing Value Robustness Testing

**Simulate increasing levels of missingness:**

```python
def evaluate_missing_robustness(model, X_test, y_test):
    """
    Test model performance with artificially introduced missing values
    """
    results = []

    # Test with different missing percentages
    for missing_pct in [0, 10, 20, 30, 40, 50]:
        X_test_masked = introduce_random_missing(X_test, missing_pct)

        y_pred = model.predict(X_test_masked)

        mae = mean_absolute_error(y_test, y_pred)
        results.append({
            'missing_percentage': missing_pct,
            'mae': mae,
            'mae_increase': mae / results[0]['mae'] if results else 1.0
        })

    return pd.DataFrame(results)
```

### 5.3 Evaluation Metrics

**Primary Metrics:**
- **MAE (Mean Absolute Error)**: Main metric, easy to interpret
- **RMSE (Root Mean Squared Error)**: Penalizes large errors
- **MAPE (Mean Absolute Percentage Error)**: Relative error
- **R² Score**: Proportion of variance explained

**Secondary Metrics:**
- **Missing Data MAE**: Performance specifically on samples with missing values
- **Prediction Interval Coverage**: For uncertainty quantification
- **Inference Time**: Speed requirement for production

**Per-Target Metrics:**
```python
# Weighted average across targets
weights = {
    'carbon_material': 0.3,
    'carbon_transport': 0.2,
    'carbon_total': 0.3,
    'water_total': 0.2
}

weighted_mae = sum(metrics[target]['mae'] * weights[target] for target in targets)
```

---

## 6. Training Pipeline Implementation

### 6.1 Data Preparation Pipeline

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import json

class FootprintDataPreprocessor:
    """
    Comprehensive preprocessing pipeline for footprint prediction
    """

    def __init__(self):
        self.material_types = []
        self.categorical_encoders = {}
        self.numerical_scaler = StandardScaler()
        self.target_scaler = {}

    def parse_materials(self, materials_json):
        """
        Parse JSON materials into one-hot encoded features

        Args:
            materials_json: String like '{"cotton_conventional":0.63,"polyester_recycled":0.37}'

        Returns:
            Dictionary with material percentages
        """
        if pd.isna(materials_json):
            return {material: 0.0 for material in self.material_types}

        try:
            materials_dict = json.loads(materials_json)
            # Ensure all material types are present
            result = {material: 0.0 for material in self.material_types}
            result.update(materials_dict)
            return result
        except:
            return {material: 0.0 for material in self.material_types}

    def extract_material_features(self, df):
        """
        Extract material composition features
        """
        # Get all unique materials from dataset
        all_materials = set()
        for materials_str in df['materials'].dropna():
            try:
                materials = json.loads(materials_str)
                all_materials.update(materials.keys())
            except:
                continue

        self.material_types = sorted(list(all_materials))

        # Create features for each material
        material_features = df['materials'].apply(self.parse_materials)
        material_df = pd.DataFrame(material_features.tolist())
        material_df.columns = [f'material_{col}' for col in material_df.columns]

        # Add derived features
        material_df['material_count'] = (material_df > 0).sum(axis=1)
        material_df['material_diversity'] = material_df.apply(
            lambda row: -sum(p * np.log(p + 1e-10) for p in row if p > 0),
            axis=1
        )

        return material_df

    def encode_categorical_features(self, df, fit=True):
        """
        Encode categorical features using target encoding or label encoding
        """
        categorical_cols = ['gender', 'parent_category', 'category', 'manufacturer_country']

        encoded_df = df.copy()

        for col in categorical_cols:
            if fit:
                self.categorical_encoders[col] = LabelEncoder()
                # Fill missing with 'Unknown'
                encoded_df[col] = encoded_df[col].fillna('Unknown')
                encoded_df[f'{col}_encoded'] = self.categorical_encoders[col].fit_transform(
                    encoded_df[col]
                )
            else:
                encoded_df[col] = encoded_df[col].fillna('Unknown')
                # Handle unseen categories
                encoded_df[f'{col}_encoded'] = encoded_df[col].apply(
                    lambda x: self.categorical_encoders[col].transform([x])[0]
                    if x in self.categorical_encoders[col].classes_
                    else -1
                )

        return encoded_df

    def create_missing_indicators(self, df):
        """
        Create binary features indicating which values are missing
        """
        important_features = ['category', 'weight_kg', 'total_distance_km', 'materials']

        for feature in important_features:
            df[f'{feature}_missing'] = df[feature].isna().astype(int)

        return df

    def prepare_features(self, df, fit=True):
        """
        Complete feature preparation pipeline
        """
        # 1. Create missing indicators
        df = self.create_missing_indicators(df)

        # 2. Extract material features
        material_features = self.extract_material_features(df)

        # 3. Encode categorical features
        df = self.encode_categorical_features(df, fit=fit)

        # 4. Combine all features
        feature_cols = (
            [f'{col}_encoded' for col in ['gender', 'parent_category', 'category', 'manufacturer_country']] +
            ['weight_kg', 'total_distance_km'] +
            [f'{col}_missing' for col in ['category', 'weight_kg', 'total_distance_km', 'materials']]
        )

        X = pd.concat([df[feature_cols], material_features], axis=1)

        # 5. Handle remaining missing values (simple imputation for numerical)
        X['weight_kg'] = X['weight_kg'].fillna(X['weight_kg'].median())
        X['total_distance_km'] = X['total_distance_km'].fillna(X['total_distance_km'].median())

        # 6. Scale numerical features
        numerical_cols = ['weight_kg', 'total_distance_km'] + list(material_features.columns)

        if fit:
            X[numerical_cols] = self.numerical_scaler.fit_transform(X[numerical_cols])
        else:
            X[numerical_cols] = self.numerical_scaler.transform(X[numerical_cols])

        return X

    def prepare_targets(self, df, fit=True):
        """
        Prepare target variables
        """
        targets = ['carbon_material', 'carbon_transport', 'carbon_total', 'water_total']
        y = df[targets].copy()

        # Optional: Scale targets for neural networks
        for target in targets:
            if fit:
                self.target_scaler[target] = StandardScaler()
                y[target] = self.target_scaler[target].fit_transform(
                    y[target].values.reshape(-1, 1)
                ).flatten()
            else:
                y[target] = self.target_scaler[target].transform(
                    y[target].values.reshape(-1, 1)
                ).flatten()

        return y

# Usage
preprocessor = FootprintDataPreprocessor()

# Load data
df = pd.read_csv('models/data_input/Product_data_with_footprints.csv')

# Split data
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
train_df, val_df = train_test_split(train_df, test_size=0.2, random_state=42)

# Prepare features and targets
X_train = preprocessor.prepare_features(train_df, fit=True)
y_train = preprocessor.prepare_targets(train_df, fit=True)

X_val = preprocessor.prepare_features(val_df, fit=False)
y_val = preprocessor.prepare_targets(val_df, fit=False)

X_test = preprocessor.prepare_features(test_df, fit=False)
y_test = preprocessor.prepare_targets(test_df, fit=False)
```

### 6.2 Model Training Script

```python
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

class FootprintModelTrainer:
    """
    Unified training interface for all model architectures
    """

    def __init__(self, model_type='xgboost'):
        self.model_type = model_type
        self.models = {}
        self.metrics = {}

    def train_xgboost(self, X_train, y_train, X_val, y_val):
        """Train XGBoost models for each target"""
        from xgboost import XGBRegressor

        targets = ['carbon_material', 'carbon_transport', 'carbon_total', 'water_total']

        for target in targets:
            print(f"\nTraining XGBoost for {target}...")

            model = XGBRegressor(
                n_estimators=500,
                max_depth=8,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                missing=np.nan,
                tree_method='hist',
                random_state=42,
                n_jobs=-1
            )

            model.fit(
                X_train, y_train[target],
                eval_set=[(X_val, y_val[target])],
                early_stopping_rounds=50,
                verbose=False
            )

            self.models[target] = model

            # Evaluate
            y_pred_val = model.predict(X_val)
            mae = mean_absolute_error(y_val[target], y_pred_val)
            rmse = np.sqrt(mean_squared_error(y_val[target], y_pred_val))
            r2 = r2_score(y_val[target], y_pred_val)

            self.metrics[target] = {
                'mae': mae,
                'rmse': rmse,
                'r2': r2
            }

            print(f"  MAE: {mae:.4f}, RMSE: {rmse:.4f}, R²: {r2:.4f}")

    def train_lightgbm(self, X_train, y_train, X_val, y_val):
        """Train LightGBM models for each target"""
        import lightgbm as lgb

        targets = ['carbon_material', 'carbon_transport', 'carbon_total', 'water_total']

        for target in targets:
            print(f"\nTraining LightGBM for {target}...")

            model = lgb.LGBMRegressor(
                n_estimators=500,
                num_leaves=64,
                learning_rate=0.05,
                feature_fraction=0.8,
                bagging_fraction=0.8,
                bagging_freq=5,
                random_state=42,
                n_jobs=-1,
                verbose=-1
            )

            model.fit(
                X_train, y_train[target],
                eval_set=[(X_val, y_val[target])],
                callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
            )

            self.models[target] = model

            # Evaluate
            y_pred_val = model.predict(X_val)
            mae = mean_absolute_error(y_val[target], y_pred_val)
            rmse = np.sqrt(mean_squared_error(y_val[target], y_pred_val))
            r2 = r2_score(y_val[target], y_pred_val)

            self.metrics[target] = {
                'mae': mae,
                'rmse': rmse,
                'r2': r2
            }

            print(f"  MAE: {mae:.4f}, RMSE: {rmse:.4f}, R²: {r2:.4f}")

    def save_models(self, path='models/saved/'):
        """Save trained models"""
        import os
        os.makedirs(path, exist_ok=True)

        for target, model in self.models.items():
            joblib.dump(model, f'{path}/{self.model_type}_{target}.pkl')

        print(f"\nModels saved to {path}")

    def load_models(self, path='models/saved/'):
        """Load trained models"""
        targets = ['carbon_material', 'carbon_transport', 'carbon_total', 'water_total']

        for target in targets:
            self.models[target] = joblib.load(f'{path}/{self.model_type}_{target}.pkl')

        print(f"Models loaded from {path}")

    def predict(self, X):
        """Make predictions for all targets"""
        predictions = {}

        for target, model in self.models.items():
            predictions[target] = model.predict(X)

        return pd.DataFrame(predictions)

# Usage
trainer = FootprintModelTrainer(model_type='xgboost')
trainer.train_xgboost(X_train, y_train, X_val, y_val)
trainer.save_models('models/saved/xgboost/')

# Make predictions
predictions = trainer.predict(X_test)
```

---

## 7. Model Comparison Framework

### 7.1 Benchmark All Architectures

```python
import time

class ModelBenchmark:
    """
    Compare all model architectures on same dataset
    """

    def __init__(self, X_train, y_train, X_val, y_val, X_test, y_test):
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.X_test = X_test
        self.y_test = y_test
        self.results = []

    def benchmark_model(self, model_name, trainer):
        """
        Benchmark a single model
        """
        print(f"\n{'='*60}")
        print(f"Benchmarking: {model_name}")
        print(f"{'='*60}")

        # Training time
        start_time = time.time()

        if model_name == 'XGBoost':
            trainer.train_xgboost(self.X_train, self.y_train, self.X_val, self.y_val)
        elif model_name == 'LightGBM':
            trainer.train_lightgbm(self.X_train, self.y_train, self.X_val, self.y_val)
        # Add other model types...

        training_time = time.time() - start_time

        # Inference time
        start_time = time.time()
        predictions = trainer.predict(self.X_test)
        inference_time = time.time() - start_time

        # Calculate metrics for each target
        targets = ['carbon_material', 'carbon_transport', 'carbon_total', 'water_total']

        for target in targets:
            mae = mean_absolute_error(self.y_test[target], predictions[target])
            rmse = np.sqrt(mean_squared_error(self.y_test[target], predictions[target]))
            mape = np.mean(np.abs((self.y_test[target] - predictions[target]) / self.y_test[target])) * 100
            r2 = r2_score(self.y_test[target], predictions[target])

            self.results.append({
                'model': model_name,
                'target': target,
                'mae': mae,
                'rmse': rmse,
                'mape': mape,
                'r2': r2,
                'training_time': training_time,
                'inference_time': inference_time
            })

    def run_all_benchmarks(self):
        """
        Run benchmarks for all model architectures
        """
        models = [
            ('XGBoost', FootprintModelTrainer('xgboost')),
            ('LightGBM', FootprintModelTrainer('lightgbm')),
            # Add other models...
        ]

        for model_name, trainer in models:
            self.benchmark_model(model_name, trainer)

    def get_results_dataframe(self):
        """
        Return results as DataFrame
        """
        return pd.DataFrame(self.results)

    def plot_comparison(self):
        """
        Visualize model comparison
        """
        import matplotlib.pyplot as plt
        import seaborn as sns

        results_df = self.get_results_dataframe()

        # Plot MAE comparison
        plt.figure(figsize=(14, 6))

        plt.subplot(1, 2, 1)
        sns.barplot(data=results_df, x='target', y='mae', hue='model')
        plt.title('Mean Absolute Error by Model and Target')
        plt.xticks(rotation=45)
        plt.ylabel('MAE')

        plt.subplot(1, 2, 2)
        sns.barplot(data=results_df, x='target', y='r2', hue='model')
        plt.title('R² Score by Model and Target')
        plt.xticks(rotation=45)
        plt.ylabel('R² Score')

        plt.tight_layout()
        plt.savefig('models/evaluation/model_comparison.png', dpi=300)
        plt.show()

# Usage
benchmark = ModelBenchmark(X_train, y_train, X_val, y_val, X_test, y_test)
benchmark.run_all_benchmarks()
results = benchmark.get_results_dataframe()
benchmark.plot_comparison()
```

---

## 8. Production Deployment Considerations

### 8.1 Model Serving API

```python
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI()

# Load models at startup
models = {}
preprocessor = None

@app.on_event("startup")
async def load_models():
    global models, preprocessor

    targets = ['carbon_material', 'carbon_transport', 'carbon_total', 'water_total']
    for target in targets:
        models[target] = joblib.load(f'models/saved/xgboost_{target}.pkl')

    preprocessor = joblib.load('models/saved/preprocessor.pkl')

class ProductInput(BaseModel):
    gender: str = None
    parent_category: str = None
    category: str = None
    manufacturer_country: str = None
    materials: str = None  # JSON string
    weight_kg: float = None
    total_distance_km: float = None

class PredictionOutput(BaseModel):
    carbon_material: float
    carbon_transport: float
    carbon_total: float
    water_total: float
    confidence_intervals: dict = None

@app.post("/predict", response_model=PredictionOutput)
async def predict_footprint(product: ProductInput):
    """
    Predict environmental footprints for a product
    """
    # Convert input to DataFrame
    input_df = pd.DataFrame([product.dict()])

    # Preprocess
    X = preprocessor.prepare_features(input_df, fit=False)

    # Predict
    predictions = {}
    for target, model in models.items():
        predictions[target] = float(model.predict(X)[0])

    return PredictionOutput(**predictions)

@app.get("/health")
async def health_check():
    return {"status": "healthy", "models_loaded": len(models)}
```

### 8.2 Model Monitoring

```python
class ModelMonitor:
    """
    Monitor model performance in production
    """

    def __init__(self):
        self.predictions_log = []
        self.performance_metrics = []

    def log_prediction(self, input_features, predictions, actual_values=None):
        """
        Log prediction for monitoring
        """
        log_entry = {
            'timestamp': pd.Timestamp.now(),
            'input': input_features,
            'predictions': predictions,
            'actual': actual_values
        }
        self.predictions_log.append(log_entry)

    def detect_data_drift(self, recent_window=1000):
        """
        Detect if input distribution has changed
        """
        from scipy.stats import ks_2samp

        # Compare recent predictions to training distribution
        recent_data = self.predictions_log[-recent_window:]

        # Kolmogorov-Smirnov test for distribution drift
        # (Compare recent vs baseline)

        return drift_detected

    def calculate_rolling_metrics(self, window=100):
        """
        Calculate rolling performance metrics
        """
        # Only if actual values are available
        recent_logs = self.predictions_log[-window:]

        for target in ['carbon_material', 'carbon_transport', 'carbon_total', 'water_total']:
            actuals = [log['actual'][target] for log in recent_logs if log['actual']]
            preds = [log['predictions'][target] for log in recent_logs if log['actual']]

            if actuals:
                mae = mean_absolute_error(actuals, preds)
                self.performance_metrics.append({
                    'timestamp': pd.Timestamp.now(),
                    'target': target,
                    'mae': mae,
                    'window_size': len(actuals)
                })
```

---

## 9. Recommended Implementation Roadmap

### Phase 1: Baseline (Week 1)
1. Implement data preprocessing pipeline
2. Train XGBoost baseline model
3. Establish evaluation metrics
4. Create simple missing value handling

**Expected MAE:** 0.15-0.20 kg CO2e for carbon, 200-300L for water

### Phase 2: Advanced Models (Week 2)
1. Implement LightGBM variant
2. Add neural network architecture
3. Implement sophisticated missing value strategies
4. Cross-validation and hyperparameter tuning

**Expected MAE:** 0.10-0.15 kg CO2e for carbon, 150-200L for water

### Phase 3: Ensemble (Week 3)
1. Build stacked ensemble
2. Implement AutoML comparison
3. Optimize for missing value robustness
4. Final model selection

**Expected MAE:** 0.08-0.12 kg CO2e for carbon, 100-150L for water

### Phase 4: Production (Week 4)
1. Deploy best model as API
2. Set up monitoring
3. Documentation
4. Testing and validation

---

## 10. Success Criteria

### Model Performance Targets

**Tier 1: Minimum Viable**
- MAE < 0.20 kg CO2e (carbon predictions)
- MAE < 300L (water predictions)
- R² > 0.70
- Inference time < 100ms

**Tier 2: Production Ready**
- MAE < 0.15 kg CO2e (carbon predictions)
- MAE < 200L (water predictions)
- R² > 0.80
- Inference time < 50ms

**Tier 3: Excellent**
- MAE < 0.10 kg CO2e (carbon predictions)
- MAE < 150L (water predictions)
- R² > 0.90
- Inference time < 20ms

### Missing Value Robustness

- Performance degradation < 20% with 30% missing values
- Graceful degradation as missingness increases
- No catastrophic failures with extreme missing patterns

---

## 11. Conclusion and Recommendations

### Top Recommended Approach

**Primary: XGBoost with Advanced Imputation (Architecture 3.1)**

**Rationale:**
1. Best performance on tabular data
2. Native missing value handling
3. Fast training and inference
4. Interpretable feature importance
5. Proven track record

**Secondary: Stacked Ensemble (Architecture 3.4)**

**Rationale:**
1. Combines multiple model strengths
2. Highest accuracy potential
3. Robust to data variations
4. More complex but worth for production

### Quick Start Implementation

1. Start with XGBoost baseline
2. Add missing indicators as features
3. Use iterative imputation for complex cases
4. Validate with cross-validation
5. Deploy and monitor

### Next Steps

1. Implement preprocessing pipeline
2. Train baseline XGBoost model
3. Evaluate on test set with missing value simulation
4. Iterate on hyperparameters
5. Compare with LightGBM and neural network
6. Build ensemble if needed
7. Deploy best model

---

**Document Version:** 1.0
**Last Updated:** 2025-12-02
**Author:** Claude Code
**Status:** Ready for Implementation
