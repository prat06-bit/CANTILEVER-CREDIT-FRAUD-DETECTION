# 🛡️ Credit Card Fraud Detection — CANTILEVER Internship

An end-to-end machine-learning pipeline for detecting fraudulent credit card
transactions, built on the standard
[Kaggle Credit Card Fraud Detection dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
(284,807 transactions by European cardholders, September 2013).

---

## 📊 Model Performance Summary

### Test Set Results (56,962 transactions — 98 fraud)

| Model | F1 (Fraud) | Precision | Recall | PR-AUC | ROC-AUC | False Alarms |
|-------|:----------:|:---------:|:------:|:------:|:-------:|:------------:|
| **Gradient Boosting** | **TBD** | **TBD** | **TBD** | **TBD** | **TBD** | **TBD** |
| Random Forest | 0.7885 | 74.6% | 83.7% | 0.8577 | 0.9877 | 29 |
| Logistic Regression | 0.0929 | 4.9% | 91.8% | 0.7692 | 0.9772 | 1,834 |

> **Note:** Gradient Boosting results will be populated after first run of v3.0.
> Random Forest and Logistic Regression scores are from the v2.0 enhanced feature run.

### 5-Fold Stratified Cross-Validation (SMOTE inside each fold)

| Model | CV F1 | CV Precision | CV Recall | CV ROC-AUC |
|-------|:-----:|:------------:|:---------:|:----------:|
| Random Forest | 0.8186 ± 0.0150 | 0.8415 ± 0.0497 | 0.8020 ± 0.0490 | 0.9801 ± 0.0156 |
| Logistic Regression | 0.0955 ± 0.0052 | 0.0504 ± 0.0030 | 0.9138 ± 0.0351 | 0.9775 ± 0.0143 |

---

## 🏗️ Project Architecture

```
CANTILEVER-CREDIT CARD FRAUD/
├── main.py              ← Entry point — run this file
├── pipeline.py          ← Core ML pipeline (all stages A-G)
├── config.py            ← All constants & hyperparameters
├── requirements.txt     ← Python dependencies
├── README.md
├── .gitignore
├── data/
│   └── creditcard.csv   ← Dataset (download from Kaggle)
├── models/
│   ├── random_forest.joblib
│   ├── gradient_boosting.joblib
│   ├── logistic_regression.joblib
│   ├── scaler_amount.joblib
│   ├── scaler_time.joblib
│   └── feature_names.joblib
└── outputs/
    ├── eda_overview.png
    ├── correlation_heatmap.png
    ├── eval_random_forest.png
    ├── eval_gradient_boosting.png
    ├── eval_logistic_regression.png
    ├── feature_importance_*.png
    └── model_comparison.csv
```

### Module Responsibilities

| Module | Lines | Purpose |
|--------|:-----:|---------|
| `config.py` | ~70 | All tuneable constants, paths, hyperparameters |
| `pipeline.py` | ~500 | Core ML logic: EDA, feature engineering, SMOTE, training, evaluation, production simulation |
| `main.py` | ~100 | Orchestrator — imports from `pipeline.py` and `config.py`, runs stages A→G sequentially |

---

## 🔬 Feature Engineering (14 Derived Features)

The pipeline creates 14 new features from the raw V1-V28, Amount, and Time columns:

| Feature | Type | Rationale |
|---------|------|-----------|
| `Hour_Sin`, `Hour_Cos` | Time cyclical | Night-time fraud pattern detection (cyclical encoding) |
| `Log_Amount` | Amount transform | Stabilises extreme right skew in transaction amounts |
| `V_Mean` | PCA aggregate | Average distance from PCA centroid per transaction |
| `V_Std` | PCA aggregate | Spread across PCA dimensions (high = unusual) |
| `V_Kurtosis` | PCA aggregate | Tail heaviness — anomalies have extreme kurtosis |
| `V_Magnitude` | L2 norm | Euclidean distance from PCA origin (frauds sit further out) |
| `V14×V4` | Interaction | Product of top-1 × top-3 features |
| `V14×V12` | Interaction | Product of top-1 × top-2 features (**becomes #1 most important**) |
| `V12×V10` | Interaction | Cross-component joint effect |
| `V10×V11` | Interaction | Cross-component joint effect |
| `V17×V3` | Interaction | Cross-component joint effect |

### Top 10 Features by Gini Importance (Random Forest)

| Rank | Feature | Type | Importance |
|:----:|---------|------|:----------:|
| 1 | **V14×V12** | Interaction (NEW) | 0.120 |
| 2 | V14 | Original PCA | 0.113 |
| 3 | **V_Magnitude** | L2 Norm (NEW) | 0.093 |
| 4 | **V14×V4** | Interaction (NEW) | 0.079 |
| 5 | V10 | Original PCA | 0.058 |
| 6 | **V_Std** | Aggregate (NEW) | 0.055 |
| 7 | **V10×V11** | Interaction (NEW) | 0.053 |
| 8 | **V_Mean** | Aggregate (NEW) | 0.050 |
| 9 | V4 | Original PCA | 0.049 |
| 10 | **V12×V10** | Interaction (NEW) | 0.039 |

> **7 out of 10** top features are derived features — proving the feature engineering adds genuine predictive signal.

---

## 🧠 Models Trained

### 1. Logistic Regression (Baseline)
- Interpretable linear model
- High recall (92%) but catastrophically low precision (5%)
- **94% of fraud alerts are false alarms** → impractical for production

### 2. Random Forest (Ensemble)
- 200 trees, max_depth=20, `balanced_subsample` class weights
- Best balance of precision (75%) and recall (84%)
- Only 29 false alarms on the test set

### 3. Gradient Boosting (NEW in v3.0)
- 200 sequential trees, learning_rate=0.1, subsample=0.8
- Sequential boosting corrects errors from previous trees
- Expected to compete with or exceed Random Forest

### Model Selection Criteria
The pipeline selects the best model by **F1-Score** (harmonic mean of precision and recall), not raw accuracy or recall alone. This prevents selecting a model that achieves high recall at the cost of unusable precision.

---

## 🛠️ Tech Stack

| Component | Library | Purpose |
|-----------|---------|---------|
| Core | Python 3.10+ | Language |
| Data | pandas, numpy | DataFrames & numerics |
| ML | scikit-learn | Models, metrics, CV |
| Imbalance | imbalanced-learn | SMOTE oversampling |
| Viz | matplotlib, seaborn | EDA & evaluation plots |
| Persistence | joblib | Model serialisation |

---

## 🚀 Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Download dataset
Download `creditcard.csv` from
[Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
and place it in `data/creditcard.csv`.

### 3. Run the pipeline
```bash
python main.py
```

The pipeline will:
1. Load & explore the data (Stage A)
2. Engineer 44 features from 31 raw columns (Stage B)
3. Split 80/20 stratified (Stage C)
4. Apply SMOTE to training set only (Stage D)
5. Train 3 models (Stage E)
6. Evaluate with train/test/CV metrics + plots (Stage F)
7. Simulate production transaction scoring (Stage G)

### 4. Load a saved model for inference
```python
from pipeline import load_pipeline

pipeline = load_pipeline("random_forest")
result = pipeline.predict({
    "Time": 100, "Amount": 49.99,
    "V1": -1.35, "V2": -0.07, ..., "V28": 0.01
})
# {'transaction_id': 'TXN-00001', 'fraud_flag': 0, 'risk_level': 'LOW', ...}
```

---

## 📋 Pipeline Stages

| Stage | Name | Description |
|:-----:|------|-------------|
| **A** | Data Loading & EDA | Load CSV, inspect imbalance (577:1 ratio), visualise distributions |
| **B** | Feature Engineering | RobustScaler on Amount/Time, 14 derived features (interactions, aggregates, cyclical time) |
| **C** | Train/Test Split | 80/20 stratified split preserving fraud proportion |
| **D** | SMOTE | Synthetic minority oversampling on training set only |
| **E** | Model Training | Logistic Regression, Random Forest, Gradient Boosting |
| **F** | Evaluation | Classification reports, confusion matrices, PR-AUC, ROC-AUC, 5-fold CV, threshold tuning, feature importance |
| **G** | Production Simulation | Real-time transaction scoring with risk bucketing (LOW/MEDIUM/HIGH/CRITICAL) |

---

## 📂 Outputs Generated

| File | Description |
|------|-------------|
| `outputs/eda_overview.png` | Class distribution, amount/time histograms |
| `outputs/correlation_heatmap.png` | Feature correlation matrix |
| `outputs/eval_*.png` | Per-model: confusion matrix, PR curve, ROC curve, threshold plot |
| `outputs/feature_importance_*.png` | Top 15 feature importances per model |
| `outputs/model_comparison.csv` | All metrics in tabular format |
| `models/*.joblib` | Serialised models and scalers for deployment |

---

## ⚙️ Configuration

All hyperparameters are in `config.py`:

```python
# Reproducibility
RANDOM_STATE = 42

# Split
TEST_SIZE = 0.2
CV_FOLDS = 5

# Random Forest
RF_N_ESTIMATORS = 200
RF_MAX_DEPTH = 20

# Gradient Boosting
GB_N_ESTIMATORS = 200
GB_LEARNING_RATE = 0.1

# Production
PRODUCTION_TARGET_RECALL = 0.80
```

---

## 📝 Design Decisions

1. **F1-Score for model selection** (not accuracy) — accuracy is misleading at 99.8% class imbalance
2. **SMOTE inside CV folds** via `imblearn.Pipeline` — prevents data leakage
3. **RobustScaler** instead of StandardScaler — resilient to extreme Amount outliers
4. **Overfitting check on original training set** — comparing SMOTE-balanced training to imbalanced test is misleading
5. **Threshold tuning for production** — default 0.5 threshold is rarely optimal for imbalanced data
6. **Three evaluation sets** (SMOTE training, original training, test) — comprehensive overfitting analysis

---

## 📄 License

This project is developed as part of the CANTILEVER internship program.
#   C A N T I L E V E R - C R E D I T - F R A U D - D E T E C T I O N  
 