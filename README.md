# CANTILEVER Credit Card Fraud Detection

This repository contains an end-to-end fraud detection pipeline for the Kaggle credit card fraud dataset. The project trains a LightGBM model with engineered features, validation-based threshold tuning, and model artifact export for deployment.

## What this project does
- Loads the Kaggle credit card fraud dataset from `data/creditcard.csv`
- Builds engineered features such as time-of-day transforms, amount transforms, PCA aggregates, and interaction terms
- Splits data into train / validation / test sets with stratification
- Tunes and trains a LightGBM classifier with Optuna
- Evaluates precision, recall, F1, PR-AUC, ROC-AUC, and saves the final model outputs

## Repository structure
- `credit_card_fraud_detection.py` — main training pipeline entry point
- `src/features.py` — feature engineering and feature selection logic
- `src/train.py` — model tuning and final model selection
- `src/utils.py` — shared helpers for loading, evaluation, and output saving
- `data/` — dataset folder (the CSV file is expected here)
- `models/` — trained model files and metrics
- `outputs/` — comparison outputs and summaries
- `requirements.txt` — Python dependencies

## Requirements
Python 3.10+ is recommended.

Install dependencies with:

```bash
pip install -r requirements.txt
```

## Dataset
Download the dataset from Kaggle and place it in:

```text
data/creditcard.csv
```

## Run the project
From the repository root, run:

```bash
python credit_card_fraud_detection.py
```

This will:
1. Load the dataset
2. Create train / validation / test splits
3. Engineer features
4. Tune LightGBM hyperparameters with Optuna
5. Select the best model using validation F1
6. Evaluate on the test set
7. Save model artifacts in `models/`

## Current model summary
The latest saved model metrics in `models/final_test_metrics.csv` show:
- Precision: 0.9186
- Recall: 0.8061
- F1: 0.8587
- PR-AUC: 0.8696
- ROC-AUC: 0.9799

These values reflect the final LightGBM model evaluated on the held-out test set.

## Outputs
The project writes the following artifacts:
- `models/lightgbm_fraud_model.txt`
- `models/selected_features.csv`
- `models/best_lightgbm_params.csv`
- `models/final_test_metrics.csv`
- `outputs/model_comparison.csv`

## Notes
- The dataset is not included in this repository because it is large and distributed by Kaggle.
- The project is designed for fraud detection tasks where class imbalance is important, so F1 and PR-AUC are prioritized over raw accuracy.
