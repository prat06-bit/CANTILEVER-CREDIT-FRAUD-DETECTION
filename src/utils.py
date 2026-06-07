import os

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split


RANDOM_STATE = 42
TEST_SIZE = 0.20
VALIDATION_SIZE = 0.20

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "creditcard.csv")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")


def load_data(path: str = DATA_PATH) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found: {path}")
    df = pd.read_csv(path)
    if df.isna().sum().sum() > 0:
        raise ValueError("Dataset contains missing values.")
    return df


def split_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_valid, test = train_test_split(
        df,
        test_size=TEST_SIZE,
        stratify=df["Class"],
        random_state=RANDOM_STATE,
    )
    train, valid = train_test_split(
        train_valid,
        test_size=VALIDATION_SIZE,
        stratify=train_valid["Class"],
        random_state=RANDOM_STATE,
    )
    return train.reset_index(drop=True), valid.reset_index(drop=True), test.reset_index(drop=True)


def class_ratio(y: pd.Series) -> float:
    positives = int((y == 1).sum())
    negatives = int((y == 0).sum())
    return negatives / max(positives, 1)


def stratified_sample(
    x: pd.DataFrame,
    y: pd.Series,
    max_rows: int,
) -> tuple[pd.DataFrame, pd.Series]:
    if len(x) <= max_rows:
        return x, y
    train_size = max_rows / len(x)
    x_sample, _, y_sample, _ = train_test_split(
        x,
        y,
        train_size=train_size,
        stratify=y,
        random_state=RANDOM_STATE,
    )
    return x_sample.reset_index(drop=True), y_sample.reset_index(drop=True)


def lgbm_callbacks(early_stopping_rounds: int, verbose: bool = False) -> list:
    log_period = 100 if verbose else 0
    return [
        lgb.early_stopping(early_stopping_rounds, verbose=False),
        lgb.log_evaluation(log_period),
    ]


def threshold_for_best_f1(
    y_true: pd.Series,
    probabilities: np.ndarray,
) -> tuple[float, float]:
    precision, recall, thresholds = precision_recall_curve(y_true, probabilities)
    f1_scores = 2.0 * precision[:-1] * recall[:-1] / (
        precision[:-1] + recall[:-1] + 1e-12
    )
    best_idx = int(np.argmax(f1_scores))
    return float(thresholds[best_idx]), float(f1_scores[best_idx])


def evaluate(
    y_true: pd.Series,
    probabilities: np.ndarray,
    threshold: float,
) -> dict[str, float | list[list[int]]]:
    predictions = (probabilities >= threshold).astype(np.int8)
    return {
        "precision": precision_score(y_true, predictions, zero_division=0),
        "recall": recall_score(y_true, predictions, zero_division=0),
        "f1": f1_score(y_true, predictions, zero_division=0),
        "pr_auc": average_precision_score(y_true, probabilities),
        "roc_auc": roc_auc_score(y_true, probabilities),
        "confusion_matrix": confusion_matrix(y_true, predictions).tolist(),
    }


def print_split_summary(train: pd.DataFrame, valid: pd.DataFrame, test: pd.DataFrame) -> None:
    print("\nDATA SPLIT")
    for name, split in [("Train", train), ("Validation", valid), ("Test", test)]:
        print(
            f"{name:<10}: {len(split):>7,} rows | "
            f"fraud={int(split['Class'].sum()):>4,} | rate={split['Class'].mean():.4%}"
        )


def print_test_report(y_true: pd.Series, probabilities: np.ndarray, threshold: float) -> None:
    predictions = (probabilities >= threshold).astype(np.int8)
    print("\nTEST CLASSIFICATION REPORT")
    print(classification_report(y_true, predictions, target_names=["Legit", "Fraud"], digits=4))


def save_outputs(
    model: lgb.LGBMClassifier,
    selected_features: list[str],
    best_params: dict,
    threshold: float,
    metrics: dict,
) -> None:
    os.makedirs(MODEL_DIR, exist_ok=True)
    model.booster_.save_model(os.path.join(MODEL_DIR, "lightgbm_fraud_model.txt"))
    pd.Series(selected_features, name="selected_features").to_csv(
        os.path.join(MODEL_DIR, "selected_features.csv"), index=False
    )
    pd.Series(best_params, name="value").to_csv(
        os.path.join(MODEL_DIR, "best_lightgbm_params.csv")
    )
    pd.Series({"threshold": threshold, **metrics}, name="value").to_csv(
        os.path.join(MODEL_DIR, "final_test_metrics.csv")
    )
