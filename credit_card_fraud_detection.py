import os
import time
import warnings
from dataclasses import dataclass

import lightgbm as lgb
import numpy as np
import optuna
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
from sklearn.preprocessing import RobustScaler


warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

RANDOM_STATE = 42
TEST_SIZE = 0.20
VALIDATION_SIZE = 0.20
N_TRIALS = int(os.getenv("OPTUNA_TRIALS", "30"))
OPTUNA_MAX_TRAIN_ROWS = int(os.getenv("OPTUNA_MAX_TRAIN_ROWS", "70000"))
EARLY_STOPPING_ROUNDS = 50
MAX_BOOST_ROUNDS = 900
MIN_SELECTED_FEATURES = 28
SELECTED_FEATURE_RATIO = 0.90

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "creditcard.csv")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")

INTERACTION_PAIRS = [
    ("V14", "V4"),
    ("V14", "V12"),
    ("V12", "V10"),
    ("V10", "V11"),
    ("V17", "V3"),
    ("V14", "V17"),
    ("V4", "V11"),
]


@dataclass
class Preprocessor:
    amount_scaler: RobustScaler
    time_scaler: RobustScaler
    feature_names: list[str]

    @classmethod
    def fit(cls, df: pd.DataFrame) -> "Preprocessor":
        amount_scaler = RobustScaler().fit(df[["Amount"]])
        time_scaler = RobustScaler().fit(df[["Time"]])
        features = build_features(df, amount_scaler, time_scaler)
        return cls(amount_scaler, time_scaler, list(features.columns))

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        features = build_features(df, self.amount_scaler, self.time_scaler)
        return features.reindex(columns=self.feature_names, fill_value=0.0)


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


def build_features(
    df: pd.DataFrame,
    amount_scaler: RobustScaler,
    time_scaler: RobustScaler,
) -> pd.DataFrame:
    x = df.drop(columns=["Class"], errors="ignore").copy()
    v_cols = [f"V{i}" for i in range(1, 29)]
    v = x[v_cols].to_numpy(dtype=np.float64)

    hour = (x["Time"].to_numpy(dtype=np.float64) % 86400.0) / 3600.0
    x["Hour_Sin"] = np.sin(2.0 * np.pi * hour / 24.0)
    x["Hour_Cos"] = np.cos(2.0 * np.pi * hour / 24.0)
    x["Log_Amount"] = np.log1p(x["Amount"])
    x["Amount_x_HourSin"] = x["Amount"] * x["Hour_Sin"]
    x["Amount_x_HourCos"] = x["Amount"] * x["Hour_Cos"]
    x["V_Mean"] = v.mean(axis=1)
    x["V_Std"] = v.std(axis=1)
    x["V_Min"] = v.min(axis=1)
    x["V_Max"] = v.max(axis=1)
    x["V_Magnitude"] = np.sqrt(np.square(v).sum(axis=1))

    centered = v - v.mean(axis=1, keepdims=True)
    m2 = np.square(centered).mean(axis=1)
    m4 = np.power(centered, 4).mean(axis=1)
    x["V_Kurtosis"] = np.where(m2 > 1e-10, m4 / np.square(m2) - 3.0, 0.0)

    for left, right in INTERACTION_PAIRS:
        x[f"{left}_x_{right}"] = x[left] * x[right]

    x["Scaled_Amount"] = amount_scaler.transform(x[["Amount"]])
    x["Scaled_Time"] = time_scaler.transform(x[["Time"]])
    x = x.drop(columns=["Amount", "Time"])

    front = ["Scaled_Amount", "Scaled_Time", "Log_Amount", "Hour_Sin", "Hour_Cos"]
    cols = front + [col for col in x.columns if col not in front]
    return x[cols].astype(np.float32)


def class_ratio(y: pd.Series) -> float:
    positives = int((y == 1).sum())
    negatives = int((y == 0).sum())
    return negatives / max(positives, 1)


def stratified_sample(
    x: pd.DataFrame,
    y: pd.Series,
    max_rows: int = OPTUNA_MAX_TRAIN_ROWS,
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


def lgbm_callbacks(verbose: bool = False) -> list:
    return [
        lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False),
        lgb.log_evaluation(100 if verbose else 0),
    ]


def threshold_for_best_f1(y_true: pd.Series, probabilities: np.ndarray) -> tuple[float, float]:
    precision, recall, thresholds = precision_recall_curve(y_true, probabilities)
    scores = 2.0 * precision[:-1] * recall[:-1] / (precision[:-1] + recall[:-1] + 1e-12)
    best_idx = int(np.argmax(scores))
    return float(thresholds[best_idx]), float(scores[best_idx])


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


def base_params(scale_pos_weight: float) -> dict:
    return {
        "objective": "binary",
        "boosting_type": "gbdt",
        "n_estimators": MAX_BOOST_ROUNDS,
        "learning_rate": 0.035,
        "num_leaves": 31,
        "max_depth": 6,
        "min_child_samples": 80,
        "subsample": 0.85,
        "subsample_freq": 1,
        "colsample_bytree": 0.85,
        "reg_alpha": 0.5,
        "reg_lambda": 8.0,
        "scale_pos_weight": scale_pos_weight,
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
        "verbosity": -1,
        "force_col_wise": True,
        "metric": "average_precision",
    }


def select_features(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_valid: pd.DataFrame,
    y_valid: pd.Series,
    scale_pos_weight: float,
) -> list[str]:
    selector = lgb.LGBMClassifier(**base_params(scale_pos_weight))
    selector.fit(
        x_train,
        y_train,
        eval_set=[(x_valid, y_valid)],
        eval_metric="average_precision",
        callbacks=lgbm_callbacks(),
    )

    gains = pd.Series(
        selector.booster_.feature_importance(importance_type="gain"),
        index=x_train.columns,
    ).sort_values(ascending=False)

    max_features = max(MIN_SELECTED_FEATURES, int(len(gains) * SELECTED_FEATURE_RATIO))
    selected = gains[gains > 0].head(max_features).index.tolist()
    if len(selected) < MIN_SELECTED_FEATURES:
        selected = gains.head(MIN_SELECTED_FEATURES).index.tolist()
    return selected


def optuna_objective(
    trial: optuna.Trial,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_valid: pd.DataFrame,
    y_valid: pd.Series,
    scale_pos_weight: float,
) -> float:
    params = {
        "objective": "binary",
        "boosting_type": "gbdt",
        "n_estimators": MAX_BOOST_ROUNDS,
        "learning_rate": trial.suggest_float("learning_rate", 0.015, 0.080, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 12, 80),
        "max_depth": trial.suggest_int("max_depth", 3, 9),
        "min_child_samples": trial.suggest_int("min_data_in_leaf", 30, 220),
        "subsample": trial.suggest_float("bagging_fraction", 0.60, 0.95),
        "subsample_freq": 1,
        "colsample_bytree": trial.suggest_float("feature_fraction", 0.55, 0.95),
        "reg_alpha": trial.suggest_float("lambda_l1", 1e-4, 10.0, log=True),
        "reg_lambda": trial.suggest_float("lambda_l2", 1e-3, 30.0, log=True),
        "min_split_gain": trial.suggest_float("min_gain_to_split", 0.0, 1.0),
        "scale_pos_weight": scale_pos_weight,
        "random_state": RANDOM_STATE + trial.number,
        "n_jobs": -1,
        "verbosity": -1,
        "force_col_wise": True,
        "metric": "average_precision",
    }

    model = lgb.LGBMClassifier(**params)
    model.fit(
        x_train,
        y_train,
        eval_set=[(x_valid, y_valid)],
        eval_metric="average_precision",
        callbacks=lgbm_callbacks(),
    )
    probabilities = model.predict_proba(x_valid)[:, 1]
    _, best_f1 = threshold_for_best_f1(y_valid, probabilities)
    pr_auc = average_precision_score(y_valid, probabilities)
    return 0.75 * best_f1 + 0.25 * pr_auc


def optimize_params(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_valid: pd.DataFrame,
    y_valid: pd.Series,
    scale_pos_weight: float,
) -> dict:
    sampler = optuna.samplers.TPESampler(seed=RANDOM_STATE, multivariate=True)
    pruner = optuna.pruners.MedianPruner(n_warmup_steps=8)
    study = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner)
    study.optimize(
        lambda trial: optuna_objective(
            trial, x_train, y_train, x_valid, y_valid, scale_pos_weight
        ),
        n_trials=N_TRIALS,
        show_progress_bar=False,
        gc_after_trial=True,
    )

    best = study.best_params.copy()
    return {
        "objective": "binary",
        "boosting_type": "gbdt",
        "n_estimators": MAX_BOOST_ROUNDS,
        "learning_rate": best["learning_rate"],
        "num_leaves": best["num_leaves"],
        "max_depth": best["max_depth"],
        "min_child_samples": best["min_data_in_leaf"],
        "subsample": best["bagging_fraction"],
        "subsample_freq": 1,
        "colsample_bytree": best["feature_fraction"],
        "reg_alpha": best["lambda_l1"],
        "reg_lambda": best["lambda_l2"],
        "min_split_gain": best["min_gain_to_split"],
        "scale_pos_weight": scale_pos_weight,
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
        "verbosity": -1,
        "force_col_wise": True,
        "metric": "average_precision",
    }


def regularized_candidate_params(scale_pos_weight: float) -> dict:
    return {
        "objective": "binary",
        "boosting_type": "gbdt",
        "n_estimators": MAX_BOOST_ROUNDS,
        "learning_rate": 0.028078987855609694,
        "num_leaves": 77,
        "max_depth": 8,
        "min_child_samples": 144,
        "subsample": 0.6546065241548528,
        "subsample_freq": 1,
        "colsample_bytree": 0.6123978081344811,
        "reg_alpha": 0.00019517224641449495,
        "reg_lambda": 7.5504986209563665,
        "min_split_gain": 0.6011150117432088,
        "scale_pos_weight": scale_pos_weight,
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
        "verbosity": -1,
        "force_col_wise": True,
        "metric": "average_precision",
    }


def train_model(
    params: dict,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_valid: pd.DataFrame,
    y_valid: pd.Series,
    verbose: bool = False,
) -> lgb.LGBMClassifier:
    model = lgb.LGBMClassifier(**params)
    model.fit(
        x_train,
        y_train,
        eval_set=[(x_valid, y_valid)],
        eval_metric="average_precision",
        callbacks=lgbm_callbacks(verbose=verbose),
    )
    return model


def printable_params(params: dict) -> dict:
    keys = {
        "learning_rate",
        "num_leaves",
        "max_depth",
        "min_child_samples",
        "subsample",
        "colsample_bytree",
        "reg_alpha",
        "reg_lambda",
        "min_split_gain",
        "scale_pos_weight",
    }
    return {key: value for key, value in params.items() if key in keys}


def select_final_model(
    candidates: dict[str, dict],
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_valid: pd.DataFrame,
    y_valid: pd.Series,
) -> dict:
    best = {
        "name": "",
        "model": None,
        "params": None,
        "threshold": 0.5,
        "validation_f1": -1.0,
    }

    print("\nFINAL CANDIDATE SELECTION")
    for name, params in candidates.items():
        model = train_model(params, x_train, y_train, x_valid, y_valid, verbose=True)
        probabilities = model.predict_proba(x_valid)[:, 1]
        threshold, validation_f1 = threshold_for_best_f1(y_valid, probabilities)
        pr_auc = average_precision_score(y_valid, probabilities)
        print(
            f"{name:<22} threshold={threshold:.6f} "
            f"val_f1={validation_f1:.4f} val_pr_auc={pr_auc:.4f}"
        )

        if validation_f1 > best["validation_f1"]:
            best.update(
                {
                    "name": name,
                    "model": model,
                    "params": params,
                    "threshold": threshold,
                    "validation_f1": validation_f1,
                }
            )
    return best


def print_split_summary(train: pd.DataFrame, valid: pd.DataFrame, test: pd.DataFrame) -> None:
    print("\nDATA SPLIT")
    for name, split in [("Train", train), ("Validation", valid), ("Test", test)]:
        print(
            f"{name:<10}: {len(split):>7,} rows | "
            f"fraud={int(split['Class'].sum()):>4,} | rate={split['Class'].mean():.4%}"
        )


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


def main() -> None:
    started = time.perf_counter()
    df = load_data()
    train_df, valid_df, test_df = split_data(df)

    print("\nLIGHTGBM FRAUD DETECTION PIPELINE")
    print(f"Rows loaded: {len(df):,}")
    print(f"Fraud rate : {df['Class'].mean():.4%}")
    print_split_summary(train_df, valid_df, test_df)

    preprocessor = Preprocessor.fit(train_df)
    x_train = preprocessor.transform(train_df)
    x_valid = preprocessor.transform(valid_df)
    x_test = preprocessor.transform(test_df)
    y_train = train_df["Class"].astype(np.int8)
    y_valid = valid_df["Class"].astype(np.int8)
    y_test = test_df["Class"].astype(np.int8)

    scale_pos_weight = class_ratio(y_train)
    x_tune, y_tune = stratified_sample(x_train, y_train)

    print(f"\nscale_pos_weight: {scale_pos_weight:.2f}")
    print(f"Optuna train rows: {len(x_tune):,}")

    selected_features = select_features(x_tune, y_tune, x_valid, y_valid, scale_pos_weight)
    x_train_selected = x_train[selected_features]
    x_tune_selected = x_tune[selected_features]
    x_valid_selected = x_valid[selected_features]
    x_test_selected = x_test[selected_features]

    print(f"\nSELECTED FEATURES ({len(selected_features)}/{x_train.shape[1]})")
    print(selected_features)

    print(f"\nOPTUNA SEARCH ({N_TRIALS} trials)")
    best_params = optimize_params(
        x_tune_selected,
        y_tune,
        x_valid_selected,
        y_valid,
        scale_pos_weight,
    )

    print("\nBEST OPTUNA HYPERPARAMETERS")
    print(printable_params(best_params))

    final = select_final_model(
        {
            "Optuna Best": best_params,
            "Regularized Candidate": regularized_candidate_params(scale_pos_weight),
        },
        x_train_selected,
        y_train,
        x_valid_selected,
        y_valid,
    )

    test_probabilities = final["model"].predict_proba(x_test_selected)[:, 1]
    test_metrics = evaluate(y_test, test_probabilities, final["threshold"])
    predictions = (test_probabilities >= final["threshold"]).astype(np.int8)

    print("\nTHRESHOLD")
    print(f"Selected final model      : {final['name']}")
    print(f"Best validation threshold : {final['threshold']:.6f}")
    print(f"Validation fraud F1       : {final['validation_f1']:.4f}")

    print("\nTEST CLASSIFICATION REPORT")
    print(classification_report(y_test, predictions, target_names=["Legit", "Fraud"], digits=4))
    print("TEST METRICS")
    print(f"Precision : {test_metrics['precision']:.4f}")
    print(f"Recall    : {test_metrics['recall']:.4f}")
    print(f"F1        : {test_metrics['f1']:.4f}")
    print(f"PR-AUC    : {test_metrics['pr_auc']:.4f}")
    print(f"ROC-AUC   : {test_metrics['roc_auc']:.4f}")
    print(f"Confusion : {test_metrics['confusion_matrix']}")

    save_outputs(
        final["model"],
        selected_features,
        printable_params(final["params"]),
        final["threshold"],
        test_metrics,
    )

    print("\nFINAL MODEL")
    print("Model saved      :", os.path.join(MODEL_DIR, "lightgbm_fraud_model.txt"))
    print("Features saved   :", os.path.join(MODEL_DIR, "selected_features.csv"))
    print("Params saved     :", os.path.join(MODEL_DIR, "best_lightgbm_params.csv"))
    print("Metrics saved    :", os.path.join(MODEL_DIR, "final_test_metrics.csv"))
    print(f"Elapsed time     : {time.perf_counter() - started:.1f}s")


if __name__ == "__main__":
    main()
