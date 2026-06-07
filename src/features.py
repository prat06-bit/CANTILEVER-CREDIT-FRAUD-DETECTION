from dataclasses import dataclass

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler


INTERACTION_PAIRS = [
    ("V14", "V4"),
    ("V14", "V12"),
    ("V12", "V10"),
    ("V10", "V11"),
    ("V17", "V3"),
    ("V14", "V17"),
    ("V4", "V11"),
]

MIN_SELECTED_FEATURES = 28
SELECTED_FEATURE_RATIO = 0.90


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
    columns = front + [column for column in x.columns if column not in front]
    return x[columns].astype(np.float32)


def select_features(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_valid: pd.DataFrame,
    y_valid: pd.Series,
    model_params: dict,
    callbacks: list,
) -> list[str]:
    selector = lgb.LGBMClassifier(**model_params)
    selector.fit(
        x_train,
        y_train,
        eval_set=[(x_valid, y_valid)],
        eval_metric="average_precision",
        callbacks=callbacks,
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
