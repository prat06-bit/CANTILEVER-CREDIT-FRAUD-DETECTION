import os
import time
import warnings

import lightgbm as lgb
import numpy as np
import optuna

from features import Preprocessor, select_features
from utils import (
    DATA_PATH,
    MODEL_DIR,
    RANDOM_STATE,
    class_ratio,
    evaluate,
    lgbm_callbacks,
    load_data,
    print_split_summary,
    print_test_report,
    save_outputs,
    split_data,
    stratified_sample,
    threshold_for_best_f1,
)


warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

N_TRIALS = int(os.getenv("OPTUNA_TRIALS", "30"))
OPTUNA_MAX_TRAIN_ROWS = int(os.getenv("OPTUNA_MAX_TRAIN_ROWS", "70000"))
EARLY_STOPPING_ROUNDS = 50
MAX_BOOST_ROUNDS = 900


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


def optuna_objective(
    trial: optuna.Trial,
    x_train,
    y_train,
    x_valid,
    y_valid,
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
        callbacks=lgbm_callbacks(EARLY_STOPPING_ROUNDS),
    )
    probabilities = model.predict_proba(x_valid)[:, 1]
    _, best_f1 = threshold_for_best_f1(y_valid, probabilities)
    return best_f1


def optimize_params(x_train, y_train, x_valid, y_valid, scale_pos_weight: float) -> dict:
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


def train_model(params: dict, x_train, y_train, x_valid, y_valid, verbose: bool = False):
    model = lgb.LGBMClassifier(**params)
    model.fit(
        x_train,
        y_train,
        eval_set=[(x_valid, y_valid)],
        eval_metric="average_precision",
        callbacks=lgbm_callbacks(EARLY_STOPPING_ROUNDS, verbose=verbose),
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


def select_final_model(candidates: dict, x_train, y_train, x_valid, y_valid):
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
        print(f"{name:<18} threshold={threshold:.6f} val_f1={validation_f1:.4f}")

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


def main() -> None:
    started = time.perf_counter()
    df = load_data(DATA_PATH)
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
    x_tune, y_tune = stratified_sample(x_train, y_train, OPTUNA_MAX_TRAIN_ROWS)
    selector_params = base_params(scale_pos_weight)

    print(f"\nscale_pos_weight: {scale_pos_weight:.2f}")
    print(f"Optuna train rows: {len(x_tune):,}")

    selected_features = select_features(
        x_tune,
        y_tune,
        x_valid,
        y_valid,
        selector_params,
        lgbm_callbacks(EARLY_STOPPING_ROUNDS),
    )
    x_train_selected = x_train[selected_features]
    x_tune_selected = x_tune[selected_features]
    x_valid_selected = x_valid[selected_features]
    x_test_selected = x_test[selected_features]

    print(f"\nSELECTED FEATURES ({len(selected_features)}/{x_train.shape[1]})")
    print(selected_features)

    print(f"\nOPTUNA SEARCH ({N_TRIALS} trials)")
    optuna_params = optimize_params(
        x_tune_selected,
        y_tune,
        x_valid_selected,
        y_valid,
        scale_pos_weight,
    )

    print("\nBEST OPTUNA HYPERPARAMETERS")
    print(printable_params(optuna_params))

    final = select_final_model(
        {
            "Optuna Best": optuna_params,
            "Regularized Candidate": regularized_candidate_params(scale_pos_weight),
        },
        x_train_selected,
        y_train,
        x_valid_selected,
        y_valid,
    )

    test_probabilities = final["model"].predict_proba(x_test_selected)[:, 1]
    test_metrics = evaluate(y_test, test_probabilities, final["threshold"])

    print("\nTHRESHOLD")
    print(f"Selected final model     : {final['name']}")
    print(f"Best validation threshold: {final['threshold']:.6f}")
    print(f"Validation fraud F1      : {final['validation_f1']:.4f}")

    print_test_report(y_test, test_probabilities, final["threshold"])
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
