
import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from xgboost import XGBClassifier

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .features import ENGINEERED_FEATURES, TelemetryFeatureEngineer


# ============================================================
# CONFIGURATION
# ============================================================

warnings.filterwarnings("ignore")

ROOT_DIR = Path(__file__).resolve().parent.parent

RAW_DATA_PATH = (
    ROOT_DIR / "data" / "raw" / "attenomy_telemetry_raw.csv"
)

MODEL_DIR = ROOT_DIR / "models"
REPORT_DIR = ROOT_DIR / "reports"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 5
N_ITER = 10

THRESHOLD = 0.50
TARGET = "is_anomaly"


# ============================================================
# FEATURE DEFINITIONS
# ============================================================

BASE_NUMERIC_FEATURES = [
    "session_duration_sec",
    "payload_size_kb",
    "latency_ms",
    "retry_count",
]

NUMERIC_FEATURES = (
    BASE_NUMERIC_FEATURES + ENGINEERED_FEATURES
)

CATEGORICAL_FEATURES = [
    "interaction_type",
    "channel",
    "error_code",
    "client_tier",
    "region",
]


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    print("\nLoading dataset...")

    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {RAW_DATA_PATH}"
        )

    df = pd.read_csv(RAW_DATA_PATH)

    if TARGET not in df.columns:
        raise ValueError(
            f"Target column '{TARGET}' is missing."
        )

    df[TARGET] = pd.to_numeric(
        df[TARGET],
        errors="raise",
    ).astype(int)

    if not set(df[TARGET].unique()).issubset({0, 1}):
        raise ValueError(
            "Target must contain only binary values 0 and 1."
        )

    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    print("Dataset shape:", df.shape)
    print("Feature shape:", X.shape)

    print("\nTarget distribution:")
    print(y.value_counts().sort_index())

    print(
        "\nAnomaly percentage:",
        round(y.mean() * 100, 2),
        "%",
    )

    return X, y


# ============================================================
# PREPROCESSING
# ============================================================

def build_preprocessor():

    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median"),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="constant",
                    fill_value="MISSING",
                ),
            ),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_pipeline,
                NUMERIC_FEATURES,
            ),
            (
                "categorical",
                categorical_pipeline,
                CATEGORICAL_FEATURES,
            ),
        ],
        remainder="drop",
        sparse_threshold=0,
    )

    return preprocessor


# ============================================================
# BUILD COMPLETE PIPELINE
# ============================================================

def build_pipeline(classifier):

    pipeline = Pipeline(
        steps=[
            (
                "feature_engineering",
                TelemetryFeatureEngineer(),
            ),
            (
                "preprocessor",
                build_preprocessor(),
            ),
            (
                "classifier",
                classifier,
            ),
        ]
    )

    return pipeline


# ============================================================
# XGBOOST CONFIGURATION
# ============================================================

def get_model_configuration(y_train):

    class_counts = np.bincount(
        np.asarray(y_train, dtype=np.int32)
    )

    if len(class_counts) < 2 or class_counts[1] == 0:
        raise ValueError(
            "Training data must contain both classes."
        )

    scale_pos_weight = (
        class_counts[0] / class_counts[1]
    )

    print(
        "\nClass imbalance ratio:",
        round(scale_pos_weight, 2),
    )

    xgboost = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        scale_pos_weight=scale_pos_weight,
        random_state=RANDOM_STATE,
        n_jobs=1,
        tree_method="hist",
        verbosity=0,
    )

    xgboost_parameters = {
        "classifier__n_estimators": [
            100, 200, 300, 500,
        ],
        "classifier__max_depth": [
            3, 4, 5, 6, 8,
        ],
        "classifier__learning_rate": [
            0.01, 0.03, 0.05, 0.10,
        ],
        "classifier__subsample": [
            0.7, 0.8, 0.9, 1.0,
        ],
        "classifier__colsample_bytree": [
            0.7, 0.8, 0.9, 1.0,
        ],
        "classifier__min_child_weight": [
            1, 3, 5, 7,
        ],
        "classifier__gamma": [
            0, 0.1, 0.2, 0.5,
        ],
        "classifier__reg_lambda": [
            0.1, 1.0, 5.0, 10.0,
        ],
    }

    estimator = build_pipeline(xgboost)

    return estimator, xgboost_parameters


# ============================================================
# RANDOMIZED SEARCH CROSS-VALIDATION
# ============================================================

def tune_model(
    estimator,
    parameters,
    X_train,
    y_train,
):

    print("\n" + "=" * 60)
    print("TUNING XGBOOST")
    print("=" * 60)

    cv = StratifiedKFold(
        n_splits=CV_FOLDS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    search = RandomizedSearchCV(
        estimator=estimator,
        param_distributions=parameters,
        n_iter=N_ITER,
        scoring="average_precision",
        cv=cv,
        random_state=RANDOM_STATE,
        n_jobs=1,
        verbose=2,
        refit=True,
        error_score="raise",
        return_train_score=True,
    )

    search.fit(X_train, y_train)

    print("\nBest CV PR-AUC:")
    print(round(search.best_score_, 6))

    print("\nBest parameters:")
    print(search.best_params_)

    return search


# ============================================================
# EVALUATE MODEL
# ============================================================

def evaluate_model(
    model,
    X_test,
    y_test,
    threshold=THRESHOLD,
):

    print("\n" + "=" * 60)
    print("XGBOOST TEST EVALUATION")
    print("=" * 60)

    probabilities = model.predict_proba(X_test)[:, 1]

    predictions = (
        probabilities >= threshold
    ).astype(int)

    metrics = {
        "model": "XGBoost",
        "threshold": threshold,
        "accuracy": accuracy_score(
            y_test,
            predictions,
        ),
        "precision": precision_score(
            y_test,
            predictions,
            zero_division=0,
        ),
        "recall": recall_score(
            y_test,
            predictions,
            zero_division=0,
        ),
        "f1": f1_score(
            y_test,
            predictions,
            zero_division=0,
        ),
        "roc_auc": roc_auc_score(
            y_test,
            probabilities,
        ),
        "pr_auc": average_precision_score(
            y_test,
            probabilities,
        ),
    }

    cm = confusion_matrix(
        y_test,
        predictions,
        labels=[0, 1],
    )

    print("\nEvaluation metrics:")

    for metric, value in metrics.items():

        if isinstance(value, (int, float, np.number)):
            print(f"{metric}: {value:.6f}")
        else:
            print(f"{metric}: {value}")

    print("\nConfusion matrix:")
    print(cm)

    predictions_df = pd.DataFrame(
        {
            "y_true": np.asarray(y_test),
            "predicted_probability": probabilities,
            "predicted_label": predictions,
            "threshold": threshold,
        }
    )

    return metrics, cm, predictions_df


# ============================================================
# SAVE RESULTS
# ============================================================

def save_results(
    search,
    metrics,
    cm,
    predictions_df,
):

    # Save best model

    model_path = MODEL_DIR / "best_model.joblib"

    joblib.dump(
        search.best_estimator_,
        model_path,
    )

    print("\nBest model saved:", model_path)

    # Save cross-validation results

    cv_results = pd.DataFrame(
        search.cv_results_
    )

    cv_results.to_csv(
        REPORT_DIR / "tuning_results.csv",
        index=False,
    )

    # Save test metrics

    pd.DataFrame([metrics]).to_csv(
        REPORT_DIR / "tuned_model_test_metrics.csv",
        index=False,
    )

    # Save predictions

    predictions_df.to_csv(
        REPORT_DIR / "test_predictions.csv",
        index=False,
    )

    # Save confusion matrix

    confusion_df = pd.DataFrame(
        cm,
        index=[
            "Actual_Normal",
            "Actual_Anomaly",
        ],
        columns=[
            "Predicted_Normal",
            "Predicted_Anomaly",
        ],
    )

    confusion_df.to_csv(
        REPORT_DIR / "confusion_matrix.csv"
    )

    # Save metadata

    metadata = {
        "best_model": "XGBoost",
        "best_cv_pr_auc": float(search.best_score_),
        "test_threshold": THRESHOLD,
        "test_metrics": {
            key: float(value)
            if isinstance(value, (np.integer, np.floating))
            else value
            for key, value in metrics.items()
        },
        "best_parameters": search.best_params_,
        "random_state": RANDOM_STATE,
        "cv_folds": CV_FOLDS,
        "n_iter": N_ITER,
    }

    with open(
        REPORT_DIR / "best_model_metadata.json",
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            metadata,
            file,
            indent=4,
        )

    print("\nGenerated files:")
    print("1. models/best_model.joblib")
    print("2. reports/tuning_results.csv")
    print("3. reports/tuned_model_test_metrics.csv")
    print("4. reports/test_predictions.csv")
    print("5. reports/confusion_matrix.csv")
    print("6. reports/best_model_metadata.json")


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n" + "=" * 65)
    print("ATTENOMY TELEMETRY ANOMALY DETECTION")
    print("WEEK 2 - XGBOOST HYPERPARAMETER TUNING")
    print("=" * 65)

    # 1. Load data

    X, y = load_data()

    # 2. Train-test split

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    print("\nTrain shape:", X_train.shape)
    print("Test shape:", X_test.shape)

    print(
        "\nTraining anomaly distribution:",
        np.unique(y_train, return_counts=True),
    )

    print(
        "Testing anomaly distribution:",
        np.unique(y_test, return_counts=True),
    )

    # 3. Build XGBoost pipeline

    estimator, parameters = get_model_configuration(
        y_train
    )

    # 4. Tune XGBoost

    search = tune_model(
        estimator=estimator,
        parameters=parameters,
        X_train=X_train,
        y_train=y_train,
    )

    # 5. Evaluate on held-out test data

    metrics, cm, predictions_df = evaluate_model(
        model=search.best_estimator_,
        X_test=X_test,
        y_test=y_test,
        threshold=THRESHOLD,
    )

    # 6. Save outputs

    save_results(
        search=search,
        metrics=metrics,
        cm=cm,
        predictions_df=predictions_df,
    )

    # 7. Final summary

    print("\n" + "=" * 65)
    print("XGBOOST TUNING COMPLETED")
    print("=" * 65)

    print("Selected model: XGBoost")

    print(
        "Best CV PR-AUC:",
        round(search.best_score_, 6),
    )

    print(
        "Test PR-AUC:",
        round(metrics["pr_auc"], 6),
    )

    print(
        "Test ROC-AUC:",
        round(metrics["roc_auc"], 6),
    )

    print(
        "Test Precision:",
        round(metrics["precision"], 6),
    )

    print(
        "Test Recall:",
        round(metrics["recall"], 6),
    )

    print(
        "Test F1:",
        round(metrics["f1"], 6),
    )


if __name__ == "__main__":
    main()



