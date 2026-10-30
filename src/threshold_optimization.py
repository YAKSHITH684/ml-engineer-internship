
# IMPORTANT: Import LightGBM before NumPy, Pandas, or Scikit-learn.
from lightgbm import LGBMClassifier

import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from xgboost import XGBClassifier

from sklearn.base import clone
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    accuracy_score,
)
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_val_predict,
)
from sklearn.pipeline import Pipeline

from src.features import TelemetryFeatureEngineer
from src.pipeline import build_feature_preprocessor


warnings.filterwarnings("ignore")

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT_DIR / "data" / "raw" / "attenomy_telemetry_raw.csv"
REPORT_DIR = ROOT_DIR / "reports"

REPORT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 5
TARGET = "is_anomaly"


def load_data():
    print("\nLoading raw dataset...")

    df = pd.read_csv(DATA_PATH)

    X = df.drop(columns=[TARGET])
    y = df[TARGET].astype(int)

    print("Dataset shape:", df.shape)
    print("Training target distribution:")
    print(y.value_counts().sort_index())

    return X, y


def build_xgboost():
    return XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        scale_pos_weight=24021 / 979,
        n_estimators=100,
        max_depth=3,
        learning_rate=0.03,
        subsample=0.9,
        colsample_bytree=0.8,
        min_child_weight=3,
        gamma=0.5,
        reg_lambda=5,
        random_state=RANDOM_STATE,
        n_jobs=1,
        tree_method="hist",
        verbosity=0,
    )


def build_lightgbm():
    return LGBMClassifier(
        objective="binary",
        n_estimators=100,
        max_depth=3,
        learning_rate=0.03,
        subsample=0.9,
        colsample_bytree=0.8,
        reg_lambda=5,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=1,
        force_col_wise=True,
        verbosity=-1,
    )


def build_pipeline(classifier):
    return Pipeline(
        steps=[
            ("feature_engineering", TelemetryFeatureEngineer()),
            ("preprocessor", build_feature_preprocessor()),
            ("classifier", classifier),
        ]
    )


def find_best_threshold(y_true, probabilities):
    precision, recall, thresholds = precision_recall_curve(
        y_true, probabilities
    )

    # The final precision/recall point has no corresponding threshold.
    precision = precision[:-1]
    recall = recall[:-1]

    f1_values = (
        2 * precision * recall /
        (precision + recall + 1e-12)
    )

    best_index = int(np.argmax(f1_values))

    return {
        "threshold": float(thresholds[best_index]),
        "oof_precision": float(precision[best_index]),
        "oof_recall": float(recall[best_index]),
        "oof_f1": float(f1_values[best_index]),
        "oof_pr_auc": float(
            average_precision_score(y_true, probabilities)
        ),
    }


def evaluate_test(model, X_test, y_test, threshold, model_name):
    probabilities = model.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= threshold).astype(int)

    return {
        "model": model_name,
        "threshold": threshold,
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(
            y_test, predictions, zero_division=0
        ),
        "recall": recall_score(
            y_test, predictions, zero_division=0
        ),
        "f1": f1_score(
            y_test, predictions, zero_division=0
        ),
        "roc_auc": roc_auc_score(y_test, probabilities),
        "pr_auc": average_precision_score(y_test, probabilities),
    }


def main():
    print("\n" + "=" * 65)
    print("WEEK 2 - CLASSIFICATION THRESHOLD OPTIMIZATION")
    print("=" * 65)

    X, y = load_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    print("\nTraining rows:", len(X_train))
    print("Testing rows:", len(X_test))

    cv = StratifiedKFold(
        n_splits=CV_FOLDS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    models = {
        "XGBoost": build_pipeline(build_xgboost()),
        "LightGBM": build_pipeline(build_lightgbm()),
    }

    threshold_results = []
    test_results = []

    for model_name, pipeline in models.items():
        print("\n" + "=" * 60)
        print("Processing:", model_name)
        print("=" * 60)

        print("Generating out-of-fold training probabilities...")

        oof_probabilities = cross_val_predict(
            clone(pipeline),
            X_train,
            y_train,
            cv=cv,
            method="predict_proba",
            n_jobs=1,
        )[:, 1]

        best = find_best_threshold(
            y_train.to_numpy(),
            oof_probabilities,
        )

        print("\nSelected threshold:", round(best["threshold"], 6))
        print("OOF F1:", round(best["oof_f1"], 4))
        print("OOF precision:", round(best["oof_precision"], 4))
        print("OOF recall:", round(best["oof_recall"], 4))
        print("OOF PR-AUC:", round(best["oof_pr_auc"], 4))

        threshold_results.append({
            "model": model_name,
            **best,
        })

        print("\nFitting model on complete training data...")

        fitted_model = clone(pipeline)
        fitted_model.fit(X_train, y_train)

        metrics = evaluate_test(
            fitted_model,
            X_test,
            y_test,
            best["threshold"],
            model_name,
        )

        test_results.append(metrics)

        print("\nUntouched test results:")
        for key, value in metrics.items():
            if isinstance(value, (int, float, np.number)):
                print(f"{key}: {value:.6f}")
            else:
                print(f"{key}: {value}")

    threshold_path = REPORT_DIR / "threshold_selection_oof.csv"
    test_path = REPORT_DIR / "threshold_optimization_test.csv"

    pd.DataFrame(threshold_results).to_csv(
        threshold_path, index=False
    )

    pd.DataFrame(test_results).to_csv(
        test_path, index=False
    )

    print("\n" + "=" * 65)
    print("THRESHOLD OPTIMIZATION COMPLETED")
    print("=" * 65)

    print("\nThreshold selection results:")
    print(pd.DataFrame(threshold_results).to_string(index=False))

    print("\nTest results:")
    print(pd.DataFrame(test_results).to_string(index=False))

    print("\nSaved reports:")
    print(threshold_path)
    print(test_path)

    print("\nExisting best_model.joblib was not modified.")


if __name__ == "__main__":
    main()
