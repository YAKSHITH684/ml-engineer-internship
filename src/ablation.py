
"""Task 2.2: Feature ablation using leak-free cross-validation."""

from pathlib import Path

import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline

from src.features import (
    ENGINEERED_FEATURES,
    TelemetryFeatureEngineer,
)
from src.pipeline import (
    BASE_NUMERIC_FEATURES,
    FEATURED_CATEGORICAL_FEATURES,
    build_feature_preprocessor,
)


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "raw" / "attenomy_telemetry_raw.csv"
REPORT_PATH = ROOT / "reports" / "ablation_results.csv"

TARGET = "is_anomaly"

RANDOM_STATE = 42
CV_FOLDS = 5

FEATURE_GROUPS = {
    "ratios_interactions": [
        "payload_per_second",
        "retry_latency_interaction",
    ],
    "category_frequencies": [
        "channel_frequency",
        "region_frequency",
    ],
    "cyclical_time": [
        "hour_sin",
        "hour_cos",
    ],
    "account_aggregation": [
        "account_mean_latency",
    ],
    "latency_bins": [
        "latency_bin",
    ],
}


def load_data():
    """Load raw data without global imputation."""

    df = pd.read_csv(DATA_PATH)

    y = df[TARGET].astype(int)

    X = df.drop(columns=[TARGET])

    return X, y


def build_ablation_pipeline(selected_features):
    """Build a leak-free pipeline for one feature configuration."""

    return Pipeline(
        steps=[
            (
                "feature_engineering",
                TelemetryFeatureEngineer(),
            ),
            (
                "preprocessor",
                build_feature_preprocessor(
                    engineered_features=selected_features
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=2000,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def main():
    print("=" * 65)
    print("TASK 2.2 - FEATURE ABLATION STUDY")
    print("=" * 65)

    X, y = load_data()

    print("Dataset shape:", X.shape)
    print("Anomaly count:", int(y.sum()))

    cv = StratifiedKFold(
        n_splits=CV_FOLDS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    experiments = {
        "baseline_original_features": [],
        "all_engineered_features": ENGINEERED_FEATURES.copy(),
    }

    for group_name, group_features in FEATURE_GROUPS.items():
        experiments[f"without_{group_name}"] = [
            feature
            for feature in ENGINEERED_FEATURES
            if feature not in group_features
        ]

    results = []

    for experiment_name, selected_features in experiments.items():

        print("\nRunning:", experiment_name)
        print("Engineered features:", len(selected_features))

        pipeline = build_ablation_pipeline(selected_features)

        scores = cross_validate(
            pipeline,
            X,
            y,
            cv=cv,
            scoring={
                "pr_auc": "average_precision",
                "roc_auc": "roc_auc",
                "f1": "f1",
                "recall": "recall",
                "precision": "precision",
            },
            n_jobs=1,
            error_score="raise",
        )

        result = {
            "experiment": experiment_name,
            "engineered_feature_count": len(selected_features),
            "pr_auc_mean": scores["test_pr_auc"].mean(),
            "pr_auc_std": scores["test_pr_auc"].std(),
            "roc_auc_mean": scores["test_roc_auc"].mean(),
            "f1_mean": scores["test_f1"].mean(),
            "recall_mean": scores["test_recall"].mean(),
            "precision_mean": scores["test_precision"].mean(),
        }

        results.append(result)

        print(
            "PR-AUC:",
            f"{result['pr_auc_mean']:.4f}",
            "+/-",
            f"{result['pr_auc_std']:.4f}",
        )

    results_df = pd.DataFrame(results)

    results_df = results_df.sort_values(
        "pr_auc_mean",
        ascending=False,
    )

    REPORT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    results_df.to_csv(
        REPORT_PATH,
        index=False,
    )

    print("\n" + "=" * 65)
    print("ABLATION RESULTS")
    print("=" * 65)

    print(results_df.to_string(index=False))

    print("\nResults saved to:", REPORT_PATH)


if __name__ == "__main__":
    main()
