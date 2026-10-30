


# IMPORTANT: Import LightGBM before Pandas.
# This import order avoids the crash observed in this environment.
from lightgbm import LGBMClassifier

import os
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from src.features import TelemetryFeatureEngineer
from src.pipeline import build_feature_preprocessor


# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

RANDOM_STATE = 42
N_SPLITS = 5

DATA_PATH = "data/raw/attenomy_telemetry_raw.csv"
OUTPUT_PATH = "reports/model_comparison.csv"

os.makedirs("reports", exist_ok=True)


# --------------------------------------------------
# 2. Load dataset
# --------------------------------------------------

print("\nLoading dataset...")

df = pd.read_csv(DATA_PATH)

y = df["is_anomaly"].to_numpy(dtype=np.int32, copy=True)
X = df.drop(columns=["is_anomaly"])

print("Dataset shape:", X.shape)
print("Class distribution:", np.unique(y, return_counts=True))


# --------------------------------------------------
# 3. Feature engineering and preprocessing
# --------------------------------------------------

def build_model_pipeline(classifier):
    return Pipeline([
        ("feature_engineering", TelemetryFeatureEngineer()),
        ("preprocessing", build_feature_preprocessor()),
        ("classifier", classifier)
    ])


# --------------------------------------------------
# 4. Define models
# --------------------------------------------------

models = {
    "XGBoost": XGBClassifier(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.03,
        subsample=0.9,
        colsample_bytree=0.8,
        min_child_weight=3,
        gamma=0.5,
        reg_lambda=5,
        random_state=RANDOM_STATE,
        eval_metric="logloss",
        n_jobs=1,
        tree_method="hist"
    ),

    "LightGBM": LGBMClassifier(
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
        verbosity=-1
    )
}


# --------------------------------------------------
# 5. Cross-validation configuration
# --------------------------------------------------

cv = StratifiedKFold(
    n_splits=N_SPLITS,
    shuffle=True,
    random_state=RANDOM_STATE
)

scoring = {
    "pr_auc": "average_precision",
    "roc_auc": "roc_auc",
    "f1": "f1",
    "recall": "recall",
    "precision": "precision"
}


# --------------------------------------------------
# 6. Evaluate models
# --------------------------------------------------

results = []

for model_name, classifier in models.items():

    print("\n" + "=" * 55)
    print("Training:", model_name)
    print("=" * 55)

    pipeline = build_model_pipeline(classifier)

    scores = cross_validate(
        estimator=pipeline,
        X=X,
        y=y,
        cv=cv,
        scoring=scoring,
        n_jobs=1,
        error_score="raise",
        return_train_score=False
    )

    result = {
        "Model": model_name
    }

    for metric in scoring:
        values = scores[f"test_{metric}"]

        result[f"{metric}_mean"] = values.mean()
        result[f"{metric}_std"] = values.std()

        print(
            f"{metric.upper()}: "
            f"{values.mean():.4f} "
            f"+/- {values.std():.4f}"
        )

    results.append(result)


# --------------------------------------------------
# 7. Save comparison results
# --------------------------------------------------

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="pr_auc_mean",
    ascending=False
)

results_df.to_csv(OUTPUT_PATH, index=False)

print("\n" + "=" * 55)
print("MODEL COMPARISON COMPLETE")
print("=" * 55)

print(results_df.to_string(index=False))

print("\nResults saved to:", OUTPUT_PATH)
