
"""Nested CV: Logistic Regression baseline vs tuned XGBoost."""

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from scipy.stats import t, ttest_rel, wilcoxon
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.metrics import average_precision_score

from src.baselines import FEATURES, RANDOM_STATE
from src.pipeline import build_preprocessor
from src.tuning import (
    build_pipeline,
    get_model_configuration,
)

warnings.filterwarnings("ignore", category=FutureWarning)

DATA_PATH = Path("data/raw/attenomy_telemetry_raw.csv")
OUTPUT_DIR = Path("reports")

TEST_SIZE = 0.20
OUTER_FOLDS = 5
INNER_FOLDS = 3
N_ITER = 10


def main():
    print("Loading dataset...")

    df = pd.read_csv(DATA_PATH)
    y = df["is_anomaly"].astype(int)
    X = df.drop(columns=["is_anomaly"])

    # Reproduce the original training split.
    train_indices, _ = train_test_split(
        np.arange(len(y)),
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    X_train = X.iloc[train_indices].reset_index(drop=True)
    y_train = y.iloc[train_indices].reset_index(drop=True)

    X_baseline = X_train[FEATURES]

    outer_cv = StratifiedKFold(
        n_splits=OUTER_FOLDS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    rows = []

    print("\nStarting nested cross-validation...")
    print("Outer folds:", OUTER_FOLDS)
    print("Inner folds:", INNER_FOLDS)
    print("Candidates per outer fold:", N_ITER)

    for fold, (train_pos, valid_pos) in enumerate(
        outer_cv.split(X_train, y_train), start=1
    ):
        print("\n" + "=" * 55)
        print("OUTER FOLD", fold)
        print("=" * 55)

        X_outer_train = X_train.iloc[train_pos]
        X_outer_valid = X_train.iloc[valid_pos]

        X_base_train = X_baseline.iloc[train_pos]
        X_base_valid = X_baseline.iloc[valid_pos]

        y_outer_train = y_train.iloc[train_pos]
        y_outer_valid = y_train.iloc[valid_pos]

        # Baseline: fit only on this outer training fold.
        baseline = Pipeline([
            ("preprocessor", build_preprocessor()),
            ("classifier", LogisticRegression(
                max_iter=1000,
                random_state=RANDOM_STATE,
            )),
        ])

        baseline.fit(X_base_train, y_outer_train)

        baseline_prob = baseline.predict_proba(
            X_base_valid
        )[:, 1]

        baseline_pr_auc = average_precision_score(
            y_outer_valid, baseline_prob
        )

        # Recalculate class imbalance using outer training data only.
        xgb_estimator, parameter_space = (
            get_model_configuration(y_outer_train)
        )

        inner_cv = StratifiedKFold(
            n_splits=INNER_FOLDS,
            shuffle=True,
            random_state=RANDOM_STATE + fold,
        )

        search = RandomizedSearchCV(
            estimator=xgb_estimator,
            param_distributions=parameter_space,
            n_iter=N_ITER,
            scoring="average_precision",
            cv=inner_cv,
            random_state=RANDOM_STATE + fold,
            n_jobs=1,
            verbose=1,
            refit=True,
            error_score="raise",
        )

        # Hyperparameter selection happens only inside outer training data.
        search.fit(X_outer_train, y_outer_train)

        # Outer validation data is used only for final fold evaluation.
        xgb_prob = search.best_estimator_.predict_proba(
            X_outer_valid
        )[:, 1]

        xgb_pr_auc = average_precision_score(
            y_outer_valid, xgb_prob
        )

        difference = xgb_pr_auc - baseline_pr_auc

        rows.append({
            "outer_fold": fold,
            "baseline_pr_auc": baseline_pr_auc,
            "nested_xgboost_pr_auc": xgb_pr_auc,
            "difference": difference,
            "inner_best_pr_auc": search.best_score_,
            "best_parameters": json.dumps(
                search.best_params_,
                sort_keys=True,
            ),
        })

        print(f"Baseline PR-AUC: {baseline_pr_auc:.6f}")
        print(f"Nested XGBoost PR-AUC: {xgb_pr_auc:.6f}")
        print(f"Difference: {difference:+.6f}")

    results = pd.DataFrame(rows)

    baseline_scores = results["baseline_pr_auc"].to_numpy()
    xgb_scores = results["nested_xgboost_pr_auc"].to_numpy()
    differences = results["difference"].to_numpy()

    mean_difference = differences.mean()
    difference_std = differences.std(ddof=1)

    # Descriptive paired t confidence interval.
    critical = t.ppf(0.975, df=OUTER_FOLDS - 1)
    margin = critical * difference_std / np.sqrt(OUTER_FOLDS)

    t_stat, t_p = ttest_rel(xgb_scores, baseline_scores)

    try:
        w_stat, w_p = wilcoxon(
            xgb_scores,
            baseline_scores,
            alternative="two-sided",
        )
    except ValueError:
        w_stat, w_p = np.nan, np.nan

    summary = pd.DataFrame([{
        "baseline_mean_pr_auc": baseline_scores.mean(),
        "baseline_fold_std": baseline_scores.std(ddof=0),
        "nested_xgboost_mean_pr_auc": xgb_scores.mean(),
        "nested_xgboost_fold_std": xgb_scores.std(ddof=0),
        "mean_paired_difference": mean_difference,
        "difference_ci_95_low": mean_difference - margin,
        "difference_ci_95_high": mean_difference + margin,
        "paired_t_statistic": t_stat,
        "paired_t_p_value": t_p,
        "wilcoxon_statistic": w_stat,
        "wilcoxon_p_value": w_p,
        "outer_folds": OUTER_FOLDS,
        "inner_folds": INNER_FOLDS,
        "iterations_per_outer_fold": N_ITER,
        "total_inner_candidate_fits": (
            OUTER_FOLDS * INNER_FOLDS * N_ITER
        ),
    }])

    OUTPUT_DIR.mkdir(exist_ok=True)

    results.to_csv(
        OUTPUT_DIR / "nested_cv_fold_results.csv",
        index=False,
    )

    summary.to_csv(
        OUTPUT_DIR / "nested_cv_summary.csv",
        index=False,
    )

    print("\n" + "=" * 55)
    print("NESTED CV SUMMARY")
    print("=" * 55)
    print(summary.T.to_string(header=False))

    print("\nSaved:")
    print("reports/nested_cv_fold_results.csv")
    print("reports/nested_cv_summary.csv")

    print(
        "\nInterpretation: nested CV reduces tuning-selection bias. "
        "Five outer folds are still a small sample; paired test "
        "p-values are exploratory, not definitive proof."
    )


if __name__ == "__main__":
    main()
