
"""Paired CV diagnostic: original Logistic Regression vs tuned XGBoost."""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from scipy.stats import t, ttest_rel, wilcoxon
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import average_precision_score

from src.baselines import FEATURES, RANDOM_STATE
from src.pipeline import build_preprocessor


DATA_PATH = Path("data/raw/attenomy_telemetry_raw.csv")
MODEL_PATH = Path("models/best_model.joblib")
OUTPUT_DIR = Path("reports")

TEST_SIZE = 0.20
N_SPLITS = 5


def main():
    print("Loading data and saved model...")

    df = pd.read_csv(DATA_PATH)
    y = df["is_anomaly"].astype(int)
    X_full = df.drop(columns=["is_anomaly"])
    X_baseline = X_full[FEATURES]

    saved_xgb = joblib.load(MODEL_PATH)

    # Reproduce the original 80/20 split using row indices.
    train_indices, _ = train_test_split(
        np.arange(len(y)),
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    X_train_full = X_full.iloc[train_indices]
    X_train_baseline = X_baseline.iloc[train_indices]
    y_train = y.iloc[train_indices]

    # Original Logistic Regression baseline.
    baseline = Pipeline([
        ("preprocessor", build_preprocessor()),
        ("classifier", LogisticRegression(
            penalty="l2",
            max_iter=1000,
            random_state=RANDOM_STATE,
        )),
    ])

    # Clone the exact saved tuned XGBoost pipeline.
    tuned_xgb = clone(saved_xgb)

    cv = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    rows = []

    print("\nRunning paired 5-fold validation...")

    for fold, (train_pos, valid_pos) in enumerate(
        cv.split(X_train_full, y_train), start=1
    ):
        y_fold_train = y_train.iloc[train_pos]
        y_fold_valid = y_train.iloc[valid_pos]

        # Logistic Regression baseline.
        baseline_fold = clone(baseline)
        baseline_fold.fit(
            X_train_baseline.iloc[train_pos],
            y_fold_train,
        )
        baseline_prob = baseline_fold.predict_proba(
            X_train_baseline.iloc[valid_pos]
        )[:, 1]

        # Tuned XGBoost pipeline.
        xgb_fold = clone(tuned_xgb)
        xgb_fold.fit(
            X_train_full.iloc[train_pos],
            y_fold_train,
        )
        xgb_prob = xgb_fold.predict_proba(
            X_train_full.iloc[valid_pos]
        )[:, 1]

        baseline_score = average_precision_score(
            y_fold_valid, baseline_prob
        )
        xgb_score = average_precision_score(
            y_fold_valid, xgb_prob
        )

        rows.append({
            "fold": fold,
            "logistic_regression_pr_auc": baseline_score,
            "tuned_xgboost_pr_auc": xgb_score,
            "difference_xgb_minus_logistic": (
                xgb_score - baseline_score
            ),
        })

        print(
            f"Fold {fold}: "
            f"Logistic={baseline_score:.6f}, "
            f"XGBoost={xgb_score:.6f}, "
            f"Difference={xgb_score - baseline_score:+.6f}"
        )

    results = pd.DataFrame(rows)
    differences = results["difference_xgb_minus_logistic"].to_numpy()

    mean_diff = differences.mean()
    std_diff = differences.std(ddof=1)

    # Descriptive paired t confidence interval.
    critical_value = t.ppf(0.975, df=N_SPLITS - 1)
    margin = critical_value * std_diff / np.sqrt(N_SPLITS)
    ci_low = mean_diff - margin
    ci_high = mean_diff + margin

    # Paired tests; interpret cautiously with only five folds.
    t_stat, t_p = ttest_rel(
        results["tuned_xgboost_pr_auc"],
        results["logistic_regression_pr_auc"],
    )

    try:
        w_stat, w_p = wilcoxon(
            results["tuned_xgboost_pr_auc"],
            results["logistic_regression_pr_auc"],
            alternative="two-sided",
        )
    except ValueError:
        w_stat, w_p = np.nan, np.nan

    summary = pd.DataFrame([{
        "baseline_mean_pr_auc": results[
            "logistic_regression_pr_auc"
        ].mean(),
        "baseline_fold_std": results[
            "logistic_regression_pr_auc"
        ].std(ddof=0),
        "xgboost_mean_pr_auc": results[
            "tuned_xgboost_pr_auc"
        ].mean(),
        "xgboost_fold_std": results[
            "tuned_xgboost_pr_auc"
        ].std(ddof=0),
        "mean_paired_difference": mean_diff,
        "difference_ci_95_low": ci_low,
        "difference_ci_95_high": ci_high,
        "paired_t_statistic": t_stat,
        "paired_t_p_value": t_p,
        "wilcoxon_statistic": w_stat,
        "wilcoxon_p_value": w_p,
        "folds": N_SPLITS,
    }])

    OUTPUT_DIR.mkdir(exist_ok=True)

    results.to_csv(
        OUTPUT_DIR / "paired_cv_fold_results.csv",
        index=False,
    )
    summary.to_csv(
        OUTPUT_DIR / "paired_cv_statistical_summary.csv",
        index=False,
    )

    print("\n========== PAIRED CV SUMMARY ==========")
    print(summary.T.to_string(header=False))

    print("\nSaved:")
    print("reports/paired_cv_fold_results.csv")
    print("reports/paired_cv_statistical_summary.csv")

    print(
        "\nIMPORTANT: Five-fold paired tests are exploratory. "
        "Tuning-selection bias remains; nested CV is needed "
        "for stronger independent significance evidence."
    )


if __name__ == "__main__":
    main()
