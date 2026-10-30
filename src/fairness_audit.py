from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
)


ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = ROOT / "data" / "raw" / "attenomy_telemetry_raw.csv"
MODEL_PATH = ROOT / "models" / "best_model.joblib"
OUTPUT_PATH = ROOT / "reports" / "subgroup_fairness_audit.csv"

RANDOM_STATE = 42
TEST_SIZE = 0.20
THRESHOLD = 0.788394

TARGET = "is_anomaly"

GROUP_COLUMNS = [
    "channel",
    "region",
    "client_tier",
    "interaction_type",
]


def calculate_metrics(group):
    y_true = group["actual"].to_numpy()
    y_pred = group["predicted"].to_numpy()

    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())

    precision = precision_score(
        y_true, y_pred, zero_division=0
    )

    recall = recall_score(
        y_true, y_pred, zero_division=0
    )

    f1 = f1_score(
        y_true, y_pred, zero_division=0
    )

    fpr = fp / (fp + tn) if (fp + tn) else np.nan
    fnr = fn / (fn + tp) if (fn + tp) else np.nan

    pr_auc = (
        average_precision_score(
            y_true, group["probability"].to_numpy()
        )
        if len(np.unique(y_true)) == 2
        else np.nan
    )

    return {
        "records": len(group),
        "actual_anomalies": int(y_true.sum()),
        "anomaly_rate": float(y_true.mean()),
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "true_negatives": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "false_positive_rate": fpr,
        "false_negative_rate": fnr,
        "pr_auc": pr_auc,
    }


def main():
    print("=" * 65)
    print("TASK 2.3 - SUBGROUP PERFORMANCE AUDIT")
    print("=" * 65)

    df = pd.read_csv(DATA_PATH)

    X = df.drop(columns=[TARGET])
    y = df[TARGET].astype(int)

    _, X_test, _, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    model = joblib.load(MODEL_PATH)

    probabilities = model.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= THRESHOLD).astype(int)

    test_data = X_test.copy()

    test_data["actual"] = y_test.to_numpy()
    test_data["predicted"] = predictions
    test_data["probability"] = probabilities

    all_results = []

    for column in GROUP_COLUMNS:
        print(f"\nAnalysing: {column}")

        test_data[column] = (
            test_data[column]
            .astype("string")
            .str.strip()
            .str.lower()
            .fillna("missing")
        )

        for group_name, group in test_data.groupby(
            column, dropna=False
        ):
            metrics = calculate_metrics(group)

            metrics["group_column"] = column
            metrics["group"] = str(group_name)

            metrics["sufficient_sample"] = (
                metrics["records"] >= 50
                and metrics["actual_anomalies"] >= 10
            )

            all_results.append(metrics)

            print(
                f"  {group_name}: "
                f"records={metrics['records']}, "
                f"recall={metrics['recall']:.3f}, "
                f"FPR={metrics['false_positive_rate']:.3f}"
            )

    results = pd.DataFrame(all_results)

    results = results[
        [
            "group_column",
            "group",
            "records",
            "actual_anomalies",
            "anomaly_rate",
            "precision",
            "recall",
            "f1",
            "false_positive_rate",
            "false_negative_rate",
            "pr_auc",
            "true_positives",
            "false_positives",
            "false_negatives",
            "true_negatives",
            "sufficient_sample",
        ]
    ]

    results.to_csv(OUTPUT_PATH, index=False)

    print("\n" + "=" * 65)
    print("SUBGROUP AUDIT RESULTS")
    print("=" * 65)

    print(results.round(4).to_string(index=False))

    print("\nSaved report:")
    print(OUTPUT_PATH)

    print(
        "\nInterpretation note: Groups with fewer than "
        "50 records or fewer than 10 actual anomalies "
        "should be treated cautiously."
    )


if __name__ == "__main__":
    main()