
import json
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

from sklearn.base import clone
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_predict
from sklearn.metrics import precision_recall_curve, f1_score


ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = ROOT / "data" / "raw" / "attenomy_telemetry_raw.csv"
MODEL_PATH = ROOT / "models" / "best_model.joblib"
REPORT_DIR = ROOT / "reports"

RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 5
TARGET = "is_anomaly"


def find_best_threshold(y_true, probabilities):
    precision, recall, thresholds = precision_recall_curve(
        y_true, probabilities
    )

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
    }


def main():
    print("=" * 65)
    print("TASK 2.3 - HIGH-CONFIDENCE ERROR ANALYSIS")
    print("=" * 65)

    # Load data and reproduce the original split.
    df = pd.read_csv(DATA_PATH)
    X = df.drop(columns=[TARGET])
    y = df[TARGET].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    # Load the exact saved pipeline.
    model = joblib.load(MODEL_PATH)

    # Recalculate OOF probabilities using the exact saved model settings.
    print("\nCalculating exact-model OOF probabilities...")

    cv = StratifiedKFold(
        n_splits=CV_FOLDS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    oof_probabilities = cross_val_predict(
        clone(model),
        X_train,
        y_train,
        cv=cv,
        method="predict_proba",
        n_jobs=1,
    )[:, 1]

    threshold_info = find_best_threshold(
        y_train.to_numpy(),
        oof_probabilities,
    )

    threshold = threshold_info["threshold"]

    print("\nSelected OOF threshold:", round(threshold, 6))
    print("OOF precision:", round(threshold_info["oof_precision"], 4))
    print("OOF recall:", round(threshold_info["oof_recall"], 4))
    print("OOF F1:", round(threshold_info["oof_f1"], 4))

    # Evaluate the saved model on the untouched test split.
    print("\nPredicting held-out test data...")

    probabilities = model.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= threshold).astype(int)

    results = X_test.copy()
    results["actual_label"] = y_test.to_numpy()
    results["predicted_probability"] = probabilities
    results["predicted_label"] = predictions

    # False positives: actual normal, predicted anomaly.
    false_positives = results[
        (results["actual_label"] == 0) &
        (results["predicted_label"] == 1)
    ].copy()

    false_positives["error_type"] = "False Positive"
    false_positives["confidence"] = (
        false_positives["predicted_probability"]
    )

    # False negatives: actual anomaly, predicted normal.
    false_negatives = results[
        (results["actual_label"] == 1) &
        (results["predicted_label"] == 0)
    ].copy()

    false_negatives["error_type"] = "False Negative"
    false_negatives["confidence"] = (
        1 - false_negatives["predicted_probability"]
    )

    # Select the 10 most confident errors from each group.
    selected_fp = false_positives.sort_values(
        "confidence", ascending=False
    ).head(10)

    selected_fn = false_negatives.sort_values(
        "confidence", ascending=False
    ).head(10)

    selected = pd.concat(
        [selected_fp, selected_fn],
        axis=0,
    )

    # Leave root-cause classification for evidence-based manual review.
    selected["root_cause_category"] = ""

    output_path = REPORT_DIR / "error_analysis_20.csv"
    selected.to_csv(output_path, index_label="original_row_index")

    threshold_path = REPORT_DIR / "error_analysis_threshold.json"

    with open(threshold_path, "w", encoding="utf-8") as file:
        json.dump(
            {
                **threshold_info,
                "model": "Saved XGBoost pipeline",
                "test_threshold": threshold,
                "false_positive_count": len(false_positives),
                "false_negative_count": len(false_negatives),
                "selected_false_positives": len(selected_fp),
                "selected_false_negatives": len(selected_fn),
            },
            file,
            indent=2,
        )

    print("\nHeld-out test error counts:")
    print("False positives:", len(false_positives))
    print("False negatives:", len(false_negatives))

    print("\nSelected records:")
    print("False positives:", len(selected_fp))
    print("False negatives:", len(selected_fn))

    print("\nSelected error records:")
    print(
        selected[
            [
                "actual_label",
                "predicted_probability",
                "predicted_label",
                "error_type",
                "confidence",
            ]
        ].to_string()
    )

    print("\nSaved:")
    print(output_path)
    print(threshold_path)


if __name__ == "__main__":
    main()
