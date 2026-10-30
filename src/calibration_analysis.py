
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.base import clone
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_val_predict,
)
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (
    brier_score_loss,
    log_loss,
)
from sklearn.calibration import calibration_curve


ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = ROOT / "data" / "raw" / "attenomy_telemetry_raw.csv"
MODEL_PATH = ROOT / "models" / "best_model.joblib"
REPORT_DIR = ROOT / "reports"
FIGURE_DIR = REPORT_DIR / "figures"

RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 5
TARGET = "is_anomaly"


def main():
    print("=" * 65)
    print("TASK 2.3 - PROBABILITY CALIBRATION")
    print("=" * 65)

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

    model = joblib.load(MODEL_PATH)

    cv = StratifiedKFold(
        n_splits=CV_FOLDS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    print("\nGenerating training OOF probabilities...")

    oof_probabilities = cross_val_predict(
        clone(model),
        X_train,
        y_train,
        cv=cv,
        method="predict_proba",
        n_jobs=1,
    )[:, 1]

    print("Generating held-out test probabilities...")

    original_probabilities = model.predict_proba(X_test)[:, 1]

    print("\nFitting isotonic calibration on training OOF predictions...")

    calibrator = IsotonicRegression(
        y_min=0,
        y_max=1,
        out_of_bounds="clip",
    )

    calibrator.fit(
        oof_probabilities,
        y_train.to_numpy(),
    )

    calibrated_probabilities = calibrator.predict(
        original_probabilities
    )

    results = {
        "original": {
            "brier_score": float(
                brier_score_loss(
                    y_test,
                    original_probabilities,
                )
            ),
            "log_loss": float(
                log_loss(
                    y_test,
                    original_probabilities,
                )
            ),
        },
        "isotonic": {
            "brier_score": float(
                brier_score_loss(
                    y_test,
                    calibrated_probabilities,
                )
            ),
            "log_loss": float(
                log_loss(
                    y_test,
                    calibrated_probabilities,
                )
            ),
        },
    }

    print("\nCALIBRATION METRICS")
    print("-" * 65)

    print("Original Brier score:", round(
        results["original"]["brier_score"], 6
    ))

    print("Isotonic Brier score:", round(
        results["isotonic"]["brier_score"], 6
    ))

    print("Original Log Loss:", round(
        results["original"]["log_loss"], 6
    ))

    print("Isotonic Log Loss:", round(
        results["isotonic"]["log_loss"], 6
    ))

    original_fraction, original_mean = calibration_curve(
        y_test,
        original_probabilities,
        n_bins=10,
        strategy="quantile",
    )

    calibrated_fraction, calibrated_mean = calibration_curve(
        y_test,
        calibrated_probabilities,
        n_bins=10,
        strategy="quantile",
    )

    plt.figure(figsize=(8, 6))

    plt.plot(
        original_mean,
        original_fraction,
        marker="o",
        label="Original XGBoost",
    )

    plt.plot(
        calibrated_mean,
        calibrated_fraction,
        marker="s",
        label="Isotonic calibrated",
    )

    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        label="Perfect calibration",
    )

    plt.xlabel("Mean predicted probability")
    plt.ylabel("Observed anomaly frequency")
    plt.title("Probability Calibration Curve")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    figure_path = FIGURE_DIR / "calibration_curve.png"
    plt.savefig(figure_path, dpi=300)
    plt.close()

    probability_output = REPORT_DIR / "calibration_test_predictions.csv"

    pd.DataFrame({
        "actual_label": y_test.to_numpy(),
        "original_probability": original_probabilities,
        "isotonic_probability": calibrated_probabilities,
    }).to_csv(probability_output, index=False)

    metrics_path = REPORT_DIR / "calibration_metrics.json"

    with open(metrics_path, "w", encoding="utf-8") as file:
        json.dump(results, file, indent=2)

    print("\nSaved:")
    print(figure_path)
    print(probability_output)
    print(metrics_path)

    print("\nCalibration evaluation completed.")


if __name__ == "__main__":
    main()
