
import warnings
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.metrics import (
    PrecisionRecallDisplay,
    RocCurveDisplay,
    ConfusionMatrixDisplay,
    precision_recall_curve,
    roc_curve,
    average_precision_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split


warnings.filterwarnings("ignore")

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT_DIR / "data" / "raw" / "attenomy_telemetry_raw.csv"
MODEL_PATH = ROOT_DIR / "models" / "best_model.joblib"
REPORT_DIR = ROOT_DIR / "reports"
FIGURE_DIR = REPORT_DIR / "figures"

FIGURE_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
TEST_SIZE = 0.20
TARGET = "is_anomaly"


def main():
    print("\nLoading dataset and saved XGBoost model...")

    df = pd.read_csv(DATA_PATH)
    X = df.drop(columns=[TARGET])
    y = df[TARGET].astype(int)

    # Reproduce the original tuning.py test split.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    model = joblib.load(MODEL_PATH)

    print("Model loaded successfully.")
    print("Test rows:", len(X_test))

    probabilities = model.predict_proba(X_test)[:, 1]

    # Use the previously selected XGBoost threshold.
    threshold_df = pd.read_csv(
        REPORT_DIR / "threshold_selection_oof.csv"
    )

    xgb_threshold = float(
        threshold_df.loc[
            threshold_df["model"] == "XGBoost",
            "threshold",
        ].iloc[0]
    )

    predictions = (
        probabilities >= xgb_threshold
    ).astype(int)

    # 1. Precision-Recall curve
    precision, recall, _ = precision_recall_curve(
        y_test, probabilities
    )

    plt.figure(figsize=(8, 6))
    plt.plot(
        recall,
        precision,
        label=f"XGBoost (AP={average_precision_score(y_test, probabilities):.3f})",
    )
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(
        FIGURE_DIR / "precision_recall_curve.png",
        dpi=300,
    )
    plt.close()

    # 2. ROC curve
    fpr, tpr, _ = roc_curve(y_test, probabilities)

    plt.figure(figsize=(8, 6))
    plt.plot(
        fpr,
        tpr,
        label=f"XGBoost (AUC={roc_auc_score(y_test, probabilities):.3f})",
    )
    plt.plot([0, 1], [0, 1], linestyle="--", label="Random classifier")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(
        FIGURE_DIR / "roc_curve.png",
        dpi=300,
    )
    plt.close()

    # 3. Confusion matrix at optimized threshold
    plt.figure(figsize=(7, 6))
    ConfusionMatrixDisplay.from_predictions(
        y_test,
        predictions,
        display_labels=["Normal", "Anomaly"],
        cmap="Blues",
        values_format="d",
    )
    plt.title(
        f"XGBoost Confusion Matrix (Threshold={xgb_threshold:.3f})"
    )
    plt.tight_layout()
    plt.savefig(
        FIGURE_DIR / "confusion_matrix_xgboost.png",
        dpi=300,
    )
    plt.close()

    # 4. Threshold comparison chart
    results = pd.read_csv(
        REPORT_DIR / "threshold_optimization_test.csv"
    )

    metrics = ["precision", "recall", "f1"]
    models = results["model"].tolist()

    x = np.arange(len(metrics))
    width = 0.35

    plt.figure(figsize=(9, 6))

    for index, model_name in enumerate(models):
        row = results[
            results["model"] == model_name
        ].iloc[0]

        values = [row[metric] for metric in metrics]

        plt.bar(
            x + (index - 0.5) * width,
            values,
            width,
            label=model_name,
        )

    plt.xticks(x, ["Precision", "Recall", "F1-score"])
    plt.ylabel("Score")
    plt.title("XGBoost vs LightGBM - Optimized Threshold Test Metrics")
    plt.ylim(0, 1)
    plt.legend()
    plt.grid(axis="y")
    plt.tight_layout()
    plt.savefig(
        FIGURE_DIR / "model_metric_comparison.png",
        dpi=300,
    )
    plt.close()

    print("\nVISUALIZATION COMPLETED")
    print("\nGenerated charts:")

    for file in FIGURE_DIR.glob("*.png"):
        print(file.name)

    print("\nAll charts saved in reports/figures/")


if __name__ == "__main__":
    main()
