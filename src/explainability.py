
import warnings
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from sklearn.model_selection import train_test_split


warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = ROOT / "data" / "raw" / "attenomy_telemetry_raw.csv"
MODEL_PATH = ROOT / "models" / "best_model.joblib"
FIGURE_DIR = ROOT / "reports" / "figures"
REPORT_DIR = ROOT / "reports"

FIGURE_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
TEST_SIZE = 0.20
SHAP_SAMPLE_SIZE = 1000
BACKGROUND_SIZE = 100


def main():
    print("=" * 65)
    print("TASK 2.3 - TREESHAP EXPLAINABILITY")
    print("=" * 65)

    # 1. Load the original dataset.
    print("\nLoading dataset...")
    df = pd.read_csv(DATA_PATH)

    X = df.drop(columns=["is_anomaly"])
    y = df["is_anomaly"].astype(int)

    # 2. Reproduce the original train-test split.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    print("Training rows:", len(X_train))
    print("Testing rows:", len(X_test))

    # 3. Load the saved pipeline without retraining.
    print("\nLoading saved model...")
    model = joblib.load(MODEL_PATH)

    feature_engineer = model.named_steps["feature_engineering"]
    preprocessor = model.named_steps["preprocessor"]
    classifier = model.named_steps["classifier"]

    # 4. Transform test data using the fitted pipeline components.
    print("\nTransforming test data...")
    engineered_test = feature_engineer.transform(X_test)
    transformed_test = preprocessor.transform(engineered_test)

    feature_names = preprocessor.get_feature_names_out()

    transformed_test = pd.DataFrame(
        transformed_test,
        columns=feature_names,
        index=X_test.index,
    )

    print("Transformed test shape:", transformed_test.shape)

    # 5. Select a reproducible sample for SHAP.
    sample_size = min(SHAP_SAMPLE_SIZE, len(transformed_test))

    X_sample = transformed_test.sample(
        n=sample_size,
        random_state=RANDOM_STATE,
    )

    background_size = min(BACKGROUND_SIZE, len(transformed_test))

    X_background = transformed_test.sample(
        n=background_size,
        random_state=RANDOM_STATE,
    )

    print("SHAP explanation sample:", X_sample.shape)
    print("SHAP background sample:", X_background.shape)

    # 6. Generate TreeSHAP explanations.
    print("\nCreating TreeSHAP explainer...")

    explainer = shap.TreeExplainer(
        classifier,
        data=X_background,
        feature_perturbation="interventional",
        model_output="raw",
    )

    print("Calculating SHAP values...")
    shap_values = explainer(X_sample)

    if shap_values.values.ndim != 2:
        raise ValueError(
            f"Unexpected SHAP dimensions: {shap_values.values.shape}"
        )

    print("SHAP values shape:", shap_values.values.shape)

    # 7. Generate global beeswarm summary.
    print("\nGenerating beeswarm plot...")

    plt.figure()
    shap.plots.beeswarm(
        shap_values,
        max_display=20,
        show=False,
    )
    plt.tight_layout()
    plt.savefig(
        FIGURE_DIR / "shap_summary.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    # 8. Generate mean absolute SHAP importance.
    print("Generating feature importance plot...")

    plt.figure()
    shap.plots.bar(
        shap_values,
        max_display=20,
        show=False,
    )
    plt.tight_layout()
    plt.savefig(
        FIGURE_DIR / "shap_importance.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    # 9. Save global feature importance as CSV.
    importance = np.abs(shap_values.values).mean(axis=0)

    importance_df = pd.DataFrame({
        "feature": feature_names,
        "mean_absolute_shap": importance,
    }).sort_values(
        "mean_absolute_shap",
        ascending=False,
    )

    importance_df.to_csv(
        REPORT_DIR / "shap_feature_importance.csv",
        index=False,
    )

    # 10. Generate three representative waterfall plots.
    print("Generating individual waterfall plots...")

    probabilities = classifier.predict_proba(X_sample)[:, 1]

    # Highest, lowest, and median predicted anomaly probability.
    selected_positions = [
        int(np.argmax(probabilities)),
        int(np.argmin(probabilities)),
        int(np.argsort(probabilities)[len(probabilities) // 2]),
    ]

    # Avoid duplicate examples.
    selected_positions = list(dict.fromkeys(selected_positions))

    for plot_number, position in enumerate(selected_positions, start=1):
        plt.figure()

        shap.plots.waterfall(
            shap_values[position],
            max_display=15,
            show=False,
        )

        plt.title(
            f"Example {plot_number} - "
            f"Predicted anomaly probability: "
            f"{probabilities[position]:.4f}"
        )

        plt.tight_layout()
        plt.savefig(
            FIGURE_DIR / f"shap_waterfall_{plot_number}.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

    # 11. Export individual SHAP values for audit.
    shap_export = pd.DataFrame(
        shap_values.values,
        columns=feature_names,
        index=X_sample.index,
    )

    shap_export.insert(
        0,
        "actual_label",
        y_test.loc[X_sample.index].values,
    )

    shap_export.insert(
        1,
        "predicted_probability",
        probabilities,
    )

    shap_export.to_csv(
        REPORT_DIR / "shap_values_sample.csv",
        index_label="original_row_index",
    )

    print("\n" + "=" * 65)
    print("TREESHAP EXPLAINABILITY COMPLETED")
    print("=" * 65)

    print("\nGenerated files:")

    for file in sorted(FIGURE_DIR.glob("shap_*.png")):
        print(file.relative_to(ROOT))

    print("reports/shap_feature_importance.csv")
    print("reports/shap_values_sample.csv")


if __name__ == "__main__":
    main()
