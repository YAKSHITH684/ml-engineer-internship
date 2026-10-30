
"""Task 2.1: Baseline models with leak-free 5-fold cross-validation."""

from pathlib import Path

import pandas as pd

from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from src.pipeline import build_preprocessor


RANDOM_STATE = 42
DATA_PATH = Path("data/raw/attenomy_telemetry_raw.csv")
REPORT_DIR = Path("reports")

TARGET = "is_anomaly"

FEATURES = [
    "session_duration_sec",
    "payload_size_kb",
    "latency_ms",
    "retry_count",
    "interaction_type",
    "channel",
    "error_code",
    "client_tier",
    "region",
]


def main():
    # 1. Load raw data. Imputation happens inside the pipeline.
    df = pd.read_csv(DATA_PATH)

    X = df[FEATURES]
    y = df[TARGET].astype(int)

    print("Dataset shape:", df.shape)
    print("\nTarget distribution:")
    print(y.value_counts())
    print("\nAnomaly percentage:", round(y.mean() * 100, 2), "%")

    # 2. Keep the test set separate from cross-validation.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    print("\nTraining rows:", len(X_train))
    print("Test rows:", len(X_test))

    # 3. Define baseline models.
    models = {
        "Dummy": DummyClassifier(strategy="most_frequent"),
        "Logistic Regression": LogisticRegression(
            penalty="l2",
            max_iter=1000,
            random_state=RANDOM_STATE,
        ),
        "KNN": KNeighborsClassifier(n_neighbors=7),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=6,
            random_state=RANDOM_STATE,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }

    # 4. Evaluate using stratified 5-fold cross-validation.
    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    scoring = {
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
        "roc_auc": "roc_auc",
        "pr_auc": "average_precision",
    }

    results = []

    for name, model in models.items():
        print(f"\nEvaluating: {name}")

        pipeline = Pipeline(
            steps=[
                ("preprocessor", build_preprocessor()),
                ("classifier", model),
            ]
        )

        scores = cross_validate(
            pipeline,
            X_train,
            y_train,
            cv=cv,
            scoring=scoring,
            return_train_score=False,
            error_score="raise",
            n_jobs=1,
        )

        row = {"model": name}

        for metric in scoring:
            values = scores[f"test_{metric}"]
            row[f"{metric}_mean"] = values.mean()
            row[f"{metric}_std"] = values.std()

        results.append(row)

        print(
            f"PR-AUC: {row['pr_auc_mean']:.4f} "
            f"+/- {row['pr_auc_std']:.4f}"
        )
        print(f"ROC-AUC: {row['roc_auc_mean']:.4f}")

    # 5. Save cross-validation results.
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(
        by="pr_auc_mean",
        ascending=False,
    )

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    results_path = REPORT_DIR / "baseline_cv_results.csv"
    results_df.to_csv(results_path, index=False)

    print("\n========== CROSS-VALIDATION RESULTS ==========")
    print(results_df.round(4).to_string(index=False))
    print("\nSaved results to:", results_path)

    # 6. Evaluate the best model on the test set only once.
    best_name = results_df.iloc[0]["model"]
    best_model = models[best_name]

    final_pipeline = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("classifier", best_model),
        ]
    )

    final_pipeline.fit(X_train, y_train)
    predictions = final_pipeline.predict(X_test)

    print("\n========== FINAL TEST RESULTS ==========")
    print("Selected model:", best_name)
    print("\nConfusion matrix:")
    print(confusion_matrix(y_test, predictions))
    print("\nClassification report:")
    print(
        classification_report(
            y_test,
            predictions,
            zero_division=0,
        )
    )


if __name__ == "__main__":
    main()
