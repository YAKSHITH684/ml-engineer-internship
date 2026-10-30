
"""Leak-free preprocessing pipeline for telemetry anomaly detection."""

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


NUMERIC_FEATURES = [
    "session_duration_sec",
    "payload_size_kb",
    "latency_ms",
    "retry_count",
]

CATEGORICAL_FEATURES = [
    "interaction_type",
    "channel",
    "error_code",
    "client_tier",
    "region",
]


def build_preprocessor():
    """Create preprocessing steps fitted only on training data."""

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="constant",
                    fill_value="MISSING",
                ),
            ),
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore"),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )

    return preprocessor


# ============================================================
# TASK 2.2: FEATURE-ENGINEERED PREPROCESSING
# ============================================================

from src.features import ENGINEERED_FEATURES


ENGINEERED_NUMERIC_FEATURES = list(ENGINEERED_FEATURES)

BASE_NUMERIC_FEATURES = [
    "session_duration_sec",
    "payload_size_kb",
    "latency_ms",
    "retry_count",
]

FEATURED_CATEGORICAL_FEATURES = [
    "interaction_type",
    "channel",
    "error_code",
    "client_tier",
    "region",
]


def build_feature_preprocessor(engineered_features=None):
    """Build preprocessing using selected engineered features."""

    if engineered_features is None:
        engineered_features = ENGINEERED_NUMERIC_FEATURES

    invalid_features = (
        set(engineered_features) - set(ENGINEERED_NUMERIC_FEATURES)
    )

    if invalid_features:
        raise ValueError(
            f"Unknown engineered features: {sorted(invalid_features)}"
        )

    numeric_features = (
        BASE_NUMERIC_FEATURES + list(engineered_features)
    )

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="constant",
                    fill_value="MISSING",
                ),
            ),
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore"),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_features),
            (
                "categorical",
                categorical_pipeline,
                FEATURED_CATEGORICAL_FEATURES,
            ),
        ],
        remainder="drop",
    )
