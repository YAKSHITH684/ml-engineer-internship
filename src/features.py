
import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin


ENGINEERED_FEATURES = [
    "payload_per_second",
    "retry_latency_interaction",
    "channel_frequency",
    "region_frequency",
    "hour_sin",
    "hour_cos",
    "account_mean_latency",
    "latency_bin",
]


class TelemetryFeatureEngineer(BaseEstimator, TransformerMixin):

    REQUIRED_COLUMNS = [
        "timestamp",
        "account_id",
        "session_duration_sec",
        "payload_size_kb",
        "latency_ms",
        "retry_count",
        "channel",
        "region",
    ]

    def _check_columns(self, X):
        missing = [
            column
            for column in self.REQUIRED_COLUMNS
            if column not in X.columns
        ]

        if missing:
            raise ValueError(
                f"Missing required columns: {missing}"
            )

    def fit(self, X, y=None):
        self._check_columns(X)

        self.frequency_maps_ = {}

        # Learn channel and region frequencies.
        for column in ["channel", "region"]:
            self.frequency_maps_[column] = (
                X[column]
                .fillna("MISSING")
                .value_counts(normalize=True)
                .to_dict()
            )

        latency = pd.to_numeric(
            X["latency_ms"],
            errors="coerce"
        )

        # Calculate account-level latency statistics.
        account_data = pd.DataFrame({
            "account_id": X["account_id"],
            "latency": latency,
        })

        grouped = account_data.groupby(
            "account_id",
            dropna=False
        )["latency"]

        self.account_latency_sum_ = grouped.sum().to_dict()

        self.account_latency_count_ = grouped.count().to_dict()

        self.account_latency_map_ = grouped.mean().to_dict()

        # Global median for missing or unavailable account averages.
        self.global_latency_median_ = latency.median()

        if pd.isna(self.global_latency_median_):
            self.global_latency_median_ = 0.0

        return self

    def transform(self, X):
        self._check_columns(X)

        result = X.copy()

        # Convert numerical columns safely.
        duration = pd.to_numeric(
            result["session_duration_sec"],
            errors="coerce"
        )

        payload = pd.to_numeric(
            result["payload_size_kb"],
            errors="coerce"
        )

        latency = pd.to_numeric(
            result["latency_ms"],
            errors="coerce"
        )

        retries = pd.to_numeric(
            result["retry_count"],
            errors="coerce"
        )

        # Handle invalid negative session durations.
        duration = duration.mask(duration < 0)

        # 1. Payload transferred per second.
        result["payload_per_second"] = (
            payload / duration.replace(0, np.nan)
        )

        # 2. Interaction between retries and latency.
        result["retry_latency_interaction"] = (
            retries * latency
        )

        # 3. Channel frequency encoding.
        result["channel_frequency"] = (
            result["channel"]
            .fillna("MISSING")
            .map(self.frequency_maps_["channel"])
            .fillna(0.0)
        )

        # 4. Region frequency encoding.
        result["region_frequency"] = (
            result["region"]
            .fillna("MISSING")
            .map(self.frequency_maps_["region"])
            .fillna(0.0)
        )

        # 5 and 6. Cyclical hour encoding.
        timestamp = pd.to_datetime(
            result["timestamp"],
            errors="coerce"
        )

        hour = timestamp.dt.hour

        result["hour_sin"] = np.sin(
            2 * np.pi * hour / 24
        )

        result["hour_cos"] = np.cos(
            2 * np.pi * hour / 24
        )

        # 7. Account-level average latency.
        result["account_mean_latency"] = (
            result["account_id"]
            .map(self.account_latency_map_)
            .fillna(self.global_latency_median_)
        )

        # 8. Latency binning.
        result["latency_bin"] = pd.cut(
            latency,
            bins=[
                -np.inf,
                50,
                100,
                200,
                np.inf
            ],
            labels=False,
            right=True
        ).astype("float64")

        return result

    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X, y)

        # Transform training data.
        result = self.transform(X)

        latency = pd.to_numeric(
            X["latency_ms"],
            errors="coerce"
        )

        # Leave-one-out account averages.
        account_sums = X["account_id"].map(
            self.account_latency_sum_
        )

        account_counts = X["account_id"].map(
            self.account_latency_count_
        )

        remaining_counts = (
            account_counts -
            latency.notna().astype(int)
        )

        remaining_sums = (
            account_sums -
            latency.fillna(0)
        )

        loo_means = (
            remaining_sums /
            remaining_counts.replace(0, np.nan)
        )

        result["account_mean_latency"] = (
            loo_means.fillna(
                self.global_latency_median_
            )
        )

        return result

