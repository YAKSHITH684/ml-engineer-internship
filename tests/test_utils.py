"""
Unit test suite verifying data utilities, defensive validation,
and leak-free partitioning mechanics.
"""


import numpy as np
import pandas as pd
import pytest

from src.utils import (
    detect_outliers_iqr,
    load_data,
    summarize_dataframe_health,
    train_val_test_split,
)


@pytest.fixture
def sample_df():
    np.random.seed(42)
    return pd.DataFrame({
        "user_id": [f"USR_{i:04d}" for i in range(100)],
        "login_failures": np.random.choice([0, 1, 2, 5], size=100),
        "session_duration_sec": np.random.exponential(scale=300, size=100),
        "device_os": np.random.choice(["Android", "iOS", "Windows", "macOS"], size=100),
        "is_fraud": np.random.choice([0, 1], p=[0.85, 0.15], size=100)
    })


def test_load_data_valid_csv(tmp_path, sample_df):
    csv_file = tmp_path / "valid.csv"
    sample_df.to_csv(csv_file, index=False)
    loaded = load_data(str(csv_file))
    assert len(loaded) == 100
    assert list(loaded.columns) == list(sample_df.columns)


def test_load_data_missing_file():
    with pytest.raises(FileNotFoundError):
        load_data("non_existent_file_path_12345.csv")


def test_load_data_empty_file(tmp_path):
    empty_file = tmp_path / "empty.csv"
    empty_file.write_text("")
    with pytest.raises(ValueError, match="empty"):
        load_data(str(empty_file))


def test_summarize_dataframe_health(sample_df):
    health = summarize_dataframe_health(sample_df)
    assert health["row_count"] == 100
    assert health["column_count"] == 5
    assert health["duplicate_rows"] == 0
    assert "memory_mb" in health
    assert "null_percentage" in health


def test_detect_outliers_iqr(sample_df):
    # Insert an obvious outlier
    sample_df.loc[0, "session_duration_sec"] = 999999.0
    outliers = detect_outliers_iqr(sample_df, "session_duration_sec")
    assert outliers.iloc[0]
    assert isinstance(outliers, pd.Series)


def test_train_val_test_split_ratios_and_no_leakage(sample_df):
    train_df, val_df, test_df = train_val_test_split(
        sample_df,
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15,
        stratify_col="is_fraud",
        seed=123
    )

    assert len(train_df) + len(val_df) + len(test_df) == 100

    # Verify strict index disjointness (zero data leakage)
    train_idx = set(train_df.index)
    val_idx = set(val_df.index)
    test_idx = set(test_df.index)

    assert train_idx.isdisjoint(val_idx)
    assert train_idx.isdisjoint(test_idx)
    assert val_idx.isdisjoint(test_idx)


def test_train_val_test_split_invalid_ratios(sample_df):
    with pytest.raises(ValueError, match="sum to 1.0"):
        train_val_test_split(
            sample_df,
            train_ratio=0.5,
            val_ratio=0.1,
            test_ratio=0.1,
        )
