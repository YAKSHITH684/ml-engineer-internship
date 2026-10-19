"""
Attenomy ML Engineering : Core Data Utilities
Provides defensive ingestion, dataset health summaries, outlier detection,
and strict leak-free train/val/test splitting.
"""

import os
from typing import Any

import numpy as np
import pandas as pd


def load_data(path: str) -> pd.DataFrame:
    """
    Ingests CSV or Parquet tabular datasets defensively.
    Args:
        path: Path to CSV or Parquet file.
    Returns:
        pd.DataFrame containing ingested dataset.
    Raises:
        FileNotFoundError: If the specified file does not exist.
        ValueError: If file extension is unsupported or file is empty.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset path not found: {path}")

    if os.path.getsize(path) == 0:
        raise ValueError(f"Dataset file is empty (0 bytes): {path}")

    ext = os.path.splitext(path)[1].lower()
    if ext == ".csv":
        df = pd.read_csv(path)
    elif ext in [".parquet", ".pq"]:
        df = pd.read_parquet(path)
    else:
        raise ValueError(f"Unsupported file format '{ext}'. Expected .csv or .parquet")

    if df.empty:
        raise ValueError(f"Dataset in {path} contains no rows.")

    return df


def summarize_dataframe_health(df: pd.DataFrame) -> dict[str, Any]:
    """
    Computes diagnostic health metrics for a given DataFrame.
    Args:
        df: Input pandas DataFrame.
    Returns:
        Dictionary containing shape, memory footprint, null counts,
        duplicate counts, and column data types.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")

    total_cells = df.shape[0] * df.shape[1]
    null_counts = df.isnull().sum().to_dict()
    total_nulls = int(sum(null_counts.values()))
    null_pct = (total_nulls / total_cells * 100.0) if total_cells > 0 else 0.0

    dup_rows = int(df.duplicated().sum())
    mem_bytes = int(df.memory_usage(deep=True).sum())

    return {
        "row_count": int(df.shape[0]),
        "column_count": int(df.shape[1]),
        "memory_mb": round(mem_bytes / (1024 * 1024), 2),
        "total_nulls": total_nulls,
        "null_percentage": round(null_pct, 2),
        "duplicate_rows": dup_rows,
        "null_counts_per_column": null_counts,
        "column_dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()}
    }


def detect_outliers_iqr(df: pd.DataFrame, column: str, multiplier: float = 1.5) -> pd.Series:
    """
    Identifies outliers using the Interquartile Range (IQR) method.
    Args
        df: Input DataFrame.
        column: Numerical column name to audit.
        multiplier: IQR multiplier (default 1.5).
    Returns:
        Boolean pd.Series indicating whether each row is an outlier.
    """
    if column not in df.columns:
        raise KeyError(f"Column '{column}' does not exist in DataFrame.")
    series = df[column].dropna()
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - (multiplier * iqr)
    upper_bound = q3 + (multiplier * iqr)

    return (df[column] < lower_bound) | (df[column] > upper_bound)

def train_val_test_split(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    stratify_col: str | None = None,
    seed: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Partitions DataFrame into clean, leak-free train, validation, and test subsets.
    Guarantees:
        1. Non-overlapping disjoint indices across all three splits.
        2. Proportional preservation of class labels when stratify_col is specified.
        3. Deterministic repeatability through fixed pseudo-random seed.
    Args:
        df: Input DataFrame.
        train_ratio: Proportion allocated to training split.
        val_ratio: Proportion allocated to validation split.
        test_ratio: Proportion allocated to test split.
        stratify_col: Column name to use for stratified sampling (optional).
        seed: Random seed for deterministic reproducibility.
    Returns:
        Tuple of (train_df, val_df, test_df).
    """
    total = train_ratio + val_ratio + test_ratio
    if not (0.999 <= total <= 1.001):
        raise ValueError(f"Split ratios must sum to 1.0. Received sum: {total}")

    rng = np.random.default_rng(seed)
    n = len(df)

    if stratify_col is not None:
        if stratify_col not in df.columns:
            raise KeyError(f"Stratify column '{stratify_col}' not found.")

        train_indices = []
        val_indices = []
        test_indices = []


        # Group by label to preserve class distribution
        for _, group in df.groupby(stratify_col):
            group_idx = group.index.to_numpy(copy=True)
            rng.shuffle(group_idx)

            g_n = len(group_idx)
            n_train = int(round(g_n * train_ratio))
            n_val = int(round(g_n * val_ratio))

            train_indices.extend(group_idx[:n_train])
            val_indices.extend(group_idx[n_train:n_train + n_val])
            test_indices.extend(group_idx[n_train + n_val:])

        train_df = df.loc[train_indices].copy()
        val_df = df.loc[val_indices].copy()
        test_df = df.loc[test_indices].copy()
    else:
        indices = np.arange(n)
        rng.shuffle(indices)

        n_train = int(round(n * train_ratio))
        n_val = int(round(n * val_ratio))

        train_df = df.iloc[indices[:n_train]].copy()
        val_df = df.iloc[indices[n_train:n_train + n_val]].copy()
        test_df = df.iloc[indices[n_train + n_val:]].copy()

    # Verify index disjointness to ensure zero leakage
    train_set = set(train_df.index)
    val_set = set(val_df.index)
    test_set = set(test_df.index)

    assert train_set.isdisjoint(val_set), "Data leakage detected: train and val share indices!"
    assert train_set.isdisjoint(test_set), "Data leakage detected: train and test share indices!"
    assert val_set.isdisjoint(test_set), "Data leakage detected: val and test share indices!"

    return train_df, val_df, test_df
