"""
Attenomy ML Engineering : Data Cleaning Pipeline CLI
Executes reproducible, defensive cleaning on raw telemetry data
and exports normalized data to Snappy-compressed Parquet.
"""

import argparse
import os

import pandas as pd

from src.utils import load_data, summarize_dataframe_health


def clean_telemetry_pipeline(input_path: str, output_path: str) -> None:
    """
    Executes end-to-end cleaning and schema normalization.
    """
    print(f"[1/4] Ingesting raw telemetry dataset: {input_path}")
    df = load_data(input_path)
    initial_health = summarize_dataframe_health(df)
    print(f"      Rows: {initial_health['row_count']:,} | Columns: {initial_health['column_count']} | Memory: {initial_health['memory_mb']} MB")

    print("[2/4] Handling missing values and schema anomalies...")
    # Clean string columns
    str_cols = df.select_dtypes(include=["object"]).columns
    for col in str_cols:
        df[col] = df[col].astype(str).str.strip()
        df[col] = df[col].replace({"nan": "UNKNOWN", "None": "UNKNOWN", "": "UNKNOWN"})

    # Numeric imputation with median
    num_cols = df.select_dtypes(include=["float64", "int64"]).columns
    for col in num_cols:
        if df[col].isnull().sum() > 0:
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)

    print("[3/4] Normalizing categorical encodings and timestamps...")
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    # Drop duplicates
    dups = df.duplicated().sum()
    if dups > 0:
        print(f"      Removing {dups} duplicate records.")
        df = df.drop_duplicates()

    print(f"[4/4] Exporting clean data to Parquet: {output_path}")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_parquet(output_path, compression="snappy", index=False)

    final_health = summarize_dataframe_health(df)
    print(f"      SUCCESS: Clean Parquet generated ({final_health['row_count']:,} rows).")


def main():
    parser = argparse.ArgumentParser(description="Attenomy Telemetry Cleaning Pipeline")
    parser.add_argument("--input", default="data/raw/attenomy_telemetry_raw.csv", help="Input dataset path")
    parser.add_argument("--output", default="data/clean/clean.parquet", help="Output Parquet path")
    args = parser.parse_args()

    clean_telemetry_pipeline(args.input, args.output)


if __name__ == "__main__":
    main()
