"""
AWS ML - Week 01: Market Arrivals Data Analysis & Cleaning
===========================================================
This script demonstrates an end-to-end data pipeline:
1. Loading raw dataset (Market Arrivals CSV).
2. Named data cleaning functions:
   - Handling missing values and duplicates
   - Text and case normalization
   - Data type conversions
   - Outlier detection and treatment using IQR
3. Statistical calculations using NumPy (mean, median, std, percentile, where, etc.)
4. Time-series data visualization.
"""

import os
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

DATA_URL = "https://raw.githubusercontent.com/selva86/datasets/master/MarketArrivals.csv"
DATA_FILE = os.path.join(os.path.dirname(__file__), "marketing.csv")


# ==============================================================================
# 1. Dataset Loading Function
# ==============================================================================
def load_raw_data(filepath: str = DATA_FILE, fallback_url: str = DATA_URL) -> pd.DataFrame:
    """
    Loads raw market dataset from local CSV file.
    If the local file is not found, fetches from fallback URL and saves it locally.
    """
    if os.path.exists(filepath):
        print(f"[INFO] Loading raw dataset from local repository: '{filepath}'")
        df = pd.read_csv(filepath)
    else:
        print(f"[INFO] Local file not found. Fetching from remote URL: {fallback_url}")
        try:
            df = pd.read_csv(fallback_url)
            df.to_csv(filepath, index=False)
            print(f"[INFO] Downloaded and saved dataset to '{filepath}'")
        except Exception as exc:
            raise RuntimeError(f"Failed to fetch dataset from URL: {exc}") from exc

    print(f"[INFO] Dataset loaded successfully with shape: {df.shape}")
    return df


# ==============================================================================
# 2. Named Cleaning Functions (Missing Values, Duplicates, Text, Types, Outliers)
# ==============================================================================
def clean_missing_and_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans missing values and duplicate records from the DataFrame.
    - Inspects missing values per column and drops/imputes as required.
    - Identifies and drops duplicate rows.
    """
    df_cleaned = df.copy()

    # 1. Check & handle missing values
    missing_counts = df_cleaned.isna().sum()
    total_missing = missing_counts.sum()
    print("\n--- [CLEANING 1: Missing Values & Duplicates] ---")
    print(f"Total missing values found: {total_missing}")
    if total_missing > 0:
        print("Missing values per column:\n", missing_counts[missing_counts > 0])
        # Drop rows where crucial columns are null
        df_cleaned = df_cleaned.dropna(subset=['date', 'priceMod', 'quantity'])
    else:
        print("No missing values detected.")

    # 2. Check & handle duplicates
    initial_rows = len(df_cleaned)
    duplicates_count = df_cleaned.duplicated().sum()
    print(f"Duplicate rows found: {duplicates_count}")
    if duplicates_count > 0:
        df_cleaned = df_cleaned.drop_duplicates().reset_index(drop=True)
        print(f"Removed {duplicates_count} duplicates. Rows remaining: {len(df_cleaned)}")
    else:
        print("No duplicate rows found.")

    return df_cleaned


def clean_text_fields(df: pd.DataFrame, text_columns: list = None) -> pd.DataFrame:
    """
    Normalizes text data:
    - Strips leading and trailing whitespace.
    - Enforces consistent uppercase casing for categorical identifiers.
    """
    df_cleaned = df.copy()
    if text_columns is None:
        text_columns = ['market', 'state', 'city']

    print("\n--- [CLEANING 2: Text & Case Fixes] ---")
    for col in text_columns:
        if col in df_cleaned.columns and (
            df_cleaned[col].dtype == 'object' or pd.api.types.is_string_dtype(df_cleaned[col])
        ):
            # Strip whitespace and convert to consistent uppercase
            df_cleaned[col] = df_cleaned[col].astype(str).str.strip().str.upper()
            print(f"Standardized text casing and stripped whitespace for column: '{col}'")

    return df_cleaned


def convert_column_types(df: pd.DataFrame) -> pd.DataFrame:
    """
    Converts columns to appropriate data types:
    - Parses 'date' column into datetime64.
    - Ensures numeric columns are explicitly typed (numeric / integer).
    - Removes redundant redundant date components (e.g. redundant 'month'/'year' strings if present).
    """
    df_cleaned = df.copy()
    print("\n--- [CLEANING 3: Type Conversions] ---")

    # Convert date column
    if 'date' in df_cleaned.columns:
        df_cleaned['date'] = pd.to_datetime(
            df_cleaned['date'], format='mixed', dayfirst=False, errors='coerce'
        )
        print("Converted 'date' column to datetime64[ns]")

    # Drop redundant month/year string columns if desired, keeping date as source of truth
    for redundant_col in ['month', 'year']:
        if redundant_col in df_cleaned.columns:
            df_cleaned = df_cleaned.drop(columns=[redundant_col])
            print(f"Dropped redundant column: '{redundant_col}'")

    # Numeric conversions
    numeric_cols = ['quantity', 'priceMin', 'priceMax', 'priceMod']
    for col in numeric_cols:
        if col in df_cleaned.columns:
            df_cleaned[col] = pd.to_numeric(df_cleaned[col], errors='coerce')
            print(f"Converted '{col}' to numeric type")

    return df_cleaned


def detect_and_handle_outliers(
    df: pd.DataFrame, column: str = 'priceMod', factor: float = 1.5
) -> pd.DataFrame:
    """
    Detects outliers using the Interquartile Range (IQR) method and caps them.
    - Calculates Q1 (25th percentile), Q3 (75th percentile), and IQR.
    - Defines lower bound (Q1 - factor * IQR) and upper bound (Q3 + factor * IQR).
    - Caps extreme values at bounds (Winsorization) to preserve data volume while handling anomalies.
    """
    df_cleaned = df.copy()
    print(f"\n--- [CLEANING 4: Outlier Detection & Handling for '{column}'] ---")

    values = df_cleaned[column].dropna().to_numpy()
    q25, q75 = np.percentile(values, [25, 75])
    iqr = q75 - q25
    lower_bound = max(0, q25 - (factor * iqr))
    upper_bound = q75 + (factor * iqr)

    outliers = (values < lower_bound) | (values > upper_bound)
    outlier_count = int(np.sum(outliers))
    print(f"Q1 (25th): {q25:.2f}, Q3 (75th): {q75:.2f}, IQR: {iqr:.2f}")
    print(f"Calculated bounds: [{lower_bound:.2f}, {upper_bound:.2f}]")
    print(f"Identified {outlier_count} outliers ({outlier_count / len(values) * 100:.2f}% of data)")

    # Cap outliers at the boundaries (Winsorization)
    df_cleaned[f'{column}_capped'] = np.clip(df_cleaned[column], lower_bound, upper_bound)
    print(f"Created capped column '{column}_capped' within bounds.")

    return df_cleaned


# ==============================================================================
# 3. NumPy Calculations Function (mean, median, std, percentile, where, min, max)
# ==============================================================================
def calculate_numpy_statistics(df: pd.DataFrame, column: str = 'priceMod') -> dict:
    """
    Performs comprehensive statistical calculations using NumPy functions:
    1. np.mean()       - Average price
    2. np.median()     - Median price (robust central tendency)
    3. np.std()        - Standard deviation (volatility/spread)
    4. np.percentile() - 10th, 25th, 75th, 90th percentiles
    5. np.where()      - Vectorized conditional classification (e.g. price categorization)
    6. np.min()        - Minimum price
    7. np.max()        - Maximum price
    """
    print(f"\n=== [NUMPY CALCULATIONS ON '{column}'] ===")
    data = df[column].dropna().to_numpy()

    # 1. np.mean
    mean_val = float(np.mean(data))

    # 2. np.median
    median_val = float(np.median(data))

    # 3. np.std
    std_val = float(np.std(data))

    # 4. np.percentile
    p10, p25, p75, p90 = np.percentile(data, [10, 25, 75, 90])

    # 5. np.min & np.max
    min_val = float(np.min(data))
    max_val = float(np.max(data))

    # 6. np.where: Vectorized conditional tagging
    # Classify prices into High vs Standard compared to 75th percentile
    price_category = np.where(data >= p75, "High (Top 25%)", "Standard/Low")
    high_count = int(np.sum(price_category == "High (Top 25%)"))

    # Print formatted results
    print(f"1. np.mean()       : {mean_val:.2f}")
    print(f"2. np.median()     : {median_val:.2f}")
    print(f"3. np.std()        : {std_val:.2f}")
    print(f"4. np.percentile() : 10th={p10:.2f}, 25th={p25:.2f}, 75th={p75:.2f}, 90th={p90:.2f}")
    print(f"5. np.min()        : {min_val:.2f}")
    print(f"6. np.max()        : {max_val:.2f}")
    print(f"7. np.where()      : Flagged {high_count} records as 'High (Top 25%)'")

    return {
        "mean": mean_val,
        "median": median_val,
        "std": std_val,
        "percentiles": {"p10": p10, "p25": p25, "p75": p75, "p90": p90},
        "min": min_val,
        "max": max_val,
        "high_tier_count": high_count,
    }


# ==============================================================================
# 4. Visualization Function
# ==============================================================================
def plot_market_trends(df: pd.DataFrame, save_path: str = None) -> None:
    """
    Plots time-series trends of market prices (Min, Max, Modal Price).
    """
    df_sorted = df.sort_values('date').dropna(subset=['date'])

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df_sorted['date'], df_sorted['priceMax'], color='#e74c3c', alpha=0.6, label='Max Price')
    ax.plot(df_sorted['date'], df_sorted['priceMin'], color='#3498db', alpha=0.6, label='Min Price')
    if 'priceMod' in df_sorted.columns:
        ax.plot(df_sorted['date'], df_sorted['priceMod'], color='#2ecc71', alpha=0.7, label='Modal Price')

    ax.set_title("Market Price Trends Over Time", fontsize=14, fontweight='bold')
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Price (INR / quintal)", fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(loc='upper left')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"\n[INFO] Saved visualization plot to: {save_path}")
    else:
        plt.show()


# ==============================================================================
# 5. Main Execution Pipeline
# ==============================================================================
def main():
    print("=" * 60)
    print("  AWS ML - Week 01 Data Pipeline & Analysis")
    print("=" * 60)

    # Step 1: Load raw data
    raw_df = load_raw_data()

    # Step 2: Named Cleaning Functions
    df_step1 = clean_missing_and_duplicates(raw_df)
    df_step2 = clean_text_fields(df_step1)
    df_step3 = convert_column_types(df_step2)
    df_clean = detect_and_handle_outliers(df_step3, column='priceMod')

    # Step 3: Dataset Summary
    print("\n--- [CLEANED DATASET OVERVIEW] ---")
    print(f"Final Cleaned Shape : {df_clean.shape}")
    print("\nColumn Data Types:")
    print(df_clean.dtypes.to_string())

    # Step 4: Statistical Analysis using NumPy
    calculate_numpy_statistics(df_clean, column='priceMod')

    # Step 5: Visualizations
    plot_path = os.path.join(os.path.dirname(__file__), "price_trends.png")
    plot_market_trends(df_clean, save_path=plot_path)

    print("\n[SUCCESS] Week 01 Data Pipeline completed successfully!")


if __name__ == "__main__":
    main()
