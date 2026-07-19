"""
data_loader.py
---------------
Handles loading the built-in sklearn Iris dataset and validating/parsing
user-uploaded CSV files that should follow the same 4-feature + species
schema.
"""

import pandas as pd
import numpy as np
import streamlit as st
from sklearn.datasets import load_iris

FEATURE_COLS = [
    "sepal length (cm)",
    "sepal width (cm)",
    "petal length (cm)",
    "petal width (cm)",
]
TARGET_COL = "species"
SPECIES_NAMES = ["setosa", "versicolor", "virginica"]

# Realistic min/max ranges pulled from the classic Iris dataset, used to
# bound the prediction-page sliders sensibly.
FEATURE_RANGES = {
    "sepal length (cm)": (4.0, 8.0, 5.8),
    "sepal width (cm)": (2.0, 4.5, 3.0),
    "petal length (cm)": (1.0, 7.0, 3.8),
    "petal width (cm)": (0.1, 2.6, 1.2),
}

# Quick-fill example measurements per species (typical values).
EXAMPLE_FLOWERS = {
    "setosa": {
        "sepal length (cm)": 5.1,
        "sepal width (cm)": 3.5,
        "petal length (cm)": 1.4,
        "petal width (cm)": 0.2,
    },
    "versicolor": {
        "sepal length (cm)": 6.0,
        "sepal width (cm)": 2.7,
        "petal length (cm)": 4.3,
        "petal width (cm)": 1.3,
    },
    "virginica": {
        "sepal length (cm)": 6.7,
        "sepal width (cm)": 3.0,
        "petal length (cm)": 5.6,
        "petal width (cm)": 2.1,
    },
}


@st.cache_data(show_spinner=False)
def load_default_dataset() -> pd.DataFrame:
    """Loads the built-in sklearn Iris dataset as a tidy DataFrame."""
    iris = load_iris(as_frame=True)
    df = iris.frame.copy()
    df.rename(columns={"target": "target_id"}, inplace=True)
    df[TARGET_COL] = df["target_id"].map(dict(enumerate(iris.target_names)))
    df = df[FEATURE_COLS + [TARGET_COL]]
    return df


def validate_csv(df: pd.DataFrame):
    """
    Validates an uploaded CSV against the expected Iris schema.

    Returns:
        (is_valid: bool, message: str, cleaned_df: pd.DataFrame | None)
    """
    if df.empty:
        return False, "The uploaded file is empty.", None

    # Normalize column names (case/whitespace tolerant matching)
    normalized = {c.strip().lower(): c for c in df.columns}
    required_keys = {
        "sepal length (cm)": ["sepal length (cm)", "sepal_length", "sepal length"],
        "sepal width (cm)": ["sepal width (cm)", "sepal_width", "sepal width"],
        "petal length (cm)": ["petal length (cm)", "petal_length", "petal length"],
        "petal width (cm)": ["petal width (cm)", "petal_width", "petal width"],
        "species": ["species", "target", "class", "label"],
    }

    resolved = {}
    missing = []
    for canonical, aliases in required_keys.items():
        found = None
        for alias in aliases:
            if alias in normalized:
                found = normalized[alias]
                break
        if found is None:
            missing.append(canonical)
        else:
            resolved[canonical] = found

    if missing:
        return False, f"Missing required column(s): {', '.join(missing)}.", None

    clean = pd.DataFrame()
    for canonical, original_col in resolved.items():
        clean[canonical] = df[original_col]

    # Type / value checks on numeric feature columns
    for col in FEATURE_COLS:
        clean[col] = pd.to_numeric(clean[col], errors="coerce")

    n_before = len(clean)
    bad_rows = clean[FEATURE_COLS].isna().any(axis=1) | clean["species"].isna()
    n_bad = int(bad_rows.sum())
    clean = clean[~bad_rows].reset_index(drop=True)

    if clean.empty:
        return False, "No valid rows remained after cleaning (check numeric columns and species labels).", None

    # Outlier flag: values wildly outside plausible ranges (not blocking, just informational)
    outlier_flags = 0
    for col in FEATURE_COLS:
        low, high, _ = FEATURE_RANGES[col]
        span = high - low
        outlier_flags += int(((clean[col] < low - span) | (clean[col] > high + span)).sum())

    msg_parts = [f"Loaded {len(clean)} valid row(s) out of {n_before}."]
    if n_bad:
        msg_parts.append(f"Dropped {n_bad} row(s) with missing/invalid values.")
    if outlier_flags:
        msg_parts.append(f"Note: {outlier_flags} value(s) look like potential outliers.")

    return True, " ".join(msg_parts), clean


def dataframe_to_csv_bytes(df: pd.DataFrame) -> bytes:
    """Serializes a DataFrame to CSV bytes for download buttons."""
    return df.to_csv(index=False).encode("utf-8")
