"""
preprocessing.py
-----------------
Train/test splitting and feature scaling helpers shared across the app.
"""

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from .data_loader import FEATURE_COLS, TARGET_COL


def split_features_target(df):
    """Splits a tidy Iris DataFrame into X (features) and y (species labels)."""
    X = df[FEATURE_COLS].copy()
    y = df[TARGET_COL].copy()
    return X, y


def train_test_split_data(X, y, test_size: float = 0.2, random_state: int = 42):
    """Wraps sklearn's train_test_split with stratification on the target."""
    return train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )


def fit_scaler(X_train):
    """Fits a StandardScaler on the training features and returns it."""
    scaler = StandardScaler()
    scaler.fit(X_train)
    return scaler


def apply_scaler(scaler, X):
    """Transforms features with a previously fitted scaler."""
    return scaler.transform(X)
