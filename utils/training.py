"""
training.py
------------
Trains and compares the six required classifiers: KNN, Decision Tree,
Logistic Regression, SVM, Random Forest, and Naive Bayes. Supports
adjustable train/test split, optional feature scaling, optional k-fold
cross-validation, and optional GridSearchCV hyperparameter tuning.
"""

import time
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from .preprocessing import (
    split_features_target,
    train_test_split_data,
    fit_scaler,
    apply_scaler,
)

# Model registry: name -> (estimator, param_grid for GridSearchCV)
MODEL_REGISTRY = {
    "KNN": (
        KNeighborsClassifier(),
        {"n_neighbors": [3, 5, 7, 9, 11]},
    ),
    "Decision Tree": (
        DecisionTreeClassifier(random_state=42),
        {"max_depth": [2, 3, 4, 5, None], "min_samples_split": [2, 4, 6]},
    ),
    "Logistic Regression": (
        LogisticRegression(max_iter=1000),
        {"C": [0.1, 1.0, 10.0]},
    ),
    "SVM": (
        SVC(probability=True, random_state=42),
        {"C": [0.5, 1.0, 5.0], "kernel": ["linear", "rbf"]},
    ),
    "Random Forest": (
        RandomForestClassifier(random_state=42),
        {"n_estimators": [50, 100, 150], "max_depth": [3, 5, None]},
    ),
    "Naive Bayes": (
        GaussianNB(),
        {"var_smoothing": [1e-9, 1e-8, 1e-7]},
    ),
}

TREE_BASED_MODELS = {"Decision Tree", "Random Forest"}


def train_all_models(
    df,
    test_size: float = 0.2,
    use_scaling: bool = True,
    use_cv: bool = False,
    use_grid_search: bool = False,
    cv_folds: int = 5,
    random_state: int = 42,
):
    """
    Trains all six registered models on the given DataFrame and returns a
    full results bundle used by the Train & Compare and Predict pages.

    Returns a dict with:
        leaderboard   : pd.DataFrame of accuracy/precision/recall/F1 per model
        models        : dict[name -> fitted estimator]
        scaler        : fitted StandardScaler or None
        X_train/X_test/y_train/y_test : split data (unscaled, for display)
        X_test_eval   : features used at evaluation time (scaled if applicable)
        best_model_name
        cv_scores     : dict[name -> np.ndarray] if use_cv else {}
        class_names   : sorted list of species labels
    """
    X, y = split_features_target(df)
    class_names = sorted(y.unique().tolist())

    X_train, X_test, y_train, y_test = train_test_split_data(
        X, y, test_size=test_size, random_state=random_state
    )

    scaler = None
    X_train_eval, X_test_eval = X_train, X_test
    if use_scaling:
        scaler = fit_scaler(X_train)
        X_train_eval = apply_scaler(scaler, X_train)
        X_test_eval = apply_scaler(scaler, X_test)

    fitted_models = {}
    rows = []
    cv_scores = {}

    for name, (estimator, param_grid) in MODEL_REGISTRY.items():
        start = time.time()

        if use_grid_search:
            skf = StratifiedKFold(n_splits=min(cv_folds, 5), shuffle=True, random_state=random_state)
            search = GridSearchCV(estimator, param_grid, cv=skf, scoring="accuracy", n_jobs=1)
            search.fit(X_train_eval, y_train)
            fitted = search.best_estimator_
        else:
            fitted = estimator
            fitted.fit(X_train_eval, y_train)

        train_time = time.time() - start

        y_pred = fitted.predict(X_test_eval)
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average="macro", zero_division=0)
        rec = recall_score(y_test, y_pred, average="macro", zero_division=0)
        f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)

        if use_cv:
            skf = StratifiedKFold(n_splits=min(cv_folds, 5), shuffle=True, random_state=random_state)
            scores = cross_val_score(fitted, X_train_eval, y_train, cv=skf, scoring="accuracy")
            cv_scores[name] = scores

        fitted_models[name] = fitted
        rows.append(
            {
                "Model": name,
                "Accuracy": acc,
                "Precision": prec,
                "Recall": rec,
                "F1 Score": f1,
                "Train Time (s)": round(train_time, 3),
            }
        )

    leaderboard = pd.DataFrame(rows).sort_values("Accuracy", ascending=False).reset_index(drop=True)
    best_model_name = leaderboard.iloc[0]["Model"]

    return {
        "leaderboard": leaderboard,
        "models": fitted_models,
        "scaler": scaler,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "X_test_eval": X_test_eval,
        "X_train_eval": X_train_eval,
        "best_model_name": best_model_name,
        "cv_scores": cv_scores,
        "class_names": class_names,
        "use_scaling": use_scaling,
    }
