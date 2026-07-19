"""
pages/2_Train_Compare_Models.py
---------------------------------
Trains all six classifiers, shows a ranked leaderboard, per-model detail
(confusion matrix, classification report, ROC curve, feature importance),
and lets the user persist the best model + scaler with joblib.
"""

import numpy as np
import pandas as pd
import streamlit as st

from utils.styling import inject_css, glass_start, glass_end, section_header
from utils.data_loader import load_default_dataset
from utils.training import train_all_models
from utils.evaluation import (
    confusion_matrix_figure,
    classification_report_df,
    roc_curve_figure,
    feature_importance_figure,
)
from utils.persistence import save_best_model

st.set_page_config(page_title="Train & Compare · IrisVision AI", page_icon="🌸", layout="wide")
inject_css()

glass_start(strong=True)
st.markdown('<div class="hero-title" style="font-size:2.1rem;">Train & Compare Models</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="hero-subtitle">Train KNN, Decision Tree, Logistic Regression, SVM, '
    "Random Forest, and Naive Bayes on the current dataset, then compare "
    "them side by side.</div>",
    unsafe_allow_html=True,
)
glass_end()

df = st.session_state.get("active_dataset", load_default_dataset())

# ---------------------------------------------------------- SIDEBAR -------
with st.sidebar:
    st.markdown("### Training controls")
    test_size = st.slider("Test set size", 0.1, 0.4, 0.2, 0.05)
    use_scaling = st.toggle("Feature scaling (StandardScaler)", value=True)
    use_cv = st.toggle("K-Fold cross-validation", value=False)
    cv_folds = st.slider("CV folds", 3, 10, 5, disabled=not use_cv)
    use_grid_search = st.toggle("Hyperparameter tuning (GridSearchCV)", value=False)
    train_clicked = st.button("🚀 Train all models", type="primary", use_container_width=True)

if train_clicked or "training_results" not in st.session_state:
    with st.spinner("Training six models on the current dataset..."):
        results = train_all_models(
            df,
            test_size=test_size,
            use_scaling=use_scaling,
            use_cv=use_cv,
            use_grid_search=use_grid_search,
            cv_folds=cv_folds,
        )
        st.session_state["training_results"] = results
        st.session_state["training_config"] = {
            "test_size": test_size,
            "use_scaling": use_scaling,
        }

results = st.session_state["training_results"]
leaderboard = results["leaderboard"]
best_model_name = results["best_model_name"]

# --------------------------------------------------------- LEADERBOARD ----
glass_start(strong=True)
section_header("Leaderboard", "Ranked by accuracy (best to worst). Best model is highlighted.")

def _highlight_best(row):
    is_best = row["Model"] == best_model_name
    return ["background-color: rgba(255,255,255,0.14)" if is_best else "" for _ in row]

styled = leaderboard.style.apply(_highlight_best, axis=1).format(
    {"Accuracy": "{:.3f}", "Precision": "{:.3f}", "Recall": "{:.3f}", "F1 Score": "{:.3f}"}
)
st.dataframe(styled, use_container_width=True, hide_index=True)
st.markdown(f'<span class="best-model-tag">🏆 Best model: {best_model_name}</span>', unsafe_allow_html=True)

col_save, _ = st.columns([1, 4])
with col_save:
    if st.button("💾 Save Best Model"):
        best_row = leaderboard.iloc[0]
        path = save_best_model(
            model=results["models"][best_model_name],
            scaler=results["scaler"],
            model_name=best_model_name,
            accuracy=float(best_row["Accuracy"]),
            use_scaling=results["use_scaling"],
        )
        st.success(f"Saved to `{path}`")
glass_end()

if results["cv_scores"]:
    glass_start()
    section_header("Cross-validation scores", f"{len(list(results['cv_scores'].values())[0])}-fold accuracy per model.")
    cv_rows = []
    for name, scores in results["cv_scores"].items():
        cv_rows.append({"Model": name, "Mean CV Accuracy": scores.mean(), "Std Dev": scores.std()})
    cv_df = pd.DataFrame(cv_rows).sort_values("Mean CV Accuracy", ascending=False)
    st.dataframe(
        cv_df.style.format({"Mean CV Accuracy": "{:.3f}", "Std Dev": "{:.3f}"}),
        use_container_width=True,
        hide_index=True,
    )
    glass_end()

# -------------------------------------------------------- MODEL DETAIL ----
glass_start()
section_header("Per-model detail", "Confusion matrix, classification report, ROC curve, feature importance.")

selected_model_name = st.selectbox(
    "Choose a model to inspect",
    leaderboard["Model"].tolist(),
    index=leaderboard["Model"].tolist().index(best_model_name),
)
model = results["models"][selected_model_name]
y_pred = model.predict(results["X_test_eval"])

col1, col2 = st.columns(2, gap="large")
with col1:
    st.markdown("**Confusion Matrix**")
    st.plotly_chart(
        confusion_matrix_figure(results["y_test"], y_pred, results["class_names"]),
        use_container_width=True,
    )
with col2:
    st.markdown("**Classification Report**")
    report_df = classification_report_df(results["y_test"], y_pred)
    st.dataframe(report_df, use_container_width=True)

roc_fig = roc_curve_figure(model, results["X_test_eval"], results["y_test"], results["class_names"])
fi_fig = feature_importance_figure(model, selected_model_name)

col3, col4 = st.columns(2, gap="large")
with col3:
    st.markdown("**ROC Curve (One-vs-Rest)**")
    if roc_fig is not None:
        st.plotly_chart(roc_fig, use_container_width=True)
    else:
        st.info("This model doesn't expose class probabilities, so an ROC curve isn't available.")
with col4:
    st.markdown("**Feature Importance**")
    if fi_fig is not None:
        st.plotly_chart(fi_fig, use_container_width=True)
    else:
        st.info("Feature importance is only shown for tree-based models (Decision Tree, Random Forest).")

glass_end()
