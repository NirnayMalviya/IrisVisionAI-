"""
explainability.py
-------------------
Generates SHAP-based explanations for a single live prediction, works with
any of the six model types (tree-based models get the fast TreeExplainer,
everything else falls back to a background-summarized generic Explainer so
it stays responsive even for KNN/SVM/Logistic Regression/Naive Bayes).
Also produces a one-line, plain-language auto-generated summary sentence.
"""

import numpy as np
import shap
import plotly.graph_objects as go

from .data_loader import FEATURE_COLS
from .training import TREE_BASED_MODELS

# Cache background summaries per (model name) call site to avoid recomputing
# kmeans every single prediction within a session.
_BACKGROUND_CACHE = {}


def _get_background(model_name, X_train_eval):
    key = model_name
    if key not in _BACKGROUND_CACHE:
        k = min(30, len(X_train_eval))
        _BACKGROUND_CACHE[key] = shap.sample(X_train_eval, k, random_state=42)
    return _BACKGROUND_CACHE[key]


def compute_shap_values(model, model_name, X_train_eval, x_input_eval, class_names, predicted_class):
    """
    Computes SHAP values for a single input row, for the predicted class.

    Returns:
        values      : np.ndarray of shape (n_features,) — per-feature contribution
        base_value  : float — the explainer's expected/base value for this class
    """
    class_idx = class_names.index(predicted_class)

    if model_name in TREE_BASED_MODELS:
        explainer = shap.TreeExplainer(model)
        raw = explainer.shap_values(x_input_eval)
        # TreeExplainer returns either a list per class, or a 3D array depending on version.
        if isinstance(raw, list):
            values = np.array(raw[class_idx])[0]
            base = explainer.expected_value
            base_value = base[class_idx] if hasattr(base, "__len__") else base
        else:
            arr = np.array(raw)
            if arr.ndim == 3:
                values = arr[0, :, class_idx]
            else:
                values = arr[0]
            base = explainer.expected_value
            base_value = base[class_idx] if hasattr(base, "__len__") else base
    else:
        background = _get_background(model_name, X_train_eval)
        explainer = shap.Explainer(model.predict_proba, background, feature_names=FEATURE_COLS)
        shap_exp = explainer(x_input_eval)
        vals = np.array(shap_exp.values)
        base_vals = np.array(shap_exp.base_values)
        if vals.ndim == 3:
            values = vals[0, :, class_idx]
        else:
            values = vals[0]
        base_value = base_vals[0, class_idx] if base_vals.ndim == 2 else float(base_vals[0])

    return np.asarray(values, dtype=float), float(base_value)


def shap_waterfall_figure(values, base_value, feature_input_values, predicted_class):
    """
    Builds a grayscale Plotly waterfall chart showing how each feature's
    SHAP contribution moves the prediction from the base value to the
    final predicted probability for the predicted class.
    """
    order = np.argsort(-np.abs(values))
    ordered_features = [FEATURE_COLS[i] for i in order]
    ordered_values = values[order]
    labels = [
        f"{FEATURE_COLS[i].replace(' (cm)', '')} = {feature_input_values[i]:.2f}"
        for i in order
    ]

    measure = ["relative"] * len(ordered_values) + ["total"]
    x_labels = labels + ["Prediction"]
    y_values = list(ordered_values) + [base_value + ordered_values.sum()]

    fig = go.Figure(
        go.Waterfall(
            orientation="v",
            measure=measure,
            x=x_labels,
            y=[float(v) for v in ordered_values] + [float(base_value + ordered_values.sum())],
            base=base_value,
            increasing=dict(marker=dict(color="rgba(255,255,255,0.85)")),
            decreasing=dict(marker=dict(color="rgba(255,255,255,0.35)")),
            totals=dict(marker=dict(color="rgba(255,255,255,0.95)")),
            connector=dict(line=dict(color="rgba(255,255,255,0.25)", width=1)),
            text=[f"{v:+.3f}" for v in ordered_values] + [""],
            textposition="outside",
        )
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#f2f2f2", family="Poppins"),
        height=420,
        margin=dict(l=40, r=20, t=30, b=90),
        showlegend=False,
        yaxis_title=f"P({predicted_class})",
    )
    return fig


def generate_plain_language_summary(values, feature_input_values, predicted_class):
    """
    Returns a one-line, human-readable sentence describing which feature
    contributed most to the prediction and in which direction.
    """
    idx = int(np.argmax(np.abs(values)))
    feature_name = FEATURE_COLS[idx].replace(" (cm)", "")
    contribution = values[idx]
    direction = "pushed the prediction toward" if contribution > 0 else "pulled the prediction away from"
    value = feature_input_values[idx]
    return (
        f"**{feature_name.title()}** ({value:.2f} cm) {direction} "
        f"**{predicted_class.title()}**, contributing the most out of all four measurements "
        f"to this prediction."
    )
