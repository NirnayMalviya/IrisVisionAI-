"""
evaluation.py
--------------
Builds evaluation visuals for a trained model: confusion matrix heatmap,
classification report table, ROC curves (one-vs-rest for multiclass), and
feature importance charts for tree-based models. All charts use a single
grayscale hue with varying opacity per class to match the monochrome theme.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.preprocessing import label_binarize

from .training import TREE_BASED_MODELS
from .data_loader import FEATURE_COLS

MONO_SCALE = [[0, "rgba(255,255,255,0.02)"], [1, "rgba(255,255,255,0.95)"]]


def _base_layout(fig, height=380):
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#f2f2f2", family="Poppins"),
        height=height,
        margin=dict(l=40, r=20, t=40, b=40),
    )
    return fig


def confusion_matrix_figure(y_test, y_pred, class_names):
    """Returns a Plotly heatmap figure for the confusion matrix."""
    cm = confusion_matrix(y_test, y_pred, labels=class_names)
    fig = go.Figure(
        data=go.Heatmap(
            z=cm,
            x=class_names,
            y=class_names,
            colorscale=MONO_SCALE,
            text=cm,
            texttemplate="%{text}",
            textfont=dict(size=16),
            showscale=False,
            hovertemplate="Actual: %{y}<br>Predicted: %{x}<br>Count: %{z}<extra></extra>",
        )
    )
    fig.update_layout(
        xaxis_title="Predicted",
        yaxis_title="Actual",
        yaxis=dict(autorange="reversed"),
    )
    return _base_layout(fig, height=380)


def classification_report_df(y_test, y_pred):
    """Returns the sklearn classification report as a tidy DataFrame."""
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    df = pd.DataFrame(report).transpose()
    df = df.round(3)
    return df


def roc_curve_figure(model, X_test_eval, y_test, class_names):
    """
    Returns a Plotly figure with one-vs-rest ROC curves for each class,
    plus macro-average AUC in the legend.
    """
    if not hasattr(model, "predict_proba"):
        return None

    y_bin = label_binarize(y_test, classes=class_names)
    y_score = model.predict_proba(X_test_eval)

    fig = go.Figure()
    opacities = np.linspace(0.35, 1.0, len(class_names))
    aucs = []

    for i, cls in enumerate(class_names):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_score[:, i])
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        fig.add_trace(
            go.Scatter(
                x=fpr,
                y=tpr,
                mode="lines",
                name=f"{cls} (AUC = {roc_auc:.3f})",
                line=dict(color=f"rgba(255,255,255,{opacities[i]:.2f})", width=2.5),
            )
        )

    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            name="Chance",
            line=dict(color="rgba(255,255,255,0.25)", width=1.5, dash="dash"),
        )
    )

    fig.update_layout(
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        legend=dict(bgcolor="rgba(0,0,0,0)"),
    )
    return _base_layout(fig, height=420)


def feature_importance_figure(model, model_name):
    """Returns a horizontal bar chart of feature importances for tree-based models, else None."""
    if model_name not in TREE_BASED_MODELS or not hasattr(model, "feature_importances_"):
        return None

    importances = model.feature_importances_
    order = np.argsort(importances)
    fig = go.Figure(
        go.Bar(
            x=importances[order],
            y=[FEATURE_COLS[i] for i in order],
            orientation="h",
            marker=dict(color="rgba(255,255,255,0.75)"),
        )
    )
    fig.update_layout(xaxis_title="Importance", yaxis_title="")
    return _base_layout(fig, height=320)
