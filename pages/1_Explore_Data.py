"""
pages/1_Explore_Data.py
-------------------------
EDA dashboard: dataset preview with search/sort, optional CSV upload with
validation, summary statistics, pairplot/scatter matrix, correlation
heatmap, box/violin plots, and class distribution — all in grayscale,
glass-panel styling.
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from utils.styling import inject_css, glass_container, section_header
from utils.data_loader import (
    load_default_dataset,
    validate_csv,
    dataframe_to_csv_bytes,
    FEATURE_COLS,
    TARGET_COL,
)

st.set_page_config(page_title="Explore Data · IrisVision AI", page_icon="🌸", layout="wide")
inject_css()

# Monochrome opacity levels per species for consistent grayscale plots.
MONO_COLORS = ["rgba(255,255,255,0.9)", "rgba(255,255,255,0.55)", "rgba(255,255,255,0.28)"]


def _mono_layout(fig, height=420):
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#f2f2f2", family="Poppins"),
        height=height,
        legend=dict(bgcolor="rgba(0,0,0,0)"),
        margin=dict(l=30, r=20, t=40, b=30),
    )
    return fig


with glass_container("hero", strong=True):
    st.markdown('<div class="hero-title" style="font-size:2.1rem;">Explore Data</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="hero-subtitle">Inspect the built-in Iris dataset, or upload your own '
        "CSV with the same schema to explore custom data.</div>",
        unsafe_allow_html=True,
    )

# ------------------------------------------------------------- UPLOAD -----
with glass_container("upload"):
    section_header("Dataset source", "Use the built-in dataset, or upload your own CSV.")

    uploaded = st.file_uploader(
        "Upload CSV (columns: sepal length (cm), sepal width (cm), petal length (cm), "
        "petal width (cm), species)",
        type=["csv"],
    )

    if uploaded is not None:
        try:
            raw_df = pd.read_csv(uploaded)
            is_valid, message, clean_df = validate_csv(raw_df)
            if is_valid:
                st.success(message)
                df = clean_df
                st.session_state["active_dataset"] = df
            else:
                st.error(message)
                df = st.session_state.get("active_dataset", load_default_dataset())
        except Exception as e:
            st.error(f"Couldn't read that file: {e}")
            df = st.session_state.get("active_dataset", load_default_dataset())
    else:
        df = st.session_state.get("active_dataset", load_default_dataset())

    st.session_state["active_dataset"] = df

# --------------------------------------------------------- PREVIEW TABLE --
with glass_container("preview"):
    section_header("Dataset preview", f"{len(df)} rows · sortable, searchable columns below.")

    search = st.text_input("Search species (optional)", "")
    display_df = df.copy()
    if search:
        display_df = display_df[display_df[TARGET_COL].str.contains(search, case=False, na=False)]

    page_size = 15
    n_pages = max(1, (len(display_df) - 1) // page_size + 1)
    page = st.number_input("Page", min_value=1, max_value=n_pages, value=1, step=1)
    start_i = (page - 1) * page_size
    st.dataframe(
        display_df.iloc[start_i : start_i + page_size],
        use_container_width=True,
        hide_index=True,
    )

    st.download_button(
        "⬇ Download dataset as CSV",
        data=dataframe_to_csv_bytes(df),
        file_name="iris_dataset.csv",
        mime="text/csv",
    )

# -------------------------------------------------------- SUMMARY STATS ---
with glass_container("summary_stats"):
    section_header("Summary statistics", "Mean, std, min, and max per feature, grouped by species.")
    summary = df.groupby(TARGET_COL)[FEATURE_COLS].agg(["mean", "std", "min", "max"]).round(2)
    st.dataframe(summary, use_container_width=True)

# --------------------------------------------------- CLASS DISTRIBUTION ---
col1, col2 = st.columns(2, gap="large")

with col1:
    with glass_container("class_distribution"):
        section_header("Class distribution")
        counts = df[TARGET_COL].value_counts().reset_index()
        counts.columns = [TARGET_COL, "count"]
        fig = px.bar(
            counts,
            x=TARGET_COL,
            y="count",
            color=TARGET_COL,
            color_discrete_sequence=MONO_COLORS,
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(_mono_layout(fig, height=360), use_container_width=True)

with col2:
    with glass_container("correlation_heatmap"):
        section_header("Correlation heatmap")
        corr = df[FEATURE_COLS].corr()
        fig = go.Figure(
            go.Heatmap(
                z=corr.values,
                x=[c.replace(" (cm)", "") for c in corr.columns],
                y=[c.replace(" (cm)", "") for c in corr.columns],
                colorscale=[[0, "rgba(255,255,255,0.03)"], [1, "rgba(255,255,255,0.95)"]],
                text=corr.round(2).values,
                texttemplate="%{text}",
                showscale=False,
            )
        )
        st.plotly_chart(_mono_layout(fig, height=360), use_container_width=True)

# -------------------------------------------------------- SCATTER MATRIX --
with glass_container("scatter_matrix"):
    section_header("Pairplot / scatter matrix", "Petal & sepal dimensions, colored by species.")
    fig = px.scatter_matrix(
        df,
        dimensions=FEATURE_COLS,
        color=TARGET_COL,
        color_discrete_sequence=MONO_COLORS,
    )
    fig.update_traces(diagonal_visible=False, marker=dict(size=4))
    st.plotly_chart(_mono_layout(fig, height=650), use_container_width=True)

# ------------------------------------------------------- BOX / VIOLIN -----
with glass_container("box_violin"):
    section_header("Box & violin plots", "Distribution of each feature per species.")
    plot_type = st.radio("Plot type", ["Box", "Violin"], horizontal=True)
    feature_choice = st.selectbox("Feature", FEATURE_COLS)

    if plot_type == "Box":
        fig = px.box(df, x=TARGET_COL, y=feature_choice, color=TARGET_COL, color_discrete_sequence=MONO_COLORS)
    else:
        fig = px.violin(df, x=TARGET_COL, y=feature_choice, color=TARGET_COL, color_discrete_sequence=MONO_COLORS, box=True)
    fig.update_layout(showlegend=False)
    st.plotly_chart(_mono_layout(fig, height=420), use_container_width=True)
