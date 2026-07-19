"""
app.py
------
IrisVision AI — Home / Hero page.

This is the entry point for the multi-page Streamlit app. Run with:
    streamlit run app.py

Other pages (Explore Data, Train & Compare Models, Predict, About) live in
the pages/ directory and are picked up automatically by Streamlit's
multi-page navigation.
"""

import streamlit as st
from utils.styling import inject_css, glass_start, glass_end, render_pills, section_header
from utils.data_loader import load_default_dataset

st.set_page_config(
    page_title="IrisVision AI",
    page_icon="🌸",
    layout="wide",
    initial_sidebar_state="expanded",
)

inject_css()

# ---------------------------------------------------------------- HERO ----
glass_start(strong=True)
st.markdown('<div class="hero-title">IrisVision AI</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="hero-subtitle">Classify iris species instantly with '
    '<em>explainable</em> machine learning — compare six algorithms, '
    'explore the data, and see exactly why each prediction was made.</div>',
    unsafe_allow_html=True,
)
render_pills(["EDA Dashboard", "Model Comparison", "Live Prediction", "Explainable AI"])
st.write("")
col1, col2 = st.columns([1, 5])
with col1:
    if st.button("Start Exploring →", type="primary"):
        st.switch_page("pages/3_Predict.py")
glass_end()

st.write("")

# ------------------------------------------------------------ ABOUT-ISH ---
col_a, col_b = st.columns([3, 2], gap="large")

with col_a:
    glass_start()
    section_header(
        "What is the Iris dataset?",
        "A classic, well-understood benchmark for classification.",
    )
    st.markdown(
        "The Iris dataset contains **150 measurements** of iris flowers "
        "from **three species** — *Setosa*, *Versicolor*, and *Virginica* — "
        "each described by four measurements: sepal length, sepal width, "
        "petal length, and petal width, all in centimeters.\n\n"
        "It's small, clean, and perfectly separable enough to make it an "
        "ideal playground for comparing how different machine learning "
        "algorithms approach the same classification problem."
    )
    glass_end()

with col_b:
    glass_start()
    section_header("How to use this app")
    st.markdown(
        "1. **Explore Data** — inspect the dataset, upload your own CSV, "
        "and view distributions.\n"
        "2. **Train & Compare Models** — train six classifiers and rank "
        "them on accuracy, precision, recall, and F1.\n"
        "3. **Predict** — set flower measurements and get a live, "
        "explained prediction.\n"
        "4. **About** — read how the best model was chosen."
    )
    glass_end()

st.write("")

# ------------------------------------------------------------- PREVIEW ----
glass_start()
section_header("Quick look at the dataset", "First few rows of the built-in Iris dataset.")
df_preview = load_default_dataset()
st.dataframe(df_preview.head(6), use_container_width=True, hide_index=True)
glass_end()
