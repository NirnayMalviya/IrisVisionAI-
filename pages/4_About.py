"""
pages/4_About.py
------------------
Plain-language explanation of the dataset, the models compared, how the
best model is selected, and how the SHAP explanation works.
"""

import streamlit as st
from utils.styling import inject_css, glass_start, glass_end, section_header

st.set_page_config(page_title="About · IrisVision AI", page_icon="🌸", layout="wide")
inject_css()

glass_start(strong=True)
st.markdown('<div class="hero-title" style="font-size:2.1rem;">About IrisVision AI</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="hero-subtitle">The dataset, the models, and how predictions are explained '
    "— in plain language.</div>",
    unsafe_allow_html=True,
)
glass_end()

col1, col2 = st.columns(2, gap="large")

with col1:
    glass_start()
    section_header("The dataset")
    st.markdown(
        "The **Iris dataset** was collected in 1936 and has become the "
        "\"hello world\" of machine learning classification. It contains "
        "**150 flowers**, split evenly across **three species**: "
        "*Setosa*, *Versicolor*, and *Virginica*.\n\n"
        "Each flower is described by four simple measurements, all in "
        "centimeters:\n"
        "- Sepal length\n"
        "- Sepal width\n"
        "- Petal length\n"
        "- Petal width\n\n"
        "Setosa is easy to tell apart from the other two — its petals are "
        "noticeably smaller — while Versicolor and Virginica overlap more, "
        "which is what makes this a genuinely interesting (if small) "
        "classification problem."
    )
    glass_end()

    glass_start()
    section_header("The six models")
    st.markdown(
        "- **K-Nearest Neighbors (KNN)** — classifies a flower based on "
        "the species of its closest neighbors in measurement-space.\n"
        "- **Decision Tree** — learns a series of yes/no splits on the "
        "measurements (e.g., \"is petal length < 2.5cm?\").\n"
        "- **Logistic Regression** — fits a linear boundary between "
        "species using probability.\n"
        "- **Support Vector Machine (SVM)** — finds the boundary that "
        "best separates species with the widest possible margin.\n"
        "- **Random Forest** — trains many decision trees and lets them "
        "vote, usually more robust than a single tree.\n"
        "- **Naive Bayes** — uses probability theory, assuming each "
        "measurement is independent given the species."
    )
    glass_end()

with col2:
    glass_start()
    section_header("How the best model is chosen")
    st.markdown(
        "All six models are trained on the same train/test split of the "
        "data. Each is then scored on the held-out test set using four "
        "metrics:\n\n"
        "- **Accuracy** — overall percentage of correct predictions\n"
        "- **Precision** — of the flowers predicted as a species, how "
        "many actually were that species\n"
        "- **Recall** — of the flowers that actually were a species, how "
        "many were correctly found\n"
        "- **F1 Score** — a balance between precision and recall\n\n"
        "The model with the highest **accuracy** on the test set is "
        "highlighted as the *best model* on the leaderboard. In practice, "
        "on the standard Iris dataset, most of these models score very "
        "similarly (often 90%+ accuracy) because the classes are fairly "
        "well separated — but tree-based models and SVM tend to edge out "
        "the others slightly on the trickier Versicolor/Virginica boundary."
    )
    glass_end()

    glass_start()
    section_header("How the SHAP explanation works")
    st.markdown(
        "**SHAP** (SHapley Additive exPlanations) is a technique borrowed "
        "from game theory. For a single prediction, it asks: *\"if I "
        "removed this measurement, how much would the predicted "
        "probability change?\"* — and does this fairly across all "
        "possible combinations of measurements.\n\n"
        "The result is a set of four numbers, one per measurement, that "
        "add up to explain the gap between an average prediction (the "
        "\"base value\") and this specific prediction. The **waterfall "
        "chart** on the Predict page shows exactly how each measurement "
        "pushed the prediction up or down, and the plain-language summary "
        "calls out whichever measurement mattered most."
    )
    glass_end()

glass_start()
section_header("Tech stack", "What this app is built with.")
st.markdown(
    "Python · Streamlit · Pandas · NumPy · Scikit-Learn · Plotly · SHAP · Joblib"
)
glass_end()
