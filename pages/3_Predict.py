"""
pages/3_Predict.py
--------------------
Live prediction interface: sliders for the 4 measurements, quick-fill
example buttons per species, real-time prediction with class probability
bar chart, SHAP waterfall explanation + plain-language summary, and a
downloadable session history of past predictions.
"""

import time
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from utils.styling import inject_css, glass_container, section_header
from utils.data_loader import load_default_dataset, FEATURE_COLS, FEATURE_RANGES, EXAMPLE_FLOWERS, dataframe_to_csv_bytes
from utils.training import train_all_models
from utils.explainability import compute_shap_values, shap_waterfall_figure, generate_plain_language_summary

st.set_page_config(page_title="Predict · IrisVision AI", page_icon="🌸", layout="wide")
inject_css()

with glass_container("hero", strong=True):
    st.markdown('<div class="hero-title" style="font-size:2.1rem;">Predict</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="hero-subtitle">Set the four flower measurements and get an '
        "instant, explained species prediction.</div>",
        unsafe_allow_html=True,
    )

# Ensure we have trained models available; auto-train with defaults if the
# user hasn't visited the Train & Compare page yet.
if "training_results" not in st.session_state:
    df = st.session_state.get("active_dataset", load_default_dataset())
    with st.spinner("Training models with default settings so predictions are ready..."):
        st.session_state["training_results"] = train_all_models(df)
        st.session_state["training_config"] = {"test_size": 0.2, "use_scaling": True}

results = st.session_state["training_results"]
best_model_name = results["best_model_name"]
class_names = results["class_names"]

# Initialize slider state defaults
for col in FEATURE_COLS:
    key = f"slider_{col}"
    if key not in st.session_state:
        st.session_state[key] = FEATURE_RANGES[col][2]

# ----------------------------------------------------------- CONTROLS -----
with glass_container("controls"):
    section_header("Flower measurements", "Adjust the sliders or try a quick example.")

    ex_cols = st.columns(3)
    for i, species in enumerate(EXAMPLE_FLOWERS.keys()):
        with ex_cols[i]:
            if st.button(f"🌼 Try {species.title()}", use_container_width=True):
                for col, val in EXAMPLE_FLOWERS[species].items():
                    st.session_state[f"slider_{col}"] = val
                st.rerun()

    st.write("")
    slider_cols = st.columns(4)
    input_values = {}
    for i, col in enumerate(FEATURE_COLS):
        low, high, _ = FEATURE_RANGES[col]
        with slider_cols[i]:
            input_values[col] = st.slider(
                col.replace(" (cm)", "").title(),
                min_value=float(low),
                max_value=float(high),
                step=0.1,
                key=f"slider_{col}",
            )

    model_choice = st.selectbox(
        "Model used for prediction",
        results["leaderboard"]["Model"].tolist(),
        index=results["leaderboard"]["Model"].tolist().index(best_model_name),
        help="Defaults to the best-performing model, but you can pick any trained model.",
    )

# ------------------------------------------------------------- PREDICT ----
x_input = np.array([[input_values[c] for c in FEATURE_COLS]])

model = results["models"][model_choice]
scaler = results["scaler"]
x_input_eval = scaler.transform(x_input) if scaler is not None else x_input

start = time.time()
try:
    pred_class = model.predict(x_input_eval)[0]
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(x_input_eval)[0]
    else:
        # Fallback: one-hot the predicted class if probabilities aren't available
        probs = np.array([1.0 if c == pred_class else 0.0 for c in class_names])
    inference_ms = (time.time() - start) * 1000
    prediction_ok = True
except Exception as e:
    prediction_ok = False
    st.error(f"Prediction failed: {e}")

if prediction_ok:
    prob_map = dict(zip(model.classes_, probs)) if hasattr(model, "classes_") else dict(zip(class_names, probs))
    confidence = prob_map.get(pred_class, max(probs))

    col_result, col_probs = st.columns([2, 3], gap="large")

    with col_result:
        with glass_container("result", strong=True):
            section_header("Prediction")
            st.markdown(
                f'<div style="font-size:2.2rem; font-weight:700; margin-bottom:0.25rem;">'
                f'<em class="serif-accent">{pred_class.title()}</em></div>',
                unsafe_allow_html=True,
            )
            st.metric("Confidence", f"{confidence * 100:.1f}%")
            st.caption(f"Model used: **{model_choice}** · Inference time: **{inference_ms:.2f} ms**")

    with col_probs:
        with glass_container("probs", strong=True):
            section_header("Class probabilities")
            ordered_classes = list(prob_map.keys())
            ordered_probs = list(prob_map.values())
            fig = go.Figure(
                go.Bar(
                    x=ordered_probs,
                    y=[c.title() for c in ordered_classes],
                    orientation="h",
                    marker=dict(color="rgba(255,255,255,0.8)"),
                    text=[f"{p*100:.1f}%" for p in ordered_probs],
                    textposition="outside",
                )
            )
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#f2f2f2", family="Poppins"),
                height=260,
                xaxis=dict(range=[0, 1]),
                margin=dict(l=10, r=30, t=20, b=20),
            )
            st.plotly_chart(fig, use_container_width=True)

    # ---------------------------------------------------------- SHAP -----
    with glass_container("shap"):
        section_header("Why this prediction? (SHAP explanation)", "How each measurement pushed the prediction toward or away from the result.")
        try:
            with st.spinner("Computing SHAP explanation..."):
                shap_values, base_value = compute_shap_values(
                    model, model_choice, results["X_train_eval"], x_input_eval, class_names, pred_class
                )
            st.plotly_chart(
                shap_waterfall_figure(shap_values, base_value, x_input[0], pred_class),
                use_container_width=True,
            )
            st.info(generate_plain_language_summary(shap_values, x_input[0], pred_class))
        except Exception as e:
            st.warning(f"SHAP explanation isn't available for this input/model combination ({e}).")

    # ------------------------------------------------------ HISTORY -------
    if "prediction_history" not in st.session_state:
        st.session_state["prediction_history"] = []

    history_key = (tuple(input_values.values()), model_choice)
    last_key = st.session_state.get("_last_history_key")
    if history_key != last_key:
        st.session_state["prediction_history"].append(
            {
                **{c.replace(" (cm)", ""): v for c, v in input_values.items()},
                "Predicted Species": pred_class.title(),
                "Confidence": f"{confidence*100:.1f}%",
                "Model": model_choice,
            }
        )
        st.session_state["_last_history_key"] = history_key

    with glass_container("history"):
        section_header("Session history", "Every prediction made this session.")
        hist_df = pd.DataFrame(st.session_state["prediction_history"])
        st.dataframe(hist_df, use_container_width=True, hide_index=True)
        col_dl, col_clear = st.columns([1, 1])
        with col_dl:
            st.download_button(
                "⬇ Download history as CSV",
                data=dataframe_to_csv_bytes(hist_df),
                file_name="prediction_history.csv",
                mime="text/csv",
            )
        with col_clear:
            if st.button("Clear history"):
                st.session_state["prediction_history"] = []
                st.rerun()
