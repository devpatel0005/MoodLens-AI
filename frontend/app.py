import streamlit as st
import torch
import pandas as pd
import plotly.express as px
import numpy as np
import shap
import matplotlib.pyplot as plt
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="MoodlensAI | Explainable Emotion Intelligence",
    page_icon="🧠",
    layout="wide"
)

# --- CUSTOM CSS ---
st.markdown("""
<style>
.main {background-color: #f8f9fa;}
.stTextArea textarea {border: 2px solid #e0e0e0; border-radius: 10px;}
.stButton>button {
    width: 100%; border-radius: 8px; height: 3.5em;
    background-color: #007BFF; color: white; font-weight: bold;
}
.hero-header {
    background: linear-gradient(135deg, #dbeafe 0%, #eef4ff 100%);
    border: 1px solid #bfdbfe;
    border-radius: 12px;
    padding: 14px 16px 12px 16px;
    margin-bottom: 12px;
}
.hero-title {
    margin: 0;
    font-size: 3.1rem;
    line-height: 1.05;
    font-weight: 800;
    color: #0b1f44 !important;
}
.hero-subtitle {
    margin: 8px 0 0 0;
    font-size: 1.1rem;
    font-weight: 600;
    color: #1e3a5f !important;
}
</style>
""", unsafe_allow_html=True)

# =========================
# ✅ LOAD MODELS
# =========================
@st.cache_resource
def load_models():
    models_dir = PROJECT_ROOT / "models"
    clf_path = models_dir / "emotion_model"

    clf_tokenizer = AutoTokenizer.from_pretrained(
        str(clf_path),
        use_fast=False,
        local_files_only=True
    )

    clf_model = AutoModelForSequenceClassification.from_pretrained(
        str(clf_path),
        local_files_only=True
    )

    device = 0 if torch.cuda.is_available() else -1
    if device == 0:
        clf_model.cuda()

    pred_pipeline = pipeline(
        "text-classification",
        model=clf_model,
        tokenizer=clf_tokenizer,
        device=device,
        return_all_scores=True
    )

    explainer = shap.Explainer(pred_pipeline)

    return clf_model, pred_pipeline, explainer


def get_class_names(model):
    fallback = ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']
    id2label = getattr(model.config, "id2label", None)
    num_labels = int(getattr(model.config, "num_labels", len(fallback)))

    if not isinstance(id2label, dict):
        return fallback

    labels = [str(id2label.get(i, id2label.get(str(i), f"label_{i}"))).lower() for i in range(num_labels)]
    generic = all(label.replace("label_", "").isdigit() for label in labels)
    if generic and len(labels) == len(fallback):
        return fallback
    return labels if labels else fallback


def get_pipeline_scores(text, pred_pipeline, classes):
    raw_scores = pred_pipeline(text)
    if isinstance(raw_scores, list) and raw_scores and isinstance(raw_scores[0], list):
        raw_scores = raw_scores[0]

    score_by_label = {}
    for item in raw_scores:
        label = str(item.get("label", "")).lower()
        score = float(item.get("score", 0.0))

        if label.startswith("label_") and label[6:].isdigit():
            idx = int(label[6:])
            if 0 <= idx < len(classes):
                label = classes[idx]

        score_by_label[label] = score

    return [score_by_label.get(label, 0.0) for label in classes]

try:
    clf_model, pred_pipeline, explainer = load_models()
    classes = get_class_names(clf_model)
except Exception as e:
    st.error(f"Model loading error: {e}")
    st.stop()

EXAMPLE_TEXTS = [
    "I failed my exam and feel really bad.",
    "I got my dream job today and I am so happy!",
    "I miss my best friend so much and feel empty.",
    "My parents surprised me with a gift and I am shocked.",
    "I am nervous about tomorrow's interview.",
    "I feel deeply loved when my family supports me.",
]


def apply_selected_example():
    selected_text = st.session_state.selected_example
    if selected_text != "Choose an example":
        st.session_state.user_input = selected_text

# =========================
# 🎯 SIDEBAR
# =========================
with st.sidebar:
    st.markdown("# 🧠 MoodLens AI")
    st.markdown("### Emotion Intelligence System")
    st.markdown("---")
    st.info("**Dev Dharmesh Patel**")
    st.markdown("[GitHub](https://github.com/devpatel0005)")
    st.markdown("[Email](mailto:devdpatel0005@gmail.com)")
    st.markdown("---")
    st.markdown("**Tech Stack**")
    st.markdown("""
- Python
- Streamlit
- PyTorch
- Hugging Face Transformers
- SHAP
- Plotly
- Pandas
- NumPy
- Matplotlib
""")

# =========================
# 🎨 MAIN UI
# =========================
st.markdown(
    """
    <div class="hero-header">
        <h1 class="hero-title">MoodlensAI</h1>
        <p class="hero-subtitle">Explainable Emotion Intelligence</p>
    </div>
    """,
    unsafe_allow_html=True
)

col1, col2 = st.columns(2)
shap_exp = None
shap_plot_error = None
pred = None

with col1:
    st.subheader("Input Text")

    if "user_input" not in st.session_state:
        st.session_state.user_input = ""

    if "selected_example" not in st.session_state:
        st.session_state.selected_example = "Choose an example"

    st.selectbox(
        "Select an example emotion text:",
        ["Choose an example"] + EXAMPLE_TEXTS,
        key="selected_example",
        on_change=apply_selected_example
    )

    user_input = st.text_area(
        "Enter text:",
        placeholder="Type something...",
        height=150,
        key="user_input"
    )

    predict_button = st.button("Analyze Emotion")

with col2:
    st.subheader("Results")

    if predict_button and user_input.strip() != "":
        with st.spinner("Analyzing text and preparing SHAP explanations..."):
            # ---- Emotion Prediction ----
            probs = get_pipeline_scores(user_input, pred_pipeline, classes)
            pred = int(np.argmax(probs))
            emotion = classes[pred]
            conf = probs[pred]

            # ---- Display Emotion ----
            st.metric("Detected Emotion", emotion.upper(), f"{conf:.2%} Confidence")

            # ---- Chart ----
            df = pd.DataFrame({
                'Emotion': [c.capitalize() for c in classes],
                'Confidence': probs
            }).sort_values("Confidence", ascending=True)

            fig = px.bar(
                df,
                x='Confidence',
                y='Emotion',
                orientation='h',
                text='Confidence',
                color_discrete_sequence=['#007BFF']
            )
            fig.update_traces(texttemplate='%{text:.2%}', textposition='outside', hovertemplate='%{y}: %{x:.2%}<extra></extra>')
            fig.update_xaxes(range=[0, 1], tickformat='.0%')
            fig.update_layout(height=320, showlegend=False, margin=dict(l=8, r=8, t=8, b=8))
            st.plotly_chart(fig, use_container_width=True)

            try:
                shap_values = explainer([user_input])
                shap_exp = shap_values[0, :, pred]
            except Exception as e:
                shap_plot_error = e

    elif predict_button:
        st.warning("Please enter text.")
    else:
        st.info("Enter text and click analyze.")

if predict_button and user_input.strip() != "":
    st.markdown("### SHAP Explanations (Predicted Emotion)")

    if shap_exp is not None:
        summary_col, force_col = st.columns(2)
        summary_col.markdown("#### Summary Plot")
        class_shap_values = np.array([shap_exp.values])
        token_names = [str(token) for token in shap_exp.data]

        plt.figure(figsize=(7, 4))
        shap.summary_plot(
            class_shap_values,
            feature_names=token_names,
            plot_type='bar',
            show=False
        )
        summary_col.pyplot(plt.gcf(), use_container_width=True)
        plt.close()

        force_col.markdown("#### Force Plot")
        plt.figure(figsize=(10, 2.6))
        shap.force_plot(
            shap_exp.base_values,
            shap_exp.values,
            shap_exp.data,
            matplotlib=True,
            show=False
        )
        force_col.pyplot(plt.gcf(), use_container_width=True)
        plt.close()
    elif shap_plot_error is not None:
        st.warning(f"Could not generate SHAP plots: {shap_plot_error}")

st.markdown("---")
st.caption("Model: local transformer emotion classifier (6 classes) | Explainability: SHAP local explanations | Limitation: short/ambiguous text can reduce reliability")
st.caption("© 2026 Dev Dharmesh Patel | MoodLens AI")