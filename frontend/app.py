import streamlit as st
import torch
import pandas as pd
import plotly.express as px
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="MoodLens AI | Dev Dharmesh Patel",
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
</style>
""", unsafe_allow_html=True)

# =========================
# ✅ LOAD MODELS
# =========================
@st.cache_resource
def load_models():
    models_dir = PROJECT_ROOT / "models"

    # =========================
    # Emotion classifier
    # =========================
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

    return clf_tokenizer, clf_model

try:
    clf_tokenizer, clf_model = load_models()
    fallback_classes = ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']
    id2label = getattr(clf_model.config, "id2label", None)
    if isinstance(id2label, dict) and len(id2label) == int(clf_model.config.num_labels):
        raw_classes = [str(id2label.get(i, id2label.get(str(i), f"label_{i}"))).lower() for i in range(int(clf_model.config.num_labels))]
        uses_generic_labels = all(label.replace("label_", "").isdigit() for label in raw_classes)
        if uses_generic_labels and len(raw_classes) == len(fallback_classes):
            classes = fallback_classes
        else:
            classes = raw_classes
    else:
        classes = fallback_classes
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
- Plotly
- Pandas
""")

# =========================
# 🎨 MAIN UI
# =========================
st.title("MoodLens AI: Emotion Detection")
st.markdown("Analyze emotions with model confidence and visualization.")

col1, col2 = st.columns(2)

# =========================
# 📥 INPUT
# =========================
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

# =========================
# 📊 OUTPUT
# =========================
with col2:
    st.subheader("Results")

    if predict_button and user_input.strip() != "":

        # ---- Emotion Prediction ----
        inputs = clf_tokenizer(user_input, return_tensors="pt", truncation=True, padding=True)

        with torch.no_grad():
            outputs = clf_model(**inputs)

        logits = outputs.logits
        probs = torch.nn.functional.softmax(logits, dim=1).flatten()

        pred = torch.argmax(logits, dim=1).item()
        emotion = classes[pred]
        conf = probs[pred].item()

        # ---- Display Emotion ----
        st.metric("Detected Emotion", emotion.upper(), f"{conf:.2%} Confidence")

        # ---- Chart ----
        df = pd.DataFrame({
            'Emotion': [c.capitalize() for c in classes],
            'Confidence': probs.tolist()
        })

        fig = px.bar(df, x='Confidence', y='Emotion',
                     orientation='h', color='Confidence',
                     text_auto='.2%')

        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)

    elif predict_button:
        st.warning("Please enter text.")
    else:
        st.info("Enter text and click analyze.")

# =========================
# FOOTER
# =========================
st.markdown("---")
st.caption("© 2026 Dev Dharmesh Patel | MoodLens AI")