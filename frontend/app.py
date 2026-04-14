import streamlit as st
import torch
import pandas as pd
import plotly.express as px
import os
import re
import time
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from dotenv import load_dotenv
from google import genai
from google.genai import errors, types

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
    id2label = getattr(clf_model.config, "id2label", None)
    if isinstance(id2label, dict) and len(id2label) == int(clf_model.config.num_labels):
        classes = [str(id2label.get(i, id2label.get(str(i), f"label_{i}"))).lower() for i in range(int(clf_model.config.num_labels))]
    else:
        classes = ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']
except Exception as e:
    st.error(f"Model loading error: {e}")
    st.stop()

load_dotenv(dotenv_path=PROJECT_ROOT / ".env", override=True)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
gemini_client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")

# =========================
# ✅ GENERATION FUNCTION
# =========================
SYSTEM_PROMPT = """
You are an Emotion Reasoning Assistant.
Analyze the user's input and emotion label.
Respond in exactly this format:
Emotion: <detected emotion>
Reasoning: <brief explanation>
Feedback: <one actionable suggestion>
"""

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


def fallback_reasoning(text, emotion):
    tips = {
        "sadness": "Feedback: Try a small grounding action: drink water, step outside, and message someone supportive.",
        "joy": "Feedback: Savor this moment by noting what went well and sharing it with someone close.",
        "love": "Feedback: Express appreciation directly to the person who made you feel supported.",
        "anger": "Feedback: Pause for 60 seconds, breathe slowly, then write what triggered you before reacting.",
        "fear": "Feedback: Break the concern into one tiny next step you can complete in the next 10 minutes.",
        "surprise": "Feedback: Take a moment to process what changed, then choose one calm next action."
    }
    short_text = (text or "").strip()
    if len(short_text) > 120:
        short_text = short_text[:117] + "..."
    return (
        f"Emotion: {emotion}\n"
        f"Reasoning: Based on the wording in your text ('{short_text}'), this appears closest to {emotion}.\n"
        f"{tips.get(emotion, 'Feedback: Take one small supportive action now, such as a short walk or breathing break.')}"
    )


def normalize_response(raw_text, emotion):
    lines = [line.strip() for line in (raw_text or "").splitlines() if line.strip()]
    emotion_line = next((l for l in lines if l.lower().startswith("emotion:")), f"Emotion: {emotion}")
    reasoning_line = next((l for l in lines if l.lower().startswith("reasoning:")), "Reasoning: The text suggests this emotional pattern.")
    feedback_line = next((l for l in lines if l.lower().startswith("feedback:")), "Feedback: Try one supportive step to improve your mood.")
    return "\n".join([emotion_line, reasoning_line, feedback_line])

def generate_reasoning_with_gemini(text, emotion):
    # Session guards keep repeated clicks from draining quota.
    if "rf_cache" not in st.session_state:
        st.session_state.rf_cache = {}
    if "rf_calls" not in st.session_state:
        st.session_state.rf_calls = 0
    if "rf_last_call" not in st.session_state:
        st.session_state.rf_last_call = 0.0
    if "rf_blocked_until" not in st.session_state:
        st.session_state.rf_blocked_until = 0.0
    if "rf_last_error" not in st.session_state:
        st.session_state.rf_last_error = ""

    max_calls = int(os.getenv("MAX_GEMINI_CALLS_PER_SESSION", "20"))
    min_gap_seconds = float(os.getenv("MIN_GEMINI_GAP_SECONDS", "1.5"))
    cache_key = f"{text.strip().lower()}::{emotion}"

    if cache_key in st.session_state.rf_cache:
        return st.session_state.rf_cache[cache_key]

    if gemini_client is None:
        return "Gemini API error: GEMINI_API_KEY is missing. Add it in .env and restart Streamlit."

    now = time.time()
    if now < st.session_state.rf_blocked_until:
        remaining = int(st.session_state.rf_blocked_until - now)
        return f"Gemini API error: temporarily rate-limited. Retry after about {remaining}s."

    if st.session_state.rf_calls >= max_calls:
        return "Gemini API error: per-session API call limit reached. Increase MAX_GEMINI_CALLS_PER_SESSION."

    wait = min_gap_seconds - (time.time() - st.session_state.rf_last_call)
    if wait > 0:
        time.sleep(wait)

    try:
        prompt = f"Text: {text[:220]}\nEmotion: {emotion}"
        response = gemini_client.models.generate_content(
            model=GEMINI_MODEL,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                temperature=0.3,
                max_output_tokens=120,
            ),
            contents=prompt,
        )
        result = normalize_response((response.text or "").strip(), emotion)
        st.session_state.rf_calls += 1
        st.session_state.rf_last_call = time.time()
        st.session_state.rf_last_error = ""
        st.session_state.rf_cache[cache_key] = result
        return result
    except errors.ClientError as exc:
        msg = str(exc)
        if "RESOURCE_EXHAUSTED" in msg or "429" in msg:
            wait_match = re.search(r"retry in\s*([0-9]+(?:\.[0-9]+)?)s", msg, re.IGNORECASE)
            wait_seconds = float(wait_match.group(1)) if wait_match else 30.0
            st.session_state.rf_blocked_until = time.time() + min(max(wait_seconds + 1, 5), 120)
            st.session_state.rf_last_error = "Gemini quota/rate limit reached."
            return "Gemini API error: quota/rate limit reached (429 RESOURCE_EXHAUSTED)."
        if "API_KEY_INVALID" in msg or "API Key not found" in msg:
            st.session_state.rf_blocked_until = time.time() + 600
            st.session_state.rf_last_error = "Invalid Gemini API key. Check GEMINI_API_KEY in .env."
            return "Gemini API error: invalid API key. Update GEMINI_API_KEY in .env and restart Streamlit."
        st.session_state.rf_last_error = "Gemini request failed."
        return f"Gemini API error: {msg}"

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
    if GEMINI_API_KEY:
        st.success("Gemini API key loaded from .env")
    else:
        st.warning("GEMINI_API_KEY missing in .env; API reasoning will fail until key is set.")
    st.caption(f"Gemini model: {GEMINI_MODEL}")
    if st.session_state.get("rf_last_error"):
        st.warning(st.session_state.rf_last_error)
    st.markdown("---")
    st.markdown("**Tech Stack**")
    st.markdown("""
- Python
- Streamlit
- PyTorch
- Hugging Face Transformers
- PEFT (LoRA)
- Plotly
- Pandas
""")

# =========================
# 🎨 MAIN UI
# =========================
st.title("MoodLens AI: Emotion + Reasoning")
st.markdown("Analyze emotions with AI-powered reasoning and supportive feedback.")

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

        # ---- Reasoning Model ----
        st.subheader("🧠 Reason + Feedback")

        with st.spinner("Generating insight..."):
            response = generate_reasoning_with_gemini(user_input, emotion)

        st.success(response)

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