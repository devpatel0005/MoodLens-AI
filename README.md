# MoodLens AI

Emotion intelligence web app that classifies text into **six emotions** using a **fine-tuned RoBERTa** model, with built-in **SHAP explainability** to show *why* the model predicted a given emotion.

**Developer:** [Dev Dharmesh Patel](https://github.com/devpatel0005)  
**Email:** [devdpatel0005@gmail.com](mailto:devdpatel0005@gmail.com)

---

## What it does
Given a piece of text, MoodLens AI predicts the emotional tone across the following classes:

- sadness
- joy
- love
- anger
- fear
- surprise

It also provides interpretable explanations (SHAP) focused on the **predicted class only**, so users can see which tokens influenced the result.

---

## Why this project matters
Most emotion-classification demos stop at “here’s the label.” MoodLens AI is built like a small product:
- fast interactive inference (Streamlit)
- confidence + per-class probability breakdown
- explainability visuals (SHAP) that make predictions auditable and easier to trust

---

## Key Contributions
- Built an end-to-end NLP web application using a **fine-tuned RoBERTa** classifier for **6-class emotion detection**.
- Implemented a **Streamlit** UI for real-time predictions, confidence scoring, and a **class-probability distribution** chart.
- Integrated **SHAP explainability** to visualize token-level influence for the **predicted emotion only** (cleaner, more actionable explanations).
- Added SHAP **Force Plot** and **Summary Plot** views inside the app for interpretability and debugging.
- Implemented robust label-mapping fallback logic to convert generic model outputs (e.g., `LABEL_0`) into human-readable emotion names.
- Improved responsiveness via cached model/tokenizer loading and local model artifacts; supports CUDA when available.

---

## UI Screenshots

### 1) Main Dashboard and Prediction Results
![Main Dashboard and Prediction Results](docs/images/moodlens-main-dashboard.png)

### 2) SHAP Explanations (Summary Plot + Force Plot)
![SHAP Explanations](docs/images/moodlens-shap-explanations.png)

---

## Model Performance (Test Set)
| Emotion | Precision | Recall | F1-Score |
| :--- | :---: | :---: | :---: |
| Sadness | 0.97 | 0.98 | 0.97 |
| Joy | 0.95 | 0.96 | 0.95 |
| Love | 0.92 | 0.90 | 0.91 |
| Anger | 0.94 | 0.94 | 0.94 |
| Fear | 0.91 | 0.91 | 0.91 |
| Surprise | 0.89 | 0.85 | 0.87 |

---

## Tech Stack
- Python, PyTorch
- Hugging Face Transformers
- Streamlit
- SHAP
- Plotly
- Pandas, NumPy, Matplotlib
- scikit-learn

---

## Project Architecture
1. User enters text in the Streamlit UI.
2. Tokenizer preprocesses input for the RoBERTa classifier.
3. Model outputs logits → softmax probabilities.
4. UI displays:
   - predicted emotion + confidence
   - per-class confidence bar chart
5. SHAP explainer computes token contributions.
6. UI renders predicted-class-only:
   - Summary Plot
   - Force Plot

---

## Quick Start
1. Clone the repository:
```bash
git clone https://github.com/devpatel0005/MoodLens-AI.git
cd MoodLens-AI
```

2. Install dependencies:
```bash
python -m pip install -r requirements.txt
```

3. Run the app:
```bash
streamlit run frontend/app.py
```

---

## Repository Structure
- `frontend/app.py` — Streamlit app, inference flow, SHAP visualizations
- `models/emotion_model/` — saved tokenizer + classifier weights
- `notebooks/Emotion_detection.ipynb` — training + evaluation workflow
- `notebooks/shap.ipynb` — SHAP experimentation and interpretability analysis
- `datasets/emotion_predictions_full.csv` — development artifacts
- `requirements.txt` — dependencies

---

## Future Improvements
- Add confidence thresholding + low-confidence fallback messaging.
- Add input cleaning + optional punctuation filtering for SHAP token display.
- Add lightweight monitoring metrics and a batch inference endpoint.

---

## License
This project is for educational and portfolio use.
