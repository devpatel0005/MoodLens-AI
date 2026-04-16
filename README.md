# MoodLens AI

MoodLens AI is an explainable NLP application that detects emotions in text using a fine-tuned RoBERTa classifier and shows why predictions are made through SHAP token-level explanations.

Live Application: https://moodlens-ai-igo7kb8nrjmvmrcqhti9xy.streamlit.app/

Developer: Dev Dharmesh Patel  
GitHub: https://github.com/devpatel0005  
Email: devdpatel0005@gmail.com

---

## Project Summary
This project solves a practical problem in applied NLP: most emotion classifiers return only a label, but do not explain decision logic. MoodLens AI provides both prediction and interpretability in one product-style interface.

For each input sentence, the app provides:
- Predicted emotion class
- Confidence score
- Per-class confidence distribution chart
- SHAP Summary Plot for token importance
- SHAP Force Plot for token contribution direction

Emotion classes supported:
- sadness
- joy
- love
- anger
- fear
- surprise

---

## Project Highlights
- Built and deployed an end-to-end explainable AI web app for emotion detection.
- Integrated Hugging Face Transformers with Streamlit for real-time NLP inference.
- Implemented SHAP explainability in production-style UI using predicted-class token attributions.
- Added confidence distribution visualization for transparent model behavior.
- Added robust class-label mapping logic to normalize LABEL_X outputs into human-readable class names.
- Optimized inference flow using cached model loading and optional CUDA support.
- Deployed the app publicly and maintained dependency reliability for cloud runtime.

---

## Model and Explainability Details

### Base Model
- Architecture: RoBERTa sequence classifier
- Framework: Hugging Face Transformers + PyTorch
- Output: six-class emotion probabilities

### Explainability Layer
- Method: SHAP (SHapley Additive exPlanations)
- Explainer input: text classification pipeline
- Scope: explanations are generated for the predicted class
- Visual outputs:
  - Summary bar plot: token impact magnitude
  - Force plot: token push-up and push-down effects from baseline to final score

### Inference Pipeline
1. User input is collected through Streamlit UI.
2. Pipeline performs tokenization and model inference.
3. Probabilities are ranked and displayed as chart + metric.
4. SHAP explainer computes token-level contributions.
5. Summary and force plots are rendered in the app.

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
- Python
- Streamlit
- PyTorch
- Torchvision
- Hugging Face Transformers
- SHAP
- Plotly
- Pandas
- NumPy
- Matplotlib
- scikit-learn

---

## Repository Structure
- frontend/app.py: Streamlit application logic, inference workflow, SHAP rendering
- models/emotion_model/: trained model artifacts and tokenizer files
- notebooks/Emotion_detection.ipynb: model development and evaluation workflow
- notebooks/shap.ipynb: SHAP experimentation and interpretation workflow
- datasets/emotion_predictions_full.csv: project data artifact
- requirements.txt: deployment and local dependency list
- docs/: documentation assets

---

## Local Setup and Run

### 1) Clone Repository
```bash
git clone https://github.com/devpatel0005/MoodLens-AI.git
cd MoodLens-AI
```

### 2) Install Dependencies
```bash
python -m pip install -r requirements.txt
```

### 3) Run Streamlit App
```bash
streamlit run frontend/app.py
```

---

## Deployment Notes
- Live deployment is running on Streamlit Cloud.
- Ensure all required libraries are listed in requirements.txt before deploy.
- Current dependency set includes plotly and torchvision to prevent cloud import errors.
- Model files are loaded from local project paths in the deployed environment.

---

## Skills Demonstrated
- End-to-end machine learning productization
- NLP model integration and inference engineering
- Explainable AI implementation in user-facing applications
- Data visualization for model confidence communication
- Cloud deployment debugging and dependency management
- UI/UX iteration for ML applications

---

## Limitations
- Predictions may be less reliable on very short or ambiguous text.
- SHAP explanations are local and input-specific, not full causal proofs.
- Domain shift can affect generalization on unseen language styles.

---

## Planned Improvements
- Confidence thresholding and low-confidence fallback messaging
- Batch inference support for multiple texts
- Additional explainability views for class comparison
- Monitoring and lightweight analytics for production usage

---

## License
This repository is for educational and portfolio purposes.
