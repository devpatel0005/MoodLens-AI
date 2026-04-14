# MoodLens AI: Emotion Detection with RoBERTa

**Developer:** [Dev Dharmesh Patel](https://github.com/devpatel0005)  
**Email:** [devdpatel0005@gmail.com](mailto:devdpatel0005@gmail.com)

## Project Summary
MoodLens AI is an NLP project that classifies user text into six emotions using a fine-tuned RoBERTa model:

- sadness
- joy
- love
- anger
- fear
- surprise

The system is deployed as a Streamlit application with real-time inference, confidence scoring, and class-wise confidence visualization.

## Resume-Ready Highlights
- Built and deployed a transformer-based multi-class emotion classifier using RoBERTa and PyTorch.
- Achieved strong performance on emotion classification with macro-level consistency across six classes.
- Designed an interactive Streamlit UI for real-time predictions and confidence analytics.
- Implemented robust model loading and inference flow with local model artifacts and cached resources.
- Added production-friendly label handling to map model indices to human-readable emotions.

## Tech Stack
- Python
- PyTorch
- Hugging Face Transformers
- Datasets
- scikit-learn
- Streamlit
- Plotly
- Pandas, NumPy, Matplotlib

## Model Performance (Test Set)
| Emotion | Precision | Recall | F1-Score |
| :--- | :---: | :---: | :---: |
| Sadness | 0.97 | 0.98 | 0.97 |
| Joy | 0.95 | 0.96 | 0.95 |
| Love | 0.92 | 0.90 | 0.91 |
| Anger | 0.94 | 0.94 | 0.94 |
| Fear | 0.91 | 0.91 | 0.91 |
| Surprise | 0.89 | 0.85 | 0.87 |

## Local Setup
1. Clone the repository:
```bash
git clone https://github.com/devpatel0005/MoodLens-AI.git
cd MoodLens-AI
```

2. Install dependencies:
```bash
python -m pip install -r requirements.txt
```

3. Run the Streamlit app:
```bash
streamlit run frontend/app.py
```

## Repository Structure
- `frontend/app.py`: Streamlit frontend and inference pipeline.
- `notebooks/Emotion_detection.ipynb`: Training and evaluation workflow.
- `models/emotion_model/`: Saved tokenizer and model weights.
- `datasets/emotion_predictions_full.csv`: Dataset/prediction artifacts used during development.
- `requirements.txt`: Project dependencies.

## Notes
- The current frontend focuses on emotion detection, confidence score, and confidence plot.
- If the model config exposes generic labels like `LABEL_0`, the app maps them to the expected emotion classes.
