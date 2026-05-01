# 🔍 Semantic Plagiarism Detection System

A end-to-end NLP pipeline that detects plagiarism using **semantic similarity** — not keyword matching. Built with pretrained Transformer embeddings, a Deep Neural Network classifier, a FastAPI backend, and a Streamlit frontend.

---

## 📊 Results

| Metric | TF-IDF + LR (Baseline) | Transformer DNN | Improvement |
|---|---|---|---|
| Accuracy | 89.29% | **94.64%** | +5.36% |
| F1 Score | 89.29% | **94.74%** | +5.45% |

**5-Fold Cross-Validation:** 94.05% Accuracy ± 1.83% — confirming the model genuinely generalizes.

---

## 🏗️ Project Structure

```
ANLP/
├── artifacts/                  # Trained model artifacts
│   ├── plagiarism_dnn.keras    # Saved Keras DNN classifier
│   └── feature_scaler.pkl      # Fitted StandardScaler
│
├── assets/images/              # Generated plots
│   ├── eda_plots.png
│   ├── training_curves.png
│   ├── model_comparison.png
│   └── pr_curve.png
│
├── data/
│   └── dataset.csv             # 370 labeled text pairs
│
├── notebooks/
│   ├── plagiarism_detection.ipynb      # Original pipeline
│   └── plagiarism_detection_v2.ipynb   # Enhanced pipeline with CV & stats
│
├── api/
│   ├── main.py                 # FastAPI backend
│   └── requirements.txt
│
├── app/
│   ├── streamlit_app.py        # Streamlit frontend
│   └── requirements.txt
│
├── requirements.txt            # All dependencies
└── run.bat                     # One-click launcher (Windows)
```

---

## ⚙️ Pipeline Overview

```
Raw Text Pairs
     │
     ▼
Preprocessing (lowercase, normalize whitespace)
     │
     ▼
Sentence Embeddings — all-mpnet-base-v2 (768-dim)
     │
     ▼
Feature Engineering — [emb1, emb2, |emb1−emb2|, emb1⊙emb2, cosine_sim]
                       = 3,073-dimensional feature vector
     │
     ▼
DNN Classifier  →  Dense(512) → Dense(256) → Dense(128) → Sigmoid
     │
     ▼
Prediction  (threshold = 0.60)
```

---

## 🚀 Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

> Or use the project's conda/venv environment directly.

### 2. Launch the app (Windows)

Double-click **`run.bat`** — it starts both services and opens the browser automatically.

| Service | URL |
|---|---|
| Streamlit UI | http://localhost:8501 |
| FastAPI Backend | http://localhost:8000 |
| API Docs (Swagger) | http://localhost:8000/docs |

### 3. Launch manually

```bash
# Terminal 1 — API
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000

# Terminal 2 — UI
python -m streamlit run app/streamlit_app.py --server.port 8501
```

---

## 🔌 API Reference

### `GET /health`
Returns the API and model status.

```json
{ "status": "ok", "encoder": "all-mpnet-base-v2", "model": "plagiarism_dnn.keras" }
```

### `POST /predict`

**Request:**
```json
{
  "text1": "Artificial intelligence is reshaping industries.",
  "text2": "AI is transforming sectors across the globe.",
  "threshold": 0.6
}
```

**Response:**
```json
{
  "prediction": "Plagiarised",
  "is_plagiarised": true,
  "confidence": 0.8721,
  "confidence_pct": "87.21%",
  "cosine_similarity": 0.9126,
  "threshold_used": 0.6
}
```

---

## 🧠 Model Details

| Component | Detail |
|---|---|
| Sentence Encoder | `all-mpnet-base-v2` (768-dim embeddings) |
| Feature Vector | 3,073-dim: `[e1, e2, \|e1−e2\|, e1⊙e2, cosine]` |
| Classifier | DNN: 512 → 256 → 128 → Sigmoid |
| Optimizer | Adam (lr=1e-3) |
| Regularization | BatchNorm + Dropout (0.3, 0.2, 0.2) |
| Class Weights | Balanced via `compute_class_weight` |
| Decision Threshold | 0.6 (adjustable in UI) |
| Training Callbacks | EarlyStopping + ReduceLROnPlateau |

---

## 📦 Dependencies

- `tensorflow` >= 2.21
- `sentence-transformers` >= 3.0
- `scikit-learn` >= 1.5
- `fastapi` + `uvicorn`
- `streamlit`
- `numpy`, `pandas`, `matplotlib`, `seaborn`
