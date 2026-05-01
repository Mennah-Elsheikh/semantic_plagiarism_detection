import re
import pickle
import numpy as np
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity as cos_sim
import tensorflow as tf

# ── Paths (resolved relative to this file's location) ───────────────
ROOT       = Path(__file__).resolve().parent.parent
MODEL_PATH  = ROOT / "artifacts" / "plagiarism_dnn.keras"
SCALER_PATH = ROOT / "artifacts" / "feature_scaler.pkl"

# ── Global model objects ──────────────────────────────────────────────
encoder = None
model   = None
scaler  = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load heavy objects once at startup."""
    global encoder, model, scaler
    print("[API] Loading sentence encoder...")
    encoder = SentenceTransformer("all-mpnet-base-v2")
    print("[API] Loading Keras model...")
    model = tf.keras.models.load_model(str(MODEL_PATH))
    print("[API] Loading scaler...")
    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)
    print("[API] All models loaded. Ready.")
    yield
    print("[API] Shutting down.")

app = FastAPI(
    title="Plagiarism Detection API",
    description="Semantic plagiarism detection using MPNet + DNN",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Schemas ───────────────────────────────────────────────────────────
class PredictRequest(BaseModel):
    text1: str
    text2: str
    threshold: float = 0.6

class PredictResponse(BaseModel):
    prediction: str
    is_plagiarised: bool
    confidence: float
    confidence_pct: str
    cosine_similarity: float
    threshold_used: float

# ── Helpers ───────────────────────────────────────────────────────────
def preprocess(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text

def build_features(e1: np.ndarray, e2: np.ndarray) -> np.ndarray:
    abs_diff = np.abs(e1 - e2)
    mult     = e1 * e2
    cosine   = cos_sim(e1, e2)[0][0]
    cos      = np.array([[cosine]])
    return np.hstack([e1, e2, abs_diff, mult, cos]), float(cosine)

# ── Endpoints ─────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {
        "status": "ok",
        "encoder": "all-mpnet-base-v2",
        "model": "plagiarism_dnn.keras",
    }

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    t1 = preprocess(req.text1)
    t2 = preprocess(req.text2)

    e1 = encoder.encode([t1])
    e2 = encoder.encode([t2])

    features, cosine = build_features(e1, e2)
    features_scaled  = scaler.transform(features)

    prob          = float(model.predict(features_scaled, verbose=0)[0][0])
    is_plagiarised = prob >= req.threshold

    return PredictResponse(
        prediction      = "Plagiarised" if is_plagiarised else "Not Plagiarised",
        is_plagiarised  = is_plagiarised,
        confidence      = round(prob, 4),
        confidence_pct  = f"{prob * 100:.2f}%",
        cosine_similarity = round(cosine, 4),
        threshold_used  = req.threshold,
    )
