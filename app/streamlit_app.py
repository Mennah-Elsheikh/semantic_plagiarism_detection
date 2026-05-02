import re
import pickle
import numpy as np
import streamlit as st
import pandas as pd
from pathlib import Path

# ── Page config ───────────────────────────────────────────────────────
st.set_page_config(
    page_title="Plagiarism Detector",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Paths ─────────────────────────────────────────────────────────────
ROOT        = Path(__file__).resolve().parent.parent
ONNX_PATH   = ROOT / "artifacts" / "plagiarism_dnn.onnx"
SCALER_PATH = ROOT / "artifacts" / "feature_scaler.pkl"

# ── Custom CSS ────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Dark gradient background */
.stApp {
    background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
    min-height: 100vh;
}

/* Header */
.hero {
    text-align: center;
    padding: 2.5rem 1rem 1rem 1rem;
}
.hero h1 {
    font-size: 3rem;
    font-weight: 700;
    background: linear-gradient(90deg, #a78bfa, #60a5fa, #34d399);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.3rem;
}
.hero p {
    color: #94a3b8;
    font-size: 1.1rem;
    margin-top: 0;
}

/* Cards */
.result-card {
    background: rgba(255,255,255,0.05);
    backdrop-filter: blur(12px);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 16px;
    padding: 1.8rem;
    margin-top: 1.5rem;
}

/* Verdict badge */
.verdict-plagiarised {
    display: inline-block;
    background: linear-gradient(135deg, #ef4444, #b91c1c);
    color: white;
    font-size: 1.6rem;
    font-weight: 700;
    padding: 0.6rem 2rem;
    border-radius: 50px;
    letter-spacing: 1px;
    margin-bottom: 1.2rem;
}
.verdict-clean {
    display: inline-block;
    background: linear-gradient(135deg, #10b981, #047857);
    color: white;
    font-size: 1.6rem;
    font-weight: 700;
    padding: 0.6rem 2rem;
    border-radius: 50px;
    letter-spacing: 1px;
    margin-bottom: 1.2rem;
}

/* Metric chip */
.metric-row {
    display: flex;
    gap: 1rem;
    flex-wrap: wrap;
    margin-top: 1rem;
}
.metric-chip {
    background: rgba(255,255,255,0.07);
    border: 1px solid rgba(255,255,255,0.12);
    border-radius: 12px;
    padding: 0.8rem 1.4rem;
    flex: 1;
    min-width: 140px;
}
.metric-chip .label {
    color: #94a3b8;
    font-size: 0.75rem;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 1px;
}
.metric-chip .value {
    color: #f1f5f9;
    font-size: 1.5rem;
    font-weight: 700;
    margin-top: 0.2rem;
}

/* Text areas */
textarea {
    background: rgba(255,255,255,0.05) !important;
    border: 1px solid rgba(255,255,255,0.15) !important;
    border-radius: 10px !important;
    color: #f1f5f9 !important;
    font-family: 'Inter', sans-serif !important;
}

/* Button */
.stButton button {
    width: 100%;
    background: linear-gradient(135deg, #7c3aed, #4f46e5) !important;
    color: white !important;
    font-weight: 600 !important;
    font-size: 1.05rem !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.75rem !important;
    transition: opacity 0.2s !important;
}
.stButton button:hover {
    opacity: 0.88 !important;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: rgba(15, 12, 41, 0.8) !important;
    border-right: 1px solid rgba(255,255,255,0.07) !important;
}

/* Status dot */
.status-ok   { color: #34d399; font-weight: 600; }
.status-err  { color: #f87171; font-weight: 600; }

/* History table */
.stDataFrame { border-radius: 12px; overflow: hidden; }
</style>
""", unsafe_allow_html=True)

# ── Session state ─────────────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []

# ── Cosine similarity (pure NumPy — no sklearn needed) ────────────────
def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a.flatten(), b.flatten()
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denom) if denom > 0 else 0.0

# ── Model loading (cached — runs only once per session) ───────────────
@st.cache_resource(show_spinner=False)
def load_models():
    """Load sentence encoder, ONNX DNN, and scaler. Cached for the session."""
    from sentence_transformers import SentenceTransformer
    import onnxruntime as ort

    encoder = SentenceTransformer("all-mpnet-base-v2")
    sess    = ort.InferenceSession(str(ONNX_PATH))
    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)
    # Store the ONNX input name so inference doesn't re-query it every time
    input_name = sess.get_inputs()[0].name
    return encoder, sess, scaler, input_name

# ── Helpers ───────────────────────────────────────────────────────────
def preprocess(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()

def build_features(e1: np.ndarray, e2: np.ndarray):
    cosine   = cosine_sim(e1, e2)
    abs_diff = np.abs(e1 - e2)
    mult     = e1 * e2
    cos_arr  = np.array([[cosine]])
    return np.hstack([e1, e2, abs_diff, mult, cos_arr]), cosine

def run_prediction(text1, text2, threshold, encoder, sess, scaler, input_name):
    e1 = encoder.encode([preprocess(text1)])
    e2 = encoder.encode([preprocess(text2)])
    features, cosine = build_features(e1, e2)
    features_scaled  = scaler.transform(features).astype(np.float32)
    prob             = float(sess.run(None, {input_name: features_scaled})[0].flat[0])
    is_plagiarised   = prob >= threshold
    return {
        "is_plagiarised":    is_plagiarised,
        "confidence":        round(prob, 4),
        "confidence_pct":    f"{prob * 100:.2f}%",
        "cosine_similarity": round(cosine, 4),
        "threshold_used":    threshold,
    }

# ── Load models ───────────────────────────────────────────────────────
with st.spinner("Loading models — first run takes ~30 s..."):
    try:
        encoder, sess, scaler, input_name = load_models()
        models_ready = True
    except Exception as e:
        models_ready = False
        model_error  = str(e)

# ── Sidebar ───────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Settings")
    threshold = st.slider(
        "Decision Threshold",
        min_value=0.30, max_value=0.95,
        value=0.60, step=0.05,
        help="Probability above which text is flagged as plagiarised.",
    )
    st.markdown("---")

    if models_ready:
        st.markdown('<p class="status-ok">&#9679; Models Loaded &mdash; Ready</p>', unsafe_allow_html=True)
    else:
        st.markdown('<p class="status-err">&#9679; Model Load Failed</p>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### Model Info")
    st.markdown("- **Encoder:** all-mpnet-base-v2")
    st.markdown("- **Classifier:** DNN (512&rarr;256&rarr;128)")
    st.markdown("- **Features:** 3073-dim")
    st.markdown("- **CV F1:** 94.19% &plusmn; 1.82%")

    st.markdown("---")
    if st.button("🗑️ Clear History"):
        st.session_state.history = []
        st.rerun()

# ── Hero header ───────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <h1>&#128269; Plagiarism Detector</h1>
    <p>Semantic similarity powered by MPNet Transformers + Deep Neural Network</p>
</div>
""", unsafe_allow_html=True)

if not models_ready:
    st.error(f"Could not load models: {model_error}\n\n"
             "Ensure `artifacts/plagiarism_dnn.onnx` and `artifacts/feature_scaler.pkl` exist.")
    st.stop()

# ── Input area ────────────────────────────────────────────────────────
col1, col2 = st.columns(2, gap="large")
with col1:
    st.markdown("#### 📄 Original Text")
    text1 = st.text_area(
        label="text1", label_visibility="collapsed",
        placeholder="Paste or type the original text here...",
        height=220, key="text1",
    )
with col2:
    st.markdown("#### 📝 Suspicious Text")
    text2 = st.text_area(
        label="text2", label_visibility="collapsed",
        placeholder="Paste or type the text to check for plagiarism...",
        height=220, key="text2",
    )

_, btn_col, _ = st.columns([1, 2, 1])
with btn_col:
    analyze = st.button("🔎 Analyze Texts", use_container_width=True)

# ── Prediction ────────────────────────────────────────────────────────
if analyze:
    if not text1.strip() or not text2.strip():
        st.warning("Please enter text in both fields before analyzing.")
    else:
        with st.spinner("Analyzing..."):
            try:
                result = run_prediction(
                    text1, text2, threshold,
                    encoder, sess, scaler, input_name
                )

                badge_class = "verdict-plagiarised" if result["is_plagiarised"] else "verdict-clean"
                icon = "&#9888;&#65039; PLAGIARISED" if result["is_plagiarised"] else "&#9989; ORIGINAL"

                st.markdown(f"""
                <div class="result-card">
                    <div style="text-align:center">
                        <span class="{badge_class}">{icon}</span>
                    </div>
                    <div class="metric-row">
                        <div class="metric-chip">
                            <div class="label">Confidence</div>
                            <div class="value">{result['confidence_pct']}</div>
                        </div>
                        <div class="metric-chip">
                            <div class="label">Cosine Similarity</div>
                            <div class="value">{result['cosine_similarity']:.4f}</div>
                        </div>
                        <div class="metric-chip">
                            <div class="label">Threshold Used</div>
                            <div class="value">{result['threshold_used']}</div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                conf = result["confidence"]
                bar_color = "#ef4444" if result["is_plagiarised"] else "#10b981"
                st.markdown("#### Confidence Level")
                st.markdown(f"""
                <div style="background:rgba(255,255,255,0.07);border-radius:10px;height:18px;overflow:hidden;margin-top:0.3rem;">
                    <div style="width:{conf*100:.1f}%;height:100%;background:{bar_color};border-radius:10px;
                                transition:width 0.5s ease;"></div>
                </div>
                <p style="color:#94a3b8;font-size:0.85rem;margin-top:0.3rem;">
                    Raw probability: {conf:.4f} (threshold: {threshold})
                </p>
                """, unsafe_allow_html=True)

                display_icon = "PLAGIARISED" if result["is_plagiarised"] else "ORIGINAL"
                st.session_state.history.insert(0, {
                    "Result": display_icon,
                    "Confidence": result["confidence_pct"],
                    "Cosine Sim": f"{result['cosine_similarity']:.4f}",
                    "Text 1 (preview)": text1[:50] + "..." if len(text1) > 50 else text1,
                    "Text 2 (preview)": text2[:50] + "..." if len(text2) > 50 else text2,
                })

            except Exception as e:
                st.error(f"Inference error: {e}")

# ── History ───────────────────────────────────────────────────────────
if st.session_state.history:
    st.markdown("---")
    st.markdown("### 📋 Session History")
    df = pd.DataFrame(st.session_state.history)
    st.dataframe(df, use_container_width=True, hide_index=True)
