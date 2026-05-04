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
html, body, [class*="css"] {
    font-family: Arial, sans-serif;
}

/* Cosmic dark background */
.stApp {
    background: #0a0e27;
}

/* Hero header */
.hero {
    text-align: center;
    padding: 2rem 1.5rem;
    border: 2px solid rgba(59, 130, 246, 0.3);
    border-radius: 2px;
    background: linear-gradient(135deg, rgba(59, 130, 246, 0.05), rgba(139, 92, 246, 0.05));
    margin-bottom: 2rem;
}
.hero h1 {
    font-size: 2rem;
    font-weight: 400;
    color: #3B82F6;
    margin-bottom: 0.5rem;
    text-shadow: 0 0 20px rgba(59, 130, 246, 0.6);
    text-transform: uppercase;
    letter-spacing: 2px;
}
.hero p {
    color: #9ca3af;
    font-size: 0.875rem;
    text-transform: uppercase;
    letter-spacing: 1px;
    font-family: 'Courier New', monospace;
}

/* Result card */
.result-card {
    background: rgba(59, 130, 246, 0.05);
    border: 2px solid rgba(59, 130, 246, 0.3);
    border-radius: 2px;
    padding: 1.5rem;
    margin-top: 1.5rem;
    box-shadow: 0 0 20px rgba(59, 130, 246, 0.1);
}

/* Verdict badges */
.verdict-plagiarised {
    display: inline-block;
    background: rgba(220, 38, 38, 0.3);
    color: #fca5a5;
    font-size: 1rem;
    font-weight: 400;
    padding: 0.5rem 1rem;
    border-radius: 2px;
    border: 1px solid #DC2626;
    letter-spacing: 1px;
    margin-bottom: 1rem;
    text-transform: uppercase;
}
.verdict-clean {
    display: inline-block;
    background: rgba(22, 163, 74, 0.3);
    color: #86efac;
    font-size: 1rem;
    font-weight: 400;
    padding: 0.5rem 1rem;
    border-radius: 2px;
    border: 1px solid #16A34A;
    letter-spacing: 1px;
    margin-bottom: 1rem;
    text-transform: uppercase;
}

/* Metric chips */
.metric-row {
    display: flex;
    gap: 1rem;
    flex-wrap: wrap;
    margin-top: 1rem;
}
.metric-chip {
    background: rgba(59, 130, 246, 0.1);
    border: 1px solid rgba(59, 130, 246, 0.3);
    border-radius: 2px;
    padding: 0.75rem 1rem;
    flex: 1;
    min-width: 120px;
}
.metric-chip .label {
    color: #9ca3af;
    font-size: 0.65rem;
    font-weight: 400;
    text-transform: uppercase;
    letter-spacing: 1px;
    font-family: 'Courier New', monospace;
}
.metric-chip .value {
    color: #3B82F6;
    font-size: 1.25rem;
    font-weight: 400;
    margin-top: 0.25rem;
    text-shadow: 0 0 8px rgba(59, 130, 246, 0.6);
}

/* Text areas */
textarea {
    background: rgba(59, 130, 246, 0.05) !important;
    border: 2px solid rgba(59, 130, 246, 0.3) !important;
    border-radius: 2px !important;
    color: #e5e7eb !important;
    font-family: 'Courier New', monospace !important;
    font-size: 0.875rem !important;
}
textarea:focus {
    border-color: #3B82F6 !important;
    background: rgba(59, 130, 246, 0.1) !important;
    box-shadow: 0 0 16px rgba(59, 130, 246, 0.4) !important;
}

/* Button */
.stButton button {
    background: rgba(59, 130, 246, 0.2) !important;
    color: #3B82F6 !important;
    font-weight: 400 !important;
    font-size: 0.75rem !important;
    border: 2px solid #3B82F6 !important;
    border-radius: 2px !important;
    padding: 0.75rem 1.5rem !important;
    transition: all 0.2s !important;
    text-transform: uppercase !important;
    letter-spacing: 1px !important;
    font-family: Arial, sans-serif !important;
}
.stButton button:hover {
    background: rgba(59, 130, 246, 0.3) !important;
    box-shadow: 0 0 20px rgba(59, 130, 246, 0.6) !important;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #0f1a3a !important;
    border-right: 2px solid rgba(59, 130, 246, 0.3) !important;
}
section[data-testid="stSidebar"] h2 {
    color: #3B82F6 !important;
    text-transform: uppercase !important;
    letter-spacing: 2px !important;
    font-weight: 400 !important;
}

/* Sliders */
.stSlider > label {
    color: #9ca3af !important;
    font-size: 0.75rem !important;
    text-transform: uppercase !important;
    letter-spacing: 1px !important;
    font-family: 'Courier New', monospace !important;
}

/* Status indicators */
.status-ok   { color: #16A34A !important; font-weight: 400 !important; }
.status-err  { color: #DC2626 !important; font-weight: 400 !important; }

/* History table */
.stDataFrame {
    border-radius: 2px !important;
    overflow: hidden !important;
}
.stDataFrame tbody tr:hover {
    background-color: rgba(59, 130, 246, 0.1) !important;
}

/* Headings */
h1, h2, h3, h4, h5, h6 {
    font-family: Arial, sans-serif !important;
    letter-spacing: 1px !important;
}

/* Confidence bar */
.confidence-bar {
    background: rgba(59, 130, 246, 0.1);
    border: 2px solid rgba(59, 130, 246, 0.3);
    border-radius: 2px;
    height: 24px;
    overflow: hidden;
    margin-top: 0.5rem;
}
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
    st.markdown("## Settings")
    threshold = st.slider(
        "Decision Threshold",
        min_value=0.30, max_value=0.95,
        value=0.60, step=0.05,
        help="Probability above which text is flagged as plagiarised.",
    )
    st.markdown("---")

    if models_ready:
        st.markdown('<p class="status-ok">Models Loaded — Ready</p>', unsafe_allow_html=True)
    else:
        st.markdown('<p class="status-err">Model Load Failed</p>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### Model Info")
    st.markdown("- **Encoder:** all-mpnet-base-v2")
    st.markdown("- **Classifier:** DNN (512→256→128)")
    st.markdown("- **Features:** 3073-dim")
    st.markdown("- **CV F1:** 94.19% ± 1.82%")

    st.markdown("---")
    if st.button("Clear History"):
        st.session_state.history = []
        st.rerun()

# ── Hero header ───────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <h1>Plagiarism Detector</h1>
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
    st.markdown("#### Original Text")
    text1 = st.text_area(
        label="text1", label_visibility="collapsed",
        placeholder="Paste or type the original text here...",
        height=220, key="text1",
    )
with col2:
    st.markdown("#### Suspicious Text")
    text2 = st.text_area(
        label="text2", label_visibility="collapsed",
        placeholder="Paste or type the text to check for plagiarism...",
        height=220, key="text2",
    )

_, btn_col, _ = st.columns([1, 2, 1])
with btn_col:
    analyze = st.button("Analyze Texts", use_container_width=True)

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
                icon = "PLAGIARISED" if result["is_plagiarised"] else "ORIGINAL"

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
    st.markdown("### Session History")
    df = pd.DataFrame(st.session_state.history)
    st.dataframe(df, use_container_width=True, hide_index=True)
