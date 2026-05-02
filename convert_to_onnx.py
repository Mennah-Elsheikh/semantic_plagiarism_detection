"""
Convert plagiarism_dnn.keras → artifacts/plagiarism_dnn.onnx
Uses the SavedModel path (more reliable with newer Keras/TF versions).
Run once locally: python convert_to_onnx.py
"""
import os, sys
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
from pathlib import Path
import numpy as np
import tempfile

ROOT       = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "artifacts" / "plagiarism_dnn.keras"
ONNX_PATH  = ROOT / "artifacts" / "plagiarism_dnn.onnx"

print("[1/4] Loading Keras model...")
import tensorflow as tf
model = tf.keras.models.load_model(str(MODEL_PATH))

# Save as SavedModel format in a temp dir
with tempfile.TemporaryDirectory() as tmp:
    saved_model_dir = tmp + "/saved_model"
    print(f"[2/4] Exporting to SavedModel -> {saved_model_dir}")
    model.export(saved_model_dir)

    print("[3/4] Converting SavedModel -> ONNX (opset 13)...")
    import subprocess
    result = subprocess.run(
        [
            sys.executable, "-m", "tf2onnx.convert",
            "--saved-model", saved_model_dir,
            "--output", str(ONNX_PATH),
            "--opset", "13",
        ],
        capture_output=True, text=True
    )
    print(result.stdout)
    if result.returncode != 0:
        print("STDERR:", result.stderr)
        sys.exit(result.returncode)

size_mb = ONNX_PATH.stat().st_size / 1e6
print(f"[4/4] ONNX model saved -> {ONNX_PATH} ({size_mb:.1f} MB)")

# ── Sanity check ──────────────────────────────────────────────────────
print("\n[Sanity check] Running dummy inference with onnxruntime...")
import onnxruntime as ort
sess  = ort.InferenceSession(str(ONNX_PATH))
inp   = sess.get_inputs()[0]
print(f"  Input name : {inp.name}  shape: {inp.shape}")
dummy = np.zeros((1, 3073), dtype=np.float32)
out   = sess.run(None, {inp.name: dummy})[0]
print(f"  Output shape: {out.shape}, value: {out.flat[0]:.6f}  ✓")
