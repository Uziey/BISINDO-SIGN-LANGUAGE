"""Jalankan server web penerjemah BISINDO."""
from pathlib import Path
import threading

import cv2
import joblib
import numpy as np
from flask import Flask, jsonify, request, send_from_directory

from data_training import MODEL, detector, extract

ROOT = Path(__file__).resolve().parent
app = Flask(__name__, static_folder=str(ROOT / "static"))
app.config["MAX_CONTENT_LENGTH"] = 2 * 1024 * 1024
inference_lock = threading.Lock()


@app.get("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.get("/api/info")
def info():
    model = app.config.get("model")
    if model is None:
        return jsonify(error="Model belum dimuat. Jalankan server dengan python app.py."), 503
    letters = sorted(str(label) for label in model.classes_ if str(label) != "UNKNOWN")
    return jsonify(letters=letters)


@app.post("/api/predict")
def predict():
    model = app.config.get("model")
    hand_detector = app.config.get("detector")
    if model is None or hand_detector is None:
        return jsonify(error="Model deteksi belum siap."), 503

    image_bytes = request.get_data()
    if not image_bytes:
        return jsonify(error="Gambar tidak valid atau kosong."), 400
    image = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        return jsonify(error="Gambar tidak valid atau kosong."), 400

    with inference_lock:
        feature, hands = extract(hand_detector, image)
        if feature is None:
            return jsonify(label=None, score=0, valid=False, top=[], hands=[
                [[float(point.x), float(point.y)] for point in hand]
                for hand in hands
            ])

        probabilities = model.predict_proba([feature])[0]
        best_index = int(np.argmax(probabilities))
        label = str(model.classes_[best_index])
        score = float(probabilities[best_index])
        allowed_counts = app.config["allowed_counts"]
        valid = label != "UNKNOWN" and len(hands) in allowed_counts.get(label, set())
        top_indices = np.argsort(probabilities)[::-1][:3]
        top = [
            {"label": str(model.classes_[index]), "score": float(probabilities[index])}
            for index in top_indices
        ]

    return jsonify(
        label=label,
        score=score,
        valid=valid,
        top=top,
        hands=[
            [[float(point.x), float(point.y)] for point in hand]
            for hand in hands
        ],
    )


def load_model():
    if not MODEL.exists():
        raise ValueError("Model belum ada. Jalankan data_training.py untuk melatih model.")
    bundle = joblib.load(MODEL)
    if not isinstance(bundle, dict) or bundle.get("feature_version") != 2:
        raise ValueError("Format model tidak valid atau versi fitur tidak cocok. Latih ulang.")
    model = bundle.get("model")
    if model is None or not hasattr(model, "predict_proba") or not hasattr(model, "classes_"):
        raise ValueError("Model tidak valid. Latih ulang.")

    allowed_counts = {}
    allowed_raw = bundle.get("allowed_counts", {})
    if isinstance(allowed_raw, dict):
        try:
            for label, counts in allowed_raw.items():
                if isinstance(counts, (list, tuple, set)):
                    allowed_counts[str(label)] = {int(value) for value in counts}
                elif isinstance(counts, int):
                    allowed_counts[str(label)] = {int(counts)}
        except (TypeError, ValueError) as exc:
            raise ValueError("Data jumlah tangan pada model tidak valid. Latih ulang.") from exc
    return model, allowed_counts


def main():
    try:
        model, allowed_counts = load_model()
        with detector() as hand_detector:
            app.config.update(
                model=model,
                allowed_counts=allowed_counts,
                detector=hand_detector,
            )
            print("Buka http://127.0.0.1:5000 untuk memakai penerjemah BISINDO.")
            app.run(host="127.0.0.1", port=5000, debug=False, threaded=True)
    except (ValueError, OSError, RuntimeError) as exc:
        raise SystemExit(f"Error: {exc}") from exc


if __name__ == "__main__":
    main()