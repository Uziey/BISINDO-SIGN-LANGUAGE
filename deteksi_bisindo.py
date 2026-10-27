"""Jalankan file ini untuk membuka kamera dan mencoba model sendiri."""
import argparse
from collections import deque
from data_training import MODEL, detector, extract, draw, camera

CAMERA_INDEX = 0
AMBANG_SKOR = 0.70

def detect(args):
    import cv2
    import numpy as np
    import joblib
    if not MODEL.exists():
        raise ValueError("Model belum ada. Jalankan data_training.py dan kumpulkan minimal 2 kelas.")
    bundle = joblib.load(MODEL)  # Hanya buka model hasil training sendiri.
    if not isinstance(bundle, dict):
        raise ValueError("Format model tidak valid. Latih ulang.")
    if bundle.get("feature_version") != 2:
        raise ValueError("Versi fitur tidak cocok. Latih ulang.")
    model = bundle.get("model")
    if model is None or not hasattr(model, "predict_proba"):
        raise ValueError("Model tidak valid. Latih ulang.")
    allowed_raw = bundle.get("allowed_counts", {})
    allowed_counts = {}
    if isinstance(allowed_raw, dict):
        for label, counts in allowed_raw.items():
            if isinstance(counts, (list, tuple, set)):
                allowed_counts[str(label)] = {int(v) for v in counts}
            elif isinstance(counts, int):
                allowed_counts[str(label)] = {int(counts)}
    history = deque(maxlen=5)
    cap = camera(args.camera)
    try:
        with detector() as det:
            while True:
                ok, frame = cap.read()
                if not ok:
                    raise ValueError("Frame kamera gagal dibaca.")
                frame = cv2.flip(frame, 1)
                feature, hands = extract(det, frame)
                text = "Tangan belum terdeteksi / ambigu"
                if feature is None:
                    history.clear()
                else:
                    probs = model.predict_proba([feature])[0]
                    idx = int(np.argmax(probs))
                    score = float(probs[idx])
                    label = str(model.classes_[idx])
                    allowed = allowed_counts.get(label, set())
                    if score < args.threshold or label == "UNKNOWN" or len(hands) not in allowed:
                        history.clear()
                        text = "Belum dikenali"
                    else:
                        history.append(label)
                        stable = len(history) == history.maxlen and len(set(history)) == 1
                        text = f"{label} | skor {score:.2f}" if stable else "Menstabilkan..."
                draw(frame, hands)
                cv2.rectangle(frame, (0,0), (frame.shape[1],60), (30,30,30), -1)
                cv2.putText(frame, text, (15,40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,255), 2)
                cv2.imshow("Deteksi huruf - Q untuk keluar", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
    finally:
        cap.release()
        cv2.destroyAllWindows()



def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", type=int, default=CAMERA_INDEX)
    parser.add_argument("--threshold", type=float, default=AMBANG_SKOR)
    args = parser.parse_args()
    if not 0 <= args.threshold <= 1:
        parser.error("threshold harus antara 0 dan 1")
    try:
        detect(args)
    except (ValueError, OSError, RuntimeError) as exc:
        parser.exit(1, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
