"""Dataset pribadi -> fitur skeleton -> Random Forest -> webcam."""
from pathlib import Path
import argparse
import time
import uuid
from collections import deque
import urllib.request
import json

# UBAH DI SINI jika tidak menggunakan opsi terminal.
HURUF = "B"                 # Ganti A, B, C, ... Z atau UNKNOWN
JUMLAH_FOTO = 100            # Jumlah foto BARU setiap sesi: 40 / 50 / 100
JUMLAH_TANGAN = 2           # 1 atau 2 sesuai pose huruf yang direkam
TRAIN_SETELAH_REKAM = True
CAMERA_INDEX = 0
INTERVAL_FOTO = 0.5          # Jeda detik antar foto
AMBANG_SKOR = 0.70           # Skor model, BUKAN jaminan probabilitas benar

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "dataset"
ASSET = ROOT / "hand_landmarker.task"
MODEL = ROOT / "model_bisindo.joblib"
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
EDGES = [(0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),
         (5,9),(9,10),(10,11),(11,12),(9,13),(13,14),(14,15),(15,16),
         (13,17),(0,17),(17,18),(18,19),(19,20)]


def setup():
    if ASSET.exists():
        print("Model skeleton sudah tersedia.")
        return
    print("Mengunduh model skeleton Google; bukan dataset huruf.")
    tmp = ASSET.with_suffix(".tmp")
    try:
        urllib.request.urlretrieve(MODEL_URL, tmp)
        tmp.replace(ASSET)
    finally:
        tmp.unlink(missing_ok=True)
    print("Selesai:", ASSET)


def detector():
    import mediapipe as mp
    if not ASSET.exists():
        raise ValueError("Jalankan data_training.py untuk mengunduh model skeleton.")
    return mp.tasks.vision.HandLandmarker.create_from_options(
        mp.tasks.vision.HandLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(ASSET)),
            running_mode=mp.tasks.vision.RunningMode.IMAGE,
            num_hands=2, min_hand_detection_confidence=0.6,
            min_hand_presence_confidence=0.6))


def extract(det, frame):
    import cv2
    import mediapipe as mp
    import numpy as np
    result = det.detect(mp.Image(image_format=mp.ImageFormat.SRGB,
                                data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
    feature = features_from_result(result, frame.shape[0], frame.shape[1])
    return feature, result.hand_landmarks


def features_from_result(result, height, width):
    """128 fitur: 2 mask + 2 x 21 x 3, slot Left/Right konsisten.

    XY relatif titik tengah pergelangan, mempertahankan jarak antar tangan.
    Z relatif pergelangan masing-masing: bukan kedalaman global antar tangan.
    """
    import numpy as np
    if not 1 <= len(result.hand_landmarks) <= 2:
        return None
    slots = {}
    for hand, categories in zip(result.hand_landmarks, result.handedness):
        if not categories or categories[0].score < 0.6:
            return None
        side = categories[0].category_name
        if side not in ("Left", "Right") or side in slots:
            return None  # Jangan menebak slot saat handedness ambigu.
        points = np.array([[q.x, q.y * height / width, q.z] for q in hand], dtype=np.float32)
        slots[side] = points
    if len(slots) != len(result.hand_landmarks):
        return None
    center = np.mean([pts[0, :2] for pts in slots.values()], axis=0)
    scale = np.mean([np.linalg.norm(pts[9, :2]-pts[0, :2]) for pts in slots.values()])
    if scale < 1e-6:
        return None
    masks, vectors = [], []
    for side in ("Left", "Right"):
        masks.append(float(side in slots))
        pts = np.zeros((21,3), dtype=np.float32)
        if side in slots:
            pts = slots[side].copy()
            pts[:, :2] -= center
            pts[:, 2] -= pts[0, 2]
            pts /= scale
        vectors.append(pts.reshape(-1))
    return np.concatenate([np.asarray(masks, dtype=np.float32), *vectors])


def draw(frame, hands):
    import cv2
    h, w = frame.shape[:2]
    for hand in hands:
        pts = [(int(p.x*w), int(p.y*h)) for p in hand]
        for a,b in EDGES:
            cv2.line(frame, pts[a], pts[b], (0,220,0), 2)
        for p in pts:
            cv2.circle(frame, p, 4, (0,80,255), -1)


def camera(index):
    import cv2
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        cap.release()
        raise ValueError("Kamera gagal dibuka. Tutup aplikasi kamera lain atau gunakan --camera 1.")
    return cap


def collect(args):
    import cv2
    label = args.label.upper()
    if label not in list("ABCDEFGHIJKLMNOPQRSTUVWXYZ") + ["UNKNOWN"]:
        raise ValueError("Label harus A-Z atau UNKNOWN.")
    if args.count < 1 or args.interval <= 0:
        raise ValueError("Jumlah foto dan interval harus positif.")
    if args.hands not in (1, 2):
        raise ValueError("JUMLAH_TANGAN harus 1 atau 2")
    session = time.strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]
    dest = DATA / label / session
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "session.json").write_text(json.dumps({"label": label, "hands": args.hands,
        "mirror": True, "feature_version": 2}), encoding="utf-8")
    saved, active, last = 0, False, 0.0
    cap = camera(args.camera)
    try:
        with detector() as det:
            while saved < args.count:
                ok, frame = cap.read()
                if not ok:
                    raise ValueError("Frame kamera gagal dibaca.")
                frame = cv2.flip(frame, 1)  # Sama persis untuk collect dan detect.
                features, hands = extract(det, frame)
                if active and features is not None and len(hands) == args.hands and time.monotonic()-last >= args.interval:
                    path = dest / f"{label}_{saved+1:04d}.jpg"
                    if not cv2.imwrite(str(path), frame):
                        raise OSError(f"Gagal menyimpan {path}")
                    saved += 1
                    last = time.monotonic()
                # Skeleton hanya untuk preview; foto tersimpan tetap bersih.
                draw(frame, hands)
                status = "REKAM" if active else "JEDA"
                cv2.putText(frame, f"{label}: {saved}/{args.count} | {status}", (15,35),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
                cv2.putText(frame, f"S: mulai/jeda | Q: keluar | tangan {len(hands)}/{args.hands}", (15,65),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 2)
                cv2.imshow("Kumpulkan dataset", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                if key == ord('s'):
                    active = not active
                    last = time.monotonic()  # Waktu bersiap setelah menekan S.
    finally:
        cap.release()
        cv2.destroyAllWindows()
    print(f"Tersimpan {saved} foto baru di {dest}")
    return saved


def train():
    import cv2
    import numpy as np
    import joblib
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import classification_report, accuracy_score
    X, y, groups = [], [], []
    skipped = 0
    allowed_counts = {}
    session_meta = {}
    with detector() as det:
        for path in sorted(DATA.glob("*/*/*.jpg")):
            label = path.parent.parent.name
            if label not in list("ABCDEFGHIJKLMNOPQRSTUVWXYZ") + ["UNKNOWN"]:
                continue
            if path.parent not in session_meta:
                meta_path = path.parent / "session.json"
                session_meta[path.parent] = json.loads(meta_path.read_text()) if meta_path.exists() else {}
            meta = session_meta[path.parent]
            if meta.get("feature_version") != 2 or meta.get("hands") not in (1,2):
                skipped += 1
                continue
            frame = cv2.imread(str(path))
            feature, hands = extract(det, frame) if frame is not None else (None, [])
            if feature is None or len(hands) != meta["hands"]:
                skipped += 1
                continue
            allowed_counts.setdefault(label, set()).add(meta["hands"])
            X.append(feature)
            y.append(label)
            groups.append(path.parent.name)
    if len(set(y)) < 2:
        print("Foto tersimpan. Model belum dilatih: perlu minimal 2 kelas, misalnya A dan B.")
        return False
    X, y, groups = np.asarray(X), np.asarray(y), np.asarray(groups)
    counts = {str(c): int(sum(y == c)) for c in np.unique(y)}
    print("Foto valid per kelas:", counts, "| Dilewati:", skipped)
    if min(counts.values()) < 10:
        raise ValueError("Setiap kelas perlu minimal 10 foto valid. Targetkan 50-100 atau lebih.")
    def new_model():
        return RandomForestClassifier(n_estimators=300, random_state=42,
                                      class_weight="balanced", n_jobs=-1)
    report = {"valid_counts": counts, "skipped": skipped}
    # Holdout seluruh sesi, bukan frame bertetangga yang hampir identik.
    sessions = {c: sorted(set(groups[y == c])) for c in np.unique(y)}
    if all(len(s) >= 2 for s in sessions.values()):
        test = np.zeros(len(y), dtype=bool)
        for c, s in sessions.items():
            test |= (y == c) & (groups == s[-1])
        evaluation_model = new_model().fit(X[~test], y[~test])
        pred = evaluation_model.predict(X[test])
        report["holdout_session_accuracy"] = float(accuracy_score(y[test], pred))
        report["classification_report"] = classification_report(y[test], pred, output_dict=True, zero_division=0)
        print("Evaluasi pada sesi terakhir per kelas (sebelum final refit):")
        print(classification_report(y[test], pred, zero_division=0))
        print("Accuracy:", report["holdout_session_accuracy"])
    else:
        report["evaluation"] = "Belum tersedia: perlu >=2 sesi valid per kelas."
        print(report["evaluation"])
    final = new_model().fit(X, y)
    temporary = MODEL.with_suffix(".tmp")
    joblib.dump({"model": final, "feature_version": 2,
                 "allowed_counts": {k: sorted(v) for k,v in allowed_counts.items()}}, temporary)
    temporary.replace(MODEL)
    (ROOT / "training_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("Model final dilatih pada semua foto valid:", MODEL)
    print("Huruf tersedia:", ", ".join(final.classes_))


    missing = sorted(set("ABCDEFGHIJKLMNOPQRSTUVWXYZ") - set(final.classes_))
    print("Belum tersedia:", ", ".join(missing) if missing else "Semua A-Z tersedia")
    return True


def main():
    parser = argparse.ArgumentParser(description="Ambil foto dan training BISINDO satu/dua tangan")
    parser.add_argument("--label", default=HURUF)
    parser.add_argument("--count", type=int, default=JUMLAH_FOTO)
    parser.add_argument("--hands", type=int, choices=[1,2], default=JUMLAH_TANGAN)
    parser.add_argument("--camera", type=int, default=CAMERA_INDEX)
    parser.add_argument("--interval", type=float, default=INTERVAL_FOTO)
    parser.add_argument("--train-only", action="store_true")
    args = parser.parse_args()
    try:
        setup()
        if args.train_only:
            train()
        elif collect(args) and TRAIN_SETELAH_REKAM:
            train()
    except (ValueError, OSError, RuntimeError) as exc:
        parser.exit(1, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
