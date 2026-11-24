import json
import os
import joblib
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

DATASET_FILE = 'bangla_dataset.json'
MODEL_FILE = 'sign_model.pkl'


def augment_frames(frames, copies=8):
    arr = np.array(frames, dtype=float)
    out = [arr]
    rng = np.random.default_rng(42)
    for _ in range(copies):
        noise = rng.normal(0.0, 0.012, size=arr.shape)
        scale = rng.uniform(0.94, 1.06)
        out.append(arr * scale + noise)
    return np.vstack(out)


def main():
    if not os.path.isfile(DATASET_FILE):
        print(f"Missing '{DATASET_FILE}'. Run 'python collect.py' first to record gestures.")
        return

    with open(DATASET_FILE, 'r', encoding='utf-8') as f:
        dataset = json.load(f)

    signs = dataset.get("signs", {})
    if len(signs) < 2:
        print("Please record at least 2 gestures (e.g., '_neutral_' and 'হ্যালো') before training.")
        return

    X_list, y_list = [], []
    for label, frames in signs.items():
        if not frames:
            continue
        aug = augment_frames(frames, copies=8)
        X_list.append(aug)
        y_list.extend([label] * len(aug))
        print(f"  - {label}: {len(frames)} raw frames -> {len(aug)} training samples")

    X = np.vstack(X_list)
    y = np.array(y_list)

    clf = make_pipeline(
        StandardScaler(),
        MLPClassifier(
            hidden_layer_sizes=(192, 96),
            activation='relu',
            max_iter=600,
            random_state=42
        )
    )
    clf.fit(X, y)
    joblib.dump(clf, MODEL_FILE)
    print(f"\n[Done] Trained '{MODEL_FILE}' successfully with {len(signs)} gesture classes!")


if __name__ == '__main__':
    main()