import csv
import joblib
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

CSV_FILE = 'data.csv'
MODEL_FILE = 'sign_model.pkl'


def main():
    X, y = [], []
    with open(CSV_FILE, mode='r', encoding='utf-8') as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            if len(row) == 208:
                y.append(row[0])
                X.append([float(v) for v in row[1:]])

    X = np.array(X, dtype=np.float32)
    y = np.array(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y
    )

    # Add 3x slight landmark jitter to training data for robustness against camera angle/distance shifts
    rng = np.random.default_rng(42)
    aug_X = [X_train]
    aug_y = [y_train]
    for noise_std in (0.012, 0.022):
        noise = rng.normal(0.0, noise_std, size=X_train.shape).astype(np.float32)
        # Keep zeroed inactive hand slots exactly zero
        zero_mask = (X_train == 0.0)
        noisy_sample = X_train + noise
        noisy_sample[zero_mask] = 0.0
        aug_X.append(noisy_sample)
        aug_y.append(y_train)

    X_train_aug = np.vstack(aug_X)
    y_train_aug = np.concatenate(aug_y)

    clf = make_pipeline(
        StandardScaler(),
        MLPClassifier(
            hidden_layer_sizes=(256, 128),
            activation='relu',
            alpha=0.01,
            max_iter=350,
            early_stopping=True,
            random_state=42
        )
    )
    clf.fit(X_train_aug, y_train_aug)

    preds = clf.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"\nTrained on {len(X_train_aug)} augmented samples across {len(set(y))} classes.")
    print(f"Validation Accuracy: {acc * 100:.2f}%\n")
    print(classification_report(y_test, preds, zero_division=0))

    joblib.dump(clf, MODEL_FILE)
    print(f"Saved model to '{MODEL_FILE}'.")


if __name__ == '__main__':
    main()