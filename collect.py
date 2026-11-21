import csv
import json
import os
import time

os.environ['GLOG_minloglevel'] = '2'

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from features import extract_holistic_features

CSV_FILE = 'data.csv'
AVATAR_FILE = 'avatar_poses.json'
SAMPLES_PER_SIGN = 100

DEFAULT_SIGNS = [
    '_neutral_',
    'hello',
    'yes',
    'no',
    'thank_you',
    'i_love_you',
    'help',
    'pain',
    'doctor',
    'water',
    'please'
]

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20)
]


def draw_skeleton(frame, hand_res):
    h, w, _ = frame.shape
    if hand_res and hand_res.hand_landmarks:
        for hand in hand_res.hand_landmarks:
            pts = [(int(lm.x * w), int(lm.y * h)) for lm in hand]
            for a, b in HAND_CONNECTIONS:
                cv2.line(frame, pts[a], pts[b], (0, 235, 130), 2, cv2.LINE_AA)
            for cx, cy in pts:
                cv2.circle(frame, (cx, cy), 3, (255, 255, 255), -1)


def load_existing_dataset():
    existing_rows = []
    existing_labels = set()
    if os.path.isfile(CSV_FILE):
        with open(CSV_FILE, mode='r', encoding='utf-8') as f:
            reader = csv.reader(f)
            next(reader, None)
            for row in reader:
                if len(row) == 208:
                    existing_rows.append(row)
                    existing_labels.add(row[0])

    avatar_poses = {}
    if os.path.isfile(AVATAR_FILE):
        with open(AVATAR_FILE, mode='r', encoding='utf-8') as f:
            avatar_poses = json.load(f)

    return existing_rows, existing_labels, avatar_poses


def main():
    print("=== AuraSign Multi-Language Dataset Collector ===")
    existing_rows, existing_labels, avatar_poses = load_existing_dataset()

    append_mode = False
    if existing_labels:
        saved_words = sorted(w for w in existing_labels if w != '_neutral_')
        print(f"Existing trained signs ({len(saved_words)}): {', '.join(saved_words)}")
        choice = input("Add/update signs into existing dataset [A] or Wipe & start fresh [W]? (A/W): ").strip().lower()
        append_mode = (choice != 'w')

    if not append_mode:
        existing_rows = []
        existing_labels = set()
        avatar_poses = {}

    custom = input(
        "Type new comma-separated words/phrases (any language, e.g. kemon_acho, bhalo, sorry) or press ENTER for defaults: "
    ).strip()

    if custom:
        words = [w.strip().lower().replace(' ', '_') for w in custom.split(',') if w.strip()]
        # Only prompt for _neutral_ if it is not already in the dataset
        if '_neutral_' not in existing_labels and '_neutral_' not in words:
            signs = ['_neutral_'] + words
        else:
            signs = words
    else:
        signs = [s for s in DEFAULT_SIGNS if not append_mode or s not in existing_labels]
        if not signs:
            print("All default signs are already recorded! Run again and type new custom words.")
            return

    hand_detector = vision.HandLandmarker.create_from_options(
        vision.HandLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path='hand_landmarker.task'),
            num_hands=2,
            min_hand_detection_confidence=0.6
        )
    )
    pose_detector = vision.PoseLandmarker.create_from_options(
        vision.PoseLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path='pose_landmarker_lite.task'),
            num_poses=1,
            min_pose_detection_confidence=0.6
        )
    )

    cap = cv2.VideoCapture(0)

    for sign in signs:
        is_neutral = (sign == '_neutral_')
        title = "NEUTRAL / RESTING HANDS (DO NOT SIGN)" if is_neutral else f"SIGN: {sign.replace('_', ' ').upper()}"
        tip = (
            "Keep hands visible near chest/lap & move naturally without signing"
            if is_neutral else
            "Hold your sign clearly; tilt/shift angle slightly during capture"
        )

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            hand_res = hand_detector.detect(mp_img)
            draw_skeleton(frame, hand_res)

            cv2.rectangle(frame, (0, 0), (frame.shape[1], 115), (18, 20, 26), -1)
            cv2.putText(frame, title, (18, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.78, (0, 230, 255), 2)
            cv2.putText(frame, tip, (18, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (200, 220, 255), 1)
            cv2.putText(frame, "Press [SPACE] to record | [S] Skip | [Q] Save & Quit", (18, 98),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (120, 255, 160), 1)
            cv2.imshow("AuraSign Collector", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord(' '):
                action = 'record'
                break
            elif key == ord('s'):
                action = 'skip'
                break
            elif key == ord('q'):
                action = 'quit'
                break

        if action == 'quit':
            break
        if action == 'skip':
            continue

        captured = []
        while len(captured) < SAMPLES_PER_SIGN:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

            hand_res = hand_detector.detect(mp_img)
            pose_res = pose_detector.detect(mp_img)
            draw_skeleton(frame, hand_res)

            if hand_res.hand_landmarks:
                feats = [round(v, 5) for v in extract_holistic_features(hand_res, pose_res)]
                captured.append(feats)

            cv2.rectangle(frame, (0, 0), (frame.shape[1], 80), (18, 20, 26), -1)
            cv2.putText(frame, f"Recording {sign.upper()}: {len(captured)}/{SAMPLES_PER_SIGN}",
                        (20, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 255, 140), 2)
            cv2.imshow("AuraSign Collector", frame)
            cv2.waitKey(1)

        if captured:
            # Replace older samples if re-recording an existing sign so the dataset stays balanced
            existing_rows = [r for r in existing_rows if r[0] != sign]
            for feats in captured:
                existing_rows.append([sign] + feats)

            if not is_neutral and len(captured) >= 4:
                n = len(captured)
                avatar_poses[sign] = [
                    captured[0],
                    captured[n // 3],
                    captured[(2 * n) // 3],
                    captured[-1]
                ]
            print(f"Saved {len(captured)} frames for '{sign}'.")
            time.sleep(0.3)

    cap.release()
    cv2.destroyAllWindows()

    with open(CSV_FILE, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['label'] + [f'f{i}' for i in range(207)])
        writer.writerows(existing_rows)

    with open(AVATAR_FILE, 'w', encoding='utf-8') as f:
        json.dump(avatar_poses, f, indent=2, ensure_ascii=False)

    total_classes = len({r[0] for r in existing_rows})
    print(f"\nDone! '{CSV_FILE}' now holds {len(existing_rows)} samples across {total_classes} classes.")
    print("Run 'python train.py' to update your model!")


if __name__ == '__main__':
    main()