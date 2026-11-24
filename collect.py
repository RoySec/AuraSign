import json
import os
import time
import cv2
import mediapipe as mp
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from features import extract_holistic_features

DATASET_FILE = 'bangla_dataset.json'
RECORD_SECONDS = 5.0

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20)
]


def load_bangla_font(size=24):
    candidates = [
        r"C:\Windows\Fonts\Nirmala.ttf",
        r"C:\Windows\Fonts\NirmalaB.ttf",
        r"C:\Windows\Fonts\vrinda.ttf",
        r"C:\Windows\Fonts\kalpurush.ttf",
    ]
    for path in candidates:
        if os.path.isfile(path):
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


FONT_BANGLA = load_bangla_font(24)


def draw_banner(frame, text_lines, bg_color=(15, 23, 42)):
    h, w, _ = frame.shape
    banner_h = 34 * len(text_lines) + 18
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, banner_h), bg_color, -1)
    cv2.addWeighted(overlay, 0.82, frame, 0.18, 0, frame)

    pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)
    for idx, (line, color) in enumerate(text_lines):
        draw.text((16, 10 + idx * 34), line, font=FONT_BANGLA, fill=color)
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


def draw_skeleton(frame, hand_res):
    if not hand_res or not hand_res.hand_landmarks:
        return
    h, w, _ = frame.shape
    for hand in hand_res.hand_landmarks:
        pts = [(int(lm.x * w), int(lm.y * h)) for lm in hand]
        for a, b in HAND_CONNECTIONS:
            cv2.line(frame, pts[a], pts[b], (0, 230, 130), 2, cv2.LINE_AA)
        for cx, cy in pts:
            cv2.circle(frame, (cx, cy), 4, (56, 189, 248), -1)


def load_dataset():
    if os.path.isfile(DATASET_FILE):
        with open(DATASET_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {"signs": {}}


def save_dataset(data):
    with open(DATASET_FILE, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def main():
    dataset = load_dataset()
    saved_keys = list(dataset["signs"].keys())
    print("\n========================================================")
    print("  AuraSign Bangla — 5-Second Gesture Motion Recorder")
    print("========================================================")
    print(f"Saved signs ({len(saved_keys)}): {saved_keys if saved_keys else 'None yet'}")
    print("Tip: Type '_neutral_' once to record resting/idle hands.\n")

    sign_label = input("Enter Bangla gesture name (e.g. হ্যালো / সাহায্য / ধন্যবাদ): ").strip()
    if not sign_label:
        print("No label entered. Cancelled.")
        return

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW) if os.name == 'nt' else cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    hand_detector = vision.HandLandmarker.create_from_options(
        vision.HandLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path='hand_landmarker.task'),
            num_hands=2,
            min_hand_detection_confidence=0.60,
            min_tracking_confidence=0.60
        )
    )
    pose_detector = vision.PoseLandmarker.create_from_options(
        vision.PoseLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path='pose_landmarker_lite.task'),
            num_poses=1,
            min_pose_detection_confidence=0.60
        )
    )

    recording = False
    record_start_time = 0.0
    countdown_start = None
    captured_frames = []

    print(f"\nCamera ready! Press [SPACE] in the camera window to record '{sign_label}' for {RECORD_SECONDS:.0f}s, or [Q] to quit.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                continue

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            hand_res = hand_detector.detect(mp_img)
            pose_res = pose_detector.detect(mp_img)

            draw_skeleton(frame, hand_res)

            if countdown_start is not None:
                rem = 2.0 - (time.time() - countdown_start)
                if rem > 0:
                    frame = draw_banner(frame, [
                        (f"Target Sign: {sign_label}", (56, 189, 248)),
                        (f"Get ready... 5s recording starts in {rem:.1f}s", (250, 204, 21))
                    ])
                else:
                    countdown_start = None
                    recording = True
                    record_start_time = time.time()
                    captured_frames = []

            elif recording:
                elapsed = time.time() - record_start_time
                remaining = max(0.0, RECORD_SECONDS - elapsed)

                if sign_label == '_neutral_' or hand_res.hand_landmarks:
                    feat = extract_holistic_features(hand_res, pose_res)
                    captured_frames.append([round(float(x), 5) for x in feat])

                frame = draw_banner(frame, [
                    (f"Recording '{sign_label}': {remaining:.1f}s left ({len(captured_frames)} frames)", (52, 211, 153)),
                    ("Perform the gesture clearly for 5 seconds...", (248, 250, 252))
                ], bg_color=(6, 45, 32))

                if elapsed >= RECORD_SECONDS:
                    if len(captured_frames) >= 5:
                        dataset["signs"][sign_label] = captured_frames
                        save_dataset(dataset)
                        print(f"[Saved] Recorded {len(captured_frames)} frames over 5.0s for '{sign_label}'.")
                    else:
                        print("[Warning] Too few hand frames detected during 5.0s. Try again with hands visible.")
                    recording = False
                    break
            else:
                frame = draw_banner(frame, [
                    (f"Ready to record (5s): {sign_label}", (56, 189, 248)),
                    ("Press [SPACE] to start 5s recording | [Q] to cancel", (226, 232, 240))
                ])

            cv2.imshow("AuraSign Bangla Collector", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord(' ') and not recording and countdown_start is None:
                countdown_start = time.time()
            elif key in (ord('q'), ord('Q'), 27):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()