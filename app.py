import asyncio
import base64
import csv
import http.server
import json
import math
import os
import socketserver
import threading
import time
import webbrowser
from collections import Counter, defaultdict, deque

os.environ['GLOG_minloglevel'] = '2'

import cv2
import joblib
import mediapipe as mp
import numpy as np
import pyttsx3
import sounddevice as sd
import speech_recognition as sr
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

try:
    import pythoncom
    import win32com.client
    HAS_SAPI = True
except ImportError:
    HAS_SAPI = False

try:
    import websockets
except ImportError:
    websockets = None
    print("[Warning] 'websockets' not installed. Run: pip install websockets")

from features import extract_holistic_features

MODEL_FILE = 'sign_model.pkl'
AVATAR_FILE = 'avatar_poses.json'
DATA_CSV_FILE = 'data.csv'
HTTP_PORT = 8000
WS_PORT = 8765

# Stage-safe anti-false-positive thresholds
MIN_CONFIDENCE = 0.78
MIN_PROB_MARGIN = 0.25
SMOOTHING_WINDOW = 8
STABLE_FRAMES_REQUIRED = 12

# Fast female TTS voice pacing
SAPI_RATE = 3
PYTTSX3_RATE = 215
WORD_GAP_SEC = 0.0

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20)
]

UPPER_BODY_CONNECTIONS = [
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (0, 11), (0, 12)
]

SYNONYMS = {
    "hi": "hello",
    "hey": "hello",
    "thanks": "thank_you",
    "assist": "help",
    "hurt": "pain",
    "ache": "pain",
    "hospital": "doctor",
    "drink": "water"
}

_tts_lock = threading.Lock()


def normalize_sign_key(text):
    """Normalizes any sign label consistently."""
    cleaned = "".join(ch for ch in text.lower() if ch.isalnum() or ch.isspace() or ch == '_')
    return "_".join(cleaned.replace('_', ' ').split())


def open_webcam_fast(cam_index=0):
    if os.name == 'nt':
        cap = cv2.VideoCapture(cam_index, cv2.CAP_DSHOW)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap.set(cv2.CAP_PROP_FPS, 30)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            return cap
        cap.release()

    cap = cv2.VideoCapture(cam_index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


def _select_female_voice_pyttsx3(engine):
    voices = engine.getProperty('voices')
    preferred = ('aria', 'jenny', 'zira', 'hazel', 'susan', 'eva', 'female')
    for keyword in preferred:
        for v in voices:
            if keyword in v.name.lower() or keyword in v.id.lower():
                engine.setProperty('voice', v.id)
                return
    if len(voices) > 1:
        engine.setProperty('voice', voices[1].id)


def speak_text_async(text):
    words = [w.strip(".,!?;:") for w in text.strip().split() if w.strip(".,!?;:")]
    if not words:
        return

    def _run():
        with _tts_lock:
            if HAS_SAPI:
                pythoncom.CoInitialize()
                try:
                    speaker = win32com.client.Dispatch("SAPI.SpVoice")
                    voices = speaker.GetVoices()
                    preferred = ('aria', 'jenny', 'zira', 'hazel', 'susan', 'eva', 'female')
                    selected = False
                    for keyword in preferred:
                        for i in range(voices.Count):
                            v = voices.Item(i)
                            if keyword in v.GetDescription().lower():
                                speaker.Voice = v
                                selected = True
                                break
                        if selected:
                            break
                    if not selected and voices.Count > 1:
                        speaker.Voice = voices.Item(1)

                    speaker.Rate = SAPI_RATE
                    speaker.Volume = 100

                    for idx, word in enumerate(words):
                        speaker.Speak(word, 0)
                        if WORD_GAP_SEC > 0 and idx < len(words) - 1:
                            time.sleep(WORD_GAP_SEC)
                except Exception as e:
                    print(f"[SAPI TTS Error]: {e}")
                finally:
                    pythoncom.CoUninitialize()
            else:
                try:
                    engine = pyttsx3.init()
                    _select_female_voice_pyttsx3(engine)
                    engine.setProperty('rate', PYTTSX3_RATE)
                    engine.setProperty('volume', 1.0)
                    engine.say(" ".join(words))
                    engine.runAndWait()
                    engine.stop()
                except Exception as e:
                    print(f"[TTS Error]: {e}")

    threading.Thread(target=_run, daemon=True).start()


def gloss_to_sentence(tokens):
    if not tokens:
        return ""
    clean = []
    for t in tokens:
        if not clean or clean[-1] != t:
            clean.append(t)

    joined = " ".join(clean)
    return joined[0].upper() + joined[1:] + "."


def draw_detection_overlay(frame, hand_res, pose_res, label="", confidence=0.0, show_dots=True):
    h, w, _ = frame.shape

    if show_dots and pose_res and pose_res.pose_landmarks:
        lms = pose_res.pose_landmarks[0]
        for a, b in UPPER_BODY_CONNECTIONS:
            pt1 = (int(lms[a].x * w), int(lms[a].y * h))
            pt2 = (int(lms[b].x * w), int(lms[b].y * h))
            cv2.line(frame, pt1, pt2, (255, 180, 0), 2, cv2.LINE_AA)

    if hand_res and hand_res.hand_landmarks:
        for idx, hand in enumerate(hand_res.hand_landmarks):
            pts = [(int(lm.x * w), int(lm.y * h)) for lm in hand]
            if show_dots:
                for a, b in HAND_CONNECTIONS:
                    cv2.line(frame, pts[a], pts[b], (0, 220, 120), 2, cv2.LINE_AA)
                for cx, cy in pts:
                    cv2.circle(frame, (cx, cy), 3, (0, 255, 150), -1)

            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            x1, y1 = max(0, min(xs) - 20), max(0, min(ys) - 20)
            x2, y2 = min(w, max(xs) + 20), min(h, max(ys) + 20)

            active = bool(label and label not in ("No Hand", "Neutral / Resting"))
            box_color = (0, 255, 140) if active else (130, 140, 155)
            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2, cv2.LINE_AA)

            if idx == 0 and active:
                badge = f"{label.upper()} {int(confidence * 100)}%"
                (tw, th), _ = cv2.getTextSize(badge, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                by1 = max(0, y1 - th - 12)
                cv2.rectangle(frame, (x1, by1), (x1 + tw + 14, y1), box_color, -1)
                cv2.putText(frame, badge, (x1 + 7, y1 - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (15, 18, 22), 2, cv2.LINE_AA)


# ==============================================================================
# 3D AVATAR DATA-DRIVEN KINEMATICS CONTROLLER
# ==============================================================================

def normalize_2d_hand_pose(raw_63):
    """Preserves exact recorded finger curl and spread while scaling to avatar hand size."""
    coords = np.array(raw_63, dtype=float).reshape(21, 3)
    xy_span = float(np.max(np.abs(coords[:, :2])))
    if xy_span < 1e-4:
        return coords.flatten()
    scale_factor = 0.86 / max(xy_span, 0.42)
    coords[:, 0] = np.clip(coords[:, 0] * scale_factor, -1.05, 1.05)
    coords[:, 1] = np.clip(coords[:, 1] * scale_factor, -1.05, 1.05)
    coords[:, 2] = np.clip(coords[:, 2], -0.60, 0.60)
    return coords.flatten()


def build_rest_hand_coords(inward_sign=1.0):
    coords = [0.0] * 63
    specs = [
        ([1, 2, 3, 4], 0.22 * inward_sign, 0.14, 0.08 * inward_sign, 0.14),
        ([5, 6, 7, 8], 0.15 * inward_sign, 0.34, 0.01 * inward_sign, 0.15),
        ([9, 10, 11, 12], 0.04 * inward_sign, 0.37, 0.0, 0.16),
        ([13, 14, 15, 16], -0.07 * inward_sign, 0.35, -0.01 * inward_sign, 0.15),
        ([17, 18, 19, 20], -0.17 * inward_sign, 0.31, -0.02 * inward_sign, 0.13),
    ]
    for joints, bx, by, dx, dy in specs:
        for step, j in enumerate(joints):
            coords[j * 3] = bx + dx * step
            coords[j * 3 + 1] = by + dy * step
    return np.array(coords, dtype=float)


def load_recorded_sequences(csv_file, json_file):
    """
    Loads the MOST RECENT recorded multi-frame sequence per sign from data.csv
    (taking the tail rows so retrained signs immediately replace older takes).
    """
    sequences = {}
    display_names = {}

    if os.path.isfile(csv_file):
        grouped = defaultdict(list)
        raw_label_map = {}
        try:
            with open(csv_file, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                header = next(reader, None)
                if header:
                    for row in reader:
                        if len(row) < 208:
                            continue
                        raw_label = row[0].strip()
                        norm_k = normalize_sign_key(raw_label)
                        if not norm_k or norm_k in ('neutral', '_neutral_'):
                            continue
                        vals = [float(x) for x in row[1:208]]
                        grouped[norm_k].append(vals)
                        raw_label_map[norm_k] = raw_label.replace('_', ' ').strip().upper()

            for norm_k, rows in grouped.items():
                arr = np.array(rows, dtype=float)
                hand_energy = np.sum(np.abs(arr[:, :63]), axis=1) + np.sum(np.abs(arr[:, 87:150]), axis=1)
                active_rows = arr[hand_energy > 0.5]
                src = active_rows if len(active_rows) >= 6 else arr
                # Take the LATEST recording at the end of data.csv (not stale earlier rows)
                clip_len = min(len(src), 24)
                clip = src[-clip_len:]
                sample_indices = np.linspace(0, len(clip) - 1, num=min(9, len(clip)), dtype=int)
                sequences[norm_k] = clip[sample_indices]
                display_names[norm_k] = raw_label_map[norm_k]
        except Exception as e:
            print(f"[Warning] Could not read {csv_file}: {e}")

    if os.path.isfile(json_file):
        with open(json_file, 'r', encoding='utf-8') as f:
            raw_poses = json.load(f)
            for k, v in raw_poses.items():
                norm_k = normalize_sign_key(k)
                if not norm_k or norm_k in ('neutral', '_neutral_'):
                    continue
                if norm_k not in sequences:
                    arr = np.array(v, dtype=float)
                    sequences[norm_k] = arr[np.newaxis, :] if arr.ndim == 1 else arr
                    display_names[norm_k] = k.replace('_', ' ').strip().upper()

    return sequences, display_names


class Avatar3DController:
    def __init__(self, pose_file, csv_file=DATA_CSV_FILE):
        self.poses, self.display_names = load_recorded_sequences(csv_file, pose_file)
        print(f"Loaded {len(self.poses)} real recorded 3D avatar gestures: {', '.join(sorted(self.poses.keys()))}")

        self.vec_dim = 207
        self.queue = deque()
        self.current_label = "IDLE / LISTENING"
        self.is_signing = False
        self.rest_vec = self._make_rest_vector()
        self.current_vec = self.rest_vec.copy()
        self.target_vec = self.rest_vec.copy()
        self.last_switch = time.time()
        self.frame_duration = 0.36  # Readable human signing cadence
        vocab_preview = ", ".join(sorted(self.display_names.values()))
        self.status_text = f"Trained signs: {vocab_preview}"

    def _make_rest_vector(self):
        vec = np.zeros(self.vec_dim, dtype=float)
        vec[:63] = build_rest_hand_coords(inward_sign=1.0)
        vec[87:150] = build_rest_hand_coords(inward_sign=-1.0)
        vec[174 + 27:174 + 30] = [-0.46, 0.90, 0.0]
        vec[174 + 30:174 + 33] = [0.46, 0.90, 0.0]
        return vec

    def _map_recorded_frame_to_avatar(self, raw_frame):
        padded = np.zeros(self.vec_dim, dtype=float)
        padded[:min(len(raw_frame), self.vec_dim)] = raw_frame[:self.vec_dim]

        h_left = padded[:63]
        h_right = padded[87:150]
        left_active = float(np.max(np.abs(h_left))) > 1e-4
        right_active = float(np.max(np.abs(h_right))) > 1e-4

        pose = padded[174:207]
        has_pose = float(np.max(np.abs(pose))) > 1e-4

        if has_pose:
            rec_lx, rec_ly = float(pose[27]), float(pose[28])
            rec_rx, rec_ry = float(pose[30]), float(pose[31])

            if rec_lx > rec_rx:
                rec_lx, rec_rx = rec_rx, rec_lx
                rec_ly, rec_ry = rec_ry, rec_ly

            lw_x = float(np.clip(rec_lx * 0.62, -0.82, 0.22))
            lw_y = float(np.clip((rec_ly - 0.28) * 0.92, -0.72, 0.68))
            rw_x = float(np.clip(rec_rx * 0.62, -0.22, 0.82))
            rw_y = float(np.clip((rec_ry - 0.28) * 0.92, -0.72, 0.68))
        else:
            lw_x, lw_y = -0.26, 0.12
            rw_x, rw_y = 0.26, 0.12

        if left_active:
            padded[:63] = normalize_2d_hand_pose(h_left)
            padded[174 + 27:174 + 29] = [lw_x, lw_y]
        else:
            padded[:63] = build_rest_hand_coords(inward_sign=1.0)
            padded[174 + 27:174 + 29] = [-0.46, 0.90]

        if right_active:
            padded[87:150] = normalize_2d_hand_pose(h_right)
            padded[174 + 30:174 + 32] = [rw_x, rw_y]
        else:
            padded[87:150] = build_rest_hand_coords(inward_sign=-1.0)
            padded[174 + 30:174 + 32] = [0.46, 0.90]

        return padded

    def _enqueue_key(self, key_name, custom_label=None):
        if key_name not in self.poses:
            return
        display_label = custom_label or self.display_names.get(key_name, key_name.replace('_', ' ').upper())
        raw_frames = self.poses[key_name]

        mapped_frames = [self._map_recorded_frame_to_avatar(fr) for fr in raw_frames]
        if len(mapped_frames) == 1:
            mapped_frames = mapped_frames * 6
        elif len(mapped_frames) < 6:
            mapped_frames = mapped_frames + mapped_frames[::-1]

        for frame_vec in mapped_frames:
            self.queue.append((display_label, frame_vec))

    def enqueue_sentence(self, sentence):
        clean = "".join(ch for ch in sentence.lower() if ch.isalnum() or ch.isspace()).strip()
        if not clean:
            return

        self.queue.clear()

        phrase_key = normalize_sign_key(clean)
        if phrase_key in self.poses:
            self._enqueue_key(phrase_key)
            self.status_text = f"Signing: {self.display_names.get(phrase_key, phrase_key.upper())}"
            return

        words = clean.split()
        matched_tokens = []
        unknown_tokens = []
        i = 0

        while i < len(words):
            matched_multi = False
            for span in (4, 3, 2):
                if i + span <= len(words):
                    sub = normalize_sign_key(" ".join(words[i:i + span]))
                    if sub in self.poses:
                        self._enqueue_key(sub)
                        matched_tokens.append(self.display_names.get(sub, sub.upper()))
                        i += span
                        matched_multi = True
                        break
            if matched_multi:
                continue

            raw_word = words[i]
            if raw_word in self.poses:
                self._enqueue_key(raw_word)
                matched_tokens.append(self.display_names.get(raw_word, raw_word.upper()))
            else:
                mapped = SYNONYMS.get(raw_word)
                if mapped and mapped in self.poses:
                    self._enqueue_key(mapped)
                    matched_tokens.append(self.display_names.get(mapped, mapped.upper()))
                elif len(raw_word) > 1 and all(ch in self.poses for ch in raw_word):
                    for ch in raw_word:
                        self._enqueue_key(ch, custom_label=f"SPELL: {ch.upper()}")
                    matched_tokens.append(f"SPELL({raw_word.upper()})")
                else:
                    unknown_tokens.append(raw_word.upper())
            i += 1

        parts = []
        if matched_tokens:
            parts.append(f"Signing: {' -> '.join(matched_tokens)}")
        if unknown_tokens:
            parts.append(f"Not trained yet: {', '.join(unknown_tokens)}")
        self.status_text = " | ".join(parts) if parts else f'No trained signs for: "{sentence}"'

    def step_and_get_state(self):
        now = time.time()
        if self.queue and (now - self.last_switch) >= self.frame_duration:
            label, vec = self.queue.popleft()
            self.current_label = label
            self.target_vec = vec
            self.last_switch = now
            self.is_signing = True
        elif not self.queue and (now - self.last_switch) >= self.frame_duration * 2.2:
            self.current_label = "IDLE / LISTENING"
            self.target_vec = self.rest_vec
            self.is_signing = False

        # Fluid human-like easing curve between keyframes
        self.current_vec = self.current_vec * 0.72 + self.target_vec * 0.28
        pose = self.current_vec[174:207]

        return {
            "label": self.current_label,
            "is_signing": self.is_signing,
            "status": self.status_text,
            "available_signs": sorted(self.display_names.values()),
            "left_hand": [round(float(x), 4) for x in self.current_vec[:63]],
            "right_hand": [round(float(x), 4) for x in self.current_vec[87:150]],
            "left_wrist": [round(float(pose[27]), 4), round(float(pose[28]), 4)],
            "right_wrist": [round(float(pose[30]), 4), round(float(pose[31]), 4)],
        }


class UnifiedStudioState:
    def __init__(self, avatar):
        self.lock = threading.Lock()
        self.avatar = avatar
        self.frame_b64 = ""
        self.live_label = "Starting Camera..."
        self.confidence = 0.0
        self.gloss_stream = []
        self.completed_sentence = ""
        self.show_dots = True
        self.is_listening = False

    def handle_command(self, data):
        action = data.get("action")
        if action == "enqueue" and data.get("text"):
            self.avatar.enqueue_sentence(str(data["text"]))
        elif action == "voice":
            self.start_voice_input()
        elif action == "speak":
            with self.lock:
                full_text = self.completed_sentence or " ".join(self.gloss_stream)
            if full_text:
                print(f"[Speaking]: {full_text}")
                speak_text_async(full_text)
        elif action == "undo":
            with self.lock:
                if self.gloss_stream:
                    self.gloss_stream.pop()
                    self.completed_sentence = gloss_to_sentence(self.gloss_stream)
        elif action == "clear":
            with self.lock:
                self.gloss_stream.clear()
                self.completed_sentence = ""
        elif action == "toggle_dots":
            with self.lock:
                self.show_dots = not self.show_dots

    def start_voice_input(self):
        if self.is_listening:
            return
        self.is_listening = True

        def _worker():
            recognizer = sr.Recognizer()
            sample_rate = 16000
            duration = 4.0
            self.avatar.status_text = "Listening... Speak into mic (4s)..."
            try:
                recording = sd.rec(int(duration * sample_rate), samplerate=sample_rate,
                                   channels=1, dtype='int16')
                sd.wait()
                audio_data = sr.AudioData(recording.tobytes(), sample_rate, 2)
                text = recognizer.recognize_google(audio_data)
                self.avatar.enqueue_sentence(text)
            except sr.UnknownValueError:
                self.avatar.status_text = "Could not understand audio. Click Voice Mic to retry."
            except Exception as e:
                self.avatar.status_text = f"Mic/ASR error: {str(e)[:36]}"
            finally:
                self.is_listening = False

        threading.Thread(target=_worker, daemon=True).start()

    def build_packet(self):
        payload = self.avatar.step_and_get_state()
        with self.lock:
            payload["cam_frame"] = self.frame_b64
            payload["live_label"] = self.live_label
            payload["confidence"] = round(self.confidence * 100)
            payload["glosses"] = list(self.gloss_stream)
            payload["sentence"] = self.completed_sentence
            payload["show_dots"] = self.show_dots
            payload["is_listening"] = self.is_listening
        return json.dumps(payload)


class QuietHTTPHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, format, *args):
        pass


def start_servers(studio):
    def _http_worker():
        socketserver.TCPServer.allow_reuse_address = True
        with socketserver.ThreadingTCPServer(("127.0.0.1", HTTP_PORT), QuietHTTPHandler) as httpd:
            httpd.serve_forever()

    threading.Thread(target=_http_worker, daemon=True).start()

    if websockets is None:
        return

    def _ws_worker():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        clients = set()

        async def handler(ws):
            clients.add(ws)
            try:
                async for msg in ws:
                    studio.handle_command(json.loads(msg))
            except Exception:
                pass
            finally:
                clients.discard(ws)

        async def broadcast_loop():
            while True:
                packet = studio.build_packet()
                if clients:
                    await asyncio.gather(
                        *(c.send(packet) for c in list(clients)),
                        return_exceptions=True
                    )
                await asyncio.sleep(0.033)  # ~30 FPS

        async def runner():
            async with websockets.serve(handler, "127.0.0.1", WS_PORT):
                await broadcast_loop()

        loop.run_until_complete(runner())

    threading.Thread(target=_ws_worker, daemon=True).start()


def main():
    if not os.path.isfile(MODEL_FILE):
        print(f"Missing '{MODEL_FILE}'. Run train.py first!")
        return

    cam_holder = {"cap": None}

    def _warm_camera():
        cam_holder["cap"] = open_webcam_fast(0)

    cam_thread = threading.Thread(target=_warm_camera, daemon=True)
    cam_thread.start()

    clf = joblib.load(MODEL_FILE)
    avatar = Avatar3DController(AVATAR_FILE, DATA_CSV_FILE)
    studio = UnifiedStudioState(avatar)
    start_servers(studio)

    studio_url = f"http://127.0.0.1:{HTTP_PORT}/avatar_3d.html"
    print("\n=== AuraSign Unified Two-Way Studio Running ===")
    print(f"Open in browser: {studio_url}")
    print("Press Ctrl+C in this terminal to stop.\n")
    webbrowser.open(studio_url)

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

    cam_thread.join()
    cap = cam_holder["cap"]

    pred_history = deque(maxlen=SMOOTHING_WINDOW)
    last_added_token = None
    stable_count = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.01)
                continue

            frame = cv2.flip(frame, 1)
            cam_panel = frame if (frame.shape[1] == 640 and frame.shape[0] == 480) else cv2.resize(frame, (640, 480))

            rgb_frame = cv2.cvtColor(cam_panel, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            hand_res = hand_detector.detect(mp_image)
            pose_res = pose_detector.detect(mp_image)

            live_label = "No Hand"
            confidence = 0.0

            if hand_res.hand_landmarks:
                feat = np.array(extract_holistic_features(hand_res, pose_res)).reshape(1, -1)
                probs = clf.predict_proba(feat)[0]

                sorted_idx = np.argsort(probs)[::-1]
                best_idx = int(sorted_idx[0])
                best_prob = float(probs[best_idx])
                second_prob = float(probs[sorted_idx[1]]) if len(sorted_idx) > 1 else 0.0
                raw_class = str(clf.classes_[best_idx])

                if raw_class == '_neutral_':
                    live_label = "Neutral / Resting"
                    confidence = best_prob
                    pred_history.clear()
                    last_added_token = None
                    stable_count = 0
                elif best_prob >= MIN_CONFIDENCE and (best_prob - second_prob) >= MIN_PROB_MARGIN:
                    confidence = best_prob
                    clean_word = raw_class.replace('_', ' ')
                    pred_history.append(clean_word)
                    most_common, freq = Counter(pred_history).most_common(1)[0]

                    if freq >= SMOOTHING_WINDOW // 2:
                        live_label = most_common
                        if live_label == last_added_token:
                            stable_count += 1
                        else:
                            last_added_token = live_label
                            stable_count = 1

                        if stable_count == STABLE_FRAMES_REQUIRED:
                            with studio.lock:
                                if not studio.gloss_stream or studio.gloss_stream[-1] != live_label:
                                    studio.gloss_stream.append(live_label)
                                    studio.completed_sentence = gloss_to_sentence(studio.gloss_stream)
                else:
                    live_label = "Neutral / Resting"
                    pred_history.clear()
                    stable_count = 0
            else:
                pred_history.clear()
                last_added_token = None
                stable_count = 0

            with studio.lock:
                dots = studio.show_dots

            draw_detection_overlay(cam_panel, hand_res, pose_res, live_label, confidence, show_dots=dots)

            ok, jpg_buf = cv2.imencode('.jpg', cam_panel, [int(cv2.IMWRITE_JPEG_QUALITY), 78])
            if ok:
                b64_str = base64.b64encode(jpg_buf).decode('ascii')
                with studio.lock:
                    studio.frame_b64 = b64_str
                    studio.live_label = live_label
                    studio.confidence = confidence
    except KeyboardInterrupt:
        print("\nShutting down AuraSign Studio...")
    finally:
        if cap is not None:
            cap.release()


if __name__ == '__main__':
    main()