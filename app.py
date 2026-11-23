import asyncio
import base64
import http.server
import json
import math
import os
import socketserver
import threading
import time
import webbrowser
from collections import Counter, deque

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
HTTP_PORT = 8000
WS_PORT = 8765

# Anti-false-positive thresholds
MIN_CONFIDENCE = 0.75
MIN_PROB_MARGIN = 0.25
SMOOTHING_WINDOW = 8

# Fast female TTS voice pacing
SAPI_RATE = 3           # Windows SAPI5 fast speed (-10 to +10; increase to 4 or 5 for even faster)
PYTTSX3_RATE = 215      # Fallback fast speaking speed (WPM)
WORD_GAP_SEC = 0.0      # Natural micro-gap from per-word SAPI calls (set e.g. 0.03 to add extra pause)

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

# Only true synonyms (never override distinct trained words like 'love' -> 'i_love_you')
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


def open_webcam_fast(cam_index=0):
    """Opens the webcam rapidly on Windows via DirectShow (CAP_DSHOW), with fallback."""
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
    """Speaks the accumulated sentence aloud quickly with crisp word separation."""
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
    """Builds clean text directly from the signed glosses without injecting extra words."""
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
# 3D AVATAR KINEMATICS CONTROLLER & UNIFIED STUDIO STATE
# ==============================================================================

def normalize_2d_hand_pose(raw_63):
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


class Avatar3DController:
    def __init__(self, pose_file):
        self.poses = {}
        if os.path.isfile(pose_file):
            with open(pose_file, 'r', encoding='utf-8') as f:
                raw_poses = json.load(f)
                self.poses = {k.strip().lower().replace(' ', '_'): v for k, v in raw_poses.items()}

        print(f"Loaded {len(self.poses)} 3D avatar gestures: {', '.join(sorted(self.poses.keys()))}")
        self.vec_dim = 207
        self.queue = deque()
        self.current_label = "IDLE / LISTENING"
        self.is_signing = False
        self.rest_vec = self._make_rest_vector()
        self.current_vec = self.rest_vec.copy()
        self.target_vec = self.rest_vec.copy()
        self.last_switch = time.time()
        self.frame_duration = 0.45
        vocab_preview = ", ".join(sorted(self.poses.keys()))
        self.status_text = f"Trained signs: {vocab_preview}"

    def _make_rest_vector(self):
        vec = np.zeros(self.vec_dim, dtype=float)
        vec[:63] = build_rest_hand_coords(inward_sign=1.0)
        vec[87:150] = build_rest_hand_coords(inward_sign=-1.0)
        vec[174 + 27:174 + 30] = [-0.46, 0.90, 0.0]
        vec[174 + 30:174 + 33] = [0.46, 0.90, 0.0]
        return vec

    def _prepare_pose_vector(self, raw_frame):
        padded = np.zeros(self.vec_dim, dtype=float)
        padded[:min(len(raw_frame), self.vec_dim)] = raw_frame[:self.vec_dim]

        h_screen_left = padded[:63]
        h_screen_right = padded[87:150]
        left_active = max(abs(v) for v in h_screen_left) > 1e-4
        right_active = max(abs(v) for v in h_screen_right) > 1e-4

        pose = padded[174:207]
        has_pose = max(abs(v) for v in pose) > 1e-4

        if has_pose:
            raw_lw = (pose[27], pose[28])
            raw_rw = (pose[30], pose[31])
            if raw_lw[0] > raw_rw[0]:
                raw_lw, raw_rw = raw_rw, raw_lw
        else:
            raw_lw = (-0.34, 0.18)
            raw_rw = (0.34, 0.18)

        left_w = (max(-0.82, min(0.14, raw_lw[0])), max(-0.68, min(0.84, raw_lw[1])))
        right_w = (max(-0.14, min(0.82, raw_rw[0])), max(-0.68, min(0.84, raw_rw[1])))

        if left_active:
            padded[:63] = normalize_2d_hand_pose(h_screen_left)
            padded[174 + 27:174 + 29] = [left_w[0], left_w[1]]
        else:
            padded[:63] = build_rest_hand_coords(inward_sign=1.0)
            padded[174 + 27:174 + 29] = [-0.46, 0.90]

        if right_active:
            padded[87:150] = normalize_2d_hand_pose(h_screen_right)
            padded[174 + 30:174 + 32] = [right_w[0], right_w[1]]
        else:
            padded[87:150] = build_rest_hand_coords(inward_sign=-1.0)
            padded[174 + 30:174 + 32] = [0.46, 0.90]

        return padded

    def _apply_kinematic_motion(self, base_vec, word_key, step_idx, total_steps):
        vec = base_vec.copy()
        left_active = max(abs(v) for v in base_vec[:63]) > 1e-4 and (base_vec[174 + 28] < 0.78)
        right_active = max(abs(v) for v in base_vec[87:150]) > 1e-4 and (base_vec[174 + 31] < 0.78)

        phase = step_idx / max(1, total_steps - 1)
        dx, dy = 0.0, 0.0

        if step_idx == 0:
            dy = 0.22
        elif 'hello' in word_key:
            dx = math.sin(phase * math.pi * 2.5) * 0.14
        elif 'yes' in word_key:
            dy = math.sin(phase * math.pi * 2.5) * 0.13
        elif 'no' in word_key:
            dx = math.sin(phase * math.pi * 3.0) * 0.14
        elif 'thank' in word_key or 'please' in word_key:
            dx = phase * 0.10
            dy = phase * 0.16
        else:
            pulse = math.sin(phase * math.pi * 2.0)
            dx = pulse * 0.08
            dy = -abs(pulse) * 0.09

        if left_active:
            vec[174 + 27] = max(-0.84, min(0.16, vec[174 + 27] - dx))
            vec[174 + 28] = max(-0.70, min(0.84, vec[174 + 28] + dy))

        if right_active:
            vec[174 + 30] = max(-0.16, min(0.84, vec[174 + 30] + dx))
            vec[174 + 31] = max(-0.70, min(0.84, vec[174 + 31] + dy))

        return vec

    def _enqueue_key(self, key_name, custom_label=None, steps=5):
        if key_name not in self.poses:
            return
        display_label = custom_label or key_name.replace('_', ' ').upper()
        arr = np.array(self.poses[key_name], dtype=float)
        raw_frames = [arr] if arr.ndim == 1 else list(arr)
        for idx in range(steps):
            src_frame = raw_frames[min(len(raw_frames) - 1, int(idx * len(raw_frames) / steps))]
            prepared = self._prepare_pose_vector(src_frame)
            animated = self._apply_kinematic_motion(prepared, display_label.lower(), idx, steps)
            self.queue.append((display_label, animated))

    def enqueue_sentence(self, sentence):
        clean = "".join(ch for ch in sentence.lower() if ch.isalnum() or ch.isspace()).strip()
        if not clean:
            return

        self.queue.clear()

        phrase_key = "_".join(clean.split())
        if phrase_key in self.poses:
            self._enqueue_key(phrase_key)
            self.status_text = f"Signing: {phrase_key.replace('_', ' ').upper()}"
            return

        words = clean.split()
        matched_tokens = []
        unknown_tokens = []
        i = 0

        while i < len(words):
            matched_multi = False
            for span in (3, 2):
                if i + span <= len(words):
                    sub = "_".join(words[i:i + span])
                    if sub in self.poses:
                        self._enqueue_key(sub)
                        matched_tokens.append(sub.replace('_', ' ').upper())
                        i += span
                        matched_multi = True
                        break
            if matched_multi:
                continue

            raw_word = words[i]
            if raw_word in self.poses:
                self._enqueue_key(raw_word)
                matched_tokens.append(raw_word.upper())
            else:
                mapped = SYNONYMS.get(raw_word)
                if mapped and mapped in self.poses:
                    self._enqueue_key(mapped)
                    matched_tokens.append(mapped.replace('_', ' ').upper())
                elif len(raw_word) > 1 and all(ch in self.poses for ch in raw_word):
                    for ch in raw_word:
                        self._enqueue_key(ch, custom_label=f"SPELL: {ch.upper()}", steps=3)
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

        self.current_vec = self.current_vec * 0.78 + self.target_vec * 0.22
        pose = self.current_vec[174:207]

        return {
            "label": self.current_label,
            "is_signing": self.is_signing,
            "status": self.status_text,
            "available_signs": sorted(self.poses.keys()),
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

    # Open webcam in parallel while MediaPipe models & servers load
    cam_holder = {"cap": None}

    def _warm_camera():
        cam_holder["cap"] = open_webcam_fast(0)

    cam_thread = threading.Thread(target=_warm_camera, daemon=True)
    cam_thread.start()

    clf = joblib.load(MODEL_FILE)
    avatar = Avatar3DController(AVATAR_FILE)
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

                        if stable_count == 10:
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