import asyncio
import base64
import ctypes
import hashlib
import http.server
import json
import os
import socketserver
import threading
import time
import traceback
import webbrowser
from collections import Counter, deque

os.environ['GLOG_minloglevel'] = '2'

import cv2
import joblib
import mediapipe as mp
import numpy as np
import sounddevice as sd
import speech_recognition as sr
from PIL import Image, ImageDraw, ImageFont
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

try:
    import edge_tts
    HAS_EDGE_TTS = True
except ImportError:
    HAS_EDGE_TTS = False

from features import extract_holistic_features

MODEL_FILE = 'sign_model.pkl'
DATASET_FILE = 'bangla_dataset.json'
CUSTOM_SIGNS_FILE = 'custom_signs.json'
TTS_CACHE_DIR = 'tts_cache'
HTTP_PORT = 8000
WS_PORT = 8765

# Stage-safe recognition thresholds
MIN_CONFIDENCE = 0.76
MIN_PROB_MARGIN = 0.22
PROTOTYPE_SIM_THRESHOLD = 0.86
SMOOTHING_WINDOW = 8
STABLE_FRAMES_REQUIRED = 12
LIVE_CAPTURE_SECONDS = 5.0

# Bangladeshi Bengali Neural Voice
BANGLA_VOICE = "bn-BD-NabanitaNeural"

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

BANGLA_SYNONYMS = {
    "নমস্কার": "হ্যালো",
    "আসসালামু আলাইকুম": "হ্যালো",
    "সালাম": "হ্যালো",
    "শুভ সকাল": "হ্যালো",
    "কৃতজ্ঞতা": "ধন্যবাদ",
    "শুকরিয়া": "ধন্যবাদ",
    "বাঁচান": "সাহায্য",
    "হেল্প": "সাহায্য",
    "ভালোবাসা": "ভালোবাসি",
    "কেমন আছো": "কেমন আছেন",
    "কেমন আছ": "কেমন আছেন",
    "ব্যথা": "ব্যথা করছে",
    "ডাক্তার": "চিকিৎসক",
    "হাসপাতাল": "চিকিৎসক",
}

_tts_lock = threading.Lock()
os.makedirs(TTS_CACHE_DIR, exist_ok=True)


def load_bangla_font(size=20):
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


CAM_BADGE_FONT = load_bangla_font(20)


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


def speak_bangla_async(text):
    clean_text = text.strip()
    if not clean_text:
        return

    def _run():
        with _tts_lock:
            text_hash = hashlib.md5(clean_text.encode('utf-8')).hexdigest()
            cache_path = os.path.abspath(os.path.join(TTS_CACHE_DIR, f"{text_hash}.mp3"))

            if not os.path.isfile(cache_path) and HAS_EDGE_TTS:
                try:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    comm = edge_tts.Communicate(clean_text, BANGLA_VOICE, rate="+5%")
                    loop.run_until_complete(comm.save(cache_path))
                    loop.close()
                except Exception as e:
                    print(f"[Bangla TTS Online Fetch Info]: {e}")

            if os.path.isfile(cache_path):
                try:
                    if os.name == 'nt':
                        alias = f"aurasigntts_{int(time.time() * 1000)}"
                        mci = ctypes.windll.winmm.mciSendStringW
                        mci(f'open "{cache_path}" type mpegvideo alias {alias}', None, 0, None)
                        mci(f'play {alias} wait', None, 0, None)
                        mci(f'close {alias}', None, 0, None)
                    else:
                        os.system(f'ffplay -nodisp -autoexit "{cache_path}" >/dev/null 2>&1')
                except Exception as e:
                    print(f"[Audio Playback Error]: {e}")
            else:
                print(f"[Bangla Speech Output (Offline Fallback)]: {clean_text}")

    threading.Thread(target=_run, daemon=True).start()


def gloss_to_bangla_sentence(tokens):
    if not tokens:
        return ""
    clean = []
    for t in tokens:
        if not clean or clean[-1] != t:
            clean.append(t)
    return " ".join(clean) + "।"


def draw_detection_overlay(frame, hand_res, pose_res, label="", confidence=0.0, show_dots=True, rec_banner=""):
    h, w, _ = frame.shape

    if show_dots and pose_res and pose_res.pose_landmarks:
        lms = pose_res.pose_landmarks[0]
        for a, b in UPPER_BODY_CONNECTIONS:
            pt1 = (int(lms[a].x * w), int(lms[a].y * h))
            pt2 = (int(lms[b].x * w), int(lms[b].y * h))
            cv2.line(frame, pt1, pt2, (255, 180, 0), 2, cv2.LINE_AA)

    active = bool(label and label not in ("হাত দেখা যাচ্ছে না", "স্বাভাবিক অবস্থা"))
    badge_to_draw = None

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

            box_color = (56, 189, 248) if rec_banner else ((0, 255, 140) if active else (130, 140, 155))
            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2, cv2.LINE_AA)

            if idx == 0 and active and not rec_banner:
                badge_to_draw = (x1, max(4, y1 - 32), f"{label} ({int(confidence * 100)}%)")

    if rec_banner or badge_to_draw is not None:
        pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_img)
        if rec_banner:
            draw.rectangle([0, h - 44, w, h], fill=(14, 116, 144))
            draw.text((16, h - 36), rec_banner, font=CAM_BADGE_FONT, fill=(255, 255, 255))
        elif badge_to_draw is not None:
            bx, by, badge_txt = badge_to_draw
            bbox = draw.textbbox((bx, by), badge_txt, font=CAM_BADGE_FONT)
            draw.rectangle([bbox[0] - 6, bbox[1] - 3, bbox[2] + 8, bbox[3] + 4], fill=(16, 185, 129))
            draw.text((bx, by), badge_txt, font=CAM_BADGE_FONT, fill=(10, 15, 22))
        frame[:] = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


# ==============================================================================
# ZERO-RETRAIN FEW-SHOT PROTOTYPE RECOGNIZER (REGIONAL & MEDICAL SIGNS)
# ==============================================================================

class FewShotSignStore:
    def __init__(self, filepath=CUSTOM_SIGNS_FILE):
        self.filepath = filepath
        self.lock = threading.Lock()
        self.signs = {}
        self._load()

    @staticmethod
    def _build_prototypes(frames):
        arr = np.array(frames, dtype=float)
        sub = arr[:, :207]
        norms = np.linalg.norm(sub, axis=1, keepdims=True)
        norms = np.where(norms < 1e-6, 1.0, norms)
        unit = sub / norms
        idx = np.linspace(0, len(unit) - 1, num=min(10, len(unit)), dtype=int)
        centroid = np.mean(unit, axis=0, keepdims=True)
        centroid /= max(float(np.linalg.norm(centroid)), 1e-6)
        return np.vstack([centroid, unit[idx]])

    def _load(self):
        if not os.path.isfile(self.filepath):
            return
        try:
            with open(self.filepath, 'r', encoding='utf-8') as f:
                raw = json.load(f)
            for label, entry in raw.get("signs", {}).items():
                frames = entry.get("frames", [])
                if frames:
                    self.signs[label] = {
                        "category": entry.get("category", "Custom"),
                        "frames": frames,
                        "prototypes": self._build_prototypes(frames),
                    }
        except Exception as e:
            print(f"[Custom Signs Load Warning]: {e}")

    def _save(self):
        serializable = {
            "signs": {
                k: {"category": v["category"], "frames": v["frames"]}
                for k, v in self.signs.items()
            }
        }
        with open(self.filepath, 'w', encoding='utf-8') as f:
            json.dump(serializable, f, ensure_ascii=False, indent=2)

    def add_or_update_sign(self, label, category, frames):
        clean_label = label.strip()
        if not clean_label or not frames:
            return
        with self.lock:
            self.signs[clean_label] = {
                "category": category or "Medical / Regional",
                "frames": frames,
                "prototypes": self._build_prototypes(frames),
            }
            self._save()

    def remove_sign(self, label):
        with self.lock:
            if label in self.signs:
                del self.signs[label]
                self._save()
                return True
        return False

    def match(self, feat_vec):
        with self.lock:
            if not self.signs:
                return None, 0.0
            q = np.array(feat_vec[:207], dtype=float)
            q_norm = float(np.linalg.norm(q))
            if q_norm < 1e-6:
                return None, 0.0
            q_unit = q / q_norm

            best_label = None
            best_score = 0.0
            for label, entry in self.signs.items():
                sims = entry["prototypes"] @ q_unit
                score = float(np.max(sims))
                if score > best_score:
                    best_score = score
                    best_label = label
            return best_label, best_score

    def list_metadata(self):
        with self.lock:
            return [
                {"label": k, "category": v["category"]}
                for k, v in self.signs.items()
            ]


# ==============================================================================
# 3D AVATAR MOTION REPLAY CONTROLLER (CHEST-TO-HEAD SIGNING SPACE)
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


def frames_to_keyframes(frames):
    arr = np.array(frames, dtype=float)
    num_steps = min(22, len(arr))
    indices = np.linspace(0, len(arr) - 1, num=num_steps, dtype=int)
    return arr[indices]


def load_bangla_motion_sequences(dataset_file):
    sequences = {}
    if not os.path.isfile(dataset_file):
        return sequences
    with open(dataset_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    for label, frames in data.get("signs", {}).items():
        clean_label = label.strip()
        if not clean_label or clean_label == '_neutral_' or not frames:
            continue
        sequences[clean_label] = frames_to_keyframes(frames)
    return sequences


class Avatar3DController:
    def __init__(self, dataset_file=DATASET_FILE, custom_store=None):
        self.sequences = load_bangla_motion_sequences(dataset_file)
        if custom_store is not None:
            with custom_store.lock:
                for label, entry in custom_store.signs.items():
                    self.sequences[label] = frames_to_keyframes(entry["frames"])

        print(f"Loaded {len(self.sequences)} Bangla signs (Base + Custom): {', '.join(self.sequences.keys())}")

        self.vec_dim = 207
        self.queue = deque()
        self.current_label = "অপেক্ষমাণ / শুনছি"
        self.is_signing = False
        self.rest_vec = self._make_rest_vector()
        self.current_vec = self.rest_vec.copy()
        self.target_vec = self.rest_vec.copy()
        self.last_switch = time.time()
        self.frame_duration = 0.20
        vocab_preview = ", ".join(self.sequences.keys()) if self.sequences else "কোনো ইশারা নেই"
        self.status_text = f"সক্রিয় ইশারা: {vocab_preview}"

    def register_live_sign(self, label, frames):
        self.sequences[label] = frames_to_keyframes(frames)
        self.status_text = f"নতুন ইশারা যুক্ত হয়েছে (রিবুট ছাড়াই): {label}"
        self.enqueue_sentence(label)

    def unregister_live_sign(self, label):
        if label in self.sequences:
            del self.sequences[label]
            self.status_text = f"ইশারা মুছে ফেলা হয়েছে: {label}"

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
            nose_y = float(pose[1])
            sh_y = 0.5 * (float(pose[4]) + float(pose[7]))
            head_span = max(abs(sh_y - nose_y), 0.12)

            rec_lx, rec_ly = float(pose[27]), float(pose[28])
            rec_rx, rec_ry = float(pose[30]), float(pose[31])

            if rec_lx > rec_rx:
                rec_lx, rec_rx = rec_rx, rec_lx
                rec_ly, rec_ry = rec_ry, rec_ly

            # Map webcam wrist Y directly into chest-to-forehead signing space:
            # -0.34 = forehead/temple, -0.10 = upper chest/collarbone, +0.16 = mid chest
            rel_ly = ((rec_ly - nose_y) / head_span) * 0.22 - 0.18
            rel_ry = ((rec_ry - nose_y) / head_span) * 0.22 - 0.18

            lw_x = float(np.clip(rec_lx * 0.48, -0.58, 0.20))
            lw_y = float(np.clip(rel_ly, -0.34, 0.18))
            rw_x = float(np.clip(rec_rx * 0.48, -0.20, 0.58))
            rw_y = float(np.clip(rel_ry, -0.34, 0.18))
        else:
            lw_x, lw_y = -0.22, -0.05
            rw_x, rw_y = 0.22, -0.05

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

    def _enqueue_sign(self, sign_name):
        if sign_name not in self.sequences:
            return
        for fr in self.sequences[sign_name]:
            self.queue.append((sign_name, self._map_recorded_frame_to_avatar(fr)))

    def enqueue_sentence(self, sentence):
        clean = sentence.replace("।", " ").replace("?", " ").replace("!", " ").replace(",", " ").strip()
        clean = " ".join(clean.split())
        if not clean:
            return

        self.queue.clear()

        if clean in self.sequences:
            self._enqueue_sign(clean)
            self.status_text = f"ইশারা দেখানো হচ্ছে: {clean}"
            return

        if clean in BANGLA_SYNONYMS and BANGLA_SYNONYMS[clean] in self.sequences:
            mapped = BANGLA_SYNONYMS[clean]
            self._enqueue_sign(mapped)
            self.status_text = f"ইশারা দেখানো হচ্ছে: {mapped}"
            return

        words = clean.split()
        matched = []
        unknown = []
        i = 0
        while i < len(words):
            found_phrase = False
            for span in (4, 3, 2):
                if i + span <= len(words):
                    sub = " ".join(words[i:i + span])
                    if sub in self.sequences:
                        self._enqueue_sign(sub)
                        matched.append(sub)
                        i += span
                        found_phrase = True
                        break
                    if sub in BANGLA_SYNONYMS and BANGLA_SYNONYMS[sub] in self.sequences:
                        m = BANGLA_SYNONYMS[sub]
                        self._enqueue_sign(m)
                        matched.append(m)
                        i += span
                        found_phrase = True
                        break
            if found_phrase:
                continue

            w = words[i]
            if w in self.sequences:
                self._enqueue_sign(w)
                matched.append(w)
            elif w in BANGLA_SYNONYMS and BANGLA_SYNONYMS[w] in self.sequences:
                m = BANGLA_SYNONYMS[w]
                self._enqueue_sign(m)
                matched.append(m)
            else:
                unknown.append(w)
            i += 1

        parts = []
        if matched:
            parts.append(f"ইশারা: {' → '.join(matched)}")
        if unknown:
            parts.append(f"যুক্ত করা নেই: {', '.join(unknown)}")
        self.status_text = " | ".join(parts) if parts else f'"{sentence}" এর জন্য ইশারা পাওয়া যায়নি'

    def step_and_get_state(self):
        now = time.time()
        if self.queue and (now - self.last_switch) >= self.frame_duration:
            label, vec = self.queue.popleft()
            self.current_label = label
            self.target_vec = vec
            self.last_switch = now
            self.is_signing = True
        elif not self.queue and (now - self.last_switch) >= self.frame_duration * 2.5:
            self.current_label = "অপেক্ষমাণ / শুনছি"
            self.target_vec = self.rest_vec
            self.is_signing = False

        self.current_vec = self.current_vec * 0.68 + self.target_vec * 0.32
        pose = self.current_vec[174:207]

        return {
            "label": self.current_label,
            "is_signing": self.is_signing,
            "status": self.status_text,
            "available_signs": list(self.sequences.keys()),
            "left_hand": [round(float(x), 4) for x in self.current_vec[:63]],
            "right_hand": [round(float(x), 4) for x in self.current_vec[87:150]],
            "left_wrist": [round(float(pose[27]), 4), round(float(pose[28]), 4)],
            "right_wrist": [round(float(pose[30]), 4), round(float(pose[31]), 4)],
        }


class UnifiedStudioState:
    def __init__(self, avatar, custom_store):
        self.lock = threading.Lock()
        self.avatar = avatar
        self.custom_store = custom_store
        self.frame_b64 = ""
        self.live_label = "ক্যামেরা চালু হচ্ছে..."
        self.confidence = 0.0
        self.gloss_stream = []
        self.completed_sentence = ""
        self.show_dots = True
        self.is_listening = False

        # 5-Second Zero-Retrain Live Capture State
        self.capture_active = False
        self.capture_countdown_end = 0.0
        self.capture_end_time = 0.0
        self.capture_label = ""
        self.capture_category = "Medical / Regional"
        self.capture_buffer = []

    def start_live_sign_capture(self, label, category):
        clean = label.strip()
        if not clean:
            return
        now = time.time()
        with self.lock:
            self.capture_label = clean
            self.capture_category = category.strip() or "Medical / Regional"
            self.capture_buffer = []
            self.capture_countdown_end = now + 1.5
            self.capture_end_time = self.capture_countdown_end + LIVE_CAPTURE_SECONDS
            self.capture_active = True
        self.avatar.status_text = f"প্রস্তুত হোন: '{clean}' ৫ সেকেন্ড রেকর্ড শুরু হচ্ছে..."

    def handle_command(self, data):
        action = data.get("action")
        if action == "enqueue" and data.get("text"):
            self.avatar.enqueue_sentence(str(data["text"]))
        elif action == "add_custom_sign" and data.get("label"):
            self.start_live_sign_capture(str(data["label"]), str(data.get("category", "Medical / Regional")))
        elif action == "delete_custom_sign" and data.get("label"):
            lbl = str(data["label"]).strip()
            if self.custom_store.remove_sign(lbl):
                self.avatar.unregister_live_sign(lbl)
        elif action == "voice":
            self.start_voice_input()
        elif action == "speak":
            with self.lock:
                full_text = self.completed_sentence or " ".join(self.gloss_stream)
            if full_text:
                print(f"[Speaking Bangla]: {full_text}")
                speak_bangla_async(full_text)
        elif action == "undo":
            with self.lock:
                if self.gloss_stream:
                    self.gloss_stream.pop()
                    self.completed_sentence = gloss_to_bangla_sentence(self.gloss_stream)
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
            self.avatar.status_text = "শুনছি... বাংলায় কথা বলুন (৪ সেকেন্ড)..."
            try:
                recording = sd.rec(int(duration * sample_rate), samplerate=sample_rate,
                                   channels=1, dtype='int16')
                sd.wait()
                audio_data = sr.AudioData(recording.tobytes(), sample_rate, 2)
                text = recognizer.recognize_google(audio_data, language="bn-BD")
                print(f"[Bangla Voice ASR]: {text}")
                self.avatar.enqueue_sentence(text)
            except sr.UnknownValueError:
                self.avatar.status_text = "কথা বোঝা যায়নি। আবার ভয়েস বাটনে ক্লিক করুন।"
            except Exception as e:
                self.avatar.status_text = f"Mic error: {str(e)[:32]}"
            finally:
                self.is_listening = False

        threading.Thread(target=_worker, daemon=True).start()

    def build_packet(self):
        payload = self.avatar.step_and_get_state()
        now = time.time()
        with self.lock:
            payload["cam_frame"] = self.frame_b64
            payload["live_label"] = self.live_label
            payload["confidence"] = round(self.confidence * 100)
            payload["glosses"] = list(self.gloss_stream)
            payload["sentence"] = self.completed_sentence
            payload["show_dots"] = self.show_dots
            payload["is_listening"] = self.is_listening
            payload["capture_active"] = self.capture_active
            payload["capture_label"] = self.capture_label
            payload["capture_frames"] = len(self.capture_buffer)
            if self.capture_active:
                if now < self.capture_countdown_end:
                    payload["capture_phase"] = "countdown"
                    payload["capture_remaining"] = round(self.capture_countdown_end - now, 1)
                else:
                    payload["capture_phase"] = "recording"
                    payload["capture_remaining"] = round(max(0.0, self.capture_end_time - now), 1)
            else:
                payload["capture_phase"] = "idle"
                payload["capture_remaining"] = 0.0
        payload["custom_signs"] = self.custom_store.list_metadata()
        return json.dumps(payload, ensure_ascii=False)


class QuietHTTPHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, format, *args):
        pass


class ReusableTCPServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True


def start_servers(studio):
    def _http_worker():
        try:
            with ReusableTCPServer(("127.0.0.1", HTTP_PORT), QuietHTTPHandler) as httpd:
                httpd.serve_forever()
        except OSError as e:
            print(f"[HTTP Server Notice]: Port {HTTP_PORT} busy ({e}) - reusing existing server.")

    threading.Thread(target=_http_worker, daemon=True).start()

    try:
        import websockets
    except ImportError:
        print("[Warning] 'websockets' not installed.")
        return

    def _ws_worker():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        clients = set()

        async def handler(ws):
            clients.add(ws)
            try:
                async for msg in ws:
                    try:
                        studio.handle_command(json.loads(msg))
                    except Exception:
                        traceback.print_exc()
            except Exception:
                pass
            finally:
                clients.discard(ws)

        async def broadcast_loop():
            while True:
                try:
                    packet = studio.build_packet()
                    if clients:
                        await asyncio.gather(
                            *(c.send(packet) for c in list(clients)),
                            return_exceptions=True
                        )
                except Exception:
                    traceback.print_exc()
                await asyncio.sleep(0.033)

        async def runner():
            try:
                async with websockets.serve(handler, "127.0.0.1", WS_PORT):
                    await broadcast_loop()
            except OSError as e:
                print(f"[WebSocket Error]: Port {WS_PORT} is already in use ({e}). Close old python.exe processes first.")

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
    custom_store = FewShotSignStore(CUSTOM_SIGNS_FILE)
    avatar = Avatar3DController(DATASET_FILE, custom_store)
    studio = UnifiedStudioState(avatar, custom_store)
    start_servers(studio)

    studio_url = f"http://127.0.0.1:{HTTP_PORT}/avatar_3d.html"
    print("\n=== AuraSign Two-Way Bangla Studio (5s Capture + Chest-Level 3D Rig) ===")
    print(f"Open in browser: {studio_url}\n")
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

            # 1. HANDLE 5-SECOND LIVE ZERO-RETRAIN CAPTURE IF ACTIVE
            rec_banner = ""
            with studio.lock:
                is_capturing = studio.capture_active
                cd_end = studio.capture_countdown_end
                rec_end = studio.capture_end_time
                cap_lbl = studio.capture_label
                cap_cat = studio.capture_category

            if is_capturing:
                now = time.time()
                if now < cd_end:
                    rec_banner = f"প্রস্তুত হোন ({cap_lbl}): {cd_end - now:.1f}s..."
                else:
                    if hand_res.hand_landmarks:
                        feat_list = [round(float(x), 5) for x in extract_holistic_features(hand_res, pose_res)]
                        with studio.lock:
                            studio.capture_buffer.append(feat_list)

                    with studio.lock:
                        cur_len = len(studio.capture_buffer)
                        done_frames = list(studio.capture_buffer) if now >= rec_end else None

                    rem_sec = max(0.0, rec_end - now)
                    rec_banner = f"রেকর্ডিং '{cap_lbl}': {rem_sec:.1f}s বাকি ({cur_len} ফ্রেম)"

                    if done_frames is not None:
                        if len(done_frames) >= 5:
                            custom_store.add_or_update_sign(cap_lbl, cap_cat, done_frames)
                            avatar.register_live_sign(cap_lbl, done_frames)
                        else:
                            avatar.status_text = "হাত স্পষ্ট দেখা যায়নি — আবার ৫ সেকেন্ড রেকর্ড করুন।"
                        with studio.lock:
                            studio.capture_active = False
                            studio.capture_buffer = []
                        pred_history.clear()
                        last_added_token = None
                        stable_count = 0
                        rec_banner = ""

            # 2. HYBRID RECOGNITION: ZERO-RETRAIN PROTOTYPES + TRAINED MLP CLASSIFIER
            live_label = "হাত দেখা যাচ্ছে না"
            confidence = 0.0

            if hand_res.hand_landmarks and not is_capturing:
                raw_feat = extract_holistic_features(hand_res, pose_res)
                feat = np.array(raw_feat).reshape(1, -1)

                proto_label, proto_sim = custom_store.match(raw_feat)

                probs = clf.predict_proba(feat)[0]
                sorted_idx = np.argsort(probs)[::-1]
                best_idx = int(sorted_idx[0])
                best_prob = float(probs[best_idx])
                second_prob = float(probs[sorted_idx[1]]) if len(sorted_idx) > 1 else 0.0
                raw_class = str(clf.classes_[best_idx])

                chosen_label = None
                chosen_conf = 0.0

                if proto_label and proto_sim >= PROTOTYPE_SIM_THRESHOLD and proto_sim >= (best_prob - 0.04):
                    chosen_label = proto_label
                    chosen_conf = proto_sim
                elif raw_class == '_neutral_':
                    live_label = "স্বাভাবিক অবস্থা"
                    confidence = best_prob
                    pred_history.clear()
                    last_added_token = None
                    stable_count = 0
                elif best_prob >= MIN_CONFIDENCE and (best_prob - second_prob) >= MIN_PROB_MARGIN:
                    chosen_label = raw_class
                    chosen_conf = best_prob

                if chosen_label:
                    confidence = chosen_conf
                    pred_history.append(chosen_label)
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
                                    studio.completed_sentence = gloss_to_bangla_sentence(studio.gloss_stream)
                elif raw_class != '_neutral_':
                    live_label = "স্বাভাবিক অবস্থা"
                    pred_history.clear()
                    stable_count = 0
            elif not is_capturing:
                pred_history.clear()
                last_added_token = None
                stable_count = 0

            with studio.lock:
                dots = studio.show_dots

            draw_detection_overlay(
                cam_panel, hand_res, pose_res, live_label, confidence,
                show_dots=dots, rec_banner=rec_banner
            )

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