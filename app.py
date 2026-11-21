import json
import math
import os
import threading
import time
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

from features import extract_holistic_features

MODEL_FILE = 'sign_model.pkl'
AVATAR_FILE = 'avatar_poses.json'

# Anti-false-positive thresholds
MIN_CONFIDENCE = 0.75
MIN_PROB_MARGIN = 0.25
SMOOTHING_WINDOW = 8

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20)
]

FINGER_CHAINS = [
    [(0, 1, 7, 6), (1, 2, 6, 5), (2, 3, 5, 4), (3, 4, 4, 3)],
    [(5, 6, 6, 5), (6, 7, 5, 4), (7, 8, 4, 3)],
    [(9, 10, 6, 5), (10, 11, 5, 4), (11, 12, 4, 3)],
    [(13, 14, 6, 5), (14, 15, 5, 4), (15, 16, 4, 3)],
    [(17, 18, 5, 4), (18, 19, 4, 3), (19, 20, 4, 3)],
]

UPPER_BODY_CONNECTIONS = [
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (0, 11), (0, 12)
]

SYNONYMS = {
    "hi": "hello", "hey": "hello", "thanks": "thank_you",
    "love": "i_love_you", "assist": "help", "hurt": "pain",
    "ache": "pain", "hospital": "doctor", "drink": "water"
}

AVATAR_THEMES = [
    {
        "name": "AURA (Interpreter)",
        "style": "male",
        "skin": (162, 196, 236),
        "skin_shadow": (134, 166, 206),
        "skin_outline": (102, 132, 172),
        "hair": (38, 34, 32),
        "outfit": (68, 82, 112),
        "outfit_border": (92, 110, 145),
        "shirt": (220, 226, 236),
        "accent": (0, 230, 255),
    },
    {
        "name": "NOVA (Interpreter)",
        "style": "female",
        "skin": (176, 206, 242),
        "skin_shadow": (146, 176, 214),
        "skin_outline": (112, 140, 178),
        "hair": (46, 32, 28),
        "outfit": (92, 118, 72),
        "outfit_border": (122, 152, 98),
        "shirt": (235, 235, 242),
        "accent": (120, 255, 180),
    },
    {
        "name": "TITAN-X (Holo-Bot)",
        "style": "robot",
        "skin": (195, 204, 216),
        "skin_shadow": (135, 146, 162),
        "skin_outline": (75, 86, 102),
        "hair": (52, 62, 78),
        "outfit": (54, 62, 76),
        "outfit_border": (0, 220, 255),
        "shirt": (0, 215, 255),
        "accent": (0, 230, 255),
    },
    {
        "name": "CUSTOM (custom_avatar.png)",
        "style": "custom_image",
        "skin": (112, 146, 196),
        "skin_shadow": (86, 118, 166),
        "skin_outline": (62, 88, 130),
        "hair": (32, 28, 26),
        "outfit": (54, 44, 38),
        "outfit_border": (78, 66, 58),
        "shirt": (72, 70, 68),
        "accent": (255, 200, 80),
    }
]

_tts_lock = threading.Lock()


def speak_text_async(text):
    """Speaks the full accumulated sentence when [SPACE] is pressed."""
    if not text.strip():
        return

    def _run():
        with _tts_lock:
            try:
                engine = pyttsx3.init()
                engine.setProperty('rate', 165)
                engine.say(text)
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

    templates = {
        "hello": "Hello!",
        "yes": "Yes.",
        "no": "No.",
        "thank you": "Thank you very much.",
        "i love you": "I love you.",
        "help": "Please help me.",
        "pain": "I am in pain.",
        "doctor": "I need a doctor.",
        "water": "Please give me some water.",
        "please": "Please."
    }
    if len(clean) == 1:
        return templates.get(clean[0], f"{clean[0].capitalize()}.")

    joined = ", ".join(clean)
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
# ANATOMICALLY CONSTRAINED 2.5D + CUSTOM IMAGE AVATAR ENGINE
# ==============================================================================

def draw_shaded_capsule(canvas, p1, p2, r1, r2, fill_color, border_color=None, highlight=True):
    x1, y1 = float(p1[0]), float(p1[1])
    x2, y2 = float(p2[0]), float(p2[1])
    dx, dy = x2 - x1, y2 - y1
    dist = math.hypot(dx, dy)
    if dist < 1e-3:
        cv2.circle(canvas, (int(x1), int(y1)), int(r1), fill_color, -1, cv2.LINE_AA)
        return

    nx, ny = -dy / dist, dx / dist
    poly = np.array([
        [int(x1 + nx * r1), int(y1 + ny * r1)],
        [int(x2 + nx * r2), int(y2 + ny * r2)],
        [int(x2 - nx * r2), int(y2 - ny * r2)],
        [int(x1 - nx * r1), int(y1 - ny * r1)],
    ], dtype=np.int32)

    cv2.fillConvexPoly(canvas, poly, fill_color, cv2.LINE_AA)
    cv2.circle(canvas, (int(x1), int(y1)), int(r1), fill_color, -1, cv2.LINE_AA)
    cv2.circle(canvas, (int(x2), int(y2)), int(r2), fill_color, -1, cv2.LINE_AA)

    if border_color is not None:
        cv2.line(canvas, (int(x1 + nx * r1), int(y1 + ny * r1)),
                 (int(x2 + nx * r2), int(y2 + ny * r2)), border_color, 1, cv2.LINE_AA)
        cv2.line(canvas, (int(x1 - nx * r1), int(y1 - ny * r1)),
                 (int(x2 - nx * r2), int(y2 - ny * r2)), border_color, 1, cv2.LINE_AA)

    if highlight and r1 >= 4:
        hi = tuple(min(255, int(c * 1.22 + 10)) for c in fill_color)
        cv2.line(canvas, (int(x1 - nx * (r1 * 0.35)), int(y1 - ny * (r1 * 0.35))),
                 (int(x2 - nx * (r2 * 0.35)), int(y2 - ny * (r2 * 0.35))),
                 hi, max(1, int(r2 * 0.45)), cv2.LINE_AA)


def solve_anatomical_ik(shoulder, wrist, len_upper, len_forearm, torso_center_x, side_sign):
    """Gravity-biased 2-bone IK that prevents impossible upward/backward elbow flips."""
    sx, sy = shoulder
    wx, wy = wrist
    dx, dy = wx - sx, wy - sy
    dist = math.hypot(dx, dy)

    min_reach = abs(len_upper - len_forearm) + 12.0
    max_reach = (len_upper + len_forearm) - 4.0
    clamped = max(min_reach, min(max_reach, dist))

    cos_a = (len_upper ** 2 + clamped ** 2 - len_forearm ** 2) / (2.0 * len_upper * clamped)
    cos_a = max(-1.0, min(1.0, cos_a))
    offset = math.acos(cos_a)
    base_angle = math.atan2(dy, dx)

    best_elbow = None
    best_score = -1e9
    for sign in (-1.0, 1.0):
        ang = base_angle + sign * offset
        ex = sx + len_upper * math.cos(ang)
        ey = sy + len_upper * math.sin(ang)
        outward_clearance = (ex - torso_center_x) * side_sign
        score = ey * 1.5 + outward_clearance * 0.8
        if score > best_score:
            best_score = score
            best_elbow = (ex, ey)

    return best_elbow


def normalize_2d_hand_pose(raw_63):
    """Normalizes hand coordinates using 2D span so Z-depth spikes never squash fingers."""
    coords = np.array(raw_63, dtype=float).reshape(21, 3)
    xy_span = np.max(np.abs(coords[:, :2]))
    if xy_span < 1e-4:
        return coords.flatten()

    scale_factor = 0.92 / max(xy_span, 0.35)
    coords[:, 0] = np.clip(coords[:, 0] * scale_factor, -1.1, 1.1)
    coords[:, 1] = np.clip(coords[:, 1] * scale_factor, -1.1, 1.1)
    coords[:, 2] = np.clip(coords[:, 2], -0.6, 0.6)
    return coords.flatten()


def rotate_hand_2d(raw_63, angle_rad):
    """Rotates a 63-element hand pose around the wrist in 2D (for waving / dynamic motion)."""
    coords = np.array(raw_63, dtype=float).reshape(21, 3)
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    x_new = coords[:, 0] * c - coords[:, 1] * s
    y_new = coords[:, 0] * s + coords[:, 1] * c
    coords[:, 0] = x_new
    coords[:, 1] = y_new
    return coords.flatten()


def draw_articulated_hand(canvas, norm_coords, wrist_xy, theme, hand_scale=70.0):
    skin_base = theme["skin"]
    skin_palm = theme["skin_shadow"]
    skin_outline = theme["skin_outline"]

    wx, wy = wrist_xy
    pts_3d = [
        (wx + norm_coords[i * 3] * hand_scale,
         wy + norm_coords[i * 3 + 1] * hand_scale,
         norm_coords[i * 3 + 2])
        for i in range(21)
    ]

    cv2.circle(canvas, (int(wx), int(wy)), 9, skin_palm, -1, cv2.LINE_AA)
    cv2.circle(canvas, (int(wx), int(wy)), 9, skin_outline, 1, cv2.LINE_AA)

    palm_poly = np.array([[int(pts_3d[i][0]), int(pts_3d[i][1])] for i in [0, 1, 5, 9, 13, 17]], dtype=np.int32)
    cv2.fillConvexPoly(canvas, palm_poly, skin_palm, cv2.LINE_AA)
    cv2.polylines(canvas, [palm_poly], True, skin_outline, 1, cv2.LINE_AA)

    segments = []
    for chain in FINGER_CHAINS:
        for a, b, r1, r2 in chain:
            segments.append(((pts_3d[a][2] + pts_3d[b][2]) * 0.5, a, b, r1, r2))
    segments.sort(key=lambda item: item[0], reverse=True)

    for _, a, b, r1, r2 in segments:
        draw_shaded_capsule(
            canvas,
            (pts_3d[a][0], pts_3d[a][1]),
            (pts_3d[b][0], pts_3d[b][1]),
            r1, r2, skin_base, skin_outline, highlight=True
        )


def build_rest_hand_coords(inward_sign=1.0):
    coords = [0.0] * 63
    specs = [
        ([1, 2, 3, 4], 0.24 * inward_sign, 0.14, 0.09 * inward_sign, 0.15),
        ([5, 6, 7, 8], 0.16 * inward_sign, 0.34, 0.02 * inward_sign, 0.16),
        ([9, 10, 11, 12], 0.04 * inward_sign, 0.37, 0.0, 0.17),
        ([13, 14, 15, 16], -0.08 * inward_sign, 0.35, -0.02 * inward_sign, 0.16),
        ([17, 18, 19, 20], -0.19 * inward_sign, 0.31, -0.03 * inward_sign, 0.14),
    ]
    for joints, bx, by, dx, dy in specs:
        for step, j in enumerate(joints):
            coords[j * 3] = bx + dx * step
            coords[j * 3 + 1] = by + dy * step
    return np.array(coords, dtype=float)


class BidirectionalAvatar:
    def __init__(self, pose_file):
        self.poses = {}
        if os.path.isfile(pose_file):
            with open(pose_file, 'r', encoding='utf-8') as f:
                self.poses = json.load(f)

        print(f"Loaded {len(self.poses)} avatar gestures: {', '.join(sorted(self.poses.keys()))}")
        self.theme_idx = 0
        self.custom_bgr = None
        self.custom_alpha = None
        self._load_custom_avatar_image()

        if self.custom_bgr is not None:
            self.theme_idx = len(AVATAR_THEMES) - 1

        self.vec_dim = 207
        self.queue = deque()
        self.current_label = "READY"
        self.is_signing = False
        self.rest_vec = self._make_rest_vector()
        self.current_vec = self.rest_vec.copy()
        self.target_vec = self.rest_vec.copy()
        self.last_switch = time.time()
        self.frame_duration = 0.26
        self.status_text = "Press [V] to Speak, [T] to Type, [A] Avatar"

    def _load_custom_avatar_image(self):
        """
        Loads custom_avatar.png/.jpg, center-crops to upper torso portrait, masks out the static
        hanging arms below the shoulders so only the 2 animated arms appear, and matches skin/hoodie colors.
        """
        target_path = None
        for fname in ('custom_avatar.png', 'custom_avatar.jpg', 'custom_avatar.jpeg'):
            if os.path.isfile(fname):
                target_path = fname
                break

        if not target_path:
            self.custom_bgr = None
            self.custom_alpha = None
            return

        img = cv2.imread(target_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            return

        # Center-crop wide 16:9 portraits to a focused upper-body portrait
        ih_orig, iw_orig = img.shape[:2]
        if iw_orig > int(ih_orig * 1.05):
            crop_w = int(ih_orig * 1.05)
            x_start = (iw_orig - crop_w) // 2
            img = img[:, x_start:x_start + crop_w]

        target_h = 294
        target_w = 308
        resized = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_AREA)

        if resized.shape[2] == 4:
            bgr = resized[:, :, :3].astype(np.float32)
            alpha = (resized[:, :, 3:4].astype(np.float32)) / 255.0
        else:
            bgr = resized.astype(np.float32)
            bg_sample = np.mean([bgr[3, 3], bgr[3, -4], bgr[-4, 3], bgr[-4, -4]], axis=0)
            diff = np.linalg.norm(bgr - bg_sample, axis=2, keepdims=True)
            alpha = np.clip((diff - 16.0) / 24.0, 0.0, 1.0)

        # Mask out the static hanging arms outside the torso below the shoulder line (y >= 134)
        yy, xx = np.mgrid[0:target_h, 0:target_w]
        cx_local = target_w * 0.5
        shoulder_y_local = 136.0
        below_shoulders = yy > shoulder_y_local
        progress = np.clip((yy - shoulder_y_local) / (target_h - shoulder_y_local), 0.0, 1.0)
        allowed_half_w = 66.0 - progress * 12.0  # Smooth torso taper from shoulder to waist
        dist_from_center = np.abs(xx - cx_local)

        arm_mask = np.where(
            below_shoulders,
            np.clip((allowed_half_w + 8.0 - dist_from_center) / 10.0, 0.0, 1.0),
            1.0
        ).astype(np.float32)[:, :, None]

        self.custom_bgr = bgr
        self.custom_alpha = alpha * arm_mask

        # Sample cheek skin tone so animated forearms & 3D hands match his face seamlessly
        cheek_y, cheek_x = int(target_h * 0.28), int(target_w * 0.54)
        if self.custom_alpha[cheek_y, cheek_x, 0] > 0.5:
            b, g, r = [int(v) for v in self.custom_bgr[cheek_y, cheek_x]]
            custom_theme = AVATAR_THEMES[-1]
            custom_theme["skin"] = (b, g, r)
            custom_theme["skin_shadow"] = (max(0, b - 26), max(0, g - 26), max(0, r - 26))
            custom_theme["skin_outline"] = (max(0, b - 54), max(0, g - 54), max(0, r - 54))

        # Sample hoodie fabric color near shoulder so animated sleeves match his hoodie
        sleeve_y, sleeve_x = int(target_h * 0.52), int(target_w * 0.34)
        if self.custom_alpha[sleeve_y, sleeve_x, 0] > 0.5:
            sb, sg, sr = [int(v) for v in self.custom_bgr[sleeve_y, sleeve_x]]
            AVATAR_THEMES[-1]["outfit"] = (sb, sg, sr)
            AVATAR_THEMES[-1]["outfit_border"] = (min(255, sb + 24), min(255, sg + 24), min(255, sr + 24))

        print(f"Loaded custom avatar image: '{target_path}'")

    def cycle_avatar(self):
        self._load_custom_avatar_image()
        self.theme_idx = (self.theme_idx + 1) % len(AVATAR_THEMES)
        print(f"Switched avatar to: {AVATAR_THEMES[self.theme_idx]['name']}")

    def _make_rest_vector(self):
        vec = np.zeros(self.vec_dim, dtype=float)
        vec[:63] = build_rest_hand_coords(inward_sign=1.0)
        vec[87:150] = build_rest_hand_coords(inward_sign=-1.0)
        vec[174 + 27:174 + 30] = [-0.48, 0.92, 0.0]
        vec[174 + 30:174 + 33] = [0.48, 0.92, 0.0]
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

        left_w = (max(-0.85, min(0.18, raw_lw[0])), max(-0.72, min(0.85, raw_lw[1])))
        right_w = (max(-0.18, min(0.85, raw_rw[0])), max(-0.72, min(0.85, raw_rw[1])))

        if left_active:
            padded[:63] = normalize_2d_hand_pose(h_screen_left)
            padded[174 + 27:174 + 29] = [left_w[0], left_w[1]]
        else:
            padded[:63] = build_rest_hand_coords(inward_sign=1.0)
            padded[174 + 27:174 + 29] = [-0.48, 0.92]

        if right_active:
            padded[87:150] = normalize_2d_hand_pose(h_screen_right)
            padded[174 + 30:174 + 32] = [right_w[0], right_w[1]]
        else:
            padded[87:150] = build_rest_hand_coords(inward_sign=-1.0)
            padded[174 + 30:174 + 32] = [0.48, 0.92]

        return padded

    def _apply_kinematic_motion(self, base_vec, word_key, step_idx, total_steps):
        """
        Adds natural multi-phase sign movement (lift -> expressive stroke -> hold)
        so static recorded poses animate smoothly with real arm & wrist motion.
        """
        vec = base_vec.copy()
        left_active = max(abs(v) for v in base_vec[:63]) > 1e-4 and (base_vec[174 + 28] < 0.80)
        right_active = max(abs(v) for v in base_vec[87:150]) > 1e-4 and (base_vec[174 + 31] < 0.80)

        phase = step_idx / max(1, total_steps - 1)  # 0.0 -> 1.0
        dx, dy, wrist_rot = 0.0, 0.0, 0.0

        if step_idx == 0:
            # Preparation lift from lower chest
            dy = 0.24
            wrist_rot = -0.15
        elif 'hello' in word_key:
            # Side-to-side friendly wave
            wave = math.sin(phase * math.pi * 2.5)
            dx = wave * 0.16
            wrist_rot = wave * 0.32
        elif 'yes' in word_key:
            # Up-and-down affirmative nod motion
            nod = math.sin(phase * math.pi * 2.5)
            dy = nod * 0.14
            wrist_rot = nod * 0.22
        elif 'no' in word_key:
            # Side-to-side shake
            shake = math.sin(phase * math.pi * 3.0)
            dx = shake * 0.15
            wrist_rot = shake * 0.25
        elif 'thank' in word_key or 'please' in word_key:
            # Outward & downward courteous arc
            dx = phase * 0.12
            dy = phase * 0.18
            wrist_rot = phase * 0.24
        else:
            # Expressive emphasis pulse for any custom sign
            pulse = math.sin(phase * math.pi * 2.0)
            dx = pulse * 0.09
            dy = -abs(pulse) * 0.10
            wrist_rot = pulse * 0.20

        if left_active:
            vec[174 + 27] = max(-0.88, min(0.20, vec[174 + 27] - dx))
            vec[174 + 28] = max(-0.75, min(0.85, vec[174 + 28] + dy))
            vec[:63] = rotate_hand_2d(vec[:63], -wrist_rot)

        if right_active:
            vec[174 + 30] = max(-0.20, min(0.88, vec[174 + 30] + dx))
            vec[174 + 31] = max(-0.75, min(0.85, vec[174 + 31] + dy))
            vec[87:150] = rotate_hand_2d(vec[87:150], wrist_rot)

        return vec

    def _enqueue_word_motion(self, display_label, pose_entry):
        arr = np.array(pose_entry, dtype=float)
        raw_frames = [arr] if arr.ndim == 1 else list(arr)
        # Expand to 5 smooth animation keyframes with kinematic motion
        expanded = []
        for idx in range(5):
            src_frame = raw_frames[min(len(raw_frames) - 1, int(idx * len(raw_frames) / 5))]
            prepared = self._prepare_pose_vector(src_frame)
            animated = self._apply_kinematic_motion(prepared, display_label.lower(), idx, 5)
            expanded.append(animated)

        for frame_vec in expanded:
            self.queue.append((display_label, frame_vec))

    def enqueue_sentence(self, sentence):
        self.status_text = f'Heard: "{sentence}"'
        clean = sentence.strip(".,!?").lower()
        if not clean:
            return

        phrase_key = clean.replace(" ", "_")
        if phrase_key in self.poses:
            self._enqueue_word_motion(clean.upper(), self.poses[phrase_key])
            return

        words = clean.split()
        i = 0
        while i < len(words):
            matched = False
            for span in (3, 2):
                if i + span <= len(words):
                    sub = "_".join(words[i:i + span])
                    if sub in self.poses:
                        self._enqueue_word_motion(sub.replace('_', ' ').upper(), self.poses[sub])
                        i += span
                        matched = True
                        break
            if matched:
                continue

            raw_word = words[i]
            mapped = SYNONYMS.get(raw_word, raw_word)
            if mapped in self.poses:
                self._enqueue_word_motion(mapped.replace('_', ' ').upper(), self.poses[mapped])
            elif raw_word in self.poses:
                self._enqueue_word_motion(raw_word.replace('_', ' ').upper(), self.poses[raw_word])
            i += 1

    def _draw_character(self, panel, t_now):
        theme = AVATAR_THEMES[self.theme_idx]
        style = theme["style"]
        h, w, _ = panel.shape
        cx = w // 2
        breath = math.sin(t_now * 2.2) * 2.2

        # Stage halo
        cv2.circle(panel, (cx, int(h * 0.44)), 165, (32, 36, 46), -1, cv2.LINE_AA)
        cv2.circle(panel, (cx, int(h * 0.44)), 165, (48, 54, 68), 1, cv2.LINE_AA)

        pose = self.current_vec[174:207]

        # --- Custom Image Character Mode ---
        if style == "custom_image":
            if self.custom_bgr is not None:
                ih, iw, _ = self.custom_bgr.shape
                y1 = int(109 + breath * 0.5)
                y2 = min(h - 116, y1 + ih)
                x1 = cx - iw // 2
                x2 = x1 + iw
                crop_h = y2 - y1
                if crop_h > 0:
                    roi = panel[y1:y2, x1:x2].astype(np.float32)
                    fg = self.custom_bgr[:crop_h]
                    alpha = self.custom_alpha[:crop_h]
                    panel[y1:y2, x1:x2] = (fg * alpha + roi * (1.0 - alpha)).astype(np.uint8)

                # Exact shoulder coordinates aligned to the cropped portrait's hoodie shoulders
                sh_y = y1 + int(ih * 0.49)
                sh_half_w = int(iw * 0.21)
                left_shoulder = (cx - sh_half_w, sh_y)
                right_shoulder = (cx + sh_half_w, sh_y)
                sw_pixels = 164.0
                left_wrist = (cx + pose[27] * sw_pixels, sh_y + pose[28] * sw_pixels)
                right_wrist = (cx + pose[30] * sw_pixels, sh_y + pose[31] * sw_pixels)

                left_elbow = solve_anatomical_ik(left_shoulder, left_wrist, 82, 78, cx, side_sign=-1.0)
                right_elbow = solve_anatomical_ik(right_shoulder, right_wrist, 82, 78, cx, side_sign=1.0)

                # Hoodie sleeves + skin forearms emerging directly from his shoulders
                draw_shaded_capsule(panel, left_shoulder, left_elbow, 17, 14, theme["outfit"], theme["outfit_border"])
                draw_shaded_capsule(panel, right_shoulder, right_elbow, 17, 14, theme["outfit"], theme["outfit_border"])
                draw_shaded_capsule(panel, left_elbow, left_wrist, 12, 9, theme["skin"], theme["skin_outline"])
                draw_shaded_capsule(panel, right_elbow, right_wrist, 12, 9, theme["skin"], theme["skin_outline"])

                draw_articulated_hand(panel, self.current_vec[:63], left_wrist, theme, hand_scale=68.0)
                draw_articulated_hand(panel, self.current_vec[87:150], right_wrist, theme, hand_scale=68.0)
            else:
                cv2.putText(panel, "Save 'custom_avatar.png' in D:\\AuraSign_Clean\\", (62, 240),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.56, (0, 220, 255), 1, cv2.LINE_AA)
                cv2.putText(panel, "and press [A] to load your character!", (115, 272),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.56, (190, 200, 215), 1, cv2.LINE_AA)
            return

        # --- Built-in Vector Characters (Aura, Nova, Titan-X) ---
        sh_y = int(h * 0.43 + breath)
        sh_half_w = 84
        left_shoulder = (cx - sh_half_w, sh_y)
        right_shoulder = (cx + sh_half_w, sh_y)
        head_center = (cx, int(h * 0.25 + breath * 0.8))

        sw_pixels = sh_half_w * 2.0
        left_wrist = (cx + pose[27] * sw_pixels, sh_y + pose[28] * sw_pixels)
        right_wrist = (cx + pose[30] * sw_pixels, sh_y + pose[31] * sw_pixels)

        left_elbow = solve_anatomical_ik(left_shoulder, left_wrist, 86, 82, cx, side_sign=-1.0)
        right_elbow = solve_anatomical_ik(right_shoulder, right_wrist, 86, 82, cx, side_sign=1.0)

        if style == "female":
            back_hair = np.array([
                [head_center[0] - 46, head_center[1] - 10],
                [head_center[0] + 46, head_center[1] - 10],
                [head_center[0] + 54, head_center[1] + 78],
                [head_center[0] - 54, head_center[1] + 78],
            ], dtype=np.int32)
            cv2.fillConvexPoly(panel, back_hair, theme["hair"], cv2.LINE_AA)

        torso_bottom_y = h - 116
        torso_pts = np.array([
            [left_shoulder[0] - 12, sh_y + 6],
            [right_shoulder[0] + 12, sh_y + 6],
            [cx + 74, torso_bottom_y],
            [cx - 74, torso_bottom_y],
        ], dtype=np.int32)
        cv2.fillConvexPoly(panel, torso_pts, theme["outfit"], cv2.LINE_AA)
        cv2.polylines(panel, [torso_pts], True, theme["outfit_border"], 2, cv2.LINE_AA)

        if style == "robot":
            cv2.circle(panel, (cx, sh_y + 46), 16, (28, 34, 44), -1, cv2.LINE_AA)
            cv2.circle(panel, (cx, sh_y + 46), 11, theme["shirt"], -1, cv2.LINE_AA)
            cv2.line(panel, (cx - 42, sh_y + 12), (cx - 18, sh_y + 46), theme["outfit_border"], 2, cv2.LINE_AA)
            cv2.line(panel, (cx + 42, sh_y + 12), (cx + 18, sh_y + 46), theme["outfit_border"], 2, cv2.LINE_AA)
        else:
            shirt_pts = np.array([[cx - 25, sh_y - 4], [cx + 25, sh_y - 4], [cx, sh_y + 68]], dtype=np.int32)
            cv2.fillConvexPoly(panel, shirt_pts, theme["shirt"], cv2.LINE_AA)

        cv2.rectangle(panel, (cx - 17, head_center[1] + 28), (cx + 17, sh_y + 8), theme["skin_shadow"], -1)
        cv2.ellipse(panel, (head_center[0] - 39, head_center[1] + 4), (7, 13), 0, 0, 360, theme["skin_shadow"], -1, cv2.LINE_AA)
        cv2.ellipse(panel, (head_center[0] + 39, head_center[1] + 4), (7, 13), 0, 0, 360, theme["skin_shadow"], -1, cv2.LINE_AA)

        if style == "robot":
            cv2.ellipse(panel, head_center, (38, 46), 0, 0, 360, theme["skin"], -1, cv2.LINE_AA)
            cv2.ellipse(panel, head_center, (38, 46), 0, 0, 360, theme["outfit_border"], 2, cv2.LINE_AA)
            cv2.rectangle(panel, (head_center[0] - 28, head_center[1] - 12),
                          (head_center[0] + 28, head_center[1] + 4), (24, 30, 38), -1)
            eye_w = 10 if (t_now % 4.2) <= 4.05 else 2
            cv2.ellipse(panel, (head_center[0] - 14, head_center[1] - 4), (8, eye_w // 2 + 2), 0, 0, 360, theme["shirt"], -1, cv2.LINE_AA)
            cv2.ellipse(panel, (head_center[0] + 14, head_center[1] - 4), (8, eye_w // 2 + 2), 0, 0, 360, theme["shirt"], -1, cv2.LINE_AA)
            eq_w = int(8 + (abs(math.sin(t_now * 10.0)) * 12 if self.is_signing else 0))
            cv2.line(panel, (cx - eq_w, head_center[1] + 22), (cx + eq_w, head_center[1] + 22), theme["shirt"], 3, cv2.LINE_AA)
        else:
            cv2.ellipse(panel, head_center, (39, 47), 0, 0, 360, theme["skin"], -1, cv2.LINE_AA)
            cv2.ellipse(panel, (head_center[0], head_center[1] - 14), (41, 35), 0, 180, 360, theme["hair"], -1, cv2.LINE_AA)

            if style == "female":
                cv2.ellipse(panel, (head_center[0] - 22, head_center[1] - 18), (22, 16), -20, 0, 360, theme["hair"], -1, cv2.LINE_AA)
                cv2.ellipse(panel, (head_center[0] + 22, head_center[1] - 18), (22, 16), 20, 0, 360, theme["hair"], -1, cv2.LINE_AA)

            brow_lift = -3 if self.is_signing else 0
            cv2.line(panel, (head_center[0] - 23, head_center[1] - 12 + brow_lift),
                     (head_center[0] - 8, head_center[1] - 14 + brow_lift), theme["hair"], 2, cv2.LINE_AA)
            cv2.line(panel, (head_center[0] + 8, head_center[1] - 14 + brow_lift),
                     (head_center[0] + 23, head_center[1] - 12 + brow_lift), theme["hair"], 2, cv2.LINE_AA)

            blink = (t_now % 4.2) > 4.05
            for eye_dx in (-15, 15):
                ex, ey = head_center[0] + eye_dx, head_center[1] - 3
                if blink:
                    cv2.line(panel, (ex - 6, ey), (ex + 6, ey), (75, 65, 60), 2, cv2.LINE_AA)
                else:
                    cv2.ellipse(panel, (ex, ey), (7, 5), 0, 0, 360, (245, 245, 245), -1, cv2.LINE_AA)
                    cv2.circle(panel, (ex, ey), 3, (45, 38, 34), -1, cv2.LINE_AA)

            mouth_y = head_center[1] + 22
            if self.is_signing:
                mouth_open = int(3 + abs(math.sin(t_now * 9.0)) * 4)
                cv2.ellipse(panel, (head_center[0], mouth_y), (9, mouth_open), 0, 0, 360, (95, 78, 125), -1, cv2.LINE_AA)
            else:
                cv2.ellipse(panel, (head_center[0], mouth_y - 2), (10, 5), 0, 15, 165, (105, 85, 115), 2, cv2.LINE_AA)

        draw_shaded_capsule(panel, left_shoulder, left_elbow, 16, 13, theme["outfit"], theme["outfit_border"])
        draw_shaded_capsule(panel, right_shoulder, right_elbow, 16, 13, theme["outfit"], theme["outfit_border"])
        draw_shaded_capsule(panel, left_elbow, left_wrist, 12, 9, theme["skin"], theme["skin_outline"])
        draw_shaded_capsule(panel, right_elbow, right_wrist, 12, 9, theme["skin"], theme["skin_outline"])

        if style == "robot":
            cv2.circle(panel, (int(left_elbow[0]), int(left_elbow[1])), 11, theme["outfit_border"], 2, cv2.LINE_AA)
            cv2.circle(panel, (int(right_elbow[0]), int(right_elbow[1])), 11, theme["outfit_border"], 2, cv2.LINE_AA)

        draw_articulated_hand(panel, self.current_vec[:63], left_wrist, theme, hand_scale=72.0)
        draw_articulated_hand(panel, self.current_vec[87:150], right_wrist, theme, hand_scale=72.0)

    def update_and_render(self, panel):
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

        self.current_vec = self.current_vec * 0.55 + self.target_vec * 0.45

        theme = AVATAR_THEMES[self.theme_idx]
        h, w, _ = panel.shape
        cv2.putText(panel, f"AVATAR: {theme['name']}", (25, 42),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.76, theme["accent"], 2)
        cv2.putText(panel, self.status_text[:48], (25, 78),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.56, (200, 200, 200), 1)
        cv2.rectangle(panel, (30, 105), (w - 30, h - 115), (45, 48, 56), 2)

        self._draw_character(panel, now)

        cv2.rectangle(panel, (30, h - 115), (w - 30, h), (24, 26, 32), -1)
        cv2.putText(panel, f"Performing: {self.current_label}", (35, h - 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.74, (120, 255, 160), 2)
        cv2.putText(panel, "[V] Voice   [T] Type   [A] Change Avatar", (35, h - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1)


def listen_microphone_async(avatar):
    def _worker():
        recognizer = sr.Recognizer()
        sample_rate = 16000
        duration = 4.0
        avatar.status_text = "Listening... Speak now (4s)..."
        try:
            recording = sd.rec(int(duration * sample_rate), samplerate=sample_rate,
                               channels=1, dtype='int16')
            sd.wait()
            audio_data = sr.AudioData(recording.tobytes(), sample_rate, 2)
            text = recognizer.recognize_google(audio_data)
            avatar.enqueue_sentence(text)
        except sr.UnknownValueError:
            avatar.status_text = "Could not understand audio. Press [V] to retry."
        except Exception as e:
            avatar.status_text = f"Mic/ASR error: {str(e)[:32]}"

    threading.Thread(target=_worker, daemon=True).start()


def main():
    if not os.path.isfile(MODEL_FILE):
        print(f"Missing '{MODEL_FILE}'. Run train.py first!")
        return

    clf = joblib.load(MODEL_FILE)
    avatar = BidirectionalAvatar(AVATAR_FILE)

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

    cap = cv2.VideoCapture(0)
    pred_history = deque(maxlen=SMOOTHING_WINDOW)
    gloss_stream = []
    completed_sentence = ""
    last_added_token = None
    stable_count = 0
    show_dots = True

    print("\n=== AuraSign Clean Studio Running ===")
    print("Controls: [SPACE] Speak Full Text | [BACKSPACE] Undo | [C] Clear | [A] Change Avatar | [V] Voice | [T] Type | [Q] Quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        cam_panel = cv2.resize(frame, (640, 520))
        avatar_panel = np.full((520, 600, 3), (24, 26, 32), dtype=np.uint8)

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
                        if not gloss_stream or gloss_stream[-1] != live_label:
                            gloss_stream.append(live_label)
                            completed_sentence = gloss_to_sentence(gloss_stream)
            else:
                live_label = "Neutral / Resting"
                pred_history.clear()
                stable_count = 0
        else:
            pred_history.clear()
            last_added_token = None
            stable_count = 0

        draw_detection_overlay(cam_panel, hand_res, pose_res, live_label, confidence, show_dots=show_dots)

        cv2.rectangle(cam_panel, (0, 0), (640, 85), (20, 22, 28), -1)
        cv2.putText(cam_panel, "HOLISTIC SIGN -> SENTENCE (LIVE)", (20, 34),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.78, (0, 255, 140), 2)
        cv2.putText(cam_panel, f"Status: {live_label} ({confidence * 100:.0f}%)", (20, 68),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2)

        cv2.rectangle(cam_panel, (0, 405), (640, 520), (20, 22, 28), -1)
        gloss_str = " -> ".join(gloss_stream[-6:]) if gloss_stream else "(Hold a sign clearly to add...)"
        sent_display = completed_sentence[-48:] if completed_sentence else "(Press [SPACE] when finished to speak all)"
        cv2.putText(cam_panel, f"Glosses ({len(gloss_stream)}): {gloss_str}", (20, 435),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.53, (180, 220, 255), 1)
        cv2.putText(cam_panel, f"Text: {sent_display}", (20, 470),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 230, 255), 2)
        cv2.putText(cam_panel, "[SPACE] Speak All  [BKSP] Undo  [C] Clear  [Q] Quit", (20, 503),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (180, 180, 180), 1)

        avatar.update_and_render(avatar_panel)
        cv2.imshow("AuraSign - Two-Way Sign Studio", np.hstack((cam_panel, avatar_panel)))

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' '):
            full_text = completed_sentence or " ".join(gloss_stream)
            if full_text:
                print(f"[Speaking]: {full_text}")
                speak_text_async(full_text)
        elif key in (8, 127):
            if gloss_stream:
                gloss_stream.pop()
                completed_sentence = gloss_to_sentence(gloss_stream)
        elif key == ord('c'):
            gloss_stream.clear()
            completed_sentence = ""
        elif key == ord('d'):
            show_dots = not show_dots
        elif key == ord('a'):
            avatar.cycle_avatar()
        elif key == ord('v'):
            listen_microphone_async(avatar)
        elif key == ord('t'):
            typed = input("\nType phrase for Avatar to sign: ").strip()
            if typed:
                avatar.enqueue_sentence(typed)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()