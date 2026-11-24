# AuraSign Studio — Two-Way Real-Time Bangla Sign Language (BdSL) Interpreter

**AuraSign** is an end-to-end, bidirectional Bangla Sign Language (BdSL) translation and 3D avatar signing studio. It bridges communication between Deaf/hard-of-hearing signers and non-signing speakers in real time using computer vision, hybrid machine learning (supervised MLP + zero-retrain few-shot learning), neural Bengali speech synthesis, and analytical 3D inverse kinematics.

---

## Core Architecture & Key Innovations

### 1. Sign Gesture → Bangla Text & Neural Speech (`app.py`, `features.py`, `train.py`)
* **Holistic Landmark Feature Engineering (`features.py`):** Extracts 207-dimensional normalized spatial feature vectors per frame using MediaPipe `HandLandmarker` (both hands, 21 landmarks × 3D) and `PoseLandmarker` (upper-body shoulder, elbow, wrist, and head anchors).
* **Hybrid Two-Tier Classifier (`app.py`, `sign_model.pkl`):**
  * **Primary Supervised Model:** A trained `scikit-learn` classifier (`sign_model.pkl`) trained on `bangla_dataset.json` with temporal smoothing (`SMOOTHING_WINDOW = 8`), confidence gating (`MIN_CONFIDENCE = 0.76`), and margin thresholding (`MIN_PROB_MARGIN = 0.22`) to prevent false triggers.
  * **Zero-Retrain Few-Shot Enrollment (`FewShotSignStore` / `custom_signs.json`):** Allows users to enroll new regional, medical, or emergency signs live via a **5-second webcam capture**—computing unit-normalized centroids and multi-frame cosine similarity prototypes (`PROTOTYPE_SIM_THRESHOLD = 0.86`) for immediate recognition and avatar playback without retraining the base classifier.
* **Bangla Gloss-to-Sentence & Neural TTS:** Converts stabilized sign glosses into punctuated Bengali sentences and speaks them aloud using `edge-tts` (`bn-BD-NabanitaNeural`) with local audio caching.

### 2. Bangla Voice / Text → 3D Sign Avatar (`avatar_3d.html`, `app.py`, `character/character.gltf`)
* **Low-Latency WebSocket Motion Streaming:** Streams interpolated 207-D pose and hand trajectories at ~30 FPS over `ws://127.0.0.1:8765` to a Three.js WebGL stage.
* **2-Bone Analytical Inverse Kinematics & Finger Curl Retargeting:** Solves shoulder-elbow-wrist 2-bone IK in world space and maps 21-point hand landmarks into individual 3-joint finger curls (thumb, index, middle, ring, pinky) and wrist pitch/yaw/roll orientations.
* **Chest-to-Head Signing Space Calibration:** Normalizes recorded webcam wrist coordinates relative to the signer's nose and shoulder span so all active signs stay naturally framed between mid-chest and temple/forehead height.
* **Multimodal Input:** Supports direct Bangla text input, one-click vocabulary pills, synonym mapping (`BANGLA_SYNONYMS`), and live Bengali speech recognition (`bn-BD` via `SpeechRecognition`).

---

## Repository Structure

| File / Directory | Purpose |
| :--- | :--- |
| `app.py` | Main studio backend: HTTP/WebSocket servers, live webcam inference, 5s sign capture, voice ASR, and Bangla TTS. |
| `avatar_3d.html` | Split-screen Three.js 3D humanoid avatar studio, analytical IK solver, and zero-retrain enrollment UI. |
| `collect.py` | CLI 5-second motion capture tool for recording new gesture sequences into `bangla_dataset.json`. |
| `train.py` | Trains and serializes the gesture classifier (`sign_model.pkl`) from recorded landmark datasets. |
| `features.py` | Shared holistic landmark normalization and feature extraction pipeline (207-D vector). |
| `bangla_dataset.json` | Recorded Bangla Sign Language landmark sequences used for training and avatar replay. |
| `custom_signs.json` | Persisted zero-retrain few-shot sign prototypes enrolled from the live web studio. |
| `sign_model.pkl` | Pre-trained `scikit-learn` classification model ready for immediate inference. |
| `character/` | 3D humanoid avatar rig (`character.gltf`, `buffer.bin`, and PBR textures). |
| `hand_landmarker.task` & `pose_landmarker_lite.task` | MediaPipe Tasks model bundles for real-time hand and upper-body pose tracking. |
| `requirements.txt` | Python package dependencies. |

---

## Quickstart & Local Execution

### Prerequisites
* Python 3.10+
* A working webcam and microphone (for live sign recognition and Bangla voice input)

### Installation & Launch
```bash
# 1. Clone the repository
git clone [https://github.com/RoySec/AuraSign.git](https://github.com/RoySec/AuraSign.git)
cd AuraSign

# 2. Create and activate a virtual environment (recommended)
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/macOS:
# source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Launch AuraSign Studio
python app.py