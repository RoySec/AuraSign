import math

FINGERTIPS = [4, 8, 12, 16, 20]
PIP_JOINTS = [3, 6, 10, 14, 18]
MCP_JOINTS = [2, 5, 9, 13, 17]
UPPER_BODY_INDICES = [0, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]


def _single_hand_features(hand_landmarks):
    wrist = hand_landmarks[0]
    raw_coords = []
    for lm in hand_landmarks:
        raw_coords.extend([lm.x - wrist.x, lm.y - wrist.y, lm.z - wrist.z])

    max_val = max(abs(v) for v in raw_coords) or 1.0
    norm_coords = [v / max_val for v in raw_coords]  # 63 features

    geom = []
    # Tip-to-tip distances (4 features)
    for i in range(len(FINGERTIPS) - 1):
        t1, t2 = FINGERTIPS[i] * 3, FINGERTIPS[i + 1] * 3
        geom.append(math.sqrt(
            (norm_coords[t1] - norm_coords[t2]) ** 2 +
            (norm_coords[t1 + 1] - norm_coords[t2 + 1]) ** 2 +
            (norm_coords[t1 + 2] - norm_coords[t2 + 2]) ** 2
        ))

    # Finger curl ratios (5 features)
    for tip_idx, pip_idx in zip(FINGERTIPS, PIP_JOINTS):
        t, p = tip_idx * 3, pip_idx * 3
        tip_dist = math.sqrt(norm_coords[t] ** 2 + norm_coords[t + 1] ** 2 + norm_coords[t + 2] ** 2)
        pip_dist = math.sqrt(norm_coords[p] ** 2 + norm_coords[p + 1] ** 2 + norm_coords[p + 2] ** 2) or 1e-6
        geom.append(tip_dist / pip_dist)

    # Knuckle-to-tip directional vectors (15 features)
    bones = []
    for tip, mcp in zip(FINGERTIPS, MCP_JOINTS):
        t, m = tip * 3, mcp * 3
        bones.extend([
            norm_coords[t] - norm_coords[m],
            norm_coords[t + 1] - norm_coords[m + 1],
            norm_coords[t + 2] - norm_coords[m + 2],
        ])

    return norm_coords + geom + bones  # 87 features


def extract_pose_features(pose_results):
    """33 shoulder-normalized upper-body & head landmarks."""
    if not pose_results or not pose_results.pose_landmarks:
        return [0.0] * 33

    lms = pose_results.pose_landmarks[0]
    l_sh, r_sh = lms[11], lms[12]
    cx = (l_sh.x + r_sh.x) / 2.0
    cy = (l_sh.y + r_sh.y) / 2.0
    cz = (l_sh.z + r_sh.z) / 2.0
    sh_width = math.hypot(l_sh.x - r_sh.x, l_sh.y - r_sh.y) or 0.25

    feats = []
    for idx in UPPER_BODY_INDICES:
        lm = lms[idx]
        feats.extend([
            (lm.x - cx) / sh_width,
            (lm.y - cy) / sh_width,
            (lm.z - cz) / sh_width,
        ])
    return feats


def extract_holistic_features(hand_results, pose_results=None):
    """
    207-feature vector with locked hand slots:
    [Slot 0: Right Hand (87), Slot 1: Left Hand (87), Upper Body/Head (33)].
    """
    right_hand = [0.0] * 87
    left_hand = [0.0] * 87

    if hand_results and hand_results.hand_landmarks:
        handedness_list = getattr(hand_results, 'handedness', [])
        for idx, hand_lms in enumerate(hand_results.hand_landmarks):
            label = 'Right'
            if idx < len(handedness_list) and handedness_list[idx]:
                label = handedness_list[idx][0].category_name

            vec = _single_hand_features(hand_lms)
            if label == 'Right':
                right_hand = vec
            else:
                left_hand = vec

    return right_hand + left_hand + extract_pose_features(pose_results)