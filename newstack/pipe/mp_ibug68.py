"""MediaPipe-478 -> iBUG/Multi-PIE-68 correspondence (SELF-AUTHORED, clean provenance).

Copied verbatim from this repo's recon/mp_flame_correspondence.py, where it was
authored for the commercial run by reading vertex indices off MediaPipe's
published canonical face mesh (Apache-2.0) against the published iBUG-68 point
definitions. No third-party correspondence file was copied.

ICT-FaceKit's `idx_to_landmark_verts` follows the Multi-PIE 68-point markup,
which is the same point layout as iBUG-68, so this table pairs the photo's
MediaPipe landmarks directly with ICT landmark vertices, index for index.

Verify visually: s1 writes out/landmarks/overlay.jpg with all 68 picks numbered.

Conventions: "Left"/"Right" are SUBJECT-anatomical. iBUG 0-based groups:
0-16 jaw contour, 17-21 right brow, 22-26 left brow, 27-30 nose bridge
(30 = tip), 31-35 nose base, 36-41 right eye, 42-47 left eye,
48-59 outer lips, 60-67 inner lips.
"""

import numpy as np

MEDIAPIPE_IBUG68 = np.array(
    [
        # jaw / face contour, subject-RIGHT down to chin to subject-LEFT (0-16)
        234, 93, 132, 58, 172, 136, 150, 176,
        152,                                   # 8 chin center (menton)
        400, 379, 365, 397, 288, 361, 323, 454,
        # subject-RIGHT eyebrow, outer -> inner (17-21)
        70, 63, 105, 66, 107,
        # subject-LEFT eyebrow, inner -> outer (22-26)
        336, 296, 334, 293, 300,
        # nose bridge, nasion -> tip (27-30)
        168, 6, 197, 4,
        # nose base row, subject-right alar -> subject-left alar (31-35)
        98, 97, 2, 326, 327,
        # subject-RIGHT eye, outer corner CCW (36-41)
        33, 160, 158, 133, 153, 144,
        # subject-LEFT eye, inner corner CW (42-47)
        362, 385, 387, 263, 373, 380,
        # OUTER lips (48-59)
        61, 40, 37, 0, 267, 270, 291, 321, 314, 17, 84, 91,
        # INNER lips (60-67)
        78, 81, 13, 311, 308, 402, 14, 178,
    ],
    dtype=np.int64,
)
assert MEDIAPIPE_IBUG68.shape == (68,), "iBUG-68 table must have exactly 68 entries"
assert len(set(MEDIAPIPE_IBUG68.tolist())) == 68, "iBUG-68 table has duplicate MediaPipe indices"

IBUG_GROUPS = {
    "contour": list(range(0, 17)),
    "brow_right": list(range(17, 22)),
    "brow_left": list(range(22, 27)),
    "nose_bridge": list(range(27, 31)),
    "nose_base": list(range(31, 36)),
    "eye_right": list(range(36, 42)),
    "eye_left": list(range(42, 48)),
    "lips_outer": list(range(48, 60)),
    "lips_inner": list(range(60, 68)),
}

# Contour is soft: MediaPipe's face-oval points are the visible silhouette,
# which only approximates the iBUG jawline.
IBUG_GROUP_WEIGHTS = {
    "contour": 0.3,
    "brow_right": 0.8,
    "brow_left": 0.8,
    "nose_bridge": 1.5,
    "nose_base": 1.5,
    "eye_right": 2.0,
    "eye_left": 2.0,
    "lips_outer": 1.5,
    "lips_inner": 1.0,
}

# iBUG index subsets used for shrinkwrap protection masks (s3b).
IBUG_EYES = list(range(36, 48))
IBUG_MOUTH = list(range(48, 68))
IBUG_NOSE = list(range(27, 36))


def per_landmark_weights() -> np.ndarray:
    """(68,) float32 weight vector from IBUG_GROUP_WEIGHTS."""
    w = np.zeros(68, dtype=np.float32)
    for group, idxs in IBUG_GROUPS.items():
        w[idxs] = IBUG_GROUP_WEIGHTS[group]
    assert (w > 0).all(), "every iBUG landmark must belong to a weighted group"
    return w
