#!/usr/bin/env python3
"""Stage 1 -- MediaPipe FaceLandmarker on the input photo (system python).

Outputs under out/landmarks/:
  input_image.png   EXIF-normalized RGB decode -- the SINGLE SOURCE OF PIXELS
                    for every later stage (s2 debug overlay, s5 bake).
  landmarks.npz     lmk478_norm, lmk478_px, ibug68_px, ibug68_mp_idx, image_hw
  overlay.jpg       all 478 points faint + the 68 correspondence picks numbered
                    (REQUIRED human check of the self-authored MP<->iBUG table)
  mp_blendshapes_photo.json   MediaPipe's ARKit-named scores for the PHOTO
                    (reference/QA only -- says nothing about rig coverage)
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import P, detect_face, die, landmarks_to_np, out_dir, save_json  # noqa: E402
from mp_ibug68 import MEDIAPIPE_IBUG68  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--photo", default=P.PHOTO)
    ap.add_argument("--task", default=P.MP_TASK)
    ap.add_argument("--out", default=P.OUT)
    ap.add_argument("--min-conf", type=float, default=0.5)
    args = ap.parse_args()
    t0 = time.time()
    od = out_dir(args.out, "landmarks")

    from PIL import Image, ImageOps
    if not Path(args.photo).is_file():
        die(f"input photo not found: {args.photo}")
    img = Image.open(args.photo)
    img = ImageOps.exif_transpose(img)  # bake EXIF rotation into pixels
    rgb = np.asarray(img.convert("RGB"), dtype=np.uint8)
    h, w = rgb.shape[:2]
    canonical = od / "input_image.png"
    Image.fromarray(rgb).save(canonical)
    print(f"[s1] canonical image {w}x{h} -> {canonical}")

    result = detect_face(rgb, args.task, args.min_conf)
    if result is None:
        die("MediaPipe detected NO face in the input photo -- cannot proceed")
    if len(result.face_landmarks) > 1:
        print(f"[s1 WARN] {len(result.face_landmarks)} faces; using face 0")
    lmk478_norm = landmarks_to_np(result)
    n = len(lmk478_norm)
    if n < int(MEDIAPIPE_IBUG68.max()) + 1:
        die(f"FaceLandmarker returned {n} landmarks; correspondence table "
            f"needs >= {int(MEDIAPIPE_IBUG68.max()) + 1}. Wrong .task model?")
    print(f"[s1] {n} landmarks (expected 478)")

    lmk478_px = np.stack([lmk478_norm[:, 0] * w, lmk478_norm[:, 1] * h], axis=1)
    ibug68_px = lmk478_px[MEDIAPIPE_IBUG68]
    np.savez(od / "landmarks.npz",
             lmk478_norm=lmk478_norm, lmk478_px=lmk478_px,
             ibug68_px=ibug68_px, ibug68_mp_idx=MEDIAPIPE_IBUG68,
             image_hw=np.array([h, w], dtype=np.int64),
             input_image=str(canonical))
    print(f"[s1] landmarks -> {od / 'landmarks.npz'}")

    # MediaPipe's ARKit-named blendshape scores for the photo (reference only)
    scores = {}
    if result.face_blendshapes:
        scores = {c.category_name: float(c.score)
                  for c in result.face_blendshapes[0]}
    save_json(od / "mp_blendshapes_photo.json",
              {"note": "photo expression state reference ONLY", "scores": scores})

    import cv2
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    for x, y in lmk478_px:
        cv2.circle(bgr, (int(round(x)), int(round(y))), 1, (90, 90, 90), -1)
    for ibug_idx, mp_idx in enumerate(MEDIAPIPE_IBUG68):
        x, y = lmk478_px[mp_idx]
        p = (int(round(x)), int(round(y)))
        cv2.circle(bgr, p, 3, (0, 255, 0), -1)
        cv2.putText(bgr, str(ibug_idx), (p[0] + 3, p[1] - 3),
                    cv2.FONT_HERSHEY_PLAIN, 0.9, (0, 0, 255), 1, cv2.LINE_AA)
    cv2.imwrite(str(od / "overlay.jpg"), bgr)
    print(f"[s1] overlay -> {od / 'overlay.jpg'}")
    print("[s1] ACTION: eyeball overlay.jpg -- every numbered pick must sit on "
          "the right facial feature.")
    print(f"[s1] DONE in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
