#!/usr/bin/env python3
"""Stage 4 -- build the ARKit blendshape set on the refined neutral
(system python, numpy only).

ICT is LINEAR, so each ARKit shape is an ADDITIVE delta valid on any neutral
of the same topology:

    shape_k = refined_neutral + sum_over_sources( ICT_source_obj - generic_neutral )

Names follow the exact ARKit-52 contract (arkit_names.py): `_L/_R` becomes
`Left/Right`; bilateral singles (browInnerUp, cheekPuff) fold `_L + _R` into
ONE summed delta when ICT ships them split; non-ARKit extras are dropped and
listed; `tongueOut` is declared unsupported (ICT has no tongue) -- 51/52.

Outputs under out/rig/:
  arkit_deltas.npz     names (S,), deltas (S,N,3) f32, refined/generic neutral,
                       topology + UVs (everything s5/s6 need -- they never
                       touch the big cache)
  arkit_manifest.json  all 52 names, supported/unsupported, sources,
                       max |delta| per shape, dropped ICT extras
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from arkit_names import ARKIT_52, UNSUPPORTED_REASON, resolve_sources  # noqa: E402
from common import N_VERTS, P, assert_topology, die, out_dir, save_json  # noqa: E402
from ict_loader import get_cache  # noqa: E402

FAIL_DELTA_CM = 1e-4   # an all-zero delta means the source axis was wrong
WARN_DELTA_CM = 0.05


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ict", default=P.ICT)
    ap.add_argument("--out", default=P.OUT)
    ap.add_argument("--neutral", default=None,
                    help="path to neutral .npy (default: out/refine/refined_neutral.npy, "
                         "falling back to out/fit/fitted_neutral.npy)")
    args = ap.parse_args()
    t0 = time.time()
    od = out_dir(args.out, "rig")

    if args.neutral:
        neutral_path = Path(args.neutral)
    else:
        neutral_path = Path(args.out) / "refine" / "refined_neutral.npy"
        if not neutral_path.is_file():
            neutral_path = Path(args.out) / "fit" / "fitted_neutral.npy"
            print("[s4 WARN] no refined neutral; using fitted neutral")
    if not neutral_path.is_file():
        die(f"neutral not found: {neutral_path}")
    neutral = np.load(neutral_path)
    assert_topology(neutral, f"neutral ({neutral_path})")
    print(f"[s4] neutral: {neutral_path}")

    z = get_cache(args.ict, Path(args.out) / "cache" / "ict_model_cache.npz")
    generic = z["generic"]
    expr_names = [str(s) for s in z["expr_names"]]
    expr_deltas = z["expr_deltas"]  # (E,N,3) = obj - generic, float32
    name_to_row = {n: i for i, n in enumerate(expr_names)}

    mapping, dropped = resolve_sources(set(expr_names))
    if dropped:
        print(f"[s4] dropped non-ARKit ICT shapes ({len(dropped)}): {', '.join(dropped)}")

    sup_names, sup_deltas = [], []
    manifest_shapes = {}
    for name in ARKIT_52:
        sources = mapping[name]
        if not sources:
            reason = UNSUPPORTED_REASON.get(
                name, "no ICT-FaceKit source OBJ found for this ARKit shape")
            manifest_shapes[name] = {"supported": False, "sources": [],
                                     "reason": reason}
            print(f"[s4]   {name:22s} UNSUPPORTED -- {reason}")
            continue
        delta = np.zeros((N_VERTS, 3), dtype=np.float64)
        for s in sources:
            delta += expr_deltas[name_to_row[s]].astype(np.float64)
        mx = float(np.linalg.norm(delta, axis=1).max())
        if mx < FAIL_DELTA_CM:
            die(f"shape {name} (sources {sources}) has max|delta|={mx:.2e} cm "
                "-- effectively zero; wrong source axis. NOT fabricating.")
        if mx < WARN_DELTA_CM:
            print(f"[s4 WARN] {name}: suspiciously small max|delta|={mx:.3f} cm")
        sup_names.append(name)
        sup_deltas.append(delta.astype(np.float32))
        fold = "  (folded)" if len(sources) > 1 else ""
        print(f"[s4]   {name:22s} <- {'+'.join(sources):28s} "
              f"max|d|={mx:6.3f} cm{fold}")
        manifest_shapes[name] = {"supported": True, "sources": sources,
                                 "max_delta_cm": round(mx, 4)}

    n_sup = len(sup_names)
    print(f"[s4] supported {n_sup}/52; unsupported: "
          + ", ".join(n for n in ARKIT_52 if not manifest_shapes[n]["supported"]))
    if n_sup < 45:
        die(f"only {n_sup} supported shapes -- ICT install looks broken")

    np.savez(
        od / "arkit_deltas.npz",
        names=np.array(sup_names),
        deltas=np.stack(sup_deltas, axis=0),
        refined_neutral=neutral.astype(np.float32),
        generic_neutral=generic.astype(np.float32),
        faces_flat=z["faces_flat"], faces_off=z["faces_off"],
        corner_vt=z["corner_vt"], vt=z["vt"].astype(np.float32),
        lmk_verts=z["lmk_verts"],
        neutral_source=np.array(str(neutral_path)),
    )
    print(f"[s4] deltas npz -> {od / 'arkit_deltas.npz'} "
          f"({n_sup} x {N_VERTS} x 3 float32)")

    save_json(od / "arkit_manifest.json", {
        "contract": "ARKit-52 exact names; GLB morph targets == MediaPipe categoryNames",
        "arkit_total": 52,
        "supported": n_sup,
        "unsupported": [n for n in ARKIT_52 if not manifest_shapes[n]["supported"]],
        "shapes": manifest_shapes,
        "dropped_ict_shapes": dropped,
        "topology": {"n_verts": N_VERTS, "n_polys": int(len(z["faces_off"]) - 1)},
        "neutral_source": str(neutral_path),
        "license": "ICT-FaceKit Light (MIT, FaceXModel only) -- no NC assets",
    })
    print(f"[s4] DONE in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
