"""Stage 3b -- shrinkwrap refinement of the fitted neutral onto the smoothed
clay (RUNS INSIDE BLENDER):

  xvfb-run -a blender --background --factory-startup \
      --python pipe/s3b_refine_blender.py -- [args]

Why Blender: the plan calls for Blender's Shrinkwrap (NEAREST_SURFACEPOINT)
against a low-passed clay. Noise control so the lumpy clay can't wreck ICT's
clean face, in order:
  1. the clay TARGET is smoothed first (Smooth modifier, evaluated in the
     depsgraph -- the shrinkwrap sees the smoothed surface);
  2. raw per-vertex displacements > --cutoff are discarded (clay missing
     there, e.g. below the neck);
  3. displacement magnitudes are clamped to --max-disp;
  4. the displacement FIELD is graph-Laplacian smoothed over ICT topology
     (only low-frequency shape/hair volume transfers, not lumps);
  5. a region+feature mask gates the result: interior verts (eyeballs, teeth,
     sockets; ids >= 11248) get ZERO, eye/mouth/nose vicinities are protected
     with a smoothstep falloff, the rest of the face gets --face-weight,
     scalp/neck gets 1.0. The mask itself is smoothed to avoid creases.

refined = fitted + mask * smoothed(clamped(delta)).  Same 26719-vert topology,
asserted. --no-shrinkwrap passes fitted through unchanged (A/B fallback).

Outputs under out/refine/: refined_neutral.npy/.obj, refine_stats.json,
refine_debug.npz (mask + displacement magnitudes).
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (EXTERIOR_END, P, assert_topology, edges_from_faces,  # noqa: E402
                    faces_as_lists, min_dist_to_points, out_dir, smooth_field,
                    smoothstep, write_obj)
from mp_ibug68 import IBUG_EYES, IBUG_MOUTH, IBUG_NOSE  # noqa: E402

import bpy  # noqa: E402


def parse_args():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    ap = argparse.ArgumentParser(description="s3b shrinkwrap refine")
    ap.add_argument("--out", default=P.OUT)
    ap.add_argument("--no-shrinkwrap", action="store_true")
    ap.add_argument("--clay-smooth-iters", type=int, default=25)
    ap.add_argument("--clay-smooth-factor", type=float, default=0.5)
    ap.add_argument("--cutoff", type=float, default=6.0, help="cm; discard bigger raw deltas")
    ap.add_argument("--max-disp", type=float, default=3.0, help="cm; clamp delta magnitude")
    ap.add_argument("--smooth-iters", type=int, default=12)
    ap.add_argument("--smooth-lam", type=float, default=0.6)
    ap.add_argument("--face-weight", type=float, default=0.35,
                    help="shrink influence on the face region (scalp/neck = 1.0)")
    ap.add_argument("--protect-r0", type=float, default=1.2, help="cm, eyes/mouth zero radius")
    ap.add_argument("--protect-r1", type=float, default=3.0, help="cm, eyes/mouth full radius")
    ap.add_argument("--nose-r0", type=float, default=0.5)
    ap.add_argument("--nose-r1", type=float, default=1.5)
    return ap.parse_args(argv)


def new_object(name, verts, faces):
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(verts.tolist(), [], faces)
    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(obj)
    return obj


def main():
    args = parse_args()
    t0 = time.time()
    od = out_dir(args.out, "refine")
    fit_dir = Path(args.out) / "fit"

    fitted = np.load(fit_dir / "fitted_neutral.npy")
    assert_topology(fitted, "fitted_neutral (s3b input)")
    topo = np.load(fit_dir / "topology.npz")
    faces_flat, faces_off = topo["faces_flat"], topo["faces_off"]
    lmk_verts = topo["lmk_verts"]

    if args.no_shrinkwrap:
        print("[s3b] --no-shrinkwrap: passing fitted neutral through unchanged")
        refined = fitted.copy()
        stats = {"mode": "passthrough"}
    else:
        clay = np.load(Path(args.out) / "clay" / "clay_aligned.npz")
        cv, cf = clay["verts"].astype(np.float64), clay["faces"]
        print(f"[s3b] clay target: {len(cv)} v / {len(cf)} f")

        bpy.ops.wm.read_factory_settings(use_empty=True)
        clay_obj = new_object("clay", cv, [tuple(int(i) for i in f) for f in cf])
        sm = clay_obj.modifiers.new("smooth", "SMOOTH")
        sm.factor = args.clay_smooth_factor
        sm.iterations = args.clay_smooth_iters
        print(f"[s3b] clay Smooth modifier: factor={sm.factor} iters={sm.iterations} "
              "(evaluated in depsgraph; shrinkwrap sees the SMOOTHED clay)")

        fit_obj = new_object("fitted", fitted, faces_as_lists(faces_flat, faces_off))
        sw = fit_obj.modifiers.new("shrink", "SHRINKWRAP")
        sw.target = clay_obj
        sw.wrap_method = "NEAREST_SURFACEPOINT"

        dg = bpy.context.evaluated_depsgraph_get()
        ev = fit_obj.evaluated_get(dg)
        me = ev.to_mesh()
        if len(me.vertices) != len(fitted):
            print(f"[s3b FATAL] evaluated mesh has {len(me.vertices)} verts "
                  f"!= {len(fitted)} -- topology drift, STOP")
            sys.exit(1)
        wrapped = np.empty(len(fitted) * 3, dtype=np.float64)
        me.vertices.foreach_get("co", wrapped)
        wrapped = wrapped.reshape(-1, 3)
        ev.to_mesh_clear()

        delta = wrapped - fitted
        mag = np.linalg.norm(delta, axis=1)
        n_cut = int((mag > args.cutoff).sum())
        delta[mag > args.cutoff] = 0.0  # clay absent there (e.g. below neck)
        mag = np.linalg.norm(delta, axis=1)
        over = mag > args.max_disp
        delta[over] *= (args.max_disp / mag[over])[:, None]

        edges = edges_from_faces(faces_flat, faces_off)
        delta_s = smooth_field(delta, edges, args.smooth_iters, args.smooth_lam)

        # ---- mask: regions + feature protection
        w = np.zeros(len(fitted))
        w[:EXTERIOR_END] = 1.0                      # face + head/neck skin only
        w[:9409] *= args.face_weight                # gentle on the face region
        eye_mouth = fitted[lmk_verts[IBUG_EYES + IBUG_MOUTH]]
        nose = fitted[lmk_verts[IBUG_NOSE]]
        w *= smoothstep(min_dist_to_points(fitted, eye_mouth),
                        args.protect_r0, args.protect_r1)
        w *= smoothstep(min_dist_to_points(fitted, nose),
                        args.nose_r0, args.nose_r1)
        w = smooth_field(w, edges, 5, 0.5)
        w[EXTERIOR_END:] = 0.0                      # interior stays EXACTLY put

        disp = w[:, None] * delta_s
        refined = fitted + disp
        dmag = np.linalg.norm(disp, axis=1)
        stats = {
            "mode": "shrinkwrap",
            "raw_delta_discarded_over_cutoff": n_cut,
            "clamped_over_max_disp": int(over.sum()),
            "disp_mean_cm": float(dmag.mean()),
            "disp_max_cm": float(dmag.max()),
            "disp_mean_face_cm": float(dmag[:9409].mean()),
            "disp_mean_scalp_cm": float(dmag[9409:EXTERIOR_END].mean()),
            "verts_moved_over_1mm": int((dmag > 0.1).sum()),
            "params": vars(args),
        }
        print(f"[s3b] disp: mean={stats['disp_mean_cm']:.2f}cm "
              f"max={stats['disp_max_cm']:.2f}cm  face-mean="
              f"{stats['disp_mean_face_cm']:.2f}cm scalp-mean="
              f"{stats['disp_mean_scalp_cm']:.2f}cm  "
              f">1mm: {stats['verts_moved_over_1mm']}/{len(fitted)}")
        np.savez(od / "refine_debug.npz", mask=w, disp_mag=dmag)

    assert_topology(refined, "refined_neutral")
    np.save(od / "refined_neutral.npy", refined)
    write_obj(od / "refined_neutral.obj", refined, faces_flat, faces_off,
              comment="newstack s3b refined ICT neutral (cm, +Y up, +Z front)")
    with open(od / "refine_stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    print(f"[s3b] refined_neutral -> {od / 'refined_neutral.obj'}")
    print(f"[s3b] DONE in {time.time()-t0:.1f}s")


main()
