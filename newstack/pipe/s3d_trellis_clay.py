#!/usr/bin/env python3
"""Stage 3d -- TRELLIS textured head -> DENSE COLORED point cloud in ICT space
(the alternative s5 color source, selected via CLAY_COLOR=trellis).

Why: TripoSR's hallucinated back-of-head palette is washed grey-beige (s5
works around it with a measured cap palette). The TRELLIS mesh
(out_trellis/head_trellis.glb, TRELLIS-image-large MIT + PyTorch3D-baked
texture) is a full-360 head from OUR photo with REAL dark-brown hair color on
the back/crown/sides -- but soft front geometry. We only take its COLOR:
the deliverable topology stays ICT 26719; this stage just builds a color
source shaped exactly like out/clay/clay_aligned.ply (trimesh w/ vertices +
uint8 vertex_colors, k-NN sampled by s5).

Pipeline:
  1. COLOR: bilinear-sample the 2048 baseColor texture at each vertex UV
     (per-vertex colors), then DENSIFY: area-weighted barycentric samples on
     the faces, each colored from the UV-interpolated texture -> a dense
     cloud (~TripoSR-like ~0.15 cm spacing) so s5's k-NN stays smooth.
  2. ALIGN (landmark route, NOT blob ICP -- measured pose-degenerate on hairy
     blobs): reuse s3a's pytorch3d view sweep on the vertex-COLORED mesh
     (TRELLIS has a real textured front, unlike TripoSG) -> MediaPipe ->
     pix_to_face/bary unprojection of the 68 iBUG landmarks -> trimmed
     Umeyama vs the fitted ICT landmark verts -> rigid (scale-frozen) trimmed
     ICP face polish onto the fitted ICT face region [0,9409), exactly like
     s3a_align_triposg.
  3. GATES (dies loudly, never ships a mis-aligned color source):
     - MediaPipe must detect + >= --min-lmk landmarks must unproject
     - trimmed landmark rms <= --max-lmk-rms cm
     - inner-landmark -> TRELLIS-surface mean <= --max-face-dist cm
     - direction-aware front-depth mean <= --max-front-err cm, 0 missing
     - nearest-surface median from the aligned TripoSG clay (fallback
       TripoSR) verts to the TRELLIS cloud <= --max-ref-med cm
     - aligned y-extent within a sane ratio of the fitted ICT head.
  4. PROOF: orthographic painter splats (front/back/right profile) of the
     aligned colored cloud + a back-view A/B against the TripoSR clay, plus
     back-region saturation metrics (TRELLIS back should be MORE saturated /
     hair-colored than TripoSR's grey-beige).

Outputs under out/clay/ (TripoSR + TripoSG files are NOT touched):
  clay_trellis_aligned.ply    mesh verts + dense samples in ICT cm, uint8
                              vertex colors; faces reference the first
                              n_mesh_verts verts (dense points unreferenced)
                              -- loads exactly like clay_aligned.ply
  clay_trellis_aligned.npz    verts float32 + faces int32 + n_mesh_verts
  clay_trellis_align.json     S, R, T + metrics + gate values + counts
  clay_trellis_debug_view.png / clay_trellis_debug_landmarks.png
  clay_trellis_check_{front,back,right}.png + clay_trellis_vs_triposr_back.png
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (ICT_REGIONS, P, bilinear_sample, detect_face, die,  # noqa: E402
                    landmarks_to_np, out_dir, save_json, umeyama)
from mp_ibug68 import MEDIAPIPE_IBUG68  # noqa: E402
from s3a_align_clay import PRE_ROTS, Renderer, face_score  # noqa: E402
from s3a_align_triposg import front_depth_errors, rigid_fit  # noqa: E402

FACE_END = ICT_REGIONS["face"][1]  # 9409
LUM_W = np.array([0.299, 0.587, 0.114])


def load_trellis(trellis_dir, glb, uv_path, tex_path):
    """Geometry from the GLB, UVs from uv.npy (fallback: GLB visual.uv),
    texture from texture_baseColor.png (fallback: GLB baseColorTexture)."""
    import trimesh
    from PIL import Image
    td = Path(trellis_dir)
    glb = Path(glb) if glb else td / "head_trellis.glb"
    m = trimesh.load(glb, process=False, force="mesh")
    v = np.asarray(m.vertices, dtype=np.float64)
    f = np.asarray(m.faces, dtype=np.int64)
    if len(f) == 0:
        die(f"TRELLIS mesh {glb} has NO faces")
    uv_path = Path(uv_path) if uv_path else td / "uv.npy"
    if uv_path.is_file():
        uv = np.load(uv_path).astype(np.float64)
    elif getattr(m.visual, "uv", None) is not None:
        uv = np.asarray(m.visual.uv, dtype=np.float64)
    else:
        die(f"no UVs: {uv_path} missing and the GLB has none")
    if len(uv) != len(v):
        die(f"UV count {len(uv)} != vertex count {len(v)} -- the TRELLIS "
            "export is expected to be seam-duplicated (per-vertex UVs)")
    tex_path = Path(tex_path) if tex_path else td / "texture_baseColor.png"
    if tex_path.is_file():
        tex = np.asarray(Image.open(tex_path).convert("RGB"),
                         dtype=np.float64) / 255.0
    else:
        img = getattr(getattr(m.visual, "material", None),
                      "baseColorTexture", None)
        if img is None:
            die(f"no texture: {tex_path} missing and the GLB has none")
        tex = np.asarray(img.convert("RGB"), dtype=np.float64) / 255.0
    return v, f, uv, tex, str(glb), str(tex_path)


def sample_texture(uv, tex):
    """Bilinear texture lookup at OpenGL-style UVs (v=0 at the BOTTOM --
    the convention verified on this export by render_compare.py)."""
    H, W = tex.shape[:2]
    px = np.stack([uv[:, 0] * W, (1.0 - uv[:, 1]) * H], axis=1)
    return np.clip(bilinear_sample(tex, px), 0.0, 1.0)


def densify(v, f, uv, tex, n, rng):
    """Area-weighted barycentric surface samples, each colored from the
    UV-interpolated texture. Valid across the whole face because the export
    duplicates seam vertices (UVs are continuous within every face)."""
    p0, p1, p2 = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
    area = 0.5 * np.linalg.norm(np.cross(p1 - p0, p2 - p0), axis=1)
    fi = rng.choice(len(f), size=n, p=area / area.sum())
    r1 = np.sqrt(rng.random(n))
    r2 = rng.random(n)
    b = np.stack([1.0 - r1, r1 * (1.0 - r2), r1 * r2], axis=1)
    pts = (b[:, 0:1] * v[f[fi, 0]] + b[:, 1:2] * v[f[fi, 1]]
           + b[:, 2:3] * v[f[fi, 2]])
    uvp = (b[:, 0:1] * uv[f[fi, 0]] + b[:, 1:2] * uv[f[fi, 1]]
           + b[:, 2:3] * uv[f[fi, 2]])
    return pts, sample_texture(uvp, tex), float(area.sum())


def splat_view(pts, cols, view, center, half, res=720):
    """Orthographic painter splat (pure numpy, 2x2 px points). Views:
    front  = camera at +Z (screen-right = +x): face should look AT you
    back   = camera at -Z (screen-right = -x): back of the skull
    right  = camera at +X (screen-right = -z): profile, face points LEFT."""
    if view == "front":
        h, v, depth = pts[:, 0], pts[:, 1], pts[:, 2]
    elif view == "back":
        h, v, depth = -pts[:, 0], pts[:, 1], -pts[:, 2]
    elif view == "right":
        h, v, depth = -pts[:, 2], pts[:, 1], pts[:, 0]
    else:
        die(f"unknown splat view {view}")
    order = np.argsort(depth)  # farthest first; nearest overwrites (painter)
    px = ((h - center[0]) / half * 0.5 + 0.5) * (res - 2)
    py = (0.5 - (v - center[1]) / half * 0.5) * (res - 2)
    px = np.clip(px.astype(np.int64), 0, res - 2)[order]
    py = np.clip(py.astype(np.int64), 0, res - 2)[order]
    img = np.full((res, res, 3), 255, dtype=np.uint8)
    c8 = (np.clip(cols, 0, 1)[order] * 255).astype(np.uint8)
    for dy in (0, 1):
        for dx in (0, 1):
            img[py + dy, px + dx] = c8
    return img


def sat_stats(c):
    """Back-region color stats. NOTE mean_sat alone is misleading here:
    TripoSR's back is a PALE SKIN-BEIGE ghost (high (mx-mn)/mx), the real
    subject hair is DARK brown (low absolute sat). Hair-coloredness on this
    subject = LOW luma + brown mean_rgb, so both are reported."""
    if len(c) == 0:
        return {"n": 0, "mean_sat": None, "chroma_mean": None,
                "mean_rgb": None, "mean_luma": None}
    mx, mn = c.max(axis=1), c.min(axis=1)
    dev = np.linalg.norm(c - (c @ LUM_W)[:, None], axis=1)
    return {"n": int(len(c)),
            "mean_sat": float(((mx - mn) / np.maximum(mx, 1e-6)).mean()),
            "chroma_mean": float(dev.mean()),
            "mean_rgb": np.round(c.mean(0), 4).tolist(),
            "mean_luma": float((c @ LUM_W).mean())}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--trellis-dir", default=P.ROOT + "/out_trellis")
    ap.add_argument("--glb", default=None, help="default <trellis-dir>/head_trellis.glb")
    ap.add_argument("--uv", default=None, help="default <trellis-dir>/uv.npy")
    ap.add_argument("--texture", default=None,
                    help="default <trellis-dir>/texture_baseColor.png")
    ap.add_argument("--task", default=P.MP_TASK)
    ap.add_argument("--out", default=P.OUT)
    ap.add_argument("--n-dense", type=int, default=100000,
                    help="barycentric color samples added on top of the "
                         "33k mesh verts (TripoSR-clay-like density)")
    ap.add_argument("--dist", type=float, default=2.4,
                    help="render camera distance (mesh normalized to max "
                         "radius 1). TRELLIS outputs are HEAD-ONLY, so s3a's "
                         "1.7 crops the face out of frame and MediaPipe never "
                         "detects -- measured: no detection at 1.7, robust "
                         "detection at 2.2-3.5")
    ap.add_argument("--elev", type=float, default=0.0)
    ap.add_argument("--min-conf", type=float, default=0.3)
    ap.add_argument("--trim-frac", type=float, default=0.15)
    ap.add_argument("--min-lmk", type=int, default=40,
                    help="GATE: minimum unprojected landmarks for umeyama")
    ap.add_argument("--face-iters", type=int, default=40)
    ap.add_argument("--face-cap", type=float, default=2.0)
    ap.add_argument("--face-trim", type=float, default=0.30)
    ap.add_argument("--n-src", type=int, default=20000)
    ap.add_argument("--max-lmk-rms", type=float, default=1.5,
                    help="cm; GATE on the trimmed landmark umeyama rms")
    ap.add_argument("--max-face-dist", type=float, default=1.0,
                    help="cm; GATE on mean INNER-landmark->surface distance")
    ap.add_argument("--max-front-err", type=float, default=1.2,
                    help="cm; GATE on mean front-depth error at inner landmarks")
    ap.add_argument("--max-ref-med", type=float, default=1.2,
                    help="cm; GATE on the median distance from the aligned "
                         "TripoSG (fallback TripoSR) clay verts to the "
                         "aligned TRELLIS cloud")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    t0 = time.time()
    od = out_dir(args.out, "clay")
    rng = np.random.default_rng(args.seed)
    from scipy.spatial import cKDTree

    import trimesh

    # ---- load + per-vertex color + densify (RAW TRELLIS frame) -----------
    v_raw, faces, uv, tex, glb_path, tex_path = load_trellis(
        args.trellis_dir, args.glb, args.uv, args.texture)
    vcols = sample_texture(uv, tex)
    print(f"[s3d] TRELLIS: {len(v_raw)} v / {len(faces)} f, texture "
          f"{tex.shape[1]}x{tex.shape[0]}, vert-color std={vcols.std():.3f}")
    if vcols.std() < 1e-3:
        die("UV-sampled vertex colors are flat -- texture/UV mismatch")
    d_pts, d_cols, area_raw = densify(v_raw, faces, uv, tex,
                                      args.n_dense, rng)
    print(f"[s3d] densified: +{len(d_pts)} barycentric samples "
          f"(raw surface area {area_raw:.3f})")

    fitted = np.load(Path(args.out) / "fit" / "fitted_neutral.npy")
    lmk_verts = np.load(Path(args.out) / "fit" / "topology.npz")["lmk_verts"]
    ict_lmk = fitted[lmk_verts]  # (68,3) cm

    # ---- landmark alignment (s3a machinery on the colored mesh) ----------
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    c = v_raw.mean(axis=0)
    sigma = float(np.linalg.norm(v_raw - c, axis=1).max()) or 1.0
    v_norm = (v_raw - c) / sigma
    best = None  # (score, up_name, azim)
    for up_name, R_pre in PRE_ROTS.items():
        rend = Renderer(v_norm @ R_pre.T, faces, vcols, device)
        for azim in range(0, 360, 45):
            rgb, _ = rend.render(azim, args.elev, args.dist, 384)
            res = detect_face(rgb, args.task, args.min_conf)
            if res is not None:
                sc = face_score(res)
                if best is None or sc > best[0]:
                    best = (sc, up_name, azim)
        del rend
    if best is None:
        die("MediaPipe found NO face on ANY TRELLIS view -- cannot "
            "landmark-align (no silent fallback)")
    sc0, up_name, azim0 = best
    print(f"[s3d] coarse best: up={up_name} azim={azim0} score={sc0:.3f}")
    R_pre = PRE_ROTS[up_name]
    rend = Renderer(v_norm @ R_pre.T, faces, vcols, device)
    best_f = (sc0, float(azim0))
    for azim in np.arange(azim0 - 30, azim0 + 30.01, 6.0):
        rgb, _ = rend.render(float(azim), args.elev, args.dist, 512)
        res = detect_face(rgb, args.task, args.min_conf)
        if res is not None:
            sc = face_score(res)
            if sc > best_f[0]:
                best_f = (sc, float(azim))
    azim = best_f[1]
    SIZE = 768
    rgb, frags = rend.render(azim, args.elev, args.dist, SIZE, want_frags=True)
    res = detect_face(rgb, args.task, args.min_conf)
    if res is None:
        die("face lost at full render resolution -- cannot landmark-align")
    import cv2
    cv2.imwrite(str(od / "clay_trellis_debug_view.png"), rgb[..., ::-1])
    lm = landmarks_to_np(res)
    src_pts, dst_pts, used = [], [], []
    dbg = rgb.copy()
    for i, mp_idx in enumerate(MEDIAPIPE_IBUG68):
        px, py = lm[mp_idx, 0] * SIZE, lm[mp_idx, 1] * SIZE
        p3 = rend.unproject(frags, px, py, SIZE)
        if p3 is None:
            continue
        src_pts.append(p3)  # prerotated-normalized frame
        dst_pts.append(ict_lmk[i])
        used.append(i)
        cv2.circle(dbg, (int(px), int(py)), 3, (0, 255, 0), -1)
    cv2.imwrite(str(od / "clay_trellis_debug_landmarks.png"), dbg[..., ::-1])
    if len(used) < args.min_lmk:
        die(f"only {len(used)} landmarks unprojected onto the TRELLIS mesh "
            f"(need >= {args.min_lmk})")
    src_pts, dst_pts = np.asarray(src_pts), np.asarray(dst_pts)
    s_u, R_u, t_u = umeyama(src_pts, dst_pts)
    resid = np.linalg.norm((s_u * (R_u @ src_pts.T).T + t_u) - dst_pts, axis=1)
    keep = resid <= np.quantile(resid, 1.0 - args.trim_frac)
    s_u, R_u, t_u = umeyama(src_pts[keep], dst_pts[keep])
    resid2 = np.linalg.norm((s_u * (R_u @ src_pts[keep].T).T + t_u)
                            - dst_pts[keep], axis=1)
    lmk_rms = float(np.sqrt((resid2 ** 2).mean()))
    # compose back through the prerotation + normalization
    S = s_u / sigma
    R = R_u @ R_pre
    T = t_u - S * (R @ c)
    lmk_info = {"up": up_name, "azim_deg": float(azim),
                "n_landmarks_used": int(len(used)),
                "n_kept_after_trim": int(keep.sum()),
                "residual_rms_cm": lmk_rms,
                "residual_max_cm": float(resid2.max())}
    print(f"[s3d] landmark umeyama: {len(used)} unprojected, "
          f"{int(keep.sum())} kept, rms={lmk_rms:.2f} cm  scale={S:.3f}")

    all_raw = np.vstack([v_raw, d_pts])
    all_cols = np.vstack([vcols, d_cols])
    v_all = S * (all_raw @ R.T) + T

    # ---- rigid (scale-frozen) trimmed ICP face polish ---------------------
    tree_face = cKDTree(fitted[:FACE_END])
    src = v_all[rng.choice(len(v_all), min(args.n_src, len(v_all)),
                           replace=False)]
    R_p, t_p = np.eye(3), np.zeros(3)
    n_pairs, face_rms = 0, None
    for _ in range(args.face_iters):
        cur = src @ R_p.T + t_p
        d, j = tree_face.query(cur, workers=-1)
        sel = d <= args.face_cap
        if sel.sum() < 500:
            die(f"face polish: only {int(sel.sum())} TRELLIS points within "
                f"{args.face_cap} cm of the fitted face -- alignment is off")
        sel &= d <= np.quantile(d[sel], 1.0 - args.face_trim)
        n_pairs = int(sel.sum())
        R_d, t_d = rigid_fit(cur[sel], fitted[:FACE_END][j[sel]])
        R_p = R_d @ R_p
        t_p = R_d @ t_p + t_d
    if args.face_iters > 0:
        d, _ = tree_face.query(src @ R_p.T + t_p, workers=-1)
        d_in = d[d <= args.face_cap]
        face_rms = float(np.sqrt((d_in[d_in <= np.quantile(
            d_in, 1.0 - args.face_trim)] ** 2).mean()))
        print(f"[s3d] face polish: {n_pairs} pairs  rigid  "
              f"face inlier rms={face_rms:.3f} cm")
        v_all = v_all @ R_p.T + t_p
        R = R_p @ R
        T = R_p @ T + t_p

    # ---- metrics + gates ---------------------------------------------------
    tree_tr = cKDTree(v_all)
    d_lmk, _ = tree_tr.query(fitted[lmk_verts], workers=-1)
    inner_mean = float(d_lmk[17:].mean())
    inner_max = float(d_lmk[17:].max())
    fd_err, fd_miss = front_depth_errors(fitted[lmk_verts[17:]], v_all)
    fd_mean = float(fd_err.mean()) if len(fd_err) else np.inf
    fd_max = float(fd_err.max()) if len(fd_err) else np.inf

    ref_name, ref_med, ref_rms3 = None, None, None
    for cand in ("clay_sg_aligned.npz", "clay_aligned.npz"):
        if (od / cand).is_file():
            ref_v = np.load(od / cand)["verts"].astype(np.float64)
            idx = rng.choice(len(ref_v), min(20000, len(ref_v)), replace=False)
            d_ref, _ = tree_tr.query(ref_v[idx], workers=-1)
            ref_name = cand
            ref_med = float(np.median(d_ref))
            ref_rms3 = float(np.sqrt((d_ref[d_ref <= 3.0] ** 2).mean()))
            break
    if ref_name is None:
        die("neither clay_sg_aligned.npz nor clay_aligned.npz present -- "
            "no reference surface to gate the TRELLIS alignment against")

    ext_tr = v_all.max(0) - v_all.min(0)
    ext_ict = fitted[:11248].max(0) - fitted[:11248].min(0)
    y_ratio = float(ext_tr[1] / max(ext_ict[1], 1e-9))

    print(f"[s3d] landmarks -> TRELLIS cloud: inner mean={inner_mean:.3f} cm "
          f"max={inner_max:.3f} cm")
    print(f"[s3d] front-depth @inner landmarks: mean={fd_mean:.3f} cm "
          f"max={fd_max:.3f} cm  missing={fd_miss}/51")
    print(f"[s3d] {ref_name} -> TRELLIS: median={ref_med:.3f} cm "
          f"rms(<=3cm)={ref_rms3:.3f} cm")
    print(f"[s3d] aligned extent={np.round(ext_tr, 1)} cm "
          f"(ICT head y-ratio {y_ratio:.2f})")

    failures = []
    if lmk_rms > args.max_lmk_rms:
        failures.append(f"landmark rms {lmk_rms:.2f} cm > {args.max_lmk_rms}")
    if inner_mean > args.max_face_dist:
        failures.append(f"inner-landmark->surface mean {inner_mean:.3f} cm > "
                        f"{args.max_face_dist} -- the FACE did not register")
    if fd_miss > 0:
        failures.append(f"{fd_miss} inner landmarks have NO TRELLIS surface "
                        "above them (wrong pose)")
    if fd_mean > args.max_front_err:
        failures.append(f"front-depth mean {fd_mean:.3f} cm > "
                        f"{args.max_front_err} (pose/scale wrong)")
    if ref_med > args.max_ref_med:
        failures.append(f"{ref_name}->TRELLIS median {ref_med:.3f} cm > "
                        f"{args.max_ref_med} -- mis-aligned color source")
    if not (0.5 <= y_ratio <= 2.0):
        failures.append(f"aligned y-extent ratio {y_ratio:.2f} is insane")

    # ---- back-region saturation A/B vs the TripoSR clay -------------------
    back_reg = "z < -4 cm and y > -2 cm (back of skull above the neck)"
    m_tr = (v_all[:, 2] < -4.0) & (v_all[:, 1] > -2.0)
    sat_cmp = {"region": back_reg, "trellis": sat_stats(all_cols[m_tr])}
    sr_ply = od / "clay_aligned.ply"
    sr_v = sr_c = None
    if sr_ply.is_file():
        sr = trimesh.load(sr_ply, process=False, force="mesh")
        sr_v = np.asarray(sr.vertices, dtype=np.float64)
        sr_c = np.asarray(sr.visual.vertex_colors,
                          dtype=np.float64)[:, :3] / 255.0
        m_sr = (sr_v[:, 2] < -4.0) & (sr_v[:, 1] > -2.0)
        sat_cmp["triposr"] = sat_stats(sr_c[m_sr])
    print(f"[s3d] back-region saturation: {sat_cmp}")

    # ---- artifacts (saved BEFORE gating, s3sg-style) -----------------------
    info = {
        "method": "UV bilinear vertex colors + area-weighted barycentric "
                  "densify + s3a colored-view-sweep mediapipe landmarks + "
                  "trimmed umeyama + rigid face polish vs fitted ICT face",
        "glb": glb_path, "texture": tex_path,
        "landmark_align": lmk_info,
        "face_polish_pairs": n_pairs,
        "face_polish_inlier_rms_cm": face_rms,
        "lmk_inner_to_surface_mean_cm": inner_mean,
        "lmk_inner_to_surface_max_cm": inner_max,
        "front_depth_mean_cm": fd_mean,
        "front_depth_max_cm": fd_max,
        "front_depth_missing": int(fd_miss),
        "ref_surface": ref_name,
        "ref_to_trellis_median_cm": ref_med,
        "ref_to_trellis_rms_le3cm_cm": ref_rms3,
        "y_extent_ratio_vs_ict": y_ratio,
        "S": S, "R": R.tolist(), "T": np.asarray(T).tolist(),
        "counts": {"mesh_verts": int(len(v_raw)),
                   "mesh_faces": int(len(faces)),
                   "dense_points": int(len(d_pts)),
                   "total_points": int(len(v_all))},
        "back_saturation": sat_cmp,
        "gates": {"max_lmk_rms_cm": args.max_lmk_rms,
                  "max_face_dist_cm": args.max_face_dist,
                  "max_front_err_cm": args.max_front_err,
                  "max_ref_med_cm": args.max_ref_med},
        "gates_passed": not failures,
        "failures": failures,
    }
    save_json(od / "clay_trellis_align.json", info)
    out_ply = od / "clay_trellis_aligned.ply"
    trimesh.Trimesh(vertices=v_all, faces=faces,
                    vertex_colors=(np.clip(all_cols, 0, 1) * 255)
                    .astype(np.uint8),
                    process=False).export(out_ply)
    np.savez(od / "clay_trellis_aligned.npz",
             verts=v_all.astype(np.float32), faces=faces.astype(np.int32),
             n_mesh_verts=np.int64(len(v_raw)))

    # reload-verify the exact s5 load path (process=False keeps the dense
    # unreferenced verts -- proven here, not assumed)
    chk = trimesh.load(out_ply, process=False, force="mesh")
    cv, cc = np.asarray(chk.vertices), np.asarray(chk.visual.vertex_colors)
    if len(cv) != len(v_all) or len(cc) != len(v_all):
        die(f"reload check failed: {len(cv)} verts / {len(cc)} colors back "
            f"from PLY, expected {len(v_all)} (dense points dropped?)")
    if cc[:, :3].std() < 1.0:
        die("reload check failed: vertex colors came back flat")
    print(f"[s3d] {out_ply.name}: {len(cv)} pts / {len(chk.faces)} f, "
          "colors OK (reload-verified)")

    # ---- proof renders -----------------------------------------------------
    # (a) painter splats of the DENSE cloud (what s5 actually samples). These
    # are see-through by nature (sparse points; far-side hair speckles through
    # the face) -- they prove the dense points + colors, not the surface.
    from PIL import Image
    center = 0.5 * (v_all.max(0) + v_all.min(0))
    half = 0.53 * float((v_all.max(0) - v_all.min(0)).max())
    for view in ("front", "back", "right"):
        img = splat_view(v_all, all_cols, view, center, half)
        Image.fromarray(img).save(od / f"clay_trellis_check_{view}.png")
    # (b) OPAQUE pytorch3d mesh renders of the aligned colored mesh -- the
    # definitive orientation + surface-color proof (no bleed-through). The
    # TripoSR A/B back view shares the SAME normalization frame, so relative
    # scale is honest.
    v_mesh = v_all[:len(v_raw)]
    rn = Renderer((v_mesh - center) / half, faces, vcols, device)
    views = (("front", 0.0), ("right", 90.0), ("back", 180.0))
    mesh_imgs = {}
    for name, az in views:
        rgb, _ = rn.render(az, 0.0, 2.4, 720)
        mesh_imgs[name] = rgb
        Image.fromarray(rgb).save(od / f"clay_trellis_render_{name}.png")
    del rn
    if sr_v is not None:
        sr_f = np.asarray(trimesh.load(sr_ply, process=False,
                                       force="mesh").faces, dtype=np.int64)
        rn = Renderer((sr_v - center) / half, sr_f, sr_c, device)
        sr_back, _ = rn.render(180.0, 0.0, 2.4, 720)
        del rn
        Image.fromarray(np.hstack([mesh_imgs["back"], sr_back])).save(
            od / "clay_trellis_vs_triposr_back.png")
    print(f"[s3d] proofs -> {od}/clay_trellis_render_*.png (opaque mesh), "
          "clay_trellis_check_*.png (dense splats), back A/B vs TripoSR")

    if failures:
        die("TRELLIS alignment GATES FAILED: " + "; ".join(failures))
    print(f"[s3d] DONE in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
