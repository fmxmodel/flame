#!/usr/bin/env python3
"""Stage 3a -- rigid-align the TripoSR clay to the fitted ICT neutral
(system python: trimesh + pytorch3d + mediapipe).

The clay's canonical orientation is not guaranteed, so we FIND its front:
render the vertex-colored clay from a sweep of candidate up-axes x yaw angles,
run MediaPipe on each render, keep the view with the widest detected face,
then refine the yaw. From the best view we unproject the 68 iBUG landmark
pixels through the rasterizer's pix_to_face/bary_coords back to 3D clay
surface points, pair them with the fitted ICT neutral's landmark VERTICES,
and solve a trimmed Umeyama similarity (scale+R+t).

Outputs under out/clay/:
  clay_aligned.ply   clay in ICT space (cm), vertex colors preserved (s5)
  clay_aligned.npz   verts float32 + tri faces int32 (s3b, Blender-readable)
  clay_align.json    S, R, T, residuals, chosen view
  debug_best_view.png / debug_landmarks.png

Fallback (--force-bbox or if no view detects a face): coarse bbox alignment
with --fallback-up/--fallback-yaw. Loudly warned; only for iteration.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import P, die, out_dir, save_json, umeyama, detect_face, landmarks_to_np  # noqa: E402
from mp_ibug68 import MEDIAPIPE_IBUG68  # noqa: E402


def rot_x(deg):
    r = np.radians(deg)
    c, s = np.cos(r), np.sin(r)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float64)


def rot_y(deg):
    r = np.radians(deg)
    c, s = np.cos(r), np.sin(r)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float64)


PRE_ROTS = {  # candidate clay-up conventions -> rotate into +Y-up
    "y_up": np.eye(3),
    "z_up": rot_x(-90.0),
    "z_down": rot_x(90.0),
    "y_down": rot_x(180.0),
}


def load_clay(path):
    import trimesh
    m = trimesh.load(path, process=False, force="mesh")
    v = np.asarray(m.vertices, dtype=np.float64)
    f = np.asarray(m.faces, dtype=np.int64)
    cols = None
    try:
        vc = m.visual.vertex_colors
        if vc is not None and len(vc) == len(v):
            cols = np.asarray(vc, dtype=np.float64)[:, :3] / 255.0
    except Exception:
        pass
    if cols is None:
        print("[s3a WARN] clay has no vertex colors; using flat gray")
        cols = np.full((len(v), 3), 0.55)
    return v, f, cols


class Renderer:
    def __init__(self, verts, faces, cols, device):
        import torch
        from pytorch3d.structures import Meshes
        from pytorch3d.renderer import TexturesVertex
        self.torch = torch
        self.device = device
        self.vt = torch.tensor(verts, dtype=torch.float32, device=device)
        self.ft = torch.tensor(faces, dtype=torch.int64, device=device)
        tex = TexturesVertex(verts_features=[torch.tensor(cols, dtype=torch.float32, device=device)])
        self.mesh = Meshes(verts=[self.vt], faces=[self.ft], textures=tex)

    def render(self, azim, elev, dist, size, fov=40.0, want_frags=False):
        import torch
        from pytorch3d.renderer import (FoVPerspectiveCameras, RasterizationSettings,
                                        MeshRasterizer, HardPhongShader, PointLights,
                                        BlendParams, look_at_view_transform)
        R, T = look_at_view_transform(dist=dist, elev=elev, azim=azim)
        cam = FoVPerspectiveCameras(device=self.device, R=R, T=T, fov=fov)
        ras = MeshRasterizer(cameras=cam, raster_settings=RasterizationSettings(
            image_size=size, blur_radius=0.0, faces_per_pixel=1, bin_size=0))
        lights = PointLights(device=self.device, location=[[0.0, 0.5, 3.0]],
                             ambient_color=((0.65, 0.65, 0.65),),
                             diffuse_color=((0.45, 0.45, 0.45),),
                             specular_color=((0.02, 0.02, 0.02),))
        shader = HardPhongShader(device=self.device, cameras=cam, lights=lights,
                                 blend_params=BlendParams(background_color=(1.0, 1.0, 1.0)))
        frags = ras(self.mesh)
        img = shader(frags, self.mesh)
        rgb = (img[0, ..., :3].clamp(0, 1) * 255).byte().cpu().numpy()
        return (rgb, frags) if want_frags else (rgb, None)

    def unproject(self, frags, px, py, size):
        """3D mesh-space point under pixel (px,py) via pix_to_face + bary."""
        xi = int(np.clip(round(px), 0, size - 1))
        yi = int(np.clip(round(py), 0, size - 1))
        f = int(frags.pix_to_face[0, yi, xi, 0].item())
        if f < 0:
            return None
        bary = frags.bary_coords[0, yi, xi, 0].cpu().numpy()
        tri = self.ft[f].cpu().numpy()
        pts = self.vt[tri].cpu().numpy()
        return bary @ pts


def face_score(res):
    """Frontality score: normalized interocular spread (mp 33 vs 263)."""
    lm = landmarks_to_np(res)
    return abs(float(lm[263, 0] - lm[33, 0]))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--clay", default=P.CLAY)
    ap.add_argument("--task", default=P.MP_TASK)
    ap.add_argument("--out", default=P.OUT)
    ap.add_argument("--dist", type=float, default=1.7)
    ap.add_argument("--elev", type=float, default=0.0)
    ap.add_argument("--min-conf", type=float, default=0.3)
    ap.add_argument("--trim-frac", type=float, default=0.15,
                    help="fraction of worst correspondences dropped in pass 2")
    ap.add_argument("--force-bbox", action="store_true")
    ap.add_argument("--fallback-up", choices=list(PRE_ROTS), default="y_up")
    ap.add_argument("--fallback-yaw", type=float, default=0.0)
    args = ap.parse_args()
    t0 = time.time()
    od = out_dir(args.out, "clay")

    fitted = np.load(Path(args.out) / "fit" / "fitted_neutral.npy")
    topo = np.load(Path(args.out) / "fit" / "topology.npz")
    lmk_verts = topo["lmk_verts"]
    ict_lmk = fitted[lmk_verts]  # (68,3) cm

    v_raw, faces, cols = load_clay(args.clay)
    print(f"[s3a] clay: {len(v_raw)} v / {len(faces)} f  colors="
          f"{'yes' if cols.std() > 1e-6 else 'flat'}")

    # normalize for rendering; composed back into the final transform
    c = v_raw.mean(axis=0)
    sigma = float(np.linalg.norm(v_raw - c, axis=1).max()) or 1.0
    v_norm = (v_raw - c) / sigma

    S = R_total = T = None
    align_info = {}

    if not args.force_bbox:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        best = None  # (score, up_name, azim)
        for up_name, R_pre in PRE_ROTS.items():
            rend = Renderer(v_norm @ R_pre.T, faces, cols, device)
            for azim in range(0, 360, 45):
                rgb, _ = rend.render(azim, args.elev, args.dist, 384)
                res = detect_face(rgb, args.task, args.min_conf)
                if res is not None:
                    sc = face_score(res)
                    if best is None or sc > best[0]:
                        best = (sc, up_name, azim)
            del rend
        if best is None:
            print("[s3a WARN] no face detected in ANY coarse view -- bbox fallback")
        else:
            sc0, up_name, azim0 = best
            print(f"[s3a] coarse best: up={up_name} azim={azim0} score={sc0:.3f}")
            R_pre = PRE_ROTS[up_name]
            rend = Renderer(v_norm @ R_pre.T, faces, cols, device)
            best_f = (sc0, azim0)
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
                print("[s3a WARN] face lost at full res -- bbox fallback")
            else:
                import cv2
                cv2.imwrite(str(od / "debug_best_view.png"), rgb[..., ::-1])
                lm = landmarks_to_np(res)
                clay_pts, ict_pts, used = [], [], []
                dbg = rgb.copy()
                for i, mp_idx in enumerate(MEDIAPIPE_IBUG68):
                    px, py = lm[mp_idx, 0] * SIZE, lm[mp_idx, 1] * SIZE
                    p3 = rend.unproject(frags, px, py, SIZE)
                    if p3 is None:
                        continue
                    clay_pts.append(p3)   # in prerotated-normalized space
                    ict_pts.append(ict_lmk[i])
                    used.append(i)
                    cv2.circle(dbg, (int(px), int(py)), 3, (0, 255, 0), -1)
                cv2.imwrite(str(od / "debug_landmarks.png"), dbg[..., ::-1])
                if len(used) < 20:
                    print(f"[s3a WARN] only {len(used)} unprojected landmarks -- bbox fallback")
                else:
                    clay_pts = np.asarray(clay_pts)
                    ict_pts = np.asarray(ict_pts)
                    s_u, R_u, t_u = umeyama(clay_pts, ict_pts)
                    resid = np.linalg.norm((s_u * (R_u @ clay_pts.T).T + t_u) - ict_pts, axis=1)
                    keep = resid <= np.quantile(resid, 1.0 - args.trim_frac)
                    s_u, R_u, t_u = umeyama(clay_pts[keep], ict_pts[keep])
                    resid2 = np.linalg.norm((s_u * (R_u @ clay_pts[keep].T).T + t_u)
                                            - ict_pts[keep], axis=1)
                    # compose: v_ict = s_u*R_u@(R_pre@(v-c)/sigma) + t_u
                    S = s_u / sigma
                    R_total = R_u @ R_pre
                    T = t_u - S * (R_total @ c)
                    align_info = {
                        "method": "mediapipe+umeyama",
                        "up": up_name, "azim_deg": float(azim),
                        "n_landmarks_used": int(len(used)),
                        "n_kept_after_trim": int(keep.sum()),
                        "residual_rms_cm": float(np.sqrt((resid2 ** 2).mean())),
                        "residual_max_cm": float(resid2.max()),
                    }
                    print(f"[s3a] umeyama: rms={align_info['residual_rms_cm']:.2f}cm "
                          f"max={align_info['residual_max_cm']:.2f}cm "
                          f"({keep.sum()}/{len(used)} pts)")

    if S is None:  # bbox fallback
        print("[s3a WARN] === BBOX FALLBACK ALIGNMENT (coarse!) -- pass "
              "--fallback-up/--fallback-yaw to steer, and re-check outputs ===")
        R_total = rot_y(args.fallback_yaw) @ PRE_ROTS[args.fallback_up]
        v_r = (R_total @ v_raw.T).T
        ext_c = v_r.max(0) - v_r.min(0)
        ext_i = fitted.max(0) - fitted.min(0)
        S = float(ext_i[1] / max(ext_c[1], 1e-9))  # match height
        T = fitted.mean(0) - S * (R_total @ v_raw.mean(0))
        align_info = {"method": "bbox_fallback", "up": args.fallback_up,
                      "yaw_deg": args.fallback_yaw}

    v_aligned = S * (R_total @ v_raw.T).T + T
    align_info.update({"S": float(S), "R": R_total.tolist(), "T": np.asarray(T).tolist(),
                       "clay_verts": int(len(v_aligned)), "clay_faces": int(len(faces))})
    save_json(od / "clay_align.json", align_info)

    import trimesh
    tm = trimesh.Trimesh(vertices=v_aligned, faces=faces,
                         vertex_colors=(np.clip(cols, 0, 1) * 255).astype(np.uint8),
                         process=False)
    tm.export(od / "clay_aligned.ply")
    np.savez(od / "clay_aligned.npz",
             verts=v_aligned.astype(np.float32), faces=faces.astype(np.int32))
    print(f"[s3a] aligned clay -> {od / 'clay_aligned.ply'} "
          f"(bbox y: {v_aligned[:,1].min():.1f}..{v_aligned[:,1].max():.1f} cm "
          f"vs ICT {fitted[:,1].min():.1f}..{fitted[:,1].max():.1f})")
    print(f"[s3a] DONE in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
