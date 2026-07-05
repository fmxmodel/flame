#!/usr/bin/env python3
"""Stage 2 -- ICT identity fit to the photo landmarks (system python, torch).

Solves, by weighted least squares (Adam), the weak-perspective camera
(s, R, t) + 100 ICT identity coefficients (+ optionally the ICT expression
coefficients, to EXPLAIN a non-neutral photo so the identity doesn't have to
distort to fake a smile; the expression is then DISCARDED from the neutral):

    landmarks2d ~= project( generic + id_basis.T@id + expr_basis.T@exp )[lmk_verts]

Only the 68 landmark VERTICES enter the optimization (fast); the full mesh is
rebuilt once at the end.

Outputs under out/fit/:
  fitted_neutral.npy/.obj  neutral identity (NO expression), ICT topology, cm
  expression_offset.npy    sum_e exp_coeff[e]*delta_e -- lets s5 project the
                           photo onto the EXPRESSED face so texture lines up
  camera.json              s, R, t + conventions (see common.py docstring)
  id_coeffs.npy exp_coeffs.npy fit_metrics.json topology.npz fit_debug.jpg
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (P, assert_topology, die, out_dir, project_weak_persp,  # noqa: E402
                    save_json, write_obj)
from ict_loader import get_cache  # noqa: E402
from mp_ibug68 import IBUG_GROUPS, per_landmark_weights  # noqa: E402


def rotmat_torch(rvec):
    import torch
    th = torch.linalg.norm(rvec) + 1e-9
    k = rvec / th
    K = torch.stack([
        torch.stack([torch.zeros((), dtype=rvec.dtype, device=rvec.device), -k[2], k[1]]),
        torch.stack([k[2], torch.zeros((), dtype=rvec.dtype, device=rvec.device), -k[0]]),
        torch.stack([-k[1], k[0], torch.zeros((), dtype=rvec.dtype, device=rvec.device)]),
    ])
    I = torch.eye(3, dtype=rvec.dtype, device=rvec.device)
    return I + torch.sin(th) * K + (1.0 - torch.cos(th)) * (K @ K)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ict", default=P.ICT)
    ap.add_argument("--out", default=P.OUT)
    ap.add_argument("--rebuild-cache", action="store_true")
    ap.add_argument("--no-fit-expressions", action="store_true",
                    help="do not co-fit expression coeffs (photo assumed neutral)")
    ap.add_argument("--lam-id", type=float, default=0.05,
                    help="L2 regularization on identity coeffs")
    ap.add_argument("--lam-exp", type=float, default=0.05)
    ap.add_argument("--id-clamp", type=float, default=3.0)
    ap.add_argument("--exp-clamp", type=float, default=1.2)
    ap.add_argument("--iters1", type=int, default=300, help="pose-only iters")
    ap.add_argument("--iters2", type=int, default=800, help="joint iters")
    ap.add_argument("--lr", type=float, default=0.02)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    t0 = time.time()
    od = out_dir(args.out, "fit")

    z = get_cache(args.ict, Path(args.out) / "cache" / "ict_model_cache.npz",
                  rebuild=args.rebuild_cache)
    G = z["generic"]                      # (N,3) cm
    id_basis = z["id_basis"]              # (100,N,3) deltas
    expr_deltas = z["expr_deltas"]        # (E,N,3) deltas
    expr_names = [str(s) for s in z["expr_names"]]
    lmk_verts = z["lmk_verts"]            # (68,)
    fit_exp = not args.no_fit_expressions

    lz = np.load(Path(args.out) / "landmarks" / "landmarks.npz")
    target = lz["ibug68_px"]              # (68,2) px, y down
    h, w = [int(v) for v in lz["image_hw"]]
    diag = float(np.hypot(h, w))
    weights = per_landmark_weights().astype(np.float64)

    import torch
    dev = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    DT = torch.float64
    G_l = torch.tensor(G[lmk_verts], dtype=DT, device=dev)                 # (68,3)
    idb_l = torch.tensor(id_basis[:, lmk_verts, :], dtype=DT, device=dev)  # (K,68,3)
    exb_l = torch.tensor(expr_deltas[:, lmk_verts, :], dtype=DT, device=dev)
    tgt = torch.tensor(target, dtype=DT, device=dev)
    wts = torch.tensor(weights, dtype=DT, device=dev)

    # analytic init: scale from outer eye corners (iBUG 36/45), translate by centroid
    d_px = float(np.linalg.norm(target[45] - target[36]))
    d_cm = float(np.linalg.norm(G[lmk_verts[45]] - G[lmk_verts[36]]))
    s0 = d_px / max(d_cm, 1e-6)
    proj0 = np.stack([s0 * G[lmk_verts][:, 0], -s0 * G[lmk_verts][:, 1]], axis=1)
    t0_uv = target.mean(0) - proj0.mean(0)
    print(f"[s2] init: s={s0:.2f} px/cm  t=({t0_uv[0]:.0f},{t0_uv[1]:.0f}) px  device={dev}")

    rvec = torch.tensor([1e-3, 1e-3, 1e-3], dtype=DT, device=dev, requires_grad=True)
    log_s = torch.tensor([np.log(s0)], dtype=DT, device=dev, requires_grad=True)
    tuv = torch.tensor(t0_uv, dtype=DT, device=dev, requires_grad=True)
    idc = torch.zeros(id_basis.shape[0], dtype=DT, device=dev, requires_grad=True)
    exc = torch.zeros(expr_deltas.shape[0], dtype=DT, device=dev, requires_grad=True)

    def predict():
        V = G_l + torch.einsum("k,kij->ij", idc, idb_l)
        if fit_exp:
            V = V + torch.einsum("e,eij->ij", exc, exb_l)
        R = rotmat_torch(rvec)
        Xc = V @ R.T
        s = torch.exp(log_s[0])
        u = s * Xc[:, 0] + tuv[0]
        v = -s * Xc[:, 1] + tuv[1]
        return torch.stack([u, v], dim=1)

    def data_loss():
        res = (predict() - tgt) / diag
        return (wts * (res ** 2).sum(dim=1)).sum() / wts.sum()

    def run_phase(params, iters, base_lr, with_reg):
        opt = torch.optim.Adam(params, lr=base_lr)
        for i in range(iters):
            for g in opt.param_groups:
                g["lr"] = base_lr * 0.5 * (1 + np.cos(np.pi * i / max(iters - 1, 1)))
            opt.zero_grad()
            loss = data_loss()
            if with_reg:
                loss = loss + args.lam_id * (idc ** 2).mean()
                if fit_exp:
                    loss = loss + args.lam_exp * (exc ** 2).mean()
            loss.backward()
            opt.step()
            with torch.no_grad():
                idc.clamp_(-args.id_clamp, args.id_clamp)
                exc.clamp_(0.0, args.exp_clamp)  # expressions are one-sided
            if i % 100 == 0 or i == iters - 1:
                print(f"    iter {i:4d}  loss {float(loss):.6f}")
        return float(data_loss())

    print(f"[s2] phase 1: pose/camera only ({args.iters1} iters)")
    run_phase([rvec, log_s, tuv], args.iters1, args.lr, with_reg=False)
    print(f"[s2] phase 2: joint pose + identity"
          + (" + expression" if fit_exp else "") + f" ({args.iters2} iters)")
    run_phase([rvec, log_s, tuv, idc] + ([exc] if fit_exp else []),
              args.iters2, args.lr, with_reg=True)

    with torch.no_grad():
        pred = predict().cpu().numpy()
        R = rotmat_torch(rvec).cpu().numpy()
        s = float(np.exp(float(log_s[0])))
        t2 = tuv.detach().cpu().numpy()
        id_np = idc.detach().cpu().numpy()
        ex_np = exc.detach().cpu().numpy()

    err = np.linalg.norm(pred - target, axis=1)
    groups_px = {g: float(err[idx].mean()) for g, idx in IBUG_GROUPS.items()}
    print(f"[s2] landmark error px: mean={err.mean():.2f} max={err.max():.2f}")
    for g, e in groups_px.items():
        print(f"      {g:12s} {e:6.2f}")
    print(f"[s2] |id| stats: max={np.abs(id_np).max():.3f} "
          f"nonzero(>0.05)={int((np.abs(id_np) > 0.05).sum())}/100")
    if fit_exp:
        top = np.argsort(-ex_np)[:6]
        print("[s2] top photo expressions: "
              + ", ".join(f"{expr_names[i]}={ex_np[i]:.2f}" for i in top))

    # full-resolution reconstruction (float64)
    fitted = G + np.einsum("k,kij->ij", id_np, id_basis.astype(np.float64))
    assert_topology(fitted, "fitted_neutral")
    expr_off = np.einsum("e,eij->ij", ex_np, expr_deltas.astype(np.float64)) \
        if fit_exp else np.zeros_like(G)

    np.save(od / "fitted_neutral.npy", fitted)
    np.save(od / "expression_offset.npy", expr_off)
    np.save(od / "id_coeffs.npy", id_np)
    np.save(od / "exp_coeffs.npy", ex_np)
    write_obj(od / "fitted_neutral.obj", fitted, z["faces_flat"], z["faces_off"],
              comment="newstack s2 fitted ICT neutral (cm, +Y up, +Z front)")
    # small topology sidecar so Blender stages never load the big cache
    np.savez(od / "topology.npz",
             faces_flat=z["faces_flat"], faces_off=z["faces_off"],
             lmk_verts=lmk_verts, n_verts=np.int64(len(G)))
    save_json(od / "camera.json", {
        "s_px_per_cm": s, "R": R.tolist(), "t_px": t2.tolist(),
        "image_hw": [h, w],
        "convention": {"u": "s*(R@X).x + tx", "v": "-s*(R@X).y + ty",
                       "depth": "(R@X).z, larger = closer",
                       "units": "model cm, image px, y down"},
    })
    save_json(od / "fit_metrics.json", {
        "px_err_mean": float(err.mean()), "px_err_max": float(err.max()),
        "px_err_by_group": groups_px,
        "id_abs_max": float(np.abs(id_np).max()),
        "exp_fitted": fit_exp,
        "exp_top": ({expr_names[i]: float(ex_np[i])
                     for i in np.argsort(-ex_np)[:10]} if fit_exp else {}),
        "expr_offset_max_cm": float(np.linalg.norm(expr_off, axis=1).max()),
    })
    print(f"[s2] fitted_neutral -> {od / 'fitted_neutral.obj'}")

    # debug overlay: green = photo landmarks, red = fitted projection
    try:
        import cv2
        img = cv2.imread(str(Path(args.out) / "landmarks" / "input_image.png"))
        uv, _ = project_weak_persp(fitted[lmk_verts] + expr_off[lmk_verts], s, R, t2)
        for (gx, gy), (rx, ry) in zip(target, uv):
            cv2.circle(img, (int(gx), int(gy)), 3, (0, 255, 0), -1)
            cv2.circle(img, (int(rx), int(ry)), 3, (0, 0, 255), 1)
        cv2.imwrite(str(od / "fit_debug.jpg"), img)
        print(f"[s2] fit overlay -> {od / 'fit_debug.jpg'}")
    except Exception as e:  # debug image is best-effort
        print(f"[s2 WARN] fit_debug.jpg skipped: {e}")
    print(f"[s2] DONE in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
