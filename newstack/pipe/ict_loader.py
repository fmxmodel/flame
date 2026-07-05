"""ICT-FaceKit "Light" (FaceXModel) loader + cache.

License note: uses ONLY the MIT-licensed FaceXModel/ shipped inside the
ICT-FaceKit repo (generic neutral, identity000..099, expression OBJs,
vertex_indices.json). No "Full" model, no NC assets.

The model is LINEAR:
    neutral(id)  = generic + sum_k id_coeff[k]  * (identity_k  - generic)
    with_expr    = neutral + sum_e exp_coeff[e] * (expression_e - generic)
so identity and expression deltas are additive on ANY neutral of the same
topology -- this is what makes the whole newstack rig valid.

First call builds out/cache/ict_model_cache.npz (~1 min of OBJ parsing);
subsequent stages load it in seconds.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import N_VERTS, assert_topology, die, read_obj  # noqa: E402

CACHE_VERSION = 2


def _find_landmark_verts(vertex_indices_path):
    with open(vertex_indices_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    key = None
    if "idx_to_landmark_verts" in data:
        key = "idx_to_landmark_verts"
    else:
        cands = [k for k in data if "landmark" in k.lower()]
        if cands:
            key = cands[0]
    if key is None:
        die("vertex_indices.json has no landmark key. Available keys: "
            + ", ".join(f"{k}(len={len(v) if isinstance(v, list) else type(v).__name__})"
                        for k, v in data.items()))
    lmk = data[key]
    if not isinstance(lmk, list):
        die(f"vertex_indices.json[{key}] is not a list ({type(lmk).__name__})")
    lmk = [int(i) for i in lmk]
    if len(lmk) < 68:
        die(f"vertex_indices.json[{key}] has {len(lmk)} entries (< 68)")
    if len(lmk) > 68:
        print(f"[ict WARN] landmark list has {len(lmk)} entries; using the "
              "FIRST 68 (Multi-PIE order). Inspect if fits look wrong.")
        lmk = lmk[:68]
    arr = np.asarray(lmk, dtype=np.int64)
    if arr.min() < 0 or arr.max() >= N_VERTS:
        die(f"landmark vertex ids out of range [0,{N_VERTS}): "
            f"min={arr.min()} max={arr.max()}")
    return arr, key


def build_cache(ict_dir, cache_path):
    ict_dir = Path(ict_dir)
    fxm = ict_dir / "FaceXModel"
    if not fxm.is_dir():
        die(f"FaceXModel not found under {ict_dir}")
    t0 = time.time()

    generic_path = fxm / "generic_neutral_mesh.obj"
    if not generic_path.is_file():
        die(f"missing {generic_path}")
    g = read_obj(generic_path)
    assert_topology(g["v"], "generic_neutral_mesh.obj")
    if g["vt"] is None or g["corner_vt"] is None:
        die("generic_neutral_mesh.obj has no UVs -- texture bake impossible")
    n_polys = len(g["faces_off"]) - 1
    print(f"[ict] generic neutral: {len(g['v'])} v / {n_polys} polys / "
          f"{len(g['vt'])} vt  ({time.time()-t0:.1f}s)")

    id_paths = sorted(fxm.glob("identity*.obj"))
    if len(id_paths) != 100:
        print(f"[ict WARN] expected 100 identity OBJs, found {len(id_paths)}")
    if not id_paths:
        die("no identity*.obj files found")
    id_basis = np.empty((len(id_paths), N_VERTS, 3), dtype=np.float32)
    for k, p in enumerate(id_paths):
        vv = read_obj(p, verts_only=True)["v"]
        assert_topology(vv, p.name)
        id_basis[k] = (vv - g["v"]).astype(np.float32)
        if (k + 1) % 25 == 0:
            print(f"[ict] identity modes parsed: {k+1}/{len(id_paths)} "
                  f"({time.time()-t0:.1f}s)")

    expr_names, expr_list, skipped = [], [], []
    for p in sorted(fxm.glob("*.obj")):
        stem = p.stem
        if stem.startswith("identity") or "generic" in stem or "unparameter" in stem:
            continue
        vv = read_obj(p, verts_only=True)["v"]
        if len(vv) != N_VERTS:
            skipped.append(f"{p.name}({len(vv)}v)")
            continue
        expr_names.append(stem)
        expr_list.append((vv - g["v"]).astype(np.float32))
    if skipped:
        print(f"[ict WARN] skipped non-topology OBJs: {', '.join(skipped)}")
    if not expr_names:
        die("no expression OBJs found in FaceXModel/")
    expr_deltas = np.stack(expr_list, axis=0)
    print(f"[ict] expression OBJs on the ICT topology: {len(expr_names)}")

    lmk_verts, lmk_key = _find_landmark_verts(fxm / "vertex_indices.json")
    print(f"[ict] landmark verts from vertex_indices.json[{lmk_key}]: "
          f"{len(lmk_verts)} (Multi-PIE-68 order)")

    Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        cache_path,
        version=np.int64(CACHE_VERSION),
        generic=g["v"].astype(np.float64),
        faces_flat=g["faces_flat"], faces_off=g["faces_off"],
        corner_vt=g["corner_vt"], vt=g["vt"].astype(np.float64),
        id_names=np.array([p.stem for p in id_paths]),
        id_basis=id_basis,
        expr_names=np.array(expr_names),
        expr_deltas=expr_deltas,
        lmk_verts=lmk_verts,
    )
    print(f"[ict] cache -> {cache_path} ({time.time()-t0:.1f}s)")


def get_cache(ict_dir, cache_path, rebuild=False):
    cache_path = Path(cache_path)
    if rebuild or not cache_path.is_file():
        build_cache(ict_dir, cache_path)
    z = np.load(cache_path, allow_pickle=False)
    if int(z["version"]) != CACHE_VERSION:
        print("[ict] cache version mismatch -- rebuilding")
        build_cache(ict_dir, cache_path)
        z = np.load(cache_path, allow_pickle=False)
    assert_topology(z["generic"], "cached generic neutral")
    return z
