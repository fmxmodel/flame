"""Stage 6 -- assemble + export head_arkit_v2.glb (RUNS INSIDE BLENDER):

  xvfb-run -a blender --background --factory-startup \
      --python pipe/s6_export_blender.py -- [args]

Deterministic assembly, NO mesh importers (so vertex ORDER can never drift):
the mesh is built with from_pydata straight from out/rig/arkit_deltas.npz
(refined neutral + polygons + ICT UVs), shape keys are set per-vertex from the
additive ARKit deltas, the baked albedo is wired to a Principled BSDF, and the
scene exports as GLB with morph targets named EXACTLY per the ARKit contract.

Axes/units: Blender coords = (x, -z, y) of ICT * 0.01. The glTF exporter's
Z-up -> Y-up conversion is (x, z, -y), so GLB coords == ICT coords in meters:
+Y up, +Z front, exactly what three.js expects.

Outputs under out/export/: head_arkit_v2.glb, head_arkit_v2.blend,
export_info.json.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import N_VERTS, P, faces_as_lists, out_dir  # noqa: E402

import bpy  # noqa: E402

CM_TO_M = 0.01


def parse_args():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    ap = argparse.ArgumentParser(description="s6 GLB export")
    ap.add_argument("--out", default=P.OUT)
    ap.add_argument("--name", default="head_arkit_v2")
    ap.add_argument("--draco", action="store_true")
    ap.add_argument("--morph-normals", action="store_true",
                    help="export per-target normals (bigger GLB)")
    return ap.parse_args(argv)


def ict_to_blender(v_cm):
    """(N,3) ICT cm (+Y up,+Z front) -> Blender m (Z up). See module docstring."""
    v = np.asarray(v_cm, dtype=np.float64)
    return np.stack([v[:, 0], -v[:, 2], v[:, 1]], axis=1) * CM_TO_M


def main():
    args = parse_args()
    t0 = time.time()
    od = out_dir(args.out, "export")
    rig_dir = Path(args.out) / "rig"

    z = np.load(rig_dir / "arkit_deltas.npz")
    names = [str(n) for n in z["names"]]
    deltas = z["deltas"].astype(np.float64)
    neutral = z["refined_neutral"].astype(np.float64)
    faces_flat, faces_off = z["faces_flat"], z["faces_off"]
    corner_vt, vt = z["corner_vt"], z["vt"].astype(np.float64)
    manifest = json.loads((rig_dir / "arkit_manifest.json").read_text())
    assert len(neutral) == N_VERTS, f"neutral {len(neutral)} != {N_VERTS} -- STOP"
    assert deltas.shape[1] == N_VERTS, "delta topology drift -- STOP"
    sup = [n for n in manifest["shapes"] if manifest["shapes"][n]["supported"]]
    assert set(names) == set(sup), "npz names != manifest supported set -- STOP"
    print(f"[s6] {len(names)} morph targets on {N_VERTS} verts "
          f"/ {len(faces_off)-1} polys")

    bpy.ops.wm.read_factory_settings(use_empty=True)

    mesh = bpy.data.meshes.new("HeadARKit")
    mesh.from_pydata(ict_to_blender(neutral).tolist(), [],
                     faces_as_lists(faces_flat, faces_off))
    mesh.update()
    assert len(mesh.vertices) == N_VERTS, "from_pydata vertex-count drift -- STOP"
    assert len(mesh.loops) == len(faces_flat), "loop-count drift -- STOP"

    # UVs: loop order after from_pydata follows the polygon corner order exactly
    uvl = mesh.uv_layers.new(name="UVMap")
    uv_flat = vt[corner_vt].astype(np.float32).ravel()
    uvl.data.foreach_set("uv", uv_flat)
    mesh.polygons.foreach_set("use_smooth",
                              np.ones(len(mesh.polygons), dtype=bool))
    mesh.update()

    obj = bpy.data.objects.new("HeadARKit", mesh)
    bpy.context.scene.collection.objects.link(obj)
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)

    # ---- shape keys: exact ARKit names, absolute coords = neutral + delta
    obj.shape_key_add(name="Basis", from_mix=False)
    for i, name in enumerate(names):
        sk = obj.shape_key_add(name=name, from_mix=False)
        co = ict_to_blender(neutral + deltas[i]).astype(np.float32).ravel()
        sk.data.foreach_set("co", co)
        sk.slider_min, sk.slider_max = 0.0, 1.0
    kb = obj.data.shape_keys.key_blocks
    got = [k.name for k in kb if k.name != "Basis"]
    assert got == names, f"shape key name drift: {set(names) ^ set(got)}"
    print(f"[s6] shape keys created: {len(got)} (+Basis)")

    # ---- material: baked albedo -> Principled BSDF
    albedo = Path(args.out) / "tex" / "albedo.png"
    mat = bpy.data.materials.new("HeadMat")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Roughness"].default_value = 0.6
    if albedo.is_file():
        img = bpy.data.images.load(str(albedo))
        img.pack()
        tex = mat.node_tree.nodes.new("ShaderNodeTexImage")
        tex.image = img
        mat.node_tree.links.new(tex.outputs["Color"], bsdf.inputs["Base Color"])
    else:
        print(f"[s6 WARN] {albedo} missing -- exporting untextured")
    obj.data.materials.append(mat)

    glb_path = od / f"{args.name}.glb"
    export_kwargs = dict(
        filepath=str(glb_path),
        export_format="GLB",
        export_morph=True,
        export_morph_normal=bool(args.morph_normals),
        export_yup=True,
        export_image_format="AUTO",
        export_draco_mesh_compression_enable=bool(args.draco),
    )
    try:
        bpy.ops.export_scene.gltf(**export_kwargs)
    except TypeError as e:  # exporter kwarg drift across Blender versions
        print(f"[s6 WARN] full-kwarg export failed ({e}); minimal retry")
        bpy.ops.export_scene.gltf(filepath=str(glb_path), export_format="GLB",
                                  export_morph=True)
    print(f"[s6] GLB -> {glb_path} ({glb_path.stat().st_size/1e6:.1f} MB)")

    blend_path = od / f"{args.name}.blend"
    bpy.ops.wm.save_as_mainfile(filepath=str(blend_path), compress=True)
    print(f"[s6] assembly blend -> {blend_path}")

    with open(od / "export_info.json", "w", encoding="utf-8") as f:
        json.dump({"glb": str(glb_path), "blend": str(blend_path),
                   "n_verts": N_VERTS, "n_polys": int(len(faces_off) - 1),
                   "morph_targets": names, "draco": bool(args.draco),
                   "textured": albedo.is_file(),
                   "units": "meters, +Y up, +Z front (glTF standard)"}, f, indent=2)
    print(f"[s6] DONE in {time.time()-t0:.1f}s")


main()
