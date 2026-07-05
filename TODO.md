# TODO — Image → ARKit Avatar → GLB (newARC)

Snapshot of the working tree. Two pipelines live here; the **new stack** is the active one.

## Folder map
```
newARC/
├── newstack/                  # ACTIVE — TripoSR + ICT-FaceKit commercial pipeline
│   ├── run_newstack.sh        # orchestrator: STAGES="1..7", REFINE, TEX_SIZE, DRACO
│   └── pipe/
│       ├── arkit_names.py     # ARKit-52 contract + ICT→ARKit name map (52/52 target)
│       ├── mp_ibug68.py       # MediaPipe-478 → iBUG/Multi-PIE-68 correspondence
│       ├── common.py          # numpy core: OBJ io, rasterizer, weak-persp cam, Umeyama, smoothing
│       ├── ict_loader.py      # ICT FaceXModel → cache npz (neutral + id modes + 51 expr + UVs + landmarks)
│       ├── s1_landmarks.py    # MediaPipe landmarks on the photo
│       ├── s2_fit_identity.py # fit ICT identity coeffs (linear model) to the photo landmarks
│       ├── s3a_align_clay.py  # align TripoSR clay → ICT space (pytorch3d view sweep + Umeyama)
│       ├── s3b_refine_blender.py  # gated shrinkwrap onto smoothed clay (hair/proportions; face protected)
│       ├── s4_build_shapes.py # additive ARKit blendshapes from ICT deltas (+ tongueOut synth, in progress)
│       ├── s5_bake_texture.py # bake photo → ICT UVs  (FIX IN PROGRESS: winding/visibility)
│       ├── s6_export_blender.py   # GLB (51/52 morphs) + .blend
│       └── s7_verify_glb.py   # stdlib GLB parser: morph count, names vs contract, texture
├── recon/ rig/ blender_build_rig.py   # ALPHA (FLAME 2023 Open pipeline; superseded, kept on branch `alpha`)
├── out/                       # reports, manifests, head_arkit.glb (FLAME), head_arkit_v2.glb (new stack)
│   ├── compliance_newstack.md # licensing: NEWSTACK-SHIP-CLEARED conditional→yes (4 build items)
│   └── ...
└── open_avatar.py             # local Blender launcher (imports a GLB)
```

## Status
- [x] New-stack env on pod (TripoSR + ICT-FaceKit + torchmcubes + rembg)
- [x] TripoSR clay from the photo (hair volume + child proportions; rough, as expected)
- [x] ICT-FaceKit Light (MIT): neutral + **51 pre-authored ARKit blendshapes** (blink/smile/frown/brows)
- [x] Full pipeline runs end-to-end → `head_arkit_v2.glb`, s7 PASS
- [x] Identity fit tight to the photo; geometry is a real win over FLAME (real head + hair, not a bald adult)
- [x] Licensing cleared (conditional): pin rembg `u2net`, ICT **Light only**, no bpy binary shipped, wire notices

## In progress / next
- [x] **s5 texture bake FIX authored** (pending pod re-run) — winding was inverted for ICT topology → dark central face. Now **measures** the camera-facing sign (like `recon/bake_texture.py`), rejects grazing texels + X-mirror fallback, exterior-priority UV rasterization so interior islands can't steal face texels, and a `central_face` sanity gate in `bake_metrics.json`. Verify `winding.facing_sign` + `central_face.pass` after re-run.
- [ ] **tongueOut → 52/52** (spec ready; NOT yet coded — implement in `arkit_names.py` + `s4` + a `tongue_synth` helper):
  - ICT HAS tongue geometry in region **"Gums and tongue" verts `[14062:17038]`**; teeth are `[17039:21451]`.
  - **Tongue = those region verts with distance-to-nearest-tooth > ~1.0 cm** (compute with cKDTree, don't hardcode ids) → ~760 central floor-of-mouth verts: centroid (0,−3.8,4.4), x∈[−3.2,3.2], y∈[−5.4,−0.8], z∈[1.0,9.1].
  - Reference: front teeth z=10.8, lips z≈11.9, tongue tip z=9.1 → push tip forward past ~12.
  - Delta: `w = smoothstep((z−z_root)/(z_tip−z_root))**1.5`; `delta = w·(AMOUNT_Z·+z + small·+y)`, AMOUNT_Z so tip moves ≈ +4–5 cm; ONLY tongue verts move (assert 0 on gums/teeth/face).
  - Mark `tongueOut` supported, source `"synthesized-ict-tongue"`; manifest → 52 supported / 0 unsupported. Gate: tip final z > lip-front z.
- [ ] Re-run `STAGES="4 5 6 7"`, re-render (correct front = Blender −Y), QA the real GLB (52/52)
- [ ] Wire `head_arkit_v2.glb` into `out/viewer/` (three.js + MediaPipe); ICT→ARKit driver name map
- [ ] Ship prep: THIRD-PARTY-NOTICES (rembg/U²-Net, TripoSR, ICT-FaceKit © USC-ICT 2020, MediaPipe, three.js, Draco)

## Pod
RunPod RTX 6000 Ada. Pipeline at `/workspace/newstack/pipe/`; ICT/TripoSR at `/workspace/newstack/`; Blender 4.2.3 (headless via `xvfb-run`).
