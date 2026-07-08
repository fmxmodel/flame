# TODO — Image → ARKit Avatar → GLB (newARC)

Snapshot of the working tree. Two pipelines live here; the **new stack** is the active one.

## Folder map
```
newARC/
├── newstack/                  # ACTIVE — TripoSR + ICT-FaceKit commercial pipeline
│   ├── run_newstack.sh        # orchestrator: STAGES="1..7", REFINE, TEX_SIZE, DRACO
│   └── pipe/
│       ├── arkit_names.py     # ARKit-52 contract + ICT→ARKit name map (52/52: 51 OBJ + 1 synth)
│       ├── mp_ibug68.py       # MediaPipe-478 → iBUG/Multi-PIE-68 correspondence
│       ├── common.py          # numpy core: OBJ io, rasterizer, weak-persp cam, Umeyama, smoothing
│       ├── ict_loader.py      # ICT FaceXModel → cache npz (neutral + id modes + 51 expr + UVs + landmarks)
│       ├── s1_landmarks.py    # MediaPipe landmarks on the photo
│       ├── s2_fit_identity.py # fit ICT identity coeffs (linear model) to the photo landmarks
│       ├── s3a_align_clay.py  # align TripoSR clay → ICT space (pytorch3d view sweep + Umeyama; --prefix)
│       ├── s3a_align_triposg.py   # TripoSG clay → ICT space (wall strip + front-render landmarks + Umeyama + rigid face polish; gated)
│       ├── s3d_trellis_clay.py    # TRELLIS textured head → dense colored ICT-space cloud (s5 color source via CLAY_COLOR=trellis; gated)
│       ├── s3b_refine_blender.py  # gated shrinkwrap (legacy TripoSR mode + face-weighted TripoSG mode w/ bbox prescale)
│       ├── s3c_verify_refine.py   # measurement gates: 26719v/23 loops, photo reprojection, back-region cleanliness
│       ├── render_neutral.py  # grey Workbench+cavity proof renders (also the s3sg detection view, --square --dump-cam)
│       ├── s4_build_shapes.py # additive ARKit blendshapes from ICT deltas + gated tongueOut/gaze synth
│       ├── tongue_synth.py    # tongueOut delta from ICT's real static tongue (cKDTree select)
│       ├── gaze_synth.py      # eyeball-rotation deltas for eyeLook* (ICT OBJs move lids only)
│       ├── _selftest_tongue.py    # offline synthetic-geometry test for tongue_synth (18 checks)
│       ├── s5_bake_texture.py # bake photo → ICT UVs + eye_left/right.png (photo-derived iris)
│       ├── eye_texture.py     # iris/pupil/sclera sampled from photo; procedural pole-centered disc
│       ├── s6_export_blender.py   # GLB (52/52 morphs, HeadMat+EyeMat, opaque-hardened) + .blend
│       ├── s7_verify_glb.py   # stdlib GLB parser: morphs/names/materials-opaque vs contract
│       └── s8_render_previews.py  # proof renders from the GLB (front/back/eyes/gaze)
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

- [x] **s3 reworked for TripoSG clay (sharper geometry)** — `CLAY_SOURCE=triposg` path, run + verified on pod:
  - TripoSG glb (`out_triposg/random-person_triposg_300k.glb`, 150k v, geometry-only) contains the photo's
    **background wall fused to the person** + ~230 confetti shards; `s3a_align_triposg.py` strips the wall
    (dominant area-weighted normal + offset peak; person side = 95th-pct depth extent, NOT vertex median —
    the shard cloud outnumbers the head) → kept 46,374 v / 91,781 f.
  - Alignment is **landmark-based, not blob ICP** (trimmed ICP was measured pose-degenerate on hairy partial
    heads: 90°-off optima, scale drift 13.5–18.9): one ortho front Blender render (MediaPipe DOES detect on a
    lit grey Workbench render; pytorch3d Phong grey = black silhouette, never detects) → columnar max-z
    unprojection → trimmed Umeyama (57/68 kept, **rms 0.82 cm**, scale 15.06) → rigid Kabsch face polish onto
    the fitted ICT face (**0.34 cm**; scale frozen — free scale vs partial template collapses).
    Gates: inner-landmark→surface **0.28 cm** (≤1.0), direction-aware front-depth **0.32 cm**/0 missing (≤1.2).
  - `s3b --weights face --prescale yz`: face region [0,9409) weight 1.0 feathering to 0 over head/neck
    (3 cm from the region boundary), eyes/mouth/nose/ears protected, interior pinned; bbox volume-match
    prescale measured k=(1.0, 1.027, 0.956) — tiny for this subject (hair pulled back). NOT x: the x factor
    is hair-width driven and pushed the jaw +2.4 cm into the hair (84 px contour reproj) before the fix.
  - `s3c` gates (all PASS): 26,719 verts, **23 boundary loops** (== raw ICT), reprojection **23.61 px** vs
    22.30 fitted / 22.57 TripoSR-run baselines, back-region shrinkwrap disp **0.016 cm** (≤0.05), face core
    → clay mean **0.18 cm** (the face genuinely sits ON the TripoSG surface). s4 rebuilt 52/52 on it
    (tongueOut gates pass: tip 13.55 > lips 11.84).
  - Renders `out/renders/neutral_triposg_*.png` vs `neutral_triposr_*.png`: TripoSR baseline has an inflated
    hair-helmet ridge across the forehead; TripoSG version has cleaner face geometry, a real hairline crease,
    clean ears and a solid smooth back. TripoSR path stays runnable (default `CLAY_SOURCE=triposr`, verified
    same session) and TripoSR stays the s5 color source.

## In progress / next
- [x] **s5 360° texture rework** (run + verified on pod, TripoSG geometry) — the back/sides now read as
  HAIR, not pale grey clay, with no backdrop "bonnet" and no second face:
  - **Person mask** (procedural: border-median flood fill from top/left/right, erode 3px + feather σ2):
    photo samples must land ON the subject — silhouette-grazing texels used to paint the white backdrop
    onto the scalp/sides. Applied to projection, mirror fallback and vertex colors alike.
  - **TripoSR fill**: k-NN (8) inverse-distance color sampling from the aligned clay + per-channel
    affine palette match (residual-trimmed LSQ on 614k photo∩clay texels) → TripoSR's washed palette
    lands on the photo's palette, so the hairline/jaw feather (`w_photo` = N·V smoothstep × soft mask)
    has no color step.
  - **Hair zone** (crown above the landmark-measured hairline + everything above ear level behind the
    temple plane): MEASURED, TripoSR hallucinates the hidden crown desaturated grey-beige with inverted
    chroma (unfixable by any fitted transfer), so there the palette comes from the MEASURED photo hair
    (20–50% luminance band of visible cap texels; 55,690 pairs → [0.52,0.37,0.29] mid-brown) and TripoSR
    contributes only luminance variation. Bald subjects self-correct (visible cap = skin then).
  - Fill feeds BOTH tile-0 albedo and `vertex_colors.npy` (RestMat scalp/back). RestMat roughness
    0.9 (specular sheen at 0.6 read as "pale plastic"). Metrics: `back_region.triposr_frac = 1.0`,
    `vertex_fill.head_neck` = 96.7% triposr / 3.3% honest default (neck below clay coverage),
    `sources_frac`, `person_mask`, `clay_match`, `cap_match`. s7 PASS (52/52, OPAQUE, doubleSided).
    Proof renders `out/renders/glb_{front,profile,back}_full.png` + eyes/gaze/tongue; measured back
    pixels are warm browns (105–169 R), zero default texels on the back region.
- [x] **s5 seam + hair-fidelity rework** (run + verified on pod, TripoSG geometry; judged against a
  fresh render of the pre-change GLB, not a stale screenshot) — the back/hair keeps TripoSR's tonal
  variation instead of being flattened to a desaturated palette mean, and the photo→fill feather is wider:
  - **`--clay-transfer lumachroma` (default)**: luma gets a trimmed affine LSQ onto the photo's luma
    (fitted gain 0.929/offset 0.075 — brightness lands right, no hairline/jaw step); CHROMA is
    preserved and rescaled by the photo/clay chroma std ratio (fitted [1.087, 1.056, 1.151] — amplify,
    not flatten; the legacy per-channel RGB affine measured gain [0.96, 0.90, 0.94] = desaturate,
    kept as `--clay-transfer affine` for A/B).
  - **Hair zone**: photo-hair base MODULATED by TripoSR's local chroma deviation about the applied-zone
    mean (`--hair-chroma 1.2`, clamped ±0.2/ch) sampled at sharper k-NN (`--hair-knn 2` vs 8 elsewhere);
    luma ratio anchored on the APPLIED zone (visible-cap anchor railed the clamp → pale-tan crown);
    zone extended down to the nape hairline (`--nape-drop 2.5` cm below ear level — 72% of visible
    skull-back verts had w_cap=0 before). RestMat re-measures both anchors on its own vertex samples
    (texel anchors blue-shifted the crown verts by −hair_chroma·dev_mu ≈ [−0.14, +0.05, +0.13]).
  - **Seam**: N·V feather widened 0.15/0.50 → `--ndotv-lo 0.08 / --ndotv-hi 0.60` (~53–85° band),
    person-mask blur σ 2→3; `--back-sat 1.35` lifts the TRANSFER fill only (backfacing-feathered;
    the measured cap palette is exempt — it already carries real hair saturation).
  - **Measured (bake_metrics.json before → after)**: back_region mean_sat 0.342 → 0.375, chroma_std
    0.032 → 0.056 (+75% tonal variation); hair_zone 76.7k → 92.3k texels, mean lands on photo hair
    [0.533, 0.394, 0.319]; render pixels: mid-back sat 0.249 → 0.301, behind-ear 0.326 → 0.377,
    hairline max local step −16%. central_face gate PASS (photo_frac 0.959, brightness_ratio 1.02),
    back_region triposr_frac 1.0 / default 0.0, s7 PASS (52/52, all mats OPAQUE + doubleSided).
    Before/after proof: `out/renders/glb_{front,profile,back}_full_{before,after}.png`;
    legacy-mode metrics kept in `out/newstack_bake_metrics_before.json`.
- [x] **s5 texture bake FIX authored** (verified on pod) — winding was inverted for ICT topology → dark central face. Now **measures** the camera-facing sign (like `recon/bake_texture.py`), rejects grazing texels + X-mirror fallback, exterior-priority UV rasterization so interior islands can't steal face texels, and a `central_face` sanity gate in `bake_metrics.json`. Verified: `winding.facing_sign` measured, `central_face.pass` true (photo_frac 0.95).
- [x] **tongueOut → 52/52** — CODED (pending pod re-run of s4+): `tongue_synth.py` selects the tongue
  from ICT's real static geometry (region `[14062:17039)` verts with cKDTree distance-to-nearest-tooth
  `[17039:21451)` > 1.0 cm, no hardcoded ids), pushes the tip `+4.5 cm` forward (`smoothstep**1.5`
  root→tip weighting, small `+0.8` y-lift) so tip z ≈ 13.6 > lips ≈ 11.9. `s4` gates it: delta EXACTLY 0
  outside the tongue set, tip final z > lip-front z (measured via mouth landmark verts 48–67), else die.
  Source `"synthesized-ict-tongue"`; manifest → 52 supported / 0 unsupported. Offline self-test
  (`_selftest_tongue.py`, synthetic geometry): 18/18 PASS. Verify on pod: `[tongue]` log line ~760 verts,
  centroid ≈ (0,−3.8,4.4), manifest `shapes.tongueOut.synth`.
- [x] **Eyes fixed** (was: blank white stare) — eyeball UVs overlap the face UVs, so eyes get
  DEDICATED photo-derived textures (`eye_texture.py`: iris color from MediaPipe iris ring
  468-472/473-477, pupil from darkest central quartile, sclera from brightest decile of the
  eye opening; procedural disc at UV (0.5,0.5) = eyeball forward pole, iris_uv_radius 0.110)
  bound to `EyeMat` in s6. ICT's transparent-purpose eye shells (lacrimal/blend/occlusion,
  verts [24591,25351)) are stripped at export — measured, they sat in FRONT of the iris and
  rendered as skin-textured lids once opaque. `eyeLook*` morphs now rotate the eyeballs
  (`gaze_synth.py`, In/Out 35°, Up 25°, Down 30°) because ICT's OBJs move lids only
  (measured eyeball delta 0.0). Proof renders: `out/renders/glb_*.png`.
- [x] **UDIM tile-1+ → RestMat** (kills the pre-existing stretched-face back of head) — ICT
  UVs are multi-tile (u up to 7); the bake fills tile 0 only and the wrapping sampler painted
  the skull back/teeth/mouth-socket with the FACE image. Polys with max corner u > 1 now use
  `RestMat` driven by s5 per-vertex colors (photo where visible, TripoSR clay hair, honest
  flat teeth/mouth defaults, eye sockets = shadowed skin) exported as COLOR_0.
- [x] **Opaque export hardened** — all materials single-sided (glTF doubleSided=false) +
  GLB JSON post-pass writes EXPLICIT `alphaMode:"OPAQUE"`; s6 re-imports the GLB and
  measures functional opacity (alpha socket unlinked & 1.0, DITHERED, culling on; fails on
  BLEND). Note: Blender 4.2's `blend_method` is a deprecated alias reading HASHED for ALL
  materials (even factory-new) — functional opacity is the real gate. s7 fails any material
  that is not explicitly OPAQUE/single-sided/textured.
- [x] Re-run `STAGES="4 5 6 7 8"` on pod — s7 PASS (52/52 names, 2 primitives, both
  materials OPAQUE), renders show iris+pupil+sclera, gaze morph moves the iris, back solid
- [x] Wire `head_arkit_v2.glb` into `out/viewer/` (three.js + MediaPipe) — loads the 52/52 GLB
  (3 opaque primitives HeadMat/EyeMat/RestMat), drives `morphTargetInfluences` by exact
  `categoryName`→`morphTargetDictionary` on ALL primitives (no remap: names ARE ARKit-52 1:1).
  `verify-names` parses the real GLB → 52/52, 51 MediaPipe categories resolve, tongueOut manual
  slider (MediaPipe never emits it); prebuild gates the build; `npm run build` PASS. UI: webcam/
  video, smoothing, head-pose toggle, FPS, tongueOut + full manual sliders. Run: `cd out/viewer
  && npm install && npm run dev`.
- [x] Ship prep: THIRD-PARTY-NOTICES (rembg/U²-Net, TripoSR, ICT-FaceKit © USC-ICT 2020, MediaPipe,
  three.js) — rewritten for the actual shipped deps in `out/viewer/public/THIRD-PARTY-NOTICES.md`
  (+ synced into `dist/`); per-license texts under `public/licenses/`. Draco is NOT used/shipped
  (GLB is uncompressed, no DRACOLoader, no decoder in dist) — deliberately excluded. FINAL gate:
  `out/compliance_newstack.md` → **NEWSTACK-SHIP-CLEARED: yes** (standing: pin `u2net`; ICT Light only).

- [x] **TRELLIS color source wired (stage `3d` + `CLAY_COLOR=trellis`)** — run + verified on pod. The
  TRELLIS spike output (`out_trellis/head_trellis.glb`, 33,096 v / 46,944 f, 2048² baked baseColor;
  TRELLIS-image-large MIT + the fork's PyTorch3D texturing — no nvdiffrast, COLOR SOURCE only, geometry
  stays ICT 26719) becomes a dense ICT-space colored cloud for s5:
  - `pipe/s3d_trellis_clay.py`: bilinear UV texture sample at each vertex + 100k area-weighted
    barycentric face samples (UV-interpolated colors) → 133,096 colored points; landmark route (s3a's
    colored pytorch3d view sweep → MediaPipe → pix_to_face unproject → trimmed Umeyama → rigid
    scale-frozen face polish vs the fitted ICT face). NOTE: TRELLIS outputs are HEAD-ONLY, so s3a's
    dist=1.7 crops the face and MediaPipe never detects — s3d defaults `--dist 2.4` (measured: robust
    at 2.2–3.5).
  - Gates (all PASS, artifacts saved before gating): landmark rms **0.80 cm** (≤1.5), inner-landmark→
    surface **0.32 cm** (≤1.0), direction-aware front-depth **0.83 cm**/0 missing (≤1.2), aligned-
    TripoSG-verts→TRELLIS median **0.325 cm** / rms(≤3cm) 0.66 (≤1.2), y-ratio 1.26 (0.5–2.0).
  - Outputs `out/clay/clay_trellis_aligned.{ply,npz}` + `clay_trellis_align.json` (S/R/T + metrics) +
    opaque mesh proof renders `clay_trellis_render_{front,right,back}.png`, dense-splat checks, and a
    back A/B `clay_trellis_vs_triposr_back.png`. TripoSR/TripoSG clays untouched.
  - Measured back-of-skull (z<−4, y>−2): TRELLIS mean RGB **[0.156, 0.142, 0.140]** luma 0.146 = real
    dark-brown hair vs TripoSR **[0.514, 0.439, 0.388]** luma 0.456 = washed pale beige.
  - Selector: `CLAY_COLOR=trellis` (default `triposr`, zero regression) → s5 `--clay-ply
    out/clay/clay_trellis_aligned.ply` (dies if missing; blend logic UNCHANGED — retuning the s5 blend
    for the darker TRELLIS palette is the next task). `bake_metrics.json` now records `clay_source`.
  - Pod note: mediapipe 0.10.35 c-bindings need `libEGL.so.1`/`libGLESv2.so.2` — root FS is ephemeral,
    re-`apt-get install -y libegl1 libgles2` after a pod restart.

## Decisions (don't re-litigate)
- **"Hole in the back of the head" was NOT geometry** — it was Blender's glTF importer assigning
  `blend_method=HASHED` + `show_transparent_back=True`, making the skin see-through so the internal
  teeth/tongue/eyeballs showed through. Proven: raw ICT, our pre-/post-shrinkwrap neutrals, and the
  exported GLB with a flat opaque material all render a **solid** closed back; topology is closed and
  identical to ICT (23 boundary loops) at every stage. Fixes: `open_avatar.py` forces opaque on import;
  s6 hardens the GLB to `alphaMode:"OPAQUE"` + single-sided. Diagnostic renders in `out/renders/`.
- **TRELLIS.2 rejected as the clay generator** — (a) it wouldn't fix the above (we always retopologize
  onto ICT, so the clay never becomes the output surface), and (b) it depends on `nvdiffrast`/`nvdiffrec`
  under NVIDIA's 1-Way *non-commercial* license, which breaks the commercial requirement (the exact NC
  trap avoided by choosing PyTorch3D). Keep TripoSR (MIT).

## Pod
RunPod RTX 6000 Ada. Pipeline at `/workspace/newstack/pipe/`; ICT/TripoSR at `/workspace/newstack/`; Blender 4.2.3 (headless via `xvfb-run`).
