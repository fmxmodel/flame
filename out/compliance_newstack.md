# Compliance Record — NEW COMMERCIAL STACK (supersedes FLAME B1 section)

**Gate:** Licensing / Compliance (blocking authority)
**Run intent:** Track B — COMMERCIAL (single photo → ARKit-52 GLB → browser-driven).
**Verified:** 2026-07-05 (all terms re-checked live this run; licenses change — re-verify next run).
**Supersedes:** the FLAME-based B1 clearance in `out/compliance_report.md`. FLAME / MPI texture
are **dropped** from the commercial path (see Dropped Set below). This document is the current
commercial gate of record.

Method note: applied the **whitewashing test** to every component — a permissive CODE license
does NOT launder non-commercial TRAINING DATA or model weights. Each weight/model was checked
against the license of the party that actually owns the trained artifact.

---

## Active stack — per-component verdict

### 1. Mask — rembg (wrapper) + U²-Net (default weights)
- **rembg license:** MIT — PyPI/GitHub `danielgatis/rembg`, verified 2026-07-05.
- **Default model:** `u2net` (confirmed default). Weights from `xuebinqin/U-2-Net`, **Apache-2.0**
  (verified on the GitHub repo page, 2026-07-05). Trained on DUTS-TR (academic saliency set), but
  the authors released the weights under Apache-2.0 — commercial use permitted.
- **Commercial verdict:** ✅ YES.
- **Obligation:** Apache-2.0 → preserve copyright + include a NOTICE/attribution entry. MIT (rembg)
  → include the MIT copyright + permission notice. **Pin the model to `u2net`** in config.
- **Flag (verify-item c):** do NOT switch the default to `silueta`, `isnet-general-use`, or
  `isnet-anime` without re-verifying. `silueta` is a size-reduced `u2net` (same Apache lineage);
  `isnet-general-use` is commonly cited Apache-2.0, but the underlying **DIS/DIS5K** research
  dataset carries academic-use language. `u2net` is the clean default — keep it pinned.

### 2. Shape+hair "clay" — TripoSR (VAST-AI-Research / Stability AI)
- **License:** MIT for **both code and weights** — `stabilityai/TripoSR` model card +
  `VAST-AI-Research/TripoSR` repo, verified 2026-07-05.
- **Training data:** a "carefully curated subset of the Objaverse dataset, available under the
  **CC-BY** license" (model card, verbatim).
- **Commercial verdict:** ✅ YES (see verify-item a for the reasoning through Objaverse).
- **Obligation:** MIT notice for code+weights. The model card carries an ethics/misuse clause
  (no disturbing/offensive/stereotype content) — that is an acceptable-use expectation, not a
  commercial-license restriction. Courtesy credit to Objaverse/Stability recommended (not required).

### 3. Topology + 52 ARKit rig "mold" — ICT-FaceKit (USC-ICT)
- **License:** MIT, © USC Institute for Creative Technologies 2020 — `USC-ICT/ICT-FaceKit`
  LICENSE + README, verified 2026-07-05.
- **What ships under MIT:** the **ICT Face Model *Light*** in the repo — base topology, 100 PCA
  identity morph modes, and the **53 expression blendshapes** (the ARKit-aligned set we use).
- **Commercial verdict:** ✅ YES for the **Light** model (see verify-item b).
- **Obligation:** MIT notice. **Hard constraint:** use ONLY the in-repo Light model. Do NOT pull
  the *Full* ICT Face Model — the README states it "will be released under a different USC specific
  license," which is NOT cleared here.

### 4. Fuse — headless Blender (bpy): shrinkwrap + deformation-transfer + bake
- **License:** GPL (Blender). Server-side/SaaS use.
- **Commercial verdict:** ✅ YES as a server-side tool (see Blender/GPL section below).
- **Obligation:** ship **no Blender/bpy binary** to end users; the GLB output is generated data,
  not a GPL derivative of Blender.

### 5. Drive — MediaPipe FaceLandmarker (`@mediapipe/tasks-vision`)
- **License:** Apache-2.0 — verified 2026-07-05. Blendshape coefficients are ARKit-named (matches
  the 52-name contract).
- **Commercial verdict:** ✅ YES.
- **Obligation:** Apache-2.0 NOTICE/attribution entry.

### 6. Render — three.js + Draco
- **License:** three.js MIT; Draco Apache-2.0 — verified 2026-07-05.
- **Commercial verdict:** ✅ YES.
- **Obligation:** MIT notice (three.js) + Apache-2.0 NOTICE (Draco).

---

## The three verify-items — resolved

### (a) TripoSR trained on Objaverse — commercially usable? → **PASS (commercial-OK)**
Objaverse-as-a-whole is **ODC-By v1.0**, and its individual objects carry a **mix** of per-object
licenses (CC0, CC-BY, and some non-commercial/share-alike). That mix, taken raw, would be a problem.
The mitigating, verified fact: Stability/Tripo state they trained on a **carefully curated CC-BY
subset** of Objaverse — i.e. the commercially-permissive (attribution) slice, deliberately excluding
the NC objects — and then released the resulting **weights under MIT**. The weights are a trained
model, not a redistribution of the source 3D objects. Whitewashing test **passes**: the training
subset was CC-BY (commercial-permissive), not NC, and the artifact owner released MIT.
- **Residual risk (low):** we rely on Stability's public representation of the CC-BY subset; there
  is no independent per-object audit. Acceptable for ship; documented here.
- **Obligation:** none binding downstream (weights are MIT). Courtesy attribution to
  Objaverse + Stability AI recommended.

### (b) ICT-FaceKit — repo MIT AND is the light-stage model + ARKit blendshapes covered? → **PASS (commercial-OK), scoped to the Light model**
The repo LICENSE is genuine **MIT** (© USC ICT 2020), and the **Light** model files — topology,
100 PCA identity modes, and the 53 ARKit-style expression blendshapes — are distributed *in that
same MIT repo*. The light-stage-derived data is baked into a model that the copyright owner (USC)
chose to release under MIT; there is **no separate research-only data license** attached to the
Light model. Whitewashing test **passes** (owner released the derived model permissively).
- **Hard caveat:** the **Full** ICT Face Model is explicitly a *different USC-specific license* and
  is NOT cleared. Ship condition: the pipeline must consume only the MIT Light model.

### (c) rembg default U²-Net weights — commercial-OK? → **PASS (commercial-OK), pin `u2net`**
Confirmed the rembg **default IS `u2net`** (not `isnet`/`silueta`). `u2net` weights are
**Apache-2.0** (`xuebinqin/U-2-Net`); the rembg wrapper is MIT. Both commercial-OK.
- **Flag:** the concern in the prompt (default silently being an isnet/silueta variant with
  different terms) does **not** apply as configured — but is real if the model is changed. Ship
  condition: pin `u2net` and re-verify before adopting any DIS/isnet-derived model.

---

## Dropped set — confirmed non-commercial / correctly excluded (one line each)
- **FaceVerse** — Tsinghua, non-commercial research only. ❌ NC. Stays out.
- **Standard FLAME + MPI FLAME texture** — MPI non-commercial (research/education/artistic). ❌ NC.
- **DECA / EMOCA** — non-commercial research fitters. ❌ NC.
- **InsightFace (WebFace42M / antelopev2)** — code MIT, but trained models + annotation data are
  non-commercial-research-only (verified 2026-07-05). ❌ NC.
- **Arc2Face** — built on SD1.5 + InsightFace embeddings; weights CC-BY-NC / OpenRAIL-class. ❌ NC.
- **AuraFace (fal)** — note: *actually* commercial-OK (built to replace ArcFace). It is dropped
  for **architecture** reasons (the identity-to-2D generation step was removed entirely — the real
  photo is fed straight to TripoSR), NOT for licensing. Documented so no one "re-adds it to be safe."
- **nvdiffrast** — NVIDIA Source Code License, non-commercial research/evaluation only (verified
  2026-07-05; commercial needs a separate NVIDIA license). ❌ NC.
- **Bria RMBG-1.4 / 2.0** — CC-BY-NC-4.0; commercial use requires a paid BRIA agreement (verified
  2026-07-05). ❌ NC (as free weights). Stays out; `u2net` is the commercial mask.

None of the dropped components are present in the active stack. The removal of the identity-to-2D
step eliminates the entire InsightFace/Arc2Face/AuraFace/SD-OpenRAIL exposure at the source.

---

## Blender GPL-in-SaaS — reasoning + caveat
Standard, well-settled reasoning, confirmed applicable here:
- Blender/bpy runs **server-side** as a processing tool. GPL obligations attach to *distribution of
  the GPL program*, not to running it as a service, and not to the data it outputs.
- We **ship no Blender/bpy binary** to end users; the browser gets only the GLB.
- The **GLB output is generated user data, not a derivative work of Blender** — it contains no
  Blender GPL code. So the output asset is unencumbered by GPL.
- **The one caveat:** do NOT redistribute the Blender/bpy build itself (e.g. bundling `bpy` in a
  downloadable app, container image handed to customers, or a shipped desktop binary). If you ever
  distribute the build, that build must carry GPL and offer corresponding source. Keep it internal
  to the render/fuse server.

---

## Attribution / NOTICE obligations to satisfy before ship (in-product credits file)
Create a `THIRD-PARTY-NOTICES` / credits screen containing:
- **MIT notices** (copyright + permission text): rembg, TripoSR (code+weights), ICT-FaceKit
  (© USC ICT 2020), three.js.
- **Apache-2.0 NOTICE/attribution:** U²-Net (`xuebinqin/U-2-Net`), MediaPipe tasks-vision, Draco.
- **Courtesy credit (recommended, not required):** Objaverse + Stability AI (TripoSR training data).
No CC-BY attribution obligation flows to us for the TripoSR training set (weights are MIT).
No EULA/lawyer question exists in this stack — the FLAME/MetaHuman lawyer questions were removed
by the pivot.

---

## Verdict

All active components are commercially clearable. There is **no non-commercial component and no
open lawyer/EULA question** in the active stack. Clearance is **conditional** only on mechanical
config/attribution hygiene that is fully within this gate's authority to require — not on any
outside legal opinion:

1. **Pin the rembg model to `u2net`** (Apache-2.0). Do not silently fall back to `isnet-*`/`silueta`.
2. **Use only the ICT-FaceKit MIT *Light* model.** Never pull the *Full* USC model.
3. **Ship no Blender/bpy binary** — server-side/SaaS use only.
4. **Include the THIRD-PARTY-NOTICES** (MIT + Apache-2.0 attributions listed above) in-product.

If and only if those four hold at build time, this stack is clean to ship commercially. If any
fails (e.g. the recon agent points ICT at the Full model, or the mask default changes), this
reverts to a hard stop.

**NEWSTACK-SHIP-CLEARED: conditional** — clears to **yes** once items 1–4 above are confirmed at
build; each is a config/notice action, none requires a lawyer.
