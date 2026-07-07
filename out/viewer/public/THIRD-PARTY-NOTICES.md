# Third-Party Notices — ARKit Avatar Viewer (new stack: TripoSR + ICT-FaceKit)

This web product is the rigged head **`head_arkit_v2.glb`** plus this **three.js + MediaPipe**
viewer (the `dist/` bundle) that loads the GLB and drives its 52 ARKit blendshape morph targets
live from a webcam. The third-party components it **redistributes** or that materially **shaped
the shipped GLB** are listed below, with their required attributions and license texts, and are
surfaced in-app via the **"Credits / Licenses"** panel.

Verified against the actually-shipped `dist/` on **2026-07-07** (all licenses re-checked live this
run; commercial gate of record: `out/compliance_newstack.md`). Full verbatim license texts also
ship alongside this file under `licenses/`.

> **Scope.** Two categories matter and are kept distinct:
> 1. **Redistributed runtime code/binaries** in `dist/` — three.js (bundled JS) and the MediaPipe
>    FaceLandmarker WASM runtime. These carry live attribution obligations and are honored below.
> 2. **GLB provenance** — components used only on the build pod to *produce* `head_arkit_v2.glb`.
>    Their code/binaries are **not** in `dist/`. Where a component's data is baked into the GLB
>    (ICT-FaceKit topology + blendshapes), its attribution is honored as if required; where it is
>    not (TripoSR clay is retopologized away; the rembg/U²-Net mask is a transient intermediate),
>    the credit is a transparency courtesy. All are recorded so nothing reads as a missed notice.
>
> The avatar's **face texture is baked from the end user's own input photo** — it carries no
> third-party image rights.

---

# A. Redistributed runtime components (attribution required)

## A.1 three.js — MIT License

The 3D viewer is built on **three.js** (`three@0.170.0`). three.js is bundled into the app's
JavaScript (`assets/index-*.js`), and its MIT banner is preserved in that bundle
(`Copyright 2010-2024 Three.js Authors`). Full text also at `licenses/three.js-MIT-LICENSE.txt`.

- **Component:** three.js `three@0.170.0` (includes `examples/jsm/loaders/GLTFLoader.js`)
- **Copyright:** © 2010–2024 three.js authors
- **License:** MIT

```
The MIT License

Copyright © 2010-2024 three.js authors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
```

> **Draco note (measured, not assumed):** the GLB is **not** Draco-compressed
> (`extensionsUsed: none`; no `KHR_draco_mesh_compression` on any primitive; generator
> `Khronos glTF Blender I/O v4.2.70`) and the viewer loads it with a plain `GLTFLoader` — it never
> imports or instantiates `DRACOLoader`, and **no Draco decoder (`.wasm`/`.js`) ships in `dist/`.**
> Draco (Apache-2.0) is therefore **not a dependency of this product** and is intentionally not
> listed as one. If a future build enables Draco compression, add the Draco Apache-2.0 NOTICE and
> ship the decoder + its license before re-shipping.

## A.2 MediaPipe `@mediapipe/tasks-vision` — Apache License 2.0

Face tracking is Google's **MediaPipe** (`@mediapipe/tasks-vision@0.10.14`). This product
**redistributes MediaPipe binaries** — the FaceLandmarker WebAssembly runtime under
`mediapipe/wasm/*` (`vision_wasm_internal.{js,wasm}`, `vision_wasm_nosimd_internal.{js,wasm}`) — so
Apache-2.0 §4(a) requires a copy of the license accompany the distribution. It is bundled below and
at `licenses/mediapipe-Apache-2.0-LICENSE.txt`.

- **Component:** MediaPipe Tasks Vision (FaceLandmarker) — `@mediapipe/tasks-vision@0.10.14`
- **Copyright:** Copyright 2019–2024 The MediaPipe Authors (Google LLC), `mediapipe@google.com`
- **License:** Apache License, Version 2.0
- **Upstream source of this license text:** `google-ai-edge/mediapipe` repository `LICENSE`
  (https://github.com/google-ai-edge/mediapipe/blob/master/LICENSE) — the npm tarball for
  `@mediapipe/tasks-vision` ships no LICENSE and no NOTICE, so both were sourced from upstream.

**NOTICE-file status (Apache-2.0 §4(d)):** as of the 2026-07-07 verification the upstream
repository ships an Apache-2.0 `LICENSE` but **no separate `NOTICE` text file**. §4(d) only
obligates propagation of a NOTICE *if the upstream Work includes one*, so there is no NOTICE content
to reproduce. The §4(a) obligation (include a copy of the License) is met by bundling the full text
below. The attribution above stands in for the customary NOTICE credit. **If a future MediaPipe
release adds a NOTICE file, its contents must be reproduced here before re-shipping.**

## A.3 MediaPipe FaceLandmarker model (`face_landmarker.task`) — Apache License 2.0

The FaceLandmarker **model bundle** (`float16/1/face_landmarker.task`) produces the 468 landmarks +
52 ARKit-named blendshape coefficients that drive the avatar. In the shipped `dist/` this model is
**loaded at runtime from the Google CDN**
(`https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task`)
and is **not redistributed** in the bundle (no `.task` file ships; `npm run fetch-model` is
optional and was not used for this build). The same model is also used at **build time** (stage s1,
photo landmarks/iris). Its license still governs commercial **use** and its **output**:

- **License (all three sub-models):** Apache License, Version 2.0.
  - **FaceDetector — MediaPipe BlazeFace (Short Range)** model card: "LICENSED UNDER Apache
    License, Version 2.0" (Google, dated 2021-06-09).
  - **FaceMesh V2** model card: Apache License, Version 2.0.
  - **Blendshape V2** model card ("A lightweight model to predict 52 facial blendshapes"): "LICENSED
    UNDER Apache License, Version 2.0" (Google, dated 2022-11-11).
- **Commercial use:** permitted (Apache-2.0). **Output (landmarks + blendshape coefficients):** no
  license restriction; the Blendshape V2 card notes the coefficients "do not provide facial
  recognition or identification and do not store any unique face representation."
- **Out-of-scope (ethics guidance, not a license term):** the cards state the models are not for
  human life-critical decisions and are not surveillance/identity tools. These are usage
  expectations, not commercial-license restrictions.
- **Obligation:** Apache-2.0 attribution to The MediaPipe Authors (Google), satisfied by A.2 above.

### Apache License 2.0 (full text, as shipped upstream by MediaPipe)

```
                                 Apache License
                           Version 2.0, January 2004
                        http://www.apache.org/licenses/

   TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION

   1. Definitions.

      "License" shall mean the terms and conditions for use, reproduction,
      and distribution as defined by Sections 1 through 9 of this document.

      "Licensor" shall mean the copyright owner or entity authorized by
      the copyright owner that is granting the License.

      "Legal Entity" shall mean the union of the acting entity and all
      other entities that control, are controlled by, or are under common
      control with that entity. For the purposes of this definition,
      "control" means (i) the power, direct or indirect, to cause the
      direction or management of such entity, whether by contract or
      otherwise, or (ii) ownership of fifty percent (50%) or more of the
      outstanding shares, or (iii) beneficial ownership of such entity.

      "You" (or "Your") shall mean an individual or Legal Entity
      exercising permissions granted by this License.

      "Source" form shall mean the preferred form for making modifications,
      including but not limited to software source code, documentation
      source, and configuration files.

      "Object" form shall mean any form resulting from mechanical
      transformation or translation of a Source form, including but
      not limited to compiled object code, generated documentation,
      and conversions to other media types.

      "Work" shall mean the work of authorship, whether in Source or
      Object form, made available under the License, as indicated by a
      copyright notice that is included in or attached to the work
      (an example is provided in the Appendix below).

      "Derivative Works" shall mean any work, whether in Source or Object
      form, that is based on (or derived from) the Work and for which the
      editorial revisions, annotations, elaborations, or other modifications
      represent, as a whole, an original work of authorship. For the purposes
      of this License, Derivative Works shall not include works that remain
      separable from, or merely link (or bind by name) to the interfaces of,
      the Work and Derivative Works thereof.

      "Contribution" shall mean any work of authorship, including
      the original version of the Work and any modifications or additions
      to that Work or Derivative Works thereof, that is intentionally
      submitted to Licensor for inclusion in the Work by the copyright owner
      or by an individual or Legal Entity authorized to submit on behalf of
      the copyright owner. For the purposes of this definition, "submitted"
      means any form of electronic, verbal, or written communication sent
      to the Licensor or its representatives, including but not limited to
      communication on electronic mailing lists, source code control systems,
      and issue tracking systems that are managed by, or on behalf of, the
      Licensor for the purpose of discussing and improving the Work, but
      excluding communication that is conspicuously marked or otherwise
      designated in writing by the copyright owner as "Not a Contribution."

      "Contributor" shall mean Licensor and any individual or Legal Entity
      on behalf of whom a Contribution has been received by Licensor and
      subsequently incorporated within the Work.

   2. Grant of Copyright License. Subject to the terms and conditions of
      this License, each Contributor hereby grants to You a perpetual,
      worldwide, non-exclusive, no-charge, royalty-free, irrevocable
      copyright license to reproduce, prepare Derivative Works of,
      publicly display, publicly perform, sublicense, and distribute the
      Work and such Derivative Works in Source or Object form.

   3. Grant of Patent License. Subject to the terms and conditions of
      this License, each Contributor hereby grants to You a perpetual,
      worldwide, non-exclusive, no-charge, royalty-free, irrevocable
      (except as stated in this section) patent license to make, have made,
      use, offer to sell, sell, import, and otherwise transfer the Work,
      where such license applies only to those patent claims licensable
      by such Contributor that are necessarily infringed by their
      Contribution(s) alone or by combination of their Contribution(s)
      with the Work to which such Contribution(s) was submitted. If You
      institute patent litigation against any entity (including a
      cross-claim or counterclaim in a lawsuit) alleging that the Work
      or a Contribution incorporated within the Work constitutes direct
      or contributory patent infringement, then any patent licenses
      granted to You under this License for that Work shall terminate
      as of the date such litigation is filed.

   4. Redistribution. You may reproduce and distribute copies of the
      Work or Derivative Works thereof in any medium, with or without
      modifications, and in Source or Object form, provided that You
      meet the following conditions:

      (a) You must give any other recipients of the Work or
          Derivative Works a copy of this License; and

      (b) You must cause any modified files to carry prominent notices
          stating that You changed the files; and

      (c) You must retain, in the Source form of any Derivative Works
          that You distribute, all copyright, patent, trademark, and
          attribution notices from the Source form of the Work,
          excluding those notices that do not pertain to any part of
          the Derivative Works; and

      (d) If the Work includes a "NOTICE" text file as part of its
          distribution, then any Derivative Works that You distribute must
          include a readable copy of the attribution notices contained
          within such NOTICE file, excluding those notices that do not
          pertain to any part of the Derivative Works, in at least one
          of the following places: within a NOTICE text file distributed
          as part of the Derivative Works; within the Source form or
          documentation, if provided along with the Derivative Works; or,
          within a display generated by the Derivative Works, if and
          wherever such third-party notices normally appear. The contents
          of the NOTICE file are for informational purposes only and
          do not modify the License. You may add Your own attribution
          notices within Derivative Works that You distribute, alongside
          or as an addendum to the NOTICE text from the Work, provided
          that such additional attribution notices cannot be construed
          as modifying the License.

      You may add Your own copyright statement to Your modifications and
      may provide additional or different license terms and conditions
      for use, reproduction, or distribution of Your modifications, or
      for any such Derivative Works as a whole, provided Your use,
      reproduction, and distribution of the Work otherwise complies with
      the conditions stated in this License.

   5. Submission of Contributions. Unless You explicitly state otherwise,
      any Contribution intentionally submitted for inclusion in the Work
      by You to the Licensor shall be under the terms and conditions of
      this License, without any additional terms or conditions.
      Notwithstanding the above, nothing herein shall supersede or modify
      the terms of any separate license agreement you may have executed
      with Licensor regarding such Contributions.

   6. Trademarks. This License does not grant permission to use the trade
      names, trademarks, service marks, or product names of the Licensor,
      except as required for reasonable and customary use in describing the
      origin of the Work and reproducing the content of the NOTICE file.

   7. Disclaimer of Warranty. Unless required by applicable law or
      agreed to in writing, Licensor provides the Work (and each
      Contributor provides its Contributions) on an "AS IS" BASIS,
      WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
      implied, including, without limitation, any warranties or conditions
      of TITLE, NON-INFRINGEMENT, MERCHANTABILITY, or FITNESS FOR A
      PARTICULAR PURPOSE. You are solely responsible for determining the
      appropriateness of using or redistributing the Work and assume any
      risks associated with Your exercise of permissions under this License.

   8. Limitation of Liability. In no event and under no legal theory,
      whether in tort (including negligence), contract, or otherwise,
      unless required by applicable law (such as deliberate and grossly
      negligent acts) or agreed to in writing, shall any Contributor be
      liable to You for damages, including any direct, indirect, special,
      incidental, or consequential damages of any character arising as a
      result of this License or out of the use or inability to use the
      Work (including but not limited to damages for loss of goodwill,
      work stoppage, computer failure or malfunction, or any and all
      other commercial damages or losses), even if such Contributor
      has been advised of the possibility of such damages.

   9. Accepting Warranty or Additional Liability. While redistributing
      the Work or Derivative Works thereof, You may choose to offer,
      and charge a fee for, acceptance of support, warranty, indemnity,
      or other liability obligations and/or rights consistent with this
      License. However, in accepting such obligations, You may act only
      on Your own behalf and on Your sole responsibility, not on behalf
      of any other Contributor, and only if You agree to indemnify,
      defend, and hold each Contributor harmless for any liability
      incurred by, or claims asserted against, such Contributor by reason
      of your accepting any such warranty or additional liability.

   END OF TERMS AND CONDITIONS

   APPENDIX: How to apply the Apache License to your work.

      To apply the Apache License to your work, attach the following
      boilerplate notice, with the fields enclosed by brackets "[]"
      replaced with your own identifying information. (Don't include
      the brackets!)  The text should be enclosed in the appropriate
      comment syntax for the file format. We also recommend that a
      file or class name and description of purpose be included on the
      same "printed page" as the copyright notice for easier
      identification within third-party archives.

   Copyright [yyyy] [name of copyright owner]

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.

===========================================================================
For files under tasks/cc/text/language_detector/custom_ops/utils/utf/
===========================================================================
/*
 * The authors of this software are Rob Pike and Ken Thompson.
 *              Copyright (c) 2002 by Lucent Technologies.
 * Permission to use, copy, modify, and distribute this software for any
 * purpose without fee is hereby granted, provided that this entire notice
 * is included in all copies of any software which is or includes a copy
 * or modification of this software and in all copies of the supporting
 * documentation for such software.
 * THIS SOFTWARE IS BEING PROVIDED "AS IS", WITHOUT ANY EXPRESS OR IMPLIED
 * WARRANTY.  IN PARTICULAR, NEITHER THE AUTHORS NOR LUCENT TECHNOLOGIES MAKE ANY
 * REPRESENTATION OR WARRANTY OF ANY KIND CONCERNING THE MERCHANTABILITY
 * OF THIS SOFTWARE OR ITS FITNESS FOR ANY PARTICULAR PURPOSE.
 */
```

> The Lucent Technologies addendum above governs only text/language-detector UTF utilities under
> `tasks/cc/text/...`; it is reproduced because it is part of MediaPipe's upstream `LICENSE` file,
> even though those source files are not part of the FaceLandmarker vision WASM this product ships.

---

# B. GLB provenance — components that shaped `head_arkit_v2.glb`

## B.1 ICT-FaceKit (Light / FaceXModel) — MIT, © 2020 USC Institute for Creative Technologies

The shipped avatar's **head topology** and its **ARKit-aligned expression blendshapes** are derived
from the **ICT Face Model (Light)** in `USC-ICT/ICT-FaceKit`. Because that geometry and the
blendshape basis are **baked into `head_arkit_v2.glb`**, this attribution is honored as a live MIT
obligation (retain the copyright + permission notice). Full text at
`licenses/ict-facekit-MIT-LICENSE.txt`.

- **Component:** ICT-FaceKit — **Light / FaceXModel only** (base topology, 100 PCA identity modes,
  and the pre-authored ARKit-style expression blendshapes).
- **Copyright / License:** Copyright (c) 2020 USC Institute for Creative Technologies — MIT License
  (LICENSE: "ICT-FaceKit is released under the MIT license.").
- **Source:** https://github.com/USC-ICT/ICT-FaceKit
- **Changes made:** the Light model was fit to an input photo (identity coefficients), had hair and
  head proportions shrink-wrapped toward a photo-derived clay, was re-textured from the user's photo,
  and re-rigged/exported as a 52-morph GLB — these are modifications of the original.
- **Hard scope constraint:** only the in-repo **Light** model is used. The **Full** ICT Face Model
  ("will be released under a different USC specific license") is **not** used and is **not** cleared.

```
MIT License

Copyright (c) 2020 USC Institute for Creative Technologies

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## B.2 TripoSR — MIT, © 2024 Tripo AI & Stability AI (build-time; not in the GLB)

TripoSR turns the single input photo into a rough head/hair **"clay"** that guides shrink-wrap for
hair volume and proportions. **The clay is retopologized onto ICT geometry and does not survive into
the shipped mesh** — no TripoSR-authored geometry is present in `head_arkit_v2.glb`, and TripoSR
code/weights are not redistributed. This credit is a transparency courtesy, not a redistribution
obligation. Full text at `licenses/triposr-MIT-LICENSE.txt`.

- **Component:** TripoSR — `VAST-AI-Research/TripoSR` code + `stabilityai/TripoSR` weights.
- **License:** MIT for **both code and weights** (repo LICENSE: "MIT License", "Copyright (c) 2024
  Tripo AI & Stability AI"; Hugging Face model card metadata: `license: mit`).
- **Training data:** "a carefully curated subset of the Objaverse dataset ... available under the
  CC-BY license" (model card). Weights are the trained artifact, released MIT; no CC-BY attribution
  flows downstream to this product. Courtesy credit: Objaverse (CC-BY) + Stability AI.

## B.3 Background mask — rembg (MIT) + U²-Net weights (Apache-2.0) (build-time; not in the GLB)

Background removal on the input photo. The mask is a transient build intermediate; neither the code
nor the weights are redistributed, and the mask is not present in the GLB. Transparency courtesy:

- **rembg:** MIT — `danielgatis/rembg` (© Daniel Gatis). Default model pinned to **`u2net`**.
- **U²-Net (`u2net`) weights:** **Apache License, Version 2.0** — `xuebinqin/U-2-Net` (Xuebin Qin
  et al.); the repo `LICENSE` is the Apache-2.0 template. Commercial use permitted. Same Apache-2.0
  text as bundled at `licenses/mediapipe-Apache-2.0-LICENSE.txt`.
- **Ship condition:** keep the rembg model pinned to `u2net`. Do **not** silently fall back to
  `isnet-*`/`silueta` (different data lineage) or to Bria RMBG (CC-BY-NC) without re-verifying.

## B.4 Face texture — the end user's own photo

The avatar's albedo/texture is baked from the **user-supplied input photograph**. It contains no
third-party imagery and carries no third-party license obligation; rights are the user's own.

---

# C. Pod-side build tools NOT redistributed (recorded decision, not an omission)

These run only on the reconstruction/rigging pod to *produce* `head_arkit_v2.glb`. Their code and
binaries are **not** in `dist/` (verified: `dist/` contains only the three.js bundle, the GLB, the
MediaPipe WASM runtime, the two `licenses/*.txt`, `index.html`, `README.txt`, and this file — **no
`bpy`/Blender/Python binary, no `.so`/`.dll`, no Draco decoder**). Under each license the
attribution/notice trigger is *redistribution of that component's code/binary*, which the shipped
product does not do.

| Pod build tool | Role | License | Notice required in shipped product? |
|---|---|---|---|
| **Blender 4.2.3** (`bpy`) | headless glTF/GLB assembly, shrink-wrap, deformation transfer, bake, export | **GPL** | **No** — Blender processes data like a compiler; its GPL does **not** reach the GLB (the GLB is generated data containing no Blender code), and **no Blender/bpy binary ships**. Do not redistribute the Blender/bpy build as part of the product. |
| **PyTorch3D** | fit/bake rasterization, cameras, Umeyama alignment | BSD-3-Clause | **No** — not redistributed. |
| **OpenCV** (`opencv-python`) | classical inpainting during texture bake | Apache-2.0 | **No** — not redistributed. |
| **transformers** (Hugging Face) | TripoSR model plumbing | Apache-2.0 | **No** — not redistributed. |
| **PyTorch** (`torch`/`torchvision`) | tensor/runtime for TripoSR + fit | BSD-3-Clause | **No** — not redistributed. |

**Trigger to revisit:** if the **pod image, pod pipeline code, or any of these binaries** is ever
itself redistributed (as opposed to only its GLB output), the corresponding notices — and for
**Blender the GPL source-offer obligation** — must be added to that distribution.

---

# Summary

| Component | Role in shipped product | License | Commercial | Obligation status |
|---|---|---|---|---|
| three.js `0.170.0` | viewer/renderer; JS bundled | MIT | ✅ | Copyright + permission notice — **wired** (in-bundle banner + this file + `licenses/`) |
| `@mediapipe/tasks-vision` WASM | face tracking runtime; binaries redistributed | Apache-2.0 | ✅ | License text bundled (§4(a)); no upstream NOTICE (§4(d) n/a) — **wired** |
| MediaPipe `face_landmarker.task` | landmarks + 52 blendshapes; loaded from Google CDN at runtime (not redistributed) | Apache-2.0 | ✅ | Attribution to The MediaPipe Authors — **wired**; output unrestricted |
| ICT-FaceKit (Light) | head topology + ARKit blendshapes baked into GLB | MIT © 2020 USC-ICT | ✅ | Copyright + permission notice — **wired** (this file + `licenses/`); **Light only** |
| TripoSR | build-time clay; retopologized away, not in GLB | MIT © 2024 Tripo AI & Stability AI | ✅ | Courtesy credit — **wired** (not a redistribution obligation) |
| rembg + U²-Net (`u2net`) | build-time background mask; not in GLB | MIT / Apache-2.0 | ✅ | Courtesy credit — **wired**; pin `u2net` |
| User photo texture | avatar albedo | user's own | ✅ | No third-party rights |
| Draco | **not used / not shipped** (GLB uncompressed) | — | n/a | Intentionally not a dependency (measured) |
| Blender / PyTorch3D / OpenCV / transformers / torch | pod build tools; NOT shipped | GPL / BSD-3 / Apache-2.0 | ✅ (as tools) | Not required for the web product (recorded decision) |

Gate of record: `out/compliance_newstack.md` — **NEWSTACK-SHIP-CLEARED (conditional on the four
build-time hygiene items).**
