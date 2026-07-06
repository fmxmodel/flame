# newstack env report — brand-new RunPod (RTX 6000 Ada)

Provisioned a fresh, empty pod (previous /workspace volume NOT attached). Rebuilt the
entire newstack toolchain from scratch. All heavy artifacts persist under `/workspace`.

**STATUS: READY**  (measured, not claimed — see verification below)

Pod: `root@195.26.233.87 -p 37551`  ·  date 2026-07-06

---

## 1. Pod GPU / driver / base
| Item | Value |
|------|-------|
| GPU | NVIDIA RTX 6000 Ada Generation, 49 GB (49140 MiB) |
| Driver | 550.127.05 |
| CUDA (driver) | 12.4 |
| CUDA toolkit | 12.4.131 at `/usr/local/cuda/bin/nvcc` (NOT on PATH by default) |
| Base python | 3.11.10 |
| /workspace | fresh MooseFS mount, 160 TB free |

GPU is present and healthy — FaceVerse-style/TripoSR fitting runs on GPU (fast), not CPU.

## 2. Python / torch / CUDA (venv = /workspace/env, --system-site-packages)
Verification (`import torch,pytorch3d,scipy,mediapipe,rembg,trimesh; torch.cuda.is_available()`):

```
REQUIRED verify: True
torch 2.4.1+cu124   cuda 12.4   avail True   device NVIDIA RTX 6000 Ada Generation
pytorch3d 0.7.8
torchmcubes: import OK ; marching_cubes ran on cuda -> ([3117,3],[1039,3])
numpy 2.4.6   scipy 1.17.1
transformers 4.35.0   tokenizers 0.14.1   huggingface_hub 0.17.3
mediapipe 0.10.35   trimesh 4.12.2   onnxruntime 1.27.0
cv2 5.0.0   PIL 12.3.0   omegaconf 2.3.1   einops 0.8.2   rembg 2.0.76
```

Base image torch 2.4.1+cu124 was **reused** (venv `--system-site-packages`); torch NOT
reinstalled.

### Pins (why)
- `transformers==4.35.0` + `tokenizers<0.15` (=0.14.1) — the known TripoSR checkpoint /
  state_dict break on newer transformers/tokenizers.
- `huggingface_hub` left UNPINNED → resolver chose **0.17.3**. Pinning it to 0.19.4 caused
  `ResolutionImpossible` because tokenizers 0.14 caps huggingface_hub `<0.18`.
- torchmcubes + pytorch3d built from source with `FORCE_CUDA=1 TORCH_CUDA_ARCH_LIST=8.9`
  (sm_89 = Ada). Both required `--no-build-isolation` + explicit `Torch_DIR` and
  `pybind11_DIR` on the CMake line — pip build-isolation otherwise hides the venv's
  torch/pybind11 from CMake (`find_package(Torch)` / `find_package(pybind11)` failures).

## 3. Asset paths + sizes (all under /workspace, persisted)
| Asset | Path | Size / count |
|-------|------|--------------|
| git repo (main) | `/workspace/newarc` | 66 MB |
| photo | `/workspace/inputs/random-person.jpeg` | 519 KB |
| ICT-FaceKit FaceXModel (MIT) | `/workspace/newstack/ICT-FaceKit/FaceXModel` | **160 files, 389 MB** (no LFS pointers left) |
| TripoSR clay | `/workspace/newstack/out_triposr/0/mesh.obj` | 8.88 MB — **81,812 verts / 163,346 faces**, vertex-colored |
| MediaPipe task | `/workspace/models/mediapipe/face_landmarker.task` | 3.76 MB |
| Blender 4.2.3 LTS | `/workspace/blender/blender-4.2.3-linux-x64/blender` | 2.1 GB (headless verified) |
| venv | `/workspace/env` | 3.0 GB |
| HF weight cache | `/workspace/cache/huggingface` (TripoSR `model.ckpt`) | 1.6 GB |
| TripoSR repo | `/workspace/TripoSR` | 91 MB |

Blender headless OK: `xvfb-run -a .../blender --background --version` → `Blender 4.2.3 LTS`.

## 4. TripoSR clay generation
`run.py` default path (built-in rembg background removal + resize) on
`/workspace/inputs/random-person.jpeg`, `--device cuda:0 --mc-resolution 256`, obj format.
`--bake-texture` deliberately NOT passed (moderngl `create_context` → XOpenDisplay crash
headless). Result: `out_triposr/0/mesh.obj`, **81,812 v / 163,346 f**, non-trivial, vertex
colors present. HF weights cached under `/workspace/cache/huggingface` (persist via HF_HOME).

## 5. s1/s2 smoke-test (env WIRING only — stages 1 & 2)
Command: `ROOT=/workspace/newstack PIPE=/workspace/newarc/newstack/pipe
PHOTO=/workspace/inputs/random-person.jpeg STAGES="1 2" bash run_newstack.sh`
(venv active, `HF_HOME=/workspace/cache/huggingface`). **RC=0.**

- **s1 (landmarks):** 478 MediaPipe landmarks (expected 478). Wrote
  `out/landmarks/landmarks.npz`, `overlay.jpg`, `mp_blendshapes_photo.json`. 2.2 s.
  (Required apt `libgles2 libegl1` — mediapipe 0.10.35 needs `libGLESv2.so.2` at import.)
- **s2 (identity fit):** converged (loss 0.000210). Landmark reproj error mean **22.30 px**,
  max 87.61 px (eyes 5–7 px, contour ~40 px). `id_abs_max=0.013`, expression fit on. Wrote
  `out/fit/fitted_neutral.npy/.obj`, `camera.json`, `fit_metrics.json`, `fit_debug.jpg`. 12.7 s.

Stages 4–7 intentionally NOT run (pipe code owned by a parallel agent).

## 6. Exact end-to-end run command
```bash
source /workspace/env/bin/activate
export HF_HOME=/workspace/cache/huggingface
export CUDA_HOME=/usr/local/cuda PATH=/usr/local/cuda/bin:$PATH
ROOT=/workspace/newstack \
PIPE=/workspace/newarc/newstack/pipe \
PHOTO=/workspace/inputs/random-person.jpeg \
ICT=/workspace/newstack/ICT-FaceKit \
CLAY=/workspace/newstack/out_triposr/0/mesh.obj \
MP_TASK=/workspace/models/mediapipe/face_landmarker.task \
BLENDER=/workspace/blender/blender-4.2.3-linux-x64/blender \
STAGES="1 2 3 4 5 6 7" \
bash /workspace/newarc/newstack/run_newstack.sh
```
Output GLB: `/workspace/newstack/out/export/head_arkit_v2.glb`. Blender stages auto-wrap in
`xvfb-run -a`. Rerun rig only: `STAGES="4 5 6 7"`. Pure-ICT (skip clay): `REFINE=0`.

## 7. Persistence / restore
- `/workspace/restore_env.sh` — idempotent rebuild (apt libs + git-lfs always; venv,
  torchmcubes, pytorch3d, TripoSR weights, ICT, Blender, clay rebuilt if missing).
- `/workspace/RESTART.md` — layout, activate line, HF_HOME, run command, pins, gotchas.
- Container FS (apt, PATH) is wiped each pod start → restore_env.sh re-runs the apt block +
  `git lfs install` every time. Everything else lives under `/workspace`.

## 8. License flag (for license-compliance gate)
- ICT-FaceKit **FaceXModel = MIT** — commercial OK. Only the MIT "Light" model was pulled;
  no non-MIT extras.
- **TripoSR weights** license must be cleared before any commercial ship (this pod = Track A).

---
### PASS / FAIL per deliverable
| # | Deliverable | Result |
|---|-------------|--------|
| 1 | SSH + GPU verified (nvidia-smi, python, torch) | **PASS** |
| 2 | apt headless-Blender libs + tools (+ libgles2/libegl1 for mediapipe) | **PASS** |
| 3 | tongueOut repo cloned, photo copied | **PASS** |
| 4 | venv + full python stack (pytorch3d 0.7.8, torchmcubes CUDA, scipy, mediapipe, rembg) | **PASS** |
| 5 | TripoSR clay `out_triposr/0/mesh.obj` (81,812 v / 163,346 f) | **PASS** |
| 6 | ICT-FaceKit FaceXModel (160 files, MIT) | **PASS** |
| 7 | MediaPipe task + Blender 4.2.3 headless verified | **PASS** |
| 8 | s1/s2 smoke-test wiring (RC=0) | **PASS** |
| 9 | restore_env.sh + RESTART.md written | **PASS** |
