# 02 — Rebuild XPU base image on oneAPI 2026.0+ + matching torch-xpu

**Branch:** `oneapi-2026-torch-xpu-base-image`  
**Base commit:** `krisclarkdev/vllm` @ `577e1a932` (production image `hal/vllm-xpu:kris-fork-577e1a932`)  
**Proposed new tag:** `hal/vllm-xpu:oneapi-2026.0-torch2.13-3e8eb5ccc` (or rebuild shortsha after further commits)  
**Status:** implemented on feature branch (Dockerfile + requirements) @ `3e8eb5ccc`. Build/smoke on `hal` still required. No push; no Ornith / DaemonSet cutover.

---

## Summary

Rebuild `docker/Dockerfile.xpu` so the runtime SYCL stack includes **Graph + `work_group_scratch_memory`** (oneAPI **2026.0+**). That is the unlock for later FA-in-graph (`03`).

**Why a torch bump is mandatory:** today’s `torch==2.12.0+xpu` pulls `intel-sycl-rt==2025.3.2`. Bumping only the apt oneAPI layer while leaving torch 2.12 leaves the process on a 2025.3 SYCL runtime — the same class of failure we already hit. Matching stack is **`torch==2.13.0+xpu`** (Requires-Dist: `intel-sycl-rt==2026.0.0`, `triton-xpu==3.7.2`).

**Does not change production:** keep Ornith on `hal/vllm-xpu:kris-fork-577e1a932` + eager until a later cutover feature. Never retag over `kris-fork-*`.

---

## Current base (audit @ `577e1a932`)

| Layer | Pin / value | Notes |
|-------|-------------|--------|
| Base OS | `ubuntu:24.04` (`vllm-base`) | Multi-stage: `rust-build` → `vllm-base` → `ucx-nixl-build` → `vllm-openai` |
| Python | 3.12 | Required by `vllm-xpu-kernels` abi3 wheel story |
| oneAPI (apt) | **2025.3** (`Pin: version 2025.3*`) | `intel-oneapi-compiler-dpcpp-cpp-2025.3`, `mkl-devel-2025.3`, `dnnl-devel-2025.3` |
| `LD_LIBRARY_PATH` | `/opt/intel/oneapi/compiler/2025.3/lib` + CCL/MPI 2021.15 | Tied to 2025.3 layout |
| UMD / compute-runtime | IGC `v2.34.4`, compute-runtime **`26.18.38308.1`**, Level Zero **`v1.28.2`** | Matches docs recommendation for kernels ≥0.1.10 |
| oneCCL (offline) | `intel-oneccl-2021.15.9` (BMG) | Comment: BMG not in default oneAPI 2025.3 CCL; pip `oneccl*` later **uninstalled** |
| torch-xpu | `torch==2.12.0` via `https://download.pytorch.org/whl/xpu` | Wheel deps: `intel-sycl-rt==2025.3.2`, … |
| `triton-xpu` | **3.7.1** (forced in Dockerfile after uninstall of `triton`) | Matches torch 2.12 |
| IPEX | **not installed** | No `intel_extension_for_pytorch` in this Dockerfile |
| Kernels | `krisclarkdev/vllm-xpu-kernels@aa156578` | Git pin in `requirements/xpu.txt`; image rebuild recompiles/reinstalls against new torch |
| Index | `PIP_EXTRA_INDEX_URL` / `UV_EXTRA_INDEX_URL` = pytorch `whl/xpu` | `UV_INDEX_STRATEGY=unsafe-best-match` |
| Cache | BuildKit `--mount=type=cache,target=/root/.cache/uv` (+ cargo caches on rust stage) | Keep; expect longer rebuild when oneAPI + torch layers invalidate |
| Production image | `hal/vllm-xpu:kris-fork-577e1a932` | Built/run on `hal`; Ornith DS in `intel_vllm_triton` (out of scope to edit live) |

**Failure this rebuild addresses:**

```text
RuntimeError: The sycl_ext_oneapi_work_group_scratch_memory feature is not yet
available for use with the SYCL Graph extension
```

Evidence: Intel oneAPI DPC++ **2026.0** release notes add Graph support for `sycl_ext_oneapi_work_group_scratch_memory`; `intel/torch-xpu-ops#3142` closed after uplift to **26.0** (verified with `torch 2.13.0.dev…+xpu`).

**Host (`hal`) assumptions (from `intel_vllm_triton` constitution, not changed here):** Arc node with compute-runtime ~26.18 class drivers; host toolkit inventory already lists DPC++/MKL/oneDNN **2026.0.x** alongside older “base toolkit 2025.3” notes. Container still ships its own UMD/oneAPI — host driver must remain ≥ image compute-runtime expectations.

---

## Target stack matrix

| Component | Target | Source / rationale |
|-----------|--------|--------------------|
| oneAPI toolkit (image) | **≥ 2026.0** (pin **`2026.0*`** first) | Minimum for Graph + work-group scratch; prefer staying on 2026.0 until 2026.1 torch wheels are required |
| Apt packages | `intel-oneapi-compiler-dpcpp-cpp-2026.0`, `intel-oneapi-mkl-devel-2026.0`, `intel-oneapi-dnnl-devel-2026.0` | Mirror current Dockerfile style with new pin file `oneapi-2026.0.pref` |
| Fallback toolkit install | DLE offline `intel-deep-learning-essentials-2026.0.0.624_offline.sh` | Same URL PyTorch CI uses for `XPU_VERSION=2026.0` if apt pin fails or layout differs |
| Python | **3.12** | Unchanged |
| torch | **`torch==2.13.0`** (`+xpu` from pytorch xpu index) | Wheel Requires-Dist: `intel-sycl-rt==2026.0.0`, `dpcpp-cpp-rt==2026.0.0`, `triton-xpu==3.7.2`, `oneccl==2022.0.0` |
| torchvision | Prefer **`0.28.0+xpu`** (present on xpu index) | Pin explicitly if resolver drifts |
| torchaudio | **Resolve risk:** xpu index currently tops out at **2.11.0** (no 2.13+xpu listed) | At implement time: pin compatible wheel, or keep unpinned and accept resolver result / make optional if install blocks |
| `triton-xpu` | **3.7.2** | Match torch 2.13 Requires-Dist; update Dockerfile force-install |
| Pip Intel RT deps | Come with torch 2.13 (`intel-sycl-rt==2026.0.0`, …) | Do **not** leave 2025.3.2 RTs from torch 2.12 |
| oneCCL | Re-evaluate BMG offline **2021.15** vs torch’s **`oneccl==2022.0.0`** | Keep BMG offline + `uv pip uninstall oneccl oneccl-devel` if TP still needs 2021.15 BMG; else align to 2022.0 and update `LD_LIBRARY_PATH` / setvars. **Smoke TP after change.** |
| Level Zero / NEO | Keep **26.18.38308.1** + L0 **1.28.2** unless smoke shows driver mismatch | Bump only if `torch.xpu` init fails on `hal` |
| Kernels | Stay on **`aa156578`** initially | Rebuild in image against torch 2.13; if AOT/ABI breaks, document kernels bump as follow-up (not automatic production cutover) |
| IPEX | Still omit | Not required for current vLLM XPU path |
| vLLM source | Fork tip **`577e1a932`** (+ this branch’s Dockerfile/reqs) | Optional later: merge `01` canary commits for force-PIECEWISE guard in the same image |

**Explicit non-goals for this tag:** Ornith DS image swap; enabling FA FULL graphs in production; Laguna; gpt-oss.

---

## Dockerfile / build recipe (to implement after acceptance)

### A. Intended file edits (preview — do not apply until accepted)

1. **`docker/Dockerfile.xpu`**
   - Replace apt pin `2025.3*` → `2026.0*` and package names `*-2026.0`.
   - Update all `/opt/intel/oneapi/compiler/2025.3` path references → `2026.0` (ld.so.conf + `LD_LIBRARY_PATH`).
   - Change `triton-xpu==3.7.1` → `triton-xpu==3.7.2`.
   - Revisit oneCCL block (see matrix).
   - Optional ARG `ONEAPI_VERSION=2026.0` for less hardcoding.
2. **`requirements/xpu.txt`**
   - `torch==2.13.0` (keep `--extra-index-url=https://download.pytorch.org/whl/xpu`).
   - Pin `torchvision==0.28.0` if needed; resolve `torchaudio` as above.
   - Keep kernels git pin `@aa156578` unless rebuild proves a bump is required.
3. **`requirements/test/xpu.txt`** (lock-style)
   - Regenerate / bump `intel-*-rt` / `dpcpp-cpp-rt` / `torch==2.13.0+xpu` / `triton-xpu==3.7.2` when implementing (today still 2025.3.2 / torch 2.12).
4. **Docs (minimal)**
   - Touch `docs/getting_started/installation/gpu.xpu.inc.md` torch/triton notes only if pins change.
   - Optional note in `intel_vllm_triton` deploy README / build notes for the **new tag only** — **no** live DaemonSet edit.

### B. Build on `hal` (step-by-step)

```bash
# On hal, from krisclarkdev/vllm checkout of this branch
git fetch origin
git checkout oneapi-2026-torch-xpu-base-image
SHORT=$(git rev-parse --short HEAD)
TAG="hal/vllm-xpu:oneapi-2026.0-torch2.13-${SHORT}"

# Do NOT reuse or overwrite: hal/vllm-xpu:kris-fork-577e1a932

export DOCKER_BUILDKIT=1
docker build \
  -f docker/Dockerfile.xpu \
  --build-arg max_jobs="$(nproc)" \
  --progress=plain \
  -t "${TAG}" \
  .

docker image ls "${TAG}"
```

**Cache / multi-stage notes:**

- Expect **full invalidation** of `vllm-base` (oneAPI layer) and all Python layers (torch bump) — first build will be long.
- Rust stage can still hit cargo cache mounts; UCX/NIXL stage rebuilds if base changes.
- Prefer BuildKit cache mounts already in Dockerfile; avoid `--no-cache` unless debugging a bad layer.
- If apt 2026.0 packages 404: switch `vllm-base` oneAPI install to DLE offline installer  
  `https://registrationcenter-download.intel.com/akdlm/IRC_NAS/8170208e-86db-4faa-a0d6-1ecf62699574/intel-deep-learning-essentials-2026.0.0.624_offline.sh`  
  (PyTorch `install_xpu.sh` `XPU_VERSION=2026.0`).

**Import to k3s (only when intentionally testing — still not Ornith cutover):**

```bash
docker save "${TAG}" | sudo /usr/local/bin/k3s ctr images import -
```

### C. Verify SYCL / scratch-in-graph presence (container smoke, not Ornith)

```bash
docker run --rm --privileged \
  --device /dev/dri:/dev/dri \
  -v /dev/dri/by-path:/dev/dri/by-path \
  --entrypoint bash "${TAG}" -lc '
set -e
source /opt/intel/oneapi/setvars.sh --force 2>/dev/null || true
python - <<PY
import torch
print("torch", torch.__version__)
print("xpu_available", torch.xpu.is_available())
assert torch.xpu.is_available(), "XPU not visible"
# Runtime package versions (pip RTs must be 2026.0.x)
import importlib.metadata as m
for pkg in ["intel-sycl-rt", "dpcpp-cpp-rt", "triton-xpu"]:
    try:
        print(pkg, m.version(pkg))
    except Exception as e:
        print(pkg, "MISSING", e)
from vllm.utils.torch_utils import supports_xpu_graph
print("supports_xpu_graph", supports_xpu_graph())
# Compiler / toolkit hint
import pathlib, glob
print("compiler dirs", glob.glob("/opt/intel/oneapi/compiler/2026*"))
PY
'
```

**Scratch-in-graph probe (safe, non-Ornith):** prefer a tiny torch SDPA / XPUGraph capture that exercises fused attention with CUDA/XPU graph, or a one-shot vLLM serve on a **small dense** model with FA + graphs **only after** toolkit check passes. Document pass/fail; if the old `work_group_scratch_memory` error persists, stop — do not proceed to `03`.

Optional compiler-side check: `sycl-ls` / confirm libsycl from 2026.0 is first on `LD_LIBRARY_PATH`.

---

## Compatibility with `01` (TRITON_ATTN + PIECEWISE)

| Check | Expectation on new base |
|-------|-------------------------|
| `VLLM_XPU_ENABLE_XPU_GRAPH=1`, omit `--enforce-eager` | Graphs allowed if `supports_xpu_graph()` true (torch ≥ 2.11 already) |
| `--attention-backend TRITON_ATTN` | Still supported; preferred safe path until `03` |
| `-cc.cudagraph_mode=PIECEWISE` | Still valid; FA stays outside capture |
| Force-PIECEWISE guard from `01` branch | **Not** on `577e1a932` tip — either merge/cherry-pick canary commits into this branch before tagging a “canary-ready” image, or rely on explicit CLI flags |

**Regression gate:** after new image builds, re-run `01` canary smoke profile on a non-Ornith workload (small dense) before enabling any FA-in-graph experiments.

---

## Test plan

Ordered, all **off** production Ornith DS:

1. **Image build** completes; tag is unique (`oneapi-2026.0-torch2.13-<shortsha>`).
2. **Import/smoke:** `torch.xpu.is_available()`, `intel-sycl-rt` **2026.0.x**, `triton-xpu==3.7.2`, `supports_xpu_graph True`.
3. **Eager baseline:** short `vllm serve` / generate on small dense (e.g. Qwen2.5-1.5B) with `--enforce-eager` — Ready + sane tokens.
4. **`01` canary flags:** graphs on + `TRITON_ATTN` + `PIECEWISE` — Ready + decode (no scratch-in-graph crash expected).
5. **FA + graph probe (prep for `03` only):** optional single-process attempt; success = no `work_group_scratch_memory` Graph error. Correctness belongs to `03`.
6. **Do not** change `intel_vllm_triton` DaemonSet image or remove `--enforce-eager` on Ornith.

---

## Risks

| Risk | Mitigation |
|------|------------|
| torch / oneAPI mismatch → broken XPU init | Pair **torch 2.13.0+xpu** with toolkit **2026.0**; verify pip `intel-sycl-rt` version in smoke |
| `torchaudio` missing 2.13+xpu on index | Pin or relax at implement; don’t block serving on audio if unused |
| oneCCL 2021.15 BMG vs torch `oneccl 2022.0` | Keep uninstall-pip + offline BMG until TP smoke; or migrate carefully |
| Larger image / longer `hal` builds | Accept; use BuildKit caches; schedule off-peak |
| Host driver older than UMD | Keep compute-runtime 26.18 unless init fails; align with host inventory |
| Accidental overwrite of `kris-fork-*` | **New tag namespace only**; never `docker tag` over production |
| Kernels `@aa156578` ABI vs torch 2.13 | Rebuild in image; if fail, document kernels follow-up before `03` |
| Apt 2026.0 packages unavailable | Fallback to DLE offline installer (PyTorch CI URL) |

---

## Effort estimate and ordered tasks

**Effort:** ~1–2 working days on `hal` (Dockerfile + first cold build + smokes). Planning done; implementation gated on acceptance.

- [x] **T0** Plan accepted (this doc).
- [x] **T1** Edit `Dockerfile.xpu` oneAPI pin → 2026.0 + path/`LD_LIBRARY_PATH` updates.
- [x] **T2** Bump `requirements/xpu.txt` to `torch==2.13.0`; fix vision/audio pins; Dockerfile `triton-xpu==3.7.2`.
- [x] **T3** Refresh `requirements/test/xpu.txt` Intel RT pins (or document regen command).
- [x] **T4** Resolve oneCCL strategy (keep BMG 2021.15 vs 2022.0); update comments.
- [ ] **T5** Build `${TAG}` on `hal`; record size/time.
- [ ] **T6** Container smokes (toolkit versions, XPU init, eager, `01` canary flags).
- [ ] **T7** Optional scratch-in-graph / FA probe; record result for `03`.
- [x] **T8** Minimal doc note (install inc + plan status). Optional `intel_vllm_triton` note deferred until tag exists on `hal`.

---

## Follow-ups → `03` FA-in-graph

- Feature `03` (`fa-in-graph-dense-validation`) consumes **`hal/vllm-xpu:oneapi-2026.0-torch2.13-<shortsha>`**.
- Only after T6–T7 show scratch-in-graph no longer throws the known RuntimeError.
- Prefer `01` PIECEWISE proven (config and/or merged guard) as rollback path.
- Still out of scope here: Ornith MXFP4 MoE graphs canary (`04`) and production cutover (`05`).

---

## References

- oneAPI DPC++ 2026.0 release notes — SYCL Graph + work-group scratch size control.
- `intel/torch-xpu-ops#3142` — same error; verified fixed on oneAPI 26.0 + torch 2.13.dev.
- PyTorch CI `install_xpu.sh` — DLE 2026.0.0.624 offline URL; wheel extra requires `intel-sycl-rt==2026.0.0`.
- Wheel METADATA: `torch-2.12.0+xpu` → RT **2025.3.2**; `torch-2.13.0+xpu` → RT **2026.0.0**, `triton-xpu==3.7.2`.
