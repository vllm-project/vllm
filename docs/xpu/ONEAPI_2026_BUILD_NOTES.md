# Build notes — oneAPI 2026.0 + torch 2.13 XPU image

**Branch:** `oneapi-2026-torch-xpu-base-image` (`krisclarkdev/vllm`)  
**Do not** retag or overwrite `hal/vllm-xpu:kris-fork-*` (production Ornith).

## Built tag (hal)

`hal/vllm-xpu:oneapi-2026.0-torch2.13-3deb3160c` (~24.6GB)  
Built 2026-07-27 from tip `3deb3160c` + local Dockerfile / `requirements/test/xpu.txt` fixes (see Done).

---

## Done

- [x] Dockerfile oneAPI **2026.0** pin; `SYCL_HOME` / compiler on `PATH`; no global `setvars` for kernels cmake (avoids 2026.1 runtime without `icpx`).
- [x] `torch==2.13.0` (+xpu index), `torchvision==0.28.0`, `torchaudio==2.11.0`; kernels `@aa156578`.
- [x] Force `triton-xpu==3.7.2`; test lockfile `triton==3.7.1` (PyPI has no `triton==3.7.2`).
- [x] Image build on `hal` succeeded after:
  - Reinstall pip `oneccl==2022.0.0` (so.9) after historical uninstall of pip CCL.
  - Prefer `/opt/venv/lib` + source `${SYCL_HOME}/env/vars.sh` when installing vLLM (cmake XPU discovery).
- [x] Container smokes on Arc Pro B70 (`--privileged --device /dev/dri`):

| Check | Result |
| --- | --- |
| `torch` | `2.13.0+xpu` (`torch.version.xpu=20260000`) |
| `intel-sycl-rt` / `dpcpp-cpp-rt` | `2026.0.0` |
| `triton-xpu` | `3.7.2` |
| `vllm-xpu-kernels` | `0.1.12.dev33+gaa15657` |
| `torch.xpu.is_available()` | True |
| matmul on XPU | OK |
| `torch.xpu.XPUGraph` capture | OK |
| `supports_xpu_graph()` | **True** (`vllm.utils.torch_utils`) |

- [x] Production Ornith left alone: DS still paused for the build window (`hal.local/vllm-xpu-paused=true`, Desired 0); image tag not cut over.
- [x] **Eager serve smoke** — `Qwen/Qwen2.5-0.5B-Instruct` (local snapshot on `hal`), `--enforce-eager`, Ready + coherent greedy tokens.
- [x] **`01` canary** — `VLLM_XPU_ENABLE_XPU_GRAPH=1` + `--attention-backend TRITON_ATTN` + `-cc.cudagraph_mode=PIECEWISE`: Ready, piecewise capture OK, decode OK.
- [x] **T7 FA-in-graph probe** — `vllm_xpu_kernels.flash_attn_varlen_func` captured inside `torch.xpu.XPUGraph` and replayed OK with a preallocated `out` tensor. **No `work_group_scratch_memory` Graph error** on oneAPI 2026.0 — the `03` blocker is cleared.
- [x] **FULL graph serve** — `-cc.cudagraph_mode=FULL` + TRITON_ATTN: Ready, full capture OK, decode OK (previously impossible on the 2025.3 base).

## A/B metrics (Arc Pro B70, Qwen2.5-0.5B-Instruct bf16, max_num_seqs=1)

Image `oneapi-2026.0-torch2.13-3deb3160c`; 8 streamed completions of 128 tokens
after warmup (greedy). Identical output text across all arms.

| Arm | TTFT mean | TTFT p50 | Decode tok/s mean | Decode tok/s p50 |
| --- | --- | --- | --- | --- |
| A: eager (`--enforce-eager`, graphs off) | 37.1 ms | 36.9 ms | 67.8 | 67.8 |
| B1: XPU graph PIECEWISE + TRITON_ATTN | 30.7 ms | 30.9 ms | 102.6 (+51%) | 103.3 |
| B2: XPU graph FULL + TRITON_ATTN | 22.8 ms | 22.8 ms | 403.7 (+496%) | 403.7 |

Raw JSON: `hal:/home/kclark/src/oneapi2026-build/ab_summary.json`
(`bench_eager/canary/full.json`, `probe_B.json`, `probe_T7_fa_graph.json`).

The A arm mirrors production Ornith's eager config on the new base; the
production image `kris-fork-577e1a932` was pruned from `hal`'s stores during
the build window, so a cross-image A/B was not possible — the eager-vs-graph
comparison on the new base is the decision metric for this feature (graphs
were hard-blocked on the old base by the `work_group_scratch_memory` error).

---

## Caveats / not done

| Item | Status |
| --- | --- |
| **BMG TP via offline oneCCL 2021.15** | **Broken on this ABI**: 2021.15 `libccl` needs `libsycl.so.8`; toolkit/torch 2026 need `.so.9`. Image uses pip `oneccl==2022.0.0` for load. Revisit when a so.9 BMG-capable CCL exists |
| **Restore Ornith DS** | Still paused. **`hal/vllm-xpu:kris-fork-577e1a932` is no longer present in docker or k3s containerd on `hal`** (pruned for disk space); unpausing now would fail to pull. Rebuild the tag from `577e1a932` (or load from a backup tar) before removing `hal.local/vllm-xpu-paused` |
| **SDPA-under-capture with external events** | `F.scaled_dot_product_attention` inside `capture_begin` without stream warmup hits `Graph nodes cannot depend on events from outside the graph` — a capture-hygiene issue, not the scratch error; the FA varlen path with prealloc `out` captures fine |
| **Production cutover** | Out of scope — do not point Ornith at `oneapi-2026.0-torch2.13-*` |

Image backup: `hal:/tank/vllm-xpu-oneapi-2026.0-torch2.13-3deb3160c.tar` (24 GB).

---

## Rebuild / re-smoke

```bash
git checkout oneapi-2026-torch-xpu-base-image
SHORT=$(git rev-parse --short HEAD)
TAG="hal/vllm-xpu:oneapi-2026.0-torch2.13-${SHORT}"

export DOCKER_BUILDKIT=1
docker build -f docker/Dockerfile.xpu --progress=plain -t "${TAG}" .

docker run --rm --privileged \
  --device /dev/dri:/dev/dri \
  -e LD_LIBRARY_PATH=/opt/venv/lib:/opt/intel/oneapi/compiler/2026.0/lib \
  --entrypoint python "${TAG}" -c '
import importlib.metadata as m, torch
from vllm.utils.torch_utils import supports_xpu_graph
print(torch.__version__, torch.xpu.is_available(),
      m.version("intel-sycl-rt"), m.version("triton-xpu"),
      supports_xpu_graph())
'
```

Expect: `2.13.0+xpu True 2026.0.0 3.7.2 True`.
