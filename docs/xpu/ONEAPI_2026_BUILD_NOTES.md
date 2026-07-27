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

---

## Caveats / not done

| Item | Status |
| --- | --- |
| **Commit/push** of Dockerfile + `requirements/test/xpu.txt` + this note | Local/hal working tree only until explicitly committed |
| **Eager serve smoke** (small dense Ready + tokens) | Not completed. `sshleifer/tiny-gpt2` hits Xe2 FA 16-byte head-dim assert (`head_dim=1`). Cached `gpt2` snapshot on `hal` has tokenizer/config only (no weights). Retry with a full local snapshot (e.g. Qwen2.5-1.5B) or `openai-community/gpt2` with network |
| **`01` canary** (`VLLM_XPU_ENABLE_XPU_GRAPH=1`, TRITON_ATTN, PIECEWISE) | Not run |
| **FA-in-graph / `work_group_scratch` probe** (prep for feature `03`) | Not run; prerequisite `supports_xpu_graph True` is satisfied |
| **BMG TP via offline oneCCL 2021.15** | **Broken on this ABI**: 2021.15 `libccl` needs `libsycl.so.8`; toolkit/torch 2026 need `.so.9`. Image uses pip `oneccl==2022.0.0` for load. Revisit when a so.9 BMG-capable CCL exists |
| **Restore Ornith DS** | Still paused; remove `hal.local/vllm-xpu-paused` when ready to bring production back (still on `kris-fork-577e1a932`) |
| **Production cutover** | Out of scope — do not point Ornith at `oneapi-2026.0-torch2.13-*` |

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
