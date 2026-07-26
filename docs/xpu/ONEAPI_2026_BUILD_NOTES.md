# Build notes — oneAPI 2026.0 + torch 2.13 XPU image

**Branch:** `oneapi-2026-torch-xpu-base-image` (`krisclarkdev/vllm`)  
**Do not** retag or overwrite `hal/vllm-xpu:kris-fork-*` (production Ornith).

```bash
git checkout oneapi-2026-torch-xpu-base-image
SHORT=$(git rev-parse --short HEAD)
TAG="hal/vllm-xpu:oneapi-2026.0-torch2.13-${SHORT}"

export DOCKER_BUILDKIT=1
docker build -f docker/Dockerfile.xpu --progress=plain -t "${TAG}" .

# Smoke (on hal with /dev/dri):
docker run --rm --privileged \
  --device /dev/dri:/dev/dri \
  -v /dev/dri/by-path:/dev/dri/by-path \
  --entrypoint bash "${TAG}" -lc '
source /opt/intel/oneapi/setvars.sh --force 2>/dev/null || true
python -c "
import importlib.metadata as m, torch
from vllm.utils.torch_utils import supports_xpu_graph
print(torch.__version__, torch.xpu.is_available(), m.version(\"intel-sycl-rt\"), m.version(\"triton-xpu\"), supports_xpu_graph())
"'
```

Expect: `torch 2.13.0+xpu`, `intel-sycl-rt 2026.0.0`, `triton-xpu 3.7.2`, `supports_xpu_graph True`.
