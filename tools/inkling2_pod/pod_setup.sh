#!/usr/bin/env bash
# Runs ON the pod. Clones the effort branch, installs vLLM (precompiled
# kernels) and downloads Inkling-2-Mini. Driven by pod_up.sh; env:
#   BRANCH        git branch of suyashpatel98/vllm to install
#   WHEEL_COMMIT  upstream commit whose precompiled wheel to use (40 hex)
#   WHEEL_VARIANT precompiled wheel CUDA variant (e.g. cu132)
set -euo pipefail

: "${BRANCH:?}" "${WHEEL_COMMIT:?}" "${WHEEL_VARIANT:?}"
MODEL=thinkingmachines/Inkling-2-Mini
REPO_URL=https://github.com/suyashpatel98/vllm.git
WORK=/root/work
export HF_HOME=/root/hf
export PATH="$HOME/.local/bin:$PATH"
mkdir -p "$WORK" "$HF_HOME"
cd "$WORK"

t0=$(date +%s)
stamp() { echo "[setup +$(( $(date +%s) - t0 ))s] $*"; }

nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
nvidia-smi | grep -o 'CUDA Version: [0-9.]*'

stamp "installing uv"
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null

# The download is network-bound and independent of the install, so overlap them.
stamp "starting model download in background"
(
  HF_XET_HIGH_PERFORMANCE=1 uvx --from 'huggingface_hub[hf_xet]' \
    hf download "$MODEL" >"$WORK/download.log" 2>&1 \
    && echo ok >"$WORK/download.status" \
    || echo fail >"$WORK/download.status"
) &
dl_pid=$!

stamp "cloning $BRANCH"
rm -rf vllm
git clone -q --depth 1 --branch "$BRANCH" "$REPO_URL" vllm
cd vllm

stamp "creating venv"
uv venv -q --python 3.14 .venv
# shellcheck disable=SC1091
source .venv/bin/activate

stamp "installing vllm (precompiled $WHEEL_VARIANT @ ${WHEEL_COMMIT:0:10})"
VLLM_USE_PRECOMPILED=1 \
VLLM_PRECOMPILED_WHEEL_COMMIT="$WHEEL_COMMIT" \
VLLM_PRECOMPILED_WHEEL_VARIANT="$WHEEL_VARIANT" \
  uv pip install -q -e . --torch-backend="$WHEEL_VARIANT"

stamp "checking install"
python - <<'EOF'
import torch, vllm
print("vllm", vllm.__version__, "| torch", torch.__version__, "cuda", torch.version.cuda)
print(torch.cuda.get_device_name(0), "SM%d%d" % torch.cuda.get_device_capability(0))
torch.ones(1, device="cuda").add_(1)
EOF

stamp "waiting for model download"
wait "$dl_pid" || true
if [[ "$(cat "$WORK/download.status")" != ok ]]; then
  tail -20 "$WORK/download.log"
  exit 1
fi
du -sh "$HF_HOME/hub"

stamp "done. Activate with: cd $WORK/vllm && source .venv/bin/activate && export HF_HOME=$HF_HOME"
