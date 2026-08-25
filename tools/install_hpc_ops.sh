#!/usr/bin/env bash
# Build and install hpc-ops from a pinned source revision.
set -euo pipefail

HPC_OPS_GIT_REPO="https://gitlab-cn-beijing.siflow.cn/inference/hpc-ops.git"
HPC_OPS_GIT_REF="4e21449375d17d4a883dd1fb40a60e5f596c31b8"
WHEEL_DIR=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --repo)
            if [[ -z "${2:-}" || "$2" =~ ^- ]]; then
                echo "Error: --repo requires an argument." >&2
                exit 1
            fi
            HPC_OPS_GIT_REPO="$2"
            shift 2
            ;;
        --ref)
            if [[ -z "${2:-}" || "$2" =~ ^- ]]; then
                echo "Error: --ref requires an argument." >&2
                exit 1
            fi
            HPC_OPS_GIT_REF="$2"
            shift 2
            ;;
        --wheel-dir)
            if [[ -z "${2:-}" || "$2" =~ ^- ]]; then
                echo "Error: --wheel-dir requires a directory path." >&2
                exit 1
            fi
            WHEEL_DIR="$2"
            shift 2
            ;;
        -h|--help)
            cat <<EOF
Usage: $0 [OPTIONS]
Options:
  --repo URL         hpc-ops Git repository (default: $HPC_OPS_GIT_REPO)
  --ref REF          Git reference to build (default: $HPC_OPS_GIT_REF)
  --wheel-dir PATH   Build a wheel into PATH without installing it
  -h, --help         Show this help message
EOF
            exit 0
            ;;
        *)
            echo "Error: unknown option '$1'." >&2
            exit 1
            ;;
    esac
done

if ! command -v nvcc >/dev/null 2>&1; then
    echo "Error: hpc-ops requires a CUDA toolkit with nvcc." >&2
    exit 1
fi

build_root=$(mktemp -d)
trap 'rm -rf "$build_root"' EXIT

echo "Preparing hpc-ops build"
echo "Repository: $HPC_OPS_GIT_REPO"
echo "Reference: $HPC_OPS_GIT_REF"

git clone --no-checkout "$HPC_OPS_GIT_REPO" "$build_root/hpc-ops"
pushd "$build_root/hpc-ops" >/dev/null
if ! git checkout --detach "$HPC_OPS_GIT_REF"; then
    git fetch --no-tags origin "$HPC_OPS_GIT_REF"
    git checkout --detach FETCH_HEAD
fi
git submodule update --init --recursive

resolved_commit=$(git rev-parse HEAD)
echo "Resolved hpc-ops commit: $resolved_commit"

rm -rf -- build dist ./*.egg-info
torch_version_before=$(python3 -c 'import torch; print(torch.__version__)')
python3 setup.py bdist_wheel

if [[ -n "$WHEEL_DIR" ]]; then
    mkdir -p "$WHEEL_DIR"
    cp dist/*.whl "$WHEEL_DIR"/
    echo "hpc-ops wheel written to $WHEEL_DIR"
    popd >/dev/null
    exit 0
fi

# The active vLLM environment owns the Torch version. hpc-ops must compile
# against it without allowing its unpinned torch dependency to replace it.
if command -v uv >/dev/null 2>&1; then
    if [[ -n "${VLLM_DOCKER_BUILD_CONTEXT:-}" ]]; then
        uv pip install --system --no-deps dist/*.whl
    else
        uv pip install --no-deps dist/*.whl
    fi
else
    python3 -m pip install --no-deps dist/*.whl
fi

popd >/dev/null

HPC_OPS_EXPECTED_COMMIT="$resolved_commit" \
TORCH_VERSION_BEFORE="$torch_version_before" \
python3 - <<'PY'
import json
import os

import hpc
import torch

assert torch.__version__ == os.environ["TORCH_VERSION_BEFORE"], (
    "hpc-ops changed torch: "
    f"{os.environ['TORCH_VERSION_BEFORE']} -> {torch.__version__}"
)
assert hasattr(hpc, "fuse_moe_bf16"), "hpc-ops is missing fuse_moe_bf16"
assert hasattr(hpc, "allocate_fuse_moe_bf16_workspace"), (
    "hpc-ops is missing the BF16 workspace allocator"
)
assert "workspace" in __import__("inspect").signature(hpc.fuse_moe_bf16).parameters, (
    "hpc-ops fuse_moe_bf16 is missing caller-owned workspace support"
)
built = hpc.__built_json__
if isinstance(built, str):
    built = json.loads(built)
built_commit = built["git-hash"]
expected_commit = os.environ["HPC_OPS_EXPECTED_COMMIT"]
assert built_commit == expected_commit[:7], (
    f"hpc-ops built from {built_commit}, expected {expected_commit[:7]}"
)
print("hpc-ops", hpc.__version__, "commit", built_commit, "torch", torch.__version__)
PY

echo "hpc-ops installation completed successfully"
