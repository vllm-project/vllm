#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
set -euo pipefail

repo_root="$(git rev-parse --show-toplevel)"
output_dir="${repo_root}/build/xpu-kernel-wheel"
kernel_repo="${VLLM_XPU_KERNEL_REPO:-https://github.com/vllm-project/vllm-xpu-kernels.git}"
kernel_ref="${VLLM_XPU_KERNEL_REF:-main}"
build_image="${VLLM_XPU_KERNEL_BUILD_IMAGE:-}"
oneapi_vars="${VLLM_XPU_KERNEL_ONEAPI_VARS:-/opt/intel/oneapi/2026.1/oneapi-vars.sh}"

if [[ "${VLLM_XPU_KERNEL_NIGHTLY:-0}" != "1" ]]; then
    echo "build_xpu_kernel.sh requires VLLM_XPU_KERNEL_NIGHTLY=1" >&2
    exit 1
fi

if ! command -v buildkite-agent >/dev/null; then
    echo "Kernel nightly requires buildkite-agent to save wheel artifacts" >&2
    exit 1
fi
kernel_source="$(mktemp -d)"
trap 'rm -rf -- "$kernel_source"' EXIT

# Do not let a retry pick up wheels from an earlier attempt.
mkdir -p "$output_dir"
find "$output_dir" -maxdepth 1 -type f -delete

git clone "$kernel_repo" "$kernel_source"
git -C "$kernel_source" fetch origin "$kernel_ref"
git -C "$kernel_source" checkout --detach FETCH_HEAD
git -C "$kernel_source" submodule update --init --recursive
kernel_sha="$(git -C "$kernel_source" rev-parse HEAD)"
printf '%s\n' "$kernel_sha" > "$output_dir/kernel-commit.txt"

if [[ -z "$build_image" ]]; then
    export PATH="$HOME/.local/bin:$PATH"
    if ! command -v uv >/dev/null; then
        export UV_INSTALL_DIR="$HOME/.local/bin"
        curl --fail --show-error --silent --location --retry 3 \
            https://astral.sh/uv/0.11.1/install.sh | sh
    fi
    workflow="${kernel_source}/.github/workflows/wheel-per-commit.yaml"
    build_image="$(uv run --isolated --no-project --python 3.12 --with pyyaml==6.0.3 python - "$workflow" <<'PY'
import re
import sys
from pathlib import Path

import yaml

path = Path(sys.argv[1])
workflow = yaml.safe_load(path.read_text())
if not isinstance(workflow, dict):
    sys.exit(f"{path}: expected a workflow mapping")
jobs = workflow.get("jobs")
if not isinstance(jobs, dict):
    sys.exit(f"{path}: missing jobs mapping")
job = jobs.get("build-wheel-on-pvc")
if not isinstance(job, dict):
    sys.exit(f"{path}: missing build-wheel-on-pvc job")
container = job.get("container")
image = container.get("image") if isinstance(container, dict) else container
if not isinstance(image, str) or not re.fullmatch(
    r"[a-zA-Z0-9][a-zA-Z0-9._/:@-]*", image
):
    sys.exit(
        f"{path}: expected a literal container image in build-wheel-on-pvc; "
        "set VLLM_XPU_KERNEL_BUILD_IMAGE explicitly if the workflow uses expressions"
    )
print(image)
PY
)"
fi
echo "Kernel build image: ${build_image}"

echo "--- Building XPU kernel wheel at ${kernel_sha}"
docker run --rm --network host \
    --user "$(id -u):$(id -g)" \
    --volume "${kernel_source}:/workspace/kernel" \
    --volume "${output_dir}:/output" \
    --workdir /workspace/kernel \
    --env "ONEAPI_VARS=${oneapi_vars}" \
    --env "MAX_JOBS=${MAX_JOBS:-4}" \
    --env HOME=/tmp/kernel-home \
    --env http_proxy --env https_proxy --env no_proxy \
    --env HTTP_PROXY --env HTTPS_PROXY --env NO_PROXY \
    --entrypoint /bin/bash \
    "$build_image" -c '
        set -eo pipefail
        source "$ONEAPI_VARS"
        mkdir -p "$HOME"
        git config --global --add safe.directory /workspace/kernel
        export UV_INSTALL_DIR="$HOME/.local/bin"
        curl -LsSf https://astral.sh/uv/0.11.1/install.sh | sh
        export PATH="$HOME/.local/bin:$PATH"
        uv venv --python 3.12 --seed .venv
        uv pip install --python .venv/bin/python \
            --index-strategy unsafe-best-match -r requirements.txt
        VLLM_CHUNK_PREFILL_CONFIG=chunk_prefill_default.conf \
        VLLM_PAGED_DECODE_CONFIG=paged_decode_default.conf \
        .venv/bin/python setup.py bdist_wheel \
            --dist-dir=/output --py-limited-api=cp38
        uv pip freeze --python .venv/bin/python > /output/build-environment.txt
    '

shopt -s nullglob
wheels=("$output_dir"/vllm_xpu_kernels-*.whl)
if [[ ${#wheels[@]} -ne 1 ]]; then
    echo "Expected exactly one kernel wheel, found ${#wheels[@]}" >&2
    exit 1
fi

cat > "$output_dir/build-info.txt" <<EOF
kernel_repository=${kernel_repo}
kernel_ref=${kernel_ref}
kernel_commit=${kernel_sha}
vllm_commit=${BUILDKITE_COMMIT}
build_image=${build_image}
oneapi_vars=${oneapi_vars}
built_at=$(date -u +%FT%TZ)
build_url=${BUILDKITE_BUILD_URL}
job_id=${BUILDKITE_JOB_ID}
EOF
(
    cd "$output_dir"
    sha256sum ./*.whl > SHA256SUMS
)

# Upload before building vLLM so the wheel survives an image/test failure.
cd "$repo_root"
buildkite-agent artifact upload "build/xpu-kernel-wheel/*"
