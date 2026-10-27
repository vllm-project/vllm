#!/bin/bash
set -eo pipefail

if [[ $# -lt 3 ]]; then
  echo "Usage: $0 <registry> <repo> <commit>"
  exit 1
fi

REGISTRY=$1
REPO=$2
BUILDKITE_COMMIT=$3
IMAGE="$REGISTRY/$REPO:$BUILDKITE_COMMIT-xpu"
nightly="${VLLM_XPU_KERNEL_NIGHTLY:-0}"
kernel_build_args=()
case "$nightly" in
  0|1) ;;
  *)
    echo "VLLM_XPU_KERNEL_NIGHTLY must be 0 or 1, got: $nightly" >&2
    exit 1
    ;;
esac

if [[ "$nightly" == "1" ]]; then
  : "${BUILDKITE_JOB_ID:?Kernel nightly requires a Buildkite job}"
  # Invalidate a previous successful attempt before retrying this step.
  buildkite-agent meta-data set xpu-kernel-nightly-image ""
  bash .buildkite/image_build/build_xpu_kernel.sh

  kernel_dir="build/xpu-kernel-wheel"
  (cd "$kernel_dir" && sha256sum -c SHA256SUMS)
  shopt -s nullglob
  wheels=("$kernel_dir"/vllm_xpu_kernels-*.whl)
  if [[ ${#wheels[@]} -ne 1 ]]; then
    echo "Expected exactly one kernel wheel, found ${#wheels[@]}" >&2
    exit 1
  fi
  wheel_name="$(basename "${wheels[0]}")"
  kernel_sha="$(cat "$kernel_dir/kernel-commit.txt")"
  IMAGE="$REGISTRY/$REPO:${BUILDKITE_COMMIT}-kernel-${kernel_sha:0:12}-${BUILDKITE_JOB_ID}-xpu"
  kernel_build_args=(--build-context "kernel-wheel=${kernel_dir}")

  requirements_backup="$(mktemp)"
  cp requirements/xpu.txt "$requirements_backup"
  trap 'cp "$requirements_backup" requirements/xpu.txt; rm -f "$requirements_backup"' EXIT
  if [[ "$(grep -Ec '^vllm[-_]xpu[-_]kernels==' requirements/xpu.txt)" != "1" ]]; then
    echo "Expected exactly one vllm-xpu-kernels pin in requirements/xpu.txt" >&2
    exit 1
  fi
  sed -i -E \
    "s|^vllm[-_]xpu[-_]kernels==.*$|vllm-xpu-kernels @ file:///opt/kernel-wheels/${wheel_name}|" \
    requirements/xpu.txt
fi

# authenticate with AWS ECR
aws ecr-public get-login-password --region us-east-1 | docker login --username AWS --password-stdin "$REGISTRY" || true
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin 936637512419.dkr.ecr.us-east-1.amazonaws.com || true

# skip build if image already exists
if [[ "$nightly" == "0" ]] && docker manifest inspect "$IMAGE" &> /dev/null; then
  echo "Image found"
else
  echo "Image not found, proceeding with build..."
  # build
  docker build \
    "${kernel_build_args[@]}" \
    --file docker/Dockerfile.xpu \
    --build-arg max_jobs=16 \
    --build-arg buildkite_commit="$BUILDKITE_COMMIT" \
    --tag "$IMAGE" \
    --progress plain .
  if [[ "$nightly" == "1" ]]; then
    kernel_version="${wheel_name#vllm_xpu_kernels-}"
    kernel_version="${kernel_version%%-*}"
    docker run --rm --env "EXPECTED_KERNEL_VERSION=${kernel_version}" \
      --entrypoint /bin/bash "$IMAGE" -c '
        set -eo pipefail
        uv pip show vllm-xpu-kernels | grep -Fx "Version: $EXPECTED_KERNEL_VERSION"
        /opt/venv/bin/python -c "import torch; import vllm_xpu_kernels._C"
      '
  fi
  # push
  docker push "$IMAGE"
fi

if [[ "$nightly" == "1" ]]; then
  buildkite-agent meta-data set xpu-kernel-nightly-image "$IMAGE"
fi

.buildkite/scripts/annotate-image-build.sh "$IMAGE"
