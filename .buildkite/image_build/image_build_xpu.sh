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
kernel_nightly="${VLLM_XPU_KERNEL_NIGHTLY:-0}"
case "$kernel_nightly" in
  0|1) ;;
  *)
    echo "VLLM_XPU_KERNEL_NIGHTLY must be 0 or 1, got: $kernel_nightly" >&2
    exit 1
    ;;
esac

if [[ "$kernel_nightly" == "1" ]]; then
  : "${BUILDKITE_JOB_ID:?Kernel nightly requires a Buildkite job}"
  # Invalidate a previous successful attempt before retrying this step.
  buildkite-agent meta-data set xpu-kernel-nightly-image pending
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
fi

# authenticate with AWS ECR
aws ecr-public get-login-password --region us-east-1 | docker login --username AWS --password-stdin "$REGISTRY" || true
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin 936637512419.dkr.ecr.us-east-1.amazonaws.com || true

# skip build if image already exists
if [[ "$kernel_nightly" == "0" ]] && docker manifest inspect "$IMAGE" &> /dev/null; then
  echo "Image found"
else
  echo "Image not found, proceeding with build..."
  build_image="$IMAGE"
  if [[ "$kernel_nightly" == "1" ]]; then
    build_image="${IMAGE}-base"
  fi

  # build
  docker build \
    --file docker/Dockerfile.xpu \
    --build-arg max_jobs=16 \
    --build-arg buildkite_commit="$BUILDKITE_COMMIT" \
    --tag "$build_image" \
    --progress plain .

  if [[ "$kernel_nightly" == "1" ]]; then
    kernel_version="${wheel_name#vllm_xpu_kernels-}"
    kernel_version="${kernel_version%%-*}"

    kernel_container="vllm-kernel-nightly-${BUILDKITE_JOB_ID}"
    cleanup_kernel_container() {
      docker rm -f "$kernel_container" >/dev/null 2>&1 || true
    }
    trap cleanup_kernel_container EXIT
    docker create \
      --name "$kernel_container" \
      --env "KERNEL_WHEEL=/tmp/${wheel_name}" \
      --entrypoint /bin/bash \
      "$build_image" \
      -c 'set -eo pipefail
        uv pip install --python /opt/venv/bin/python \
          --reinstall --no-deps "$KERNEL_WHEEL"
      '
    docker cp "${wheels[0]}" "${kernel_container}:/tmp/${wheel_name}"
    docker start --attach "$kernel_container"
    docker commit "$kernel_container" "$IMAGE"
    docker rm "$kernel_container"
    docker image rm "$build_image"

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

if [[ "$kernel_nightly" == "1" ]]; then
  buildkite-agent meta-data set xpu-kernel-nightly-image "$IMAGE"
fi

.buildkite/scripts/annotate-image-build.sh "$IMAGE"
