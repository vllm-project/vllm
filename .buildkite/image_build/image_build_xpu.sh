#!/bin/bash
set -e

if [[ $# -lt 3 ]]; then
  echo "Usage: $0 <registry> <repo> <commit>"
  exit 1
fi

REGISTRY=$1
REPO=$2
BUILDKITE_COMMIT=$3
IMAGE="$REGISTRY/$REPO:$BUILDKITE_COMMIT-xpu"
IMAGE_BUILT=0

if [[ -n "${VLLM_WHEEL_URL:-}" && -z "${VLLM_WHEEL_VERSION:-}" ]]; then
  echo "VLLM_WHEEL_VERSION is required when VLLM_WHEEL_URL is set" >&2
  exit 1
fi

# authenticate with AWS ECR
aws ecr-public get-login-password --region us-east-1 | docker login --username AWS --password-stdin "$REGISTRY" || true
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin 936637512419.dkr.ecr.us-east-1.amazonaws.com || true

# skip build if image already exists
if docker manifest inspect "$IMAGE" &> /dev/null; then
  echo "Image found"
  docker pull "$IMAGE"
else
  echo "Image not found, proceeding with build..."
  # build
  docker build \
    --file docker/Dockerfile.xpu \
    --build-arg max_jobs=16 \
    --build-arg buildkite_commit="$BUILDKITE_COMMIT" \
    --build-arg VLLM_WHEEL_URL="${VLLM_WHEEL_URL:-}" \
    --tag "$IMAGE" \
    --progress plain .
  IMAGE_BUILT=1
fi

VLLM_VERSION=$(docker run --rm --entrypoint /opt/venv/bin/python "$IMAGE" \
  -c 'import importlib.metadata; print(importlib.metadata.version("vllm"))')
if [[ -n "${VLLM_WHEEL_VERSION:-}" && "$VLLM_VERSION" != "$VLLM_WHEEL_VERSION" ]]; then
  echo "Installed vLLM version $VLLM_VERSION does not match expected $VLLM_WHEEL_VERSION" >&2
  exit 1
fi
if [[ "$IMAGE_BUILT" == "1" ]]; then
  docker push "$IMAGE"
fi
WHEEL_COMMIT="${VLLM_WHEEL_COMMIT:-$(python3 -c 'import re, sys; match = re.search(r"\+g([0-9a-f]{7,40})", sys.argv[1]); print(match.group(1) if match else "unknown")' "$VLLM_VERSION")}"

if command -v buildkite-agent >/dev/null 2>&1; then
  if ! printf '### Installed XPU vLLM wheel\n\n- **Version:** \140%s\140\n- **Wheel source commit:** [\140%s\140](https://github.com/vllm-project/vllm/commit/%s)\n- **Image:** \140%s\140\n' \
    "$VLLM_VERSION" "$WHEEL_COMMIT" "$WHEEL_COMMIT" "$IMAGE" \
    | buildkite-agent annotate --append --style info --context xpu-installed-wheel; then
    echo "warning: failed to annotate the installed XPU vLLM wheel"
  fi
fi

.buildkite/scripts/annotate-image-build.sh "$IMAGE"
