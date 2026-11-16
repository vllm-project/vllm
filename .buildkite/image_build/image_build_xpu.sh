#!/bin/bash
set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "Usage: $0 <registry> <repo> <commit>"
  exit 1
fi

REGISTRY=$1
REPO=$2
BUILDKITE_COMMIT=$3
IMAGE="$REGISTRY/$REPO:$BUILDKITE_COMMIT-xpu"

# Retry registry errors as in .buildkite/scripts/rocm/build.sh, but not missing images.
image_digest() {
  local output digest attempt
  for attempt in 1 2 3; do
    if output=$(docker buildx imagetools inspect "$1" 2>&1); then
      digest=$(awk '$1 == "Digest:" { print $2; exit }' <<<"$output")
      if [[ $digest =~ ^sha256:[0-9a-f]{64}$ ]]; then
        echo "$digest"
        return 0
      fi
    elif grep -qiE 'manifest unknown|no such manifest|name unknown|manifest for .* not found|: not found' <<<"$output"; then
      return 1
    fi
    echo "Failed to inspect $1 (attempt $attempt/3): $output" >&2
    (( attempt == 3 )) || sleep 5
  done
  return 2
}

# authenticate with AWS ECR
aws ecr-public get-login-password --region us-east-1 | docker login --username AWS --password-stdin "$REGISTRY" || true
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin 936637512419.dkr.ecr.us-east-1.amazonaws.com || true

publish_base=false
if [[ ${BUILDKITE_PULL_REQUEST:-} == false && ${BUILDKITE_BRANCH:-} == "${BUILDKITE_PIPELINE_DEFAULT_BRANCH:-main}" ]]; then
  publish_base=true
fi

build_xpu_image() {
  local image_exists=false base_key base_image base_digest="" metadata_file status
  local -a base_args=()

  if image_digest "$IMAGE" > /dev/null; then
    echo "Image found"
    image_exists=true
  else
    status=$?
    (( status == 1 )) || return "$status"
    echo "Image not found, proceeding with build..."
  fi

  # Main still needs to publish the base if the final image already exists.
  if [[ $image_exists == true && $publish_base == false ]]; then
    return
  fi

  # The shared base is for CI builders with an amd64 default platform.
  base_key=$(
    { sha256sum docker/Dockerfile.xpu_base && printf 'linux/amd64\n'; } |
      sha256sum | cut -c1-32
  )
  base_image="$REGISTRY/$REPO:xpu-base-$base_key"
  if base_digest=$(image_digest "$base_image"); then
    echo "XPU base image found: $base_image@$base_digest"
  else
    status=$?
    (( status == 1 )) || return "$status"
    if [[ $publish_base == true ]]; then
      # Only main publishes a shared base; PRs build it locally on a miss.
      echo "Building and publishing $base_image"
      metadata_file=$(mktemp)
      docker buildx build --file docker/Dockerfile.xpu_base \
        --tag "$base_image" --push --metadata-file "$metadata_file" --progress plain . || {
        status=$?
        rm -f "$metadata_file"
        return "$status"
      }
      base_digest=$(awk -F'"' '$2 == "containerimage.digest" {print $4; exit}' "$metadata_file")
      rm -f "$metadata_file"
      if [[ ! $base_digest =~ ^sha256:[0-9a-f]{64}$ ]]; then
        echo "Cannot resolve the pushed XPU base image digest" >&2
        return 1
      fi
      echo "Published $base_image@$base_digest"
    else
      echo "XPU base image not found; building from the Dockerfile"
    fi
  fi

  if [[ $image_exists == true ]]; then
    return
  fi

  if [[ -n $base_digest ]]; then
    # Pin the base so later tag updates cannot change this build.
    base_args=(--set "vllm-openai.contexts.vllm-base=docker-image://$base_image@$base_digest")
  fi
  docker buildx bake \
    --file docker/docker-bake-xpu.hcl --load \
    --set vllm-openai.args.max_jobs=16 \
    --set "vllm-openai.args.buildkite_commit=$BUILDKITE_COMMIT" \
    --set "vllm-openai.tags=$IMAGE" \
    "${base_args[@]}" \
    --progress plain vllm-openai
  docker push "$IMAGE"
}

build_xpu_image
.buildkite/scripts/annotate-image-build.sh "$IMAGE"
