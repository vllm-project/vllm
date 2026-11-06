#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Publish a release Docker image family from ECR to DockerHub.
# Pulls per-arch images, tags with latest and versioned tags, pushes them,
# then creates and pushes multi-arch manifests.

set -euo pipefail

TARGET="${1:-all}"
case "${TARGET}" in
  cuda-13-0 | cuda-13-0-efa | rocm | xpu | cpu | all) ;;
  *)
    echo "Usage: $0 {cuda-13-0|cuda-13-0-efa|rocm|xpu|cpu|all}"
    exit 2
    ;;
esac

target_enabled() {
  [ "${TARGET}" = "all" ] || [ "${TARGET}" = "$1" ]
}

# Promote one image family from ECR staging to DockerHub: per-arch tags plus the
# latest and versioned multi-arch manifests. $1 is the tag suffix shared by both
# registries: "" for the default CUDA family, else e.g. "-efa".
publish_image_family() {
  local suffix="$1" arch
  local staging="public.ecr.aws/q9t5s3a7/vllm-release-repo"
  local latest="vllm/vllm-openai:latest${suffix}"
  local versioned="vllm/vllm-openai:v${RELEASE_VERSION}${suffix}"

  for arch in x86_64 aarch64; do
    docker pull "${staging}:${COMMIT}-${arch}${suffix}"
  done

  for arch in x86_64 aarch64; do
    docker tag "${staging}:${COMMIT}-${arch}${suffix}" "vllm/vllm-openai:latest-${arch}${suffix}"
    docker tag "${staging}:${COMMIT}-${arch}${suffix}" "vllm/vllm-openai:v${RELEASE_VERSION}-${arch}${suffix}"
    docker push "vllm/vllm-openai:latest-${arch}${suffix}"
    docker push "vllm/vllm-openai:v${RELEASE_VERSION}-${arch}${suffix}"
  done

  docker manifest rm "${latest}" || true
  docker manifest rm "${versioned}" || true
  docker manifest create "${latest}" \
    "vllm/vllm-openai:latest-x86_64${suffix}" \
    "vllm/vllm-openai:latest-aarch64${suffix}"
  docker manifest create "${versioned}" \
    "vllm/vllm-openai:v${RELEASE_VERSION}-x86_64${suffix}" \
    "vllm/vllm-openai:v${RELEASE_VERSION}-aarch64${suffix}"
  docker manifest push "${latest}"
  docker manifest push "${versioned}"
}

# Publish an opt-in family all-or-nothing, like the CPU images below: skip when
# neither arch was built, fail on a partial build so it is not silenced.
publish_optional_family() {
  local suffix="$1" arch missing=() present=()
  local staging="public.ecr.aws/q9t5s3a7/vllm-release-repo"

  for arch in x86_64 aarch64; do
    if docker manifest inspect "${staging}:${COMMIT}-${arch}${suffix}" >/dev/null 2>&1; then
      present+=("${arch}")
    else
      missing+=("${arch}")
    fi
  done

  if [ "${#present[@]}" -eq 0 ]; then
    echo "WARNING: no ${suffix} images found in ECR for ${COMMIT}, skipping the ${suffix} publish"
  elif [ "${#missing[@]}" -eq 0 ]; then
    publish_image_family "${suffix}"
  else
    echo "ERROR: partial ${suffix} build (present: ${present[*]}; missing: ${missing[*]})." >&2
    echo "       Refusing to publish to avoid per-arch and multi-arch tags disagreeing." >&2
    exit 1
  fi
}

RELEASE_VERSION=$(buildkite-agent meta-data get release-version --default "" | sed 's/^v//')
if [ -z "${RELEASE_VERSION}" ]; then
  echo "ERROR: release-version metadata not set"
  exit 1
fi

COMMIT="$BUILDKITE_COMMIT"

echo "========================================"
echo "Publishing ${TARGET} release images v${RELEASE_VERSION}"
echo "  Commit: ${COMMIT}"
echo "========================================"

# Login to ECR to pull staging images
aws ecr-public get-login-password --region us-east-1 | \
  docker login --username AWS --password-stdin public.ecr.aws/q9t5s3a7

# ---- CUDA (default: 13.0) ----

if target_enabled cuda-13-0; then
  publish_image_family ""

  ZSTD_DIGEST=$(docker buildx imagetools inspect \
    "public.ecr.aws/q9t5s3a7/vllm-release-repo:${COMMIT}-x86_64" \
    --format '{{json .Manifest.Digest}}' | tr -d '"')
  .buildkite/scripts/publish-zstd-image.sh \
    "public.ecr.aws/q9t5s3a7/vllm-release-repo@${ZSTD_DIGEST}" \
    "vllm/vllm-openai:v${RELEASE_VERSION}-x86_64-zstd"
fi

# ---- AWS EFA variant (docker/Dockerfile.efa) ----
# No cu134 entry: Rubin CUDA 13.4 is nightly-only and is not promoted to
# DockerHub for the base variant either.

if target_enabled cuda-13-0-efa; then
  publish_optional_family "-efa"
fi

# ---- ROCm ----
# Copied from ECR (publish.sh). The default stack (rocm) also owns the plain :latest / :v<ver> tags.

# pin_rocm <ref>  — <ref>@<digest>, so what was checked is what is published
pin_rocm() {
  local d
  d=$(docker buildx imagetools inspect "$1" | awk '$1 == "Digest:" { print $2; exit }')
  [[ $d =~ ^sha256:[0-9a-f]{64}$ ]] || { echo "cannot resolve the digest of $1" >&2; return 1; }
  echo "$1@$d"
}

publish_rocm_stack() {
  local v s sfx img base ver ecr=public.ecr.aws/q9t5s3a7/vllm-release-repo:${COMMIT} hub=vllm/vllm-openai-rocm
  v=$(buildkite-agent meta-data get "rocm-$1-variant")
  img=$(pin_rocm "$ecr-$v")
  base=$(pin_rocm "$ecr-$v-base")
  # Another build of this commit (e.g. a branch push) may have retagged it with its own version.
  ver=$(docker buildx imagetools inspect "$img" \
    --format '{{index .Image.Config.Labels "org.opencontainers.image.version"}}')
  if [[ ${ver%%+*} != "$RELEASE_VERSION" ]]; then
    echo "ERROR: $img is version '$ver', not $RELEASE_VERSION" >&2
    exit 1
  fi
  sfx=("-$v"); if [[ $1 == rocm ]]; then sfx+=(""); fi
  for s in "${sfx[@]}"; do
    docker buildx imagetools create -t "$hub:latest$s" -t "$hub:v${RELEASE_VERSION}$s" "$img"
    docker buildx imagetools create -t "$hub:latest$s-base" -t "$hub:v${RELEASE_VERSION}$s-base" "$base"
  done
}

if target_enabled rocm; then
  publish_rocm_stack rocm
fi

# ---- XPU ----

if target_enabled xpu; then
  docker pull "public.ecr.aws/q9t5s3a7/vllm-release-repo:${COMMIT}-x86_64-xpu"

  docker tag "public.ecr.aws/q9t5s3a7/vllm-release-repo:${COMMIT}-x86_64-xpu" vllm/vllm-openai-xpu:latest-x86_64
  docker tag "public.ecr.aws/q9t5s3a7/vllm-release-repo:${COMMIT}-x86_64-xpu" "vllm/vllm-openai-xpu:v${RELEASE_VERSION}-x86_64"
  docker push vllm/vllm-openai-xpu:latest-x86_64
  docker push "vllm/vllm-openai-xpu:v${RELEASE_VERSION}-x86_64"

  docker manifest rm vllm/vllm-openai-xpu:latest || true
  docker manifest rm "vllm/vllm-openai-xpu:v${RELEASE_VERSION}" || true
  docker manifest create vllm/vllm-openai-xpu:latest vllm/vllm-openai-xpu:latest-x86_64 --amend
  docker manifest create "vllm/vllm-openai-xpu:v${RELEASE_VERSION}" "vllm/vllm-openai-xpu:v${RELEASE_VERSION}-x86_64" --amend
  docker manifest push vllm/vllm-openai-xpu:latest
  docker manifest push "vllm/vllm-openai-xpu:v${RELEASE_VERSION}"
fi

# ---- CPU ----
# CPU images are behind separate block steps and may not have been built.
# All-or-nothing: inspect both arches first, then either publish everything
# (per-arch + multi-arch manifest) or skip everything. Publishing only one
# arch would leave `:latest-x86_64` pointing at the new release while the
# `:latest` multi-arch manifest still resolves to the previous release.

if target_enabled cpu; then
  CPU_X86_TAG=public.ecr.aws/q9t5s3a7/vllm-cpu-release-repo:${COMMIT}-x86_64
  CPU_ARM_TAG=public.ecr.aws/q9t5s3a7/vllm-arm64-cpu-release-repo:${COMMIT}-arm64

  CPU_X86_AVAILABLE=false
  CPU_ARM_AVAILABLE=false
  docker manifest inspect "${CPU_X86_TAG}" >/dev/null 2>&1 && CPU_X86_AVAILABLE=true
  docker manifest inspect "${CPU_ARM_TAG}" >/dev/null 2>&1 && CPU_ARM_AVAILABLE=true

  if [ "$CPU_X86_AVAILABLE" = "true" ] && [ "$CPU_ARM_AVAILABLE" = "true" ]; then
    docker pull "${CPU_X86_TAG}"
    docker tag "${CPU_X86_TAG}" vllm/vllm-openai-cpu:latest-x86_64
    docker tag "${CPU_X86_TAG}" "vllm/vllm-openai-cpu:v${RELEASE_VERSION}-x86_64"
    docker push vllm/vllm-openai-cpu:latest-x86_64
    docker push "vllm/vllm-openai-cpu:v${RELEASE_VERSION}-x86_64"

    docker pull "${CPU_ARM_TAG}"
    docker tag "${CPU_ARM_TAG}" vllm/vllm-openai-cpu:latest-arm64
    docker tag "${CPU_ARM_TAG}" "vllm/vllm-openai-cpu:v${RELEASE_VERSION}-arm64"
    docker push vllm/vllm-openai-cpu:latest-arm64
    docker push "vllm/vllm-openai-cpu:v${RELEASE_VERSION}-arm64"

    docker manifest rm vllm/vllm-openai-cpu:latest || true
    docker manifest rm "vllm/vllm-openai-cpu:v${RELEASE_VERSION}" || true
    docker manifest create vllm/vllm-openai-cpu:latest vllm/vllm-openai-cpu:latest-x86_64 vllm/vllm-openai-cpu:latest-arm64
    docker manifest create "vllm/vllm-openai-cpu:v${RELEASE_VERSION}" "vllm/vllm-openai-cpu:v${RELEASE_VERSION}-x86_64" "vllm/vllm-openai-cpu:v${RELEASE_VERSION}-arm64"
    docker manifest push vllm/vllm-openai-cpu:latest
    docker manifest push "vllm/vllm-openai-cpu:v${RELEASE_VERSION}"
  elif [ "$CPU_X86_AVAILABLE" = "false" ] && [ "$CPU_ARM_AVAILABLE" = "false" ]; then
    echo "WARNING: Neither CPU image found in ECR, skipping CPU publish (ensure block-cpu-release-image-build and block-arm64-cpu-release-image-build were unblocked and the builds finished pushing)"
  else
    # Partial state: one arch built, the other did not. Fail loudly rather than
    # ship a Docker Hub state where `:latest-${arch}` and `:latest` (multi-arch)
    # disagree on which release they point at.
    echo "ERROR: Partial CPU build detected (x86_64=${CPU_X86_AVAILABLE}, arm64=${CPU_ARM_AVAILABLE})."
    echo "       Refusing to publish to avoid split-tag drift between per-arch and multi-arch tags."
    echo "       Re-run the missing CPU build and retry, or manually publish if a single-arch release is intended."
    exit 1
  fi
fi

echo ""
echo "Successfully published ${TARGET} release images for v${RELEASE_VERSION}"
