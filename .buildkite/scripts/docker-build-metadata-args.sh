#!/bin/bash
# Emit docker build flags for release image provenance metadata.

set -euo pipefail

# Variant examples: "", "cu129", "ubuntu2404", "cu129-ubuntu2404".
variant="${1:-}"
variant_suffix=""
if [[ -n "${variant}" ]]; then
  variant_suffix="-${variant}"
fi

build_kind_default="${VLLM_BUILD_KIND_DEFAULT:-local}"
if [[ "${NIGHTLY:-}" == "1" ]]; then
  build_kind_default="nightly"
elif [[ -n "${BUILDKITE:-}" && "${BUILDKITE_PIPELINE_SLUG:-}" == *release* ]]; then
  build_kind_default="release"
elif [[ -n "${BUILDKITE:-}" && "${build_kind_default}" == "local" ]]; then
  build_kind_default="ci"
fi

image_name="${VLLM_DOCKER_IMAGE_NAME:-vllm/vllm-openai}"
staging_repo="${VLLM_STAGING_IMAGE_REPO:-public.ecr.aws/q9t5s3a7/vllm-release-repo}"
build_commit="${VLLM_BUILD_COMMIT:-${BUILDKITE_COMMIT:-unknown}}"
tag_commit="${BUILDKITE_COMMIT:-${build_commit}}"

if [[ -n "${BUILDKITE:-}" || -n "${BUILDKITE_COMMIT:-}" ]]; then
  release_version="${RELEASE_VERSION:-}"
  if command -v buildkite-agent >/dev/null 2>&1; then
    release_version="${release_version:-$(buildkite-agent meta-data get release-version 2>/dev/null || true)}"
  fi
  release_version="${release_version#v}"
  release_version="${release_version:-${tag_commit}}"

  staging_image_ref="${staging_repo}:${tag_commit}-$(uname -m)${variant_suffix}"

  if [[ "${NIGHTLY:-}" == "1" ]]; then
    if [[ -z "${variant}" ]]; then
      image_ref="${image_name}:nightly-${tag_commit}"
    elif [[ "${variant}" == cu* ]]; then
      cuda_variant="${variant%%-*}"
      remaining_variant="${variant#${cuda_variant}}"
      image_ref="${image_name}:${cuda_variant}-nightly-${tag_commit}${remaining_variant}"
    else
      image_ref="${image_name}:nightly-${tag_commit}${variant_suffix}"
    fi
  else
    image_ref="${image_name}:v${release_version}${variant_suffix}"
  fi
else
  image_ref="${VLLM_IMAGE_REF:-local/vllm-openai:dev}"
  staging_image_ref="${image_ref}"
fi

emit_arg() {
  printf -- "--build-arg %s=%s " "$1" "$2"
}

emit_arg VLLM_BUILD_KIND "${VLLM_BUILD_KIND:-${build_kind_default}}"
emit_arg VLLM_BUILD_REPO "${VLLM_BUILD_REPO:-${BUILDKITE_REPO:-https://github.com/vllm-project/vllm}}"
emit_arg VLLM_BUILD_PIPELINE "${VLLM_BUILD_PIPELINE:-${BUILDKITE_PIPELINE_SLUG:-local}}"
emit_arg VLLM_BUILD_COMMIT "${build_commit}"
emit_arg VLLM_BUILD_URL "${VLLM_BUILD_URL:-${BUILDKITE_BUILD_URL:-}}"
emit_arg VLLM_IMAGE_REF "${image_ref}"
printf -- "--tag %s " "${staging_image_ref}"
