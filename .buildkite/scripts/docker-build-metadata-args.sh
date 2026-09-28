#!/bin/bash
# Emit docker build flags for release image provenance metadata.
# Keep this helper best-effort: missing Buildkite metadata should fall back to
# local/default values instead of blocking the Docker build.

# Variant examples: "", "cu129", "ubuntu2404", "cu129-ubuntu2404".
variant="${1:-}"
variant_suffix="${variant:+-${variant}}"

image_name="${VLLM_DOCKER_IMAGE_NAME:-vllm/vllm-openai}"
staging_repo="${VLLM_STAGING_IMAGE_REPO:-public.ecr.aws/q9t5s3a7/vllm-release-repo}"
build_commit="${VLLM_BUILD_COMMIT:-${BUILDKITE_COMMIT:-unknown}}"
build_pipeline="${VLLM_BUILD_PIPELINE:-${BUILDKITE_PIPELINE_ID:-${BUILDKITE_PIPELINE_SLUG:-local}}}"
build_url="${VLLM_BUILD_URL:-${BUILDKITE_BUILD_URL:-}}"
tag_commit="${BUILDKITE_COMMIT:-${build_commit}}"

if [[ -n "${BUILDKITE:-}" || -n "${BUILDKITE_COMMIT:-}" ]]; then
  release_version="${RELEASE_VERSION:-}"
  if command -v buildkite-agent >/dev/null 2>&1; then
    release_version="${release_version:-$(buildkite-agent meta-data get release-version 2>/dev/null)}"
  fi
  release_version="${release_version#v}"
  release_version="${release_version:-${tag_commit}}"

  staging_image_ref="${staging_repo}:${tag_commit}-$(uname -m)${variant_suffix}"

  if [[ "${NIGHTLY:-}" == "1" ]]; then
    if [[ -z "${variant}" ]]; then
      image_tag="${image_name}:nightly-${tag_commit}"
    elif [[ "${variant}" == cu* ]]; then
      cuda_variant="${variant%%-*}"
      remaining_variant="${variant#${cuda_variant}}"
      image_tag="${image_name}:${cuda_variant}-nightly-${tag_commit}${remaining_variant}"
    else
      image_tag="${image_name}:nightly-${tag_commit}${variant_suffix}"
    fi
  else
    image_tag="${image_name}:v${release_version}${variant_suffix}"
  fi
else
  image_tag="${VLLM_IMAGE_TAG:-local/vllm-openai:dev}"
  staging_image_ref="${image_tag}"
fi

emit_arg() {
  printf -- "--build-arg %s=%s " "$1" "$2"
}

emit_arg VLLM_BUILD_COMMIT "${build_commit}"
emit_arg VLLM_BUILD_PIPELINE "${build_pipeline}"
emit_arg VLLM_BUILD_URL "${build_url}"
# This is the intended public tag. The final digest is only known after push.
emit_arg VLLM_IMAGE_TAG "${image_tag}"
printf -- "--tag %s " "${staging_image_ref}"
