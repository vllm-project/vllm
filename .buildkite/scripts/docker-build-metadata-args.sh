#!/bin/bash
# Emit docker build flags for image provenance metadata.

set -euo pipefail

image_name="${VLLM_DOCKER_IMAGE_NAME:-vllm/vllm-openai}"
staging_repo="${VLLM_STAGING_IMAGE_REPO:-public.ecr.aws/q9t5s3a7/vllm-release-repo}"

mode="build-args"
variant=""
explicit_image_ref=""
explicit_staging_ref=""

case "${1:-}" in
  --build-flags)
    mode="build-flags"
    variant="${2:-}"
    ;;
  --image-ref)
    mode="image-ref"
    variant="${2:-}"
    ;;
  --staging-ref)
    mode="staging-ref"
    variant="${2:-}"
    ;;
  *)
    explicit_image_ref="${1:-}"
    explicit_staging_ref="${2:-}"
    ;;
esac

build_kind_default="${VLLM_BUILD_KIND_DEFAULT:-local}"
if [[ "${NIGHTLY:-}" == "1" ]]; then
  build_kind_default="nightly"
elif [[ -n "${BUILDKITE:-}" && "${BUILDKITE_PIPELINE_SLUG:-}" == *release* ]]; then
  build_kind_default="release"
elif [[ -n "${BUILDKITE:-}" && "${build_kind_default}" == "local" ]]; then
  build_kind_default="ci"
fi

release_version="${RELEASE_VERSION:-}"
if command -v buildkite-agent >/dev/null 2>&1; then
  release_version="${release_version:-$(buildkite-agent meta-data get release-version 2>/dev/null || true)}"
fi
release_version="${release_version#v}"
release_version="${release_version:-${BUILDKITE_COMMIT:-unknown}}"

commit="${VLLM_BUILD_COMMIT:-${BUILDKITE_COMMIT:-unknown}}"
tag_commit="${BUILDKITE_COMMIT:-${commit}}"
variant_suffix=""
if [[ -n "${variant}" ]]; then
  variant_suffix="-${variant}"
fi

if [[ -n "${explicit_image_ref}" || -n "${explicit_staging_ref}" ]]; then
  image_ref="${explicit_image_ref:-local/vllm-openai:dev}"
  staging_image_ref="${explicit_staging_ref:-}"
elif [[ -n "${BUILDKITE:-}" || -n "${BUILDKITE_COMMIT:-}" || -n "${variant}" ]]; then
  staging_image_ref="${staging_repo}:${tag_commit}-$(uname -m)${variant_suffix}"
  if [[ "${NIGHTLY:-}" == "1" ]]; then
    nightly_suffix="${variant_suffix}"
    nightly_prefix=""
    if [[ "${variant}" == cu* ]]; then
      nightly_prefix="${variant%%-*}-"
      nightly_suffix="${variant#${variant%%-*}}"
    fi
    image_ref="${image_name}:${nightly_prefix}nightly-${tag_commit}${nightly_suffix}"
  else
    image_ref="${image_name}:v${release_version}${variant_suffix}"
  fi
else
  image_ref="local/vllm-openai:dev"
  staging_image_ref="${image_ref}"
fi

emit_arg() {
  printf -- "--build-arg %s=%s " "$1" "$2"
}

emit_build_args() {
  emit_arg VLLM_BUILD_KIND "${VLLM_BUILD_KIND:-${build_kind_default}}"
  emit_arg VLLM_BUILD_REPO "${VLLM_BUILD_REPO:-${BUILDKITE_REPO:-https://github.com/vllm-project/vllm}}"
  emit_arg VLLM_BUILD_PIPELINE "${VLLM_BUILD_PIPELINE:-${BUILDKITE_PIPELINE_SLUG:-local}}"
  emit_arg VLLM_BUILD_COMMIT "${commit}"
  emit_arg VLLM_BUILD_URL "${VLLM_BUILD_URL:-${BUILDKITE_BUILD_URL:-}}"
  emit_arg VLLM_IMAGE_REF "${image_ref}"
  emit_arg VLLM_STAGING_IMAGE_REF "${staging_image_ref}"
}

case "${mode}" in
  build-args)
    emit_build_args
    ;;
  build-flags)
    emit_build_args
    printf -- "--tag %s " "${staging_image_ref}"
    ;;
  image-ref)
    printf "%s\n" "${image_ref}"
    ;;
  staging-ref)
    printf "%s\n" "${staging_image_ref}"
    ;;
esac
