#!/bin/bash
# Emit docker build args for image provenance metadata.

set -euo pipefail

image_ref="${1:-local/vllm-openai:dev}"
staging_image_ref="${2:-}"

build_kind_default="${VLLM_BUILD_KIND_DEFAULT:-local}"
if [[ "${NIGHTLY:-}" == "1" ]]; then
  build_kind_default="nightly"
elif [[ -n "${BUILDKITE:-}" && "${build_kind_default}" == "local" ]]; then
  build_kind_default="ci"
fi

emit_arg() {
  printf -- "--build-arg %s=%s " "$1" "$2"
}

emit_arg VLLM_BUILD_KIND "${VLLM_BUILD_KIND:-${build_kind_default}}"
emit_arg VLLM_BUILD_REPO "${VLLM_BUILD_REPO:-${BUILDKITE_REPO:-https://github.com/vllm-project/vllm}}"
emit_arg VLLM_BUILD_PIPELINE "${VLLM_BUILD_PIPELINE:-${BUILDKITE_PIPELINE_SLUG:-local}}"
emit_arg VLLM_BUILD_COMMIT "${VLLM_BUILD_COMMIT:-${BUILDKITE_COMMIT:-unknown}}"
emit_arg VLLM_BUILD_URL "${VLLM_BUILD_URL:-${BUILDKITE_BUILD_URL:-}}"
emit_arg VLLM_IMAGE_REF "${image_ref}"
emit_arg VLLM_STAGING_IMAGE_REF "${staging_image_ref}"
