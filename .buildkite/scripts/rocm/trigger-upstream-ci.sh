#!/usr/bin/env bash
# Dynamically trigger the normal upstream CI after a ROCm base-image refresh.

set -euo pipefail

metadata_get() {
    local key="$1"
    if command -v buildkite-agent >/dev/null 2>&1; then
        buildkite-agent meta-data get "${key}" 2>/dev/null || true
    fi
}

yaml_escape() {
    local value="$1"
    value="${value//\\/\\\\}"
    value="${value//\"/\\\"}"
    printf '%s' "${value}"
}

main() {
    local refresh=""
    local pipeline="${ROCM_BASE_REFRESH_UPSTREAM_PIPELINE:-ci}"
    local message=""
    local base_image=""
    local ci_base_image=""
    local ci_image=""

    if [[ "${ROCM_BASE_REFRESH_TRIGGER_UPSTREAM:-1}" != "1" ]]; then
        echo "ROCM_BASE_REFRESH_TRIGGER_UPSTREAM is disabled; skipping upstream trigger"
        return 0
    fi

    refresh="$(metadata_get rocm-base-refresh)"
    if [[ "${refresh}" != "1" ]]; then
        echo "No ROCm base refresh happened in this build; skipping upstream trigger"
        return 0
    fi

    if [[ "${ROCM_BASE_REFRESH_SKIP:-0}" == "1" ]]; then
        echo "ROCM_BASE_REFRESH_SKIP=1 set; skipping upstream trigger"
        return 0
    fi

    if ! command -v buildkite-agent >/dev/null 2>&1; then
        echo "buildkite-agent not found; cannot upload upstream trigger"
        return 0
    fi

    base_image="$(metadata_get rocm-base-image)"
    ci_base_image="$(metadata_get rocm-ci-base-image)"
    ci_image="${VLLM_CI_FALLBACK_IMAGE:-rocm/vllm-ci:${BUILDKITE_COMMIT:-}}"
    message="[ROCm base refresh] ${BUILDKITE_MESSAGE:-${BUILDKITE_COMMIT:-manual build}}"
    message="${message//$'\n'/ }"

    echo "--- :buildkite: Triggering upstream CI"
    echo "Pipeline: ${pipeline}"
    echo "Base image: ${base_image}"
    echo "CI base image: ${ci_base_image}"
    echo "CI image: ${ci_image}"

    buildkite-agent pipeline upload <<YAML
steps:
  - trigger: "$(yaml_escape "${pipeline}")"
    label: "Upstream CI for ROCm base refresh"
    build:
      commit: "$(yaml_escape "${BUILDKITE_COMMIT:-}")"
      branch: "$(yaml_escape "${BUILDKITE_BRANCH:-}")"
      message: "$(yaml_escape "${message}")"
      env:
        RUN_ALL: "1"
        ROCM_BASE_REFRESH_SKIP: "1"
        BASE_IMAGE: "$(yaml_escape "${base_image}")"
        CI_BASE_IMAGE: "$(yaml_escape "${ci_base_image}")"
        VLLM_CI_BASE_IMAGE: "$(yaml_escape "${ci_base_image}")"
        ROCM_BASE_IMAGE: "$(yaml_escape "${base_image}")"
        ROCM_CI_BASE_IMAGE: "$(yaml_escape "${ci_base_image}")"
        VLLM_CI_FALLBACK_IMAGE: "$(yaml_escape "${ci_image}")"
        ROCM_BASE_REFRESH_PARENT_BUILD: "$(yaml_escape "${BUILDKITE_BUILD_URL:-}")"
YAML
}

main "$@"
