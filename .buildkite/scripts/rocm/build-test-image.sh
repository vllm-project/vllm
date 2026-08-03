#!/usr/bin/env bash
# Build the ROCm CI image or wheel artifact from the immutable ci_base selected
# by the preceding pipeline step.

set -euo pipefail

metadata_get() {
    local key="$1"
    if command -v buildkite-agent >/dev/null 2>&1; then
        buildkite-agent meta-data get "${key}" 2>/dev/null || true
    fi
}

metadata_set() {
    local key="$1"
    local value="$2"

    if command -v buildkite-agent >/dev/null 2>&1; then
        buildkite-agent meta-data set "${key}" "${value}"
    elif [[ "${BUILDKITE:-false}" == "true" ]]; then
        echo "buildkite-agent not found; cannot publish ${key}" >&2
        return 1
    fi
}

normalize_repo_slug() {
    local repo="${1:-}"

    repo="${repo%/}"
    repo="${repo%.git}"
    repo="${repo#git@github.com:}"
    repo="${repo#ssh://git@github.com/}"
    repo="${repo#https://github.com/}"
    repo="${repo#http://github.com/}"
    printf '%s\n' "${repo}"
}

is_trusted_main_build() {
    local actual_repo=""
    local trusted_repo=""

    [[ "${BUILDKITE:-false}" == "true" ]] || return 1
    [[ "${BUILDKITE_PULL_REQUEST:-false}" == "false" ]] || return 1
    [[ "${BUILDKITE_BRANCH:-}" == "${ROCM_BASE_STABLE_BRANCH:-main}" ]] || return 1
    actual_repo=$(normalize_repo_slug "${BUILDKITE_REPO:-}")
    trusted_repo=$(normalize_repo_slug \
        "${ROCM_BASE_STABLE_REPO_SLUG:-vllm-project/vllm}")
    [[ -n "${actual_repo}" && "${actual_repo}" == "${trusted_repo}" ]]
}

is_digest_pinned_image() {
    local image_ref="${1:-}"
    [[ "${image_ref}" =~ ^[^[:space:]@]+@sha256:[0-9a-f]{64}$ ]]
}

use_ci_base_if_present() {
    local ci_base_image=""

    ci_base_image="$(metadata_get rocm-ci-base-image)"
    if [[ -z "${ci_base_image}" ]]; then
        return 1
    fi
    if ! is_digest_pinned_image "${ci_base_image}"; then
        echo "ROCm ci_base handoff is not digest-pinned: ${ci_base_image}" >&2
        return 1
    fi

    export CI_BASE_IMAGE="${ci_base_image}"
    echo "Using ROCm ci_base image selected by the preceding build step: ${CI_BASE_IMAGE}"
}

main() {
    local base_build_required=""
    local ci_base_build_required=""

    if ! use_ci_base_if_present; then
        if [[ "${BUILDKITE:-false}" == "true" ]]; then
            echo "Required ROCm ci_base handoff metadata is missing or invalid" >&2
            return 1
        fi
        echo "No ROCm ci_base handoff metadata found; using the local default"
    fi

    base_build_required="$(metadata_get rocm-base-build-required)"
    ci_base_build_required="$(metadata_get rocm-ci-base-build-required)"
    if [[ "${ROCM_CI_ARTIFACT_ONLY:-0}" == "1" \
        && "${base_build_required}" == "0" \
        && "${ci_base_build_required}" == "0" ]] \
        && ! is_trusted_main_build; then
        echo "ROCM_CI_ARTIFACT_ONLY=1; building ROCm wheel artifact only"
        metadata_set "rocm-ci-image-smoke-required" "0"
        metadata_set "rocm-ci-image-smoked" "0"
        IMAGE_TAG="" bash .buildkite/scripts/ci-bake-rocm.sh \
            test-rocm-ci-with-artifacts
        return 0
    fi

    metadata_set "rocm-ci-image-smoke-required" "1"
    metadata_set "rocm-ci-image-smoked" "0"
    bash .buildkite/scripts/ci-bake-rocm.sh test-rocm-ci-with-wheel
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    main "$@"
fi
