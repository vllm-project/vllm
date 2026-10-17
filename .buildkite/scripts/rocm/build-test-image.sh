#!/usr/bin/env bash
# Build the ROCm CI image or wheel artifact from the selected ci_base.

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
    repo="${repo#github.com/}"
    printf '%s\n' "${repo}"
}

is_trusted_main_build() {
    local stable_branch="${ROCM_BASE_STABLE_BRANCH:-${CI_BASE_STABLE_BRANCH:-main}}"
    local stable_repo="${ROCM_BASE_STABLE_REPO_SLUG:-${CI_BASE_STABLE_REPO_SLUG:-vllm-project/vllm}}"

    [[ "${BUILDKITE:-false}" == "true" ]] \
        && [[ "${BUILDKITE_PULL_REQUEST:-false}" == "false" ]] \
        && [[ "${BUILDKITE_BRANCH:-}" == "${stable_branch}" ]] \
        && [[ "$(normalize_repo_slug "${BUILDKITE_REPO:-}")" == \
            "$(normalize_repo_slug "${stable_repo}")" ]]
}

is_digest_pinned_image() {
    [[ "${1:-}" =~ ^[^[:space:]@]+@sha256:[0-9a-f]{64}$ ]]
}

load_digest_handoff() {
    local metadata_key="$1"
    local env_name="$2"
    local description="$3"
    local image_ref=""

    image_ref="$(metadata_get "${metadata_key}")"
    if [[ -z "${image_ref}" ]]; then
        return 1
    fi
    if ! is_digest_pinned_image "${image_ref}"; then
        echo "${description} is not digest-pinned: ${image_ref}" >&2
        return 1
    fi

    printf -v "${env_name}" '%s' "${image_ref}"
    export "${env_name?}"
    echo "Using ${description}: ${image_ref}"
}

validate_selected_base() {
    local selected_base=""

    selected_base="$(metadata_get rocm-base-image)"
    if ! is_digest_pinned_image "${selected_base}"; then
        echo "Selected ROCm base handoff is missing or invalid: ${selected_base:-<empty>}" >&2
        return 1
    fi
    if [[ "${selected_base}" != "${BASE_IMAGE:-}" ]]; then
        echo "ROCm base handoffs disagree:" >&2
        echo "  selected base: ${selected_base}" >&2
        echo "  ci_base parent: ${BASE_IMAGE:-<empty>}" >&2
        return 1
    fi
}

main() {
    local base_build_required=""
    local base_built_in_build=""
    local ci_base_build_required=""
    local ci_base_built_in_build=""
    local expected_smoke_image=""
    local smoke_image="${VLLM_CI_SMOKE_IMAGE:-${IMAGE_TAG:-}}"

    # This job always builds the checked-out commit. Some externally generated
    # pipeline templates still inject remote-fetch settings; do not let those
    # settings make the source identity commit-specific or bypass local edits.
    export REMOTE_VLLM=0
    unset VLLM_BRANCH

    if ! load_digest_handoff \
        rocm-ci-base-image CI_BASE_IMAGE "ROCm ci_base handoff"; then
        if [[ "${BUILDKITE:-false}" == "true" ]]; then
            echo "Required ROCm ci_base handoff metadata is missing or invalid" >&2
            return 1
        fi
        echo "No ROCm ci_base handoff metadata found; using the local default"
    fi

    if ! load_digest_handoff rocm-ci-base-parent-image BASE_IMAGE \
        "ROCm ci_base parent handoff"; then
        if [[ "${BUILDKITE:-false}" == "true" ]]; then
            echo "Required ROCm ci_base parent handoff metadata is missing or invalid" >&2
            return 1
        fi
        echo "No ROCm ci_base parent handoff metadata found; using the local default"
    fi
    if [[ "${BUILDKITE:-false}" == "true" ]]; then
        validate_selected_base || return 1
    fi

    base_build_required="$(metadata_get rocm-base-build-required)"
    base_built_in_build="$(metadata_get rocm-base-built-in-build)"
    ci_base_build_required="$(metadata_get rocm-ci-base-build-required)"
    ci_base_built_in_build="$(metadata_get rocm-ci-base-built-in-build)"
    if [[ "${ROCM_CI_ARTIFACT_ONLY:-0}" == "1" \
        && "${base_build_required}" == "0" \
        && "${ci_base_build_required}" == "0" \
        && "${base_built_in_build}" == "0" \
        && "${ci_base_built_in_build}" == "0" ]] \
        && ! is_trusted_main_build; then
        echo "ROCM_CI_ARTIFACT_ONLY=1; building ROCm wheel artifact only"
        metadata_set rocm-ci-image-smoke-required 0
        metadata_set rocm-ci-image-smoked 0
        IMAGE_TAG="" bash .buildkite/scripts/ci-bake-rocm.sh test-rocm-ci-with-artifacts
        return 0
    fi

    if [[ "${BUILDKITE:-false}" == "true" ]]; then
        expected_smoke_image="${ROCM_CI_IMAGE_REPO:-rocm/vllm-ci}:build-${BUILDKITE_BUILD_ID:?BUILDKITE_BUILD_ID is required}"
        if [[ "${IMAGE_TAG:-}" != "${expected_smoke_image}" \
            || "${smoke_image}" != "${expected_smoke_image}" ]]; then
            echo "ROCm build and smoke images must use ${expected_smoke_image}" >&2
            return 1
        fi
        unset IMAGE_TAG_LATEST
    fi
    metadata_set rocm-ci-image-smoke-required 1
    metadata_set rocm-ci-image-smoked 0
    metadata_set rocm-ci-image-smoke-ref "${smoke_image}"
    bash .buildkite/scripts/ci-bake-rocm.sh test-rocm-ci-with-wheel
}

main "$@"
