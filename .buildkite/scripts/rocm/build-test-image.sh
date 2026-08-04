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
    printf '%s\n' "${repo}"
}

is_trusted_main_build() {
    [[ "${BUILDKITE:-false}" == "true" ]] \
        && [[ "${BUILDKITE_PULL_REQUEST:-false}" == "false" ]] \
        && [[ "${BUILDKITE_BRANCH:-}" == "${ROCM_BASE_STABLE_BRANCH:-main}" ]] \
        && [[ "$(normalize_repo_slug "${BUILDKITE_REPO:-}")" == \
            "$(normalize_repo_slug \
                "${ROCM_BASE_STABLE_REPO_SLUG:-vllm-project/vllm}")" ]]
}

is_digest_pinned_image() {
    [[ "${1:-}" =~ ^[^[:space:]@]+@sha256:[0-9a-f]{64}$ ]]
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

use_ci_base_parent_if_present() {
    local parent_image=""

    parent_image="$(metadata_get rocm-ci-base-parent-image)"
    if [[ -z "${parent_image}" ]]; then
        return 1
    fi
    if ! is_digest_pinned_image "${parent_image}"; then
        echo "ROCm ci_base parent handoff is not digest-pinned: ${parent_image}" >&2
        return 1
    fi

    export BASE_IMAGE="${parent_image}"
    echo "Using the exact parent selected for ci_base: ${BASE_IMAGE}"
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
    local ci_base_build_required=""
    local expected_smoke_image=""
    local legacy_commit_image=""
    local smoke_image="${VLLM_CI_SMOKE_IMAGE:-${IMAGE_TAG:-}}"

    # This job always builds the checked-out commit. Some externally generated
    # pipeline templates still inject remote-fetch settings; do not let those
    # settings make the source identity commit-specific or bypass local edits.
    export REMOTE_VLLM=0
    unset VLLM_BRANCH

    if ! use_ci_base_if_present; then
        if [[ "${BUILDKITE:-false}" == "true" ]]; then
            echo "Required ROCm ci_base handoff metadata is missing or invalid" >&2
            return 1
        fi
        echo "No ROCm ci_base handoff metadata found; using the local default"
    fi

    if ! use_ci_base_parent_if_present; then
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
    ci_base_build_required="$(metadata_get rocm-ci-base-build-required)"
    if [[ "${ROCM_CI_ARTIFACT_ONLY:-0}" == "1" \
        && "${base_build_required}" == "0" \
        && "${ci_base_build_required}" == "0" ]] \
        && ! is_trusted_main_build; then
        echo "ROCM_CI_ARTIFACT_ONLY=1; building ROCm wheel artifact only"
        metadata_set rocm-ci-image-smoke-required 0
        metadata_set rocm-ci-image-smoked 0
        IMAGE_TAG="" bash .buildkite/scripts/ci-bake-rocm.sh test-rocm-ci-with-artifacts
        return 0
    fi

    if [[ "${BUILDKITE:-false}" == "true" ]]; then
        expected_smoke_image="${ROCM_CI_IMAGE_REPO:-rocm/vllm-ci}:build-${BUILDKITE_BUILD_ID:?BUILDKITE_BUILD_ID is required}"
        legacy_commit_image="${ROCM_CI_IMAGE_REPO:-rocm/vllm-ci}:${BUILDKITE_COMMIT:?BUILDKITE_COMMIT is required}"
        if [[ -z "${VLLM_CI_SMOKE_IMAGE:-}" \
            && "${IMAGE_TAG:-}" == "${legacy_commit_image}" ]]; then
            echo "Normalizing legacy commit-scoped image configuration to ${expected_smoke_image}"
            export IMAGE_TAG="${expected_smoke_image}"
            export IMAGE_TAG_LATEST="${legacy_commit_image}"
            export VLLM_CI_SMOKE_IMAGE="${expected_smoke_image}"
            smoke_image="${expected_smoke_image}"
        fi
        if [[ "${IMAGE_TAG:-}" != "${expected_smoke_image}" \
            || "${smoke_image}" != "${expected_smoke_image}" ]]; then
            echo "ROCm build and smoke images must use ${expected_smoke_image}" >&2
            return 1
        fi
        if [[ -n "${IMAGE_TAG_LATEST:-}" \
            && "${IMAGE_TAG_LATEST}" != "${legacy_commit_image}" ]]; then
            echo "ROCm compatibility image must use ${legacy_commit_image}" >&2
            return 1
        fi
        export IMAGE_TAG_LATEST="${legacy_commit_image}"
    fi
    metadata_set rocm-ci-image-smoke-required 1
    metadata_set rocm-ci-image-smoked 0
    metadata_set rocm-ci-image-smoke-ref "${smoke_image}"
    bash .buildkite/scripts/ci-bake-rocm.sh test-rocm-ci-with-wheel
}

main "$@"
