#!/usr/bin/env bash
# Build the ROCm CI test image or wheel artifact.
#
# When Dockerfile.rocm_base changes, always build the full image so downstream
# ROCm tests can validate the freshly rebuilt base -> ci_base -> ci image chain.

set -euo pipefail

metadata_get() {
    local key="$1"
    if command -v buildkite-agent >/dev/null 2>&1; then
        buildkite-agent meta-data get "${key}" 2>/dev/null || true
    fi
}

use_ci_base_if_present() {
    local ci_base_image=""

    ci_base_image="$(metadata_get rocm-ci-base-image)"
    if [[ -z "${ci_base_image}" ]]; then
        return 1
    fi
    if [[ ! "${ci_base_image}" =~ @sha256:[0-9a-f]{64}$ ]]; then
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
    if [[ ! "${parent_image}" =~ @sha256:[0-9a-f]{64}$ ]]; then
        echo "ROCm ci_base parent handoff is not digest-pinned: ${parent_image}" >&2
        return 1
    fi

    export BASE_IMAGE="${parent_image}"
    echo "Using the exact parent selected for ci_base: ${BASE_IMAGE}"
}

use_refreshed_base_if_present() {
    local base_refreshed=""
    local refreshed_base_image=""

    base_refreshed="$(metadata_get rocm-base-refresh)"
    if [[ "${base_refreshed}" != "1" ]]; then
        return 1
    fi

    export IMAGE_TAG_LATEST

    refreshed_base_image="$(metadata_get rocm-base-image)"
    IMAGE_TAG_LATEST="$(metadata_get rocm-ci-image-descriptive)"
    if [[ ! "${refreshed_base_image}" =~ @sha256:[0-9a-f]{64}$ ]]; then
        echo "Refreshed ROCm base handoff is missing or not digest-pinned: ${refreshed_base_image:-<empty>}" >&2
        return 2
    fi
    if [[ "${refreshed_base_image}" != "${BASE_IMAGE:-}" ]]; then
        echo "ROCm base handoffs disagree:" >&2
        echo "  ci_base parent: ${BASE_IMAGE:-<empty>}" >&2
        echo "  refreshed base: ${refreshed_base_image}" >&2
        return 2
    fi

    echo "Validated refreshed ROCm base handoff: ${refreshed_base_image}"
    if [[ -n "${IMAGE_TAG_LATEST}" ]]; then
        echo "Also tagging full ROCm CI image as: ${IMAGE_TAG_LATEST}"
    fi

    return 0
}

main() {
    local base_refreshed=0
    local refreshed_status=0

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

    if use_refreshed_base_if_present; then
        base_refreshed=1
    else
        refreshed_status=$?
        if [[ ${refreshed_status} -gt 1 ]]; then
            return "${refreshed_status}"
        fi
    fi

    if [[ "${ROCM_CI_ARTIFACT_ONLY:-0}" == "1" && "${base_refreshed}" != "1" ]]; then
        echo "ROCM_CI_ARTIFACT_ONLY=1; building ROCm wheel artifact only"
        IMAGE_TAG="" bash .buildkite/scripts/ci-bake-rocm.sh test-rocm-ci-with-artifacts
        return
    fi

    bash .buildkite/scripts/ci-bake-rocm.sh test-rocm-ci-with-wheel
}

main "$@"
