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

use_refreshed_base_if_present() {
    local base_refreshed=""

    base_refreshed="$(metadata_get rocm-base-refresh)"
    if [[ "${base_refreshed}" != "1" ]]; then
        return 1
    fi

    export BASE_IMAGE
    export CI_BASE_IMAGE
    export IMAGE_TAG_LATEST

    BASE_IMAGE="$(metadata_get rocm-base-image)"
    CI_BASE_IMAGE="$(metadata_get rocm-ci-base-image)"
    IMAGE_TAG_LATEST="$(metadata_get rocm-ci-image-descriptive)"

    echo "Using refreshed ROCm base image for test image: ${BASE_IMAGE}"
    echo "Using refreshed ROCm ci_base image for test image: ${CI_BASE_IMAGE}"
    if [[ -n "${IMAGE_TAG_LATEST}" ]]; then
        echo "Also tagging full ROCm CI image as: ${IMAGE_TAG_LATEST}"
    fi

    return 0
}

main() {
    local base_refreshed=0

    if use_refreshed_base_if_present; then
        base_refreshed=1
    fi

    if [[ "${ROCM_CI_ARTIFACT_ONLY:-0}" == "1" && "${base_refreshed}" != "1" ]]; then
        echo "ROCM_CI_ARTIFACT_ONLY=1; building ROCm wheel artifact only"
        IMAGE_TAG="" bash .buildkite/scripts/ci-bake-rocm.sh test-rocm-ci-with-artifacts
        return
    fi

    bash .buildkite/scripts/ci-bake-rocm.sh test-rocm-ci-with-wheel
}

main "$@"
