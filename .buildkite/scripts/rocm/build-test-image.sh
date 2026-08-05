#!/usr/bin/env bash
# Build the ROCm CI test image or wheel artifact.
#
# When base selection changes, build the full image so downstream ROCm tests
# validate the selected base -> ci_base -> CI image chain.

set -euo pipefail

metadata_get() {
    local key="$1"
    if command -v buildkite-agent >/dev/null 2>&1; then
        buildkite-agent meta-data get "${key}" 2>/dev/null || true
    fi
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
    if [[ ! "${image_ref}" =~ @sha256:[0-9a-f]{64}$ ]]; then
        echo "${description} is not digest-pinned: ${image_ref}" >&2
        return 1
    fi

    printf -v "${env_name}" '%s' "${image_ref}"
    export "${env_name}"
    echo "Using ${description}: ${image_ref}"
}

main() {
    local base_refreshed=0

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

    if ! load_digest_handoff \
        rocm-ci-base-parent-image BASE_IMAGE "ROCm ci_base parent handoff"; then
        if [[ "${BUILDKITE:-false}" == "true" ]]; then
            echo "Required ROCm ci_base parent handoff metadata is missing or invalid" >&2
            return 1
        fi
        echo "No ROCm ci_base parent handoff metadata found; using the local default"
    fi

    if [[ "$(metadata_get rocm-base-refresh)" == "1" ]]; then
        echo "The selected ROCm base differs from the current stable base"
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
