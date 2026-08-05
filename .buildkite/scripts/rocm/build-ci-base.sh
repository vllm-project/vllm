#!/usr/bin/env bash
# Build the ROCm ci_base image, optionally from a freshly rebuilt ROCm base.

set -euo pipefail

metadata_get() {
    local key="$1"
    if command -v buildkite-agent >/dev/null 2>&1; then
        buildkite-agent meta-data get "${key}" 2>/dev/null || true
    fi
}

main() {
    local base_refreshed=""
    local selected_base=""

    # The content identity below describes this checkout. Do not let legacy
    # external pipeline settings switch the build to a remote source tree.
    export REMOTE_VLLM=0
    unset VLLM_BRANCH

    base_refreshed="$(metadata_get rocm-base-refresh)"
    selected_base="$(metadata_get rocm-base-image)"
    if [[ -n "${selected_base}" ]]; then
        if [[ ! "${selected_base}" =~ @sha256:[0-9a-f]{64}$ ]]; then
            echo "Selected ROCm base handoff is not digest-pinned: ${selected_base}" >&2
            return 1
        fi
        export BASE_IMAGE="${selected_base}"
        echo "Using selected ROCm base image for ci_base: ${BASE_IMAGE}"
    elif [[ "${BUILDKITE:-false}" == "true" ]]; then
        echo "Required ROCm base handoff metadata is missing" >&2
        return 1
    fi

    if [[ "${base_refreshed}" == "1" ]]; then
        export CI_BASE_PUSH_STABLE_TAG

        CI_BASE_PUSH_STABLE_TAG="$(metadata_get rocm-base-push-stable-tag)"
        CI_BASE_PUSH_STABLE_TAG="${CI_BASE_PUSH_STABLE_TAG:-0}"

        echo "Push stable ci_base tag: ${CI_BASE_PUSH_STABLE_TAG}"
    fi

    bash .buildkite/scripts/ci-bake-rocm.sh ci-base-rocm-ci-with-deps
}

main "$@"
