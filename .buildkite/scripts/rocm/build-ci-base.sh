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

    base_refreshed="$(metadata_get rocm-base-refresh)"
    if [[ "${base_refreshed}" == "1" ]]; then
        export BASE_IMAGE
        export CI_BASE_PUSH_STABLE_TAG

        BASE_IMAGE="$(metadata_get rocm-base-image)"
        CI_BASE_PUSH_STABLE_TAG="$(metadata_get rocm-base-push-stable-tag)"
        CI_BASE_PUSH_STABLE_TAG="${CI_BASE_PUSH_STABLE_TAG:-0}"
        if [[ ! "${BASE_IMAGE}" =~ @sha256:[0-9a-f]{64}$ ]]; then
            echo "Refreshed ROCm base handoff is missing or not digest-pinned: ${BASE_IMAGE:-<empty>}" >&2
            return 1
        fi

        echo "Using refreshed ROCm base image for ci_base: ${BASE_IMAGE}"
        echo "Push stable ci_base tag: ${CI_BASE_PUSH_STABLE_TAG}"
    fi

    bash .buildkite/scripts/ci-bake-rocm.sh ci-base-rocm-ci-with-deps
}

main "$@"
