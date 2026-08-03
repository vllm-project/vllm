#!/usr/bin/env bash
# Build the ROCm ci_base image from the immutable base selected earlier in the
# pipeline. Stable aliases are written only by the serialized promotion step.

set -euo pipefail

metadata_get() {
    local key="$1"
    if command -v buildkite-agent >/dev/null 2>&1; then
        buildkite-agent meta-data get "${key}" 2>/dev/null || true
    fi
}

is_digest_pinned_image() {
    local image_ref="${1:-}"
    [[ "${image_ref}" =~ ^[^[:space:]@]+@sha256:[0-9a-f]{64}$ ]]
}

main() {
    local base_image=""

    base_image="$(metadata_get rocm-base-image)"
    if is_digest_pinned_image "${base_image}"; then
        export BASE_IMAGE="${base_image}"
        echo "Using selected ROCm base image for ci_base: ${BASE_IMAGE}"
    elif [[ "${BUILDKITE:-false}" == "true" ]]; then
        echo "Required ROCm base image handoff metadata is missing or invalid: ${base_image:-<empty>}" >&2
        return 1
    else
        echo "No digest-pinned ROCm base image handoff found; using the local default"
    fi

    # The long build publishes content and commit tags only. Mutable aliases
    # are updated after freshness validation by promote-stable-images.sh.
    export CI_BASE_PUSH_STABLE_TAG=0
    bash .buildkite/scripts/ci-bake-rocm.sh ci-base-rocm-ci-with-deps
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    main "$@"
fi
