#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Wraps a test command with nsys profiling when VLLM_ENABLE_PROFILING=1.
# Only profiles steps whose BUILDKITE_LABEL is in the NSYS_PROFILED_TESTS
# allowlist (semicolon-separated). Profiles all steps if the list is empty.
# Usage: bash run-with-profiling.sh <command...>

set -euo pipefail

VLLM_ENABLE_PROFILING="${VLLM_ENABLE_PROFILING:-0}"
NSYS_PROFILED_TESTS="${NSYS_PROFILED_TESTS:-}"
PROFILE_OUTPUT_DIR="${PROFILE_OUTPUT_DIR:-/tmp/profiles}"
BUILDKITE_LABEL=$(echo "${BUILDKITE_LABEL:-}" | xargs)

SANITIZED_LABEL=$(echo "${BUILDKITE_LABEL}" | tr ' /:' '_' | tr -cd '[:alnum:]_-')
PROFILE_NAME="${PROFILE_NAME:-${SANITIZED_LABEL:-vllm_profile}}"

if [ "$#" -eq 0 ]; then
    echo "Usage: $0 <command...>"
    exit 1
fi

should_profile() {
    if [ "${VLLM_ENABLE_PROFILING}" != "1" ]; then
        return 1
    fi

    if [ -z "${NSYS_PROFILED_TESTS}" ]; then
        return 0
    fi

    IFS=';' read -ra ENTRIES <<< "${NSYS_PROFILED_TESTS}"
    for entry in "${ENTRIES[@]}"; do
        entry=$(echo "${entry}" | xargs)
        if [ "${entry}" = "${BUILDKITE_LABEL}" ]; then
            return 0
        fi
    done

    echo "=== nsys profiling SKIPPED (step '${BUILDKITE_LABEL}' not in NSYS_PROFILED_TESTS) ==="
    return 1
}

upload_profiles() {
    if command -v buildkite-agent &> /dev/null; then
        echo "=== Uploading nsys profiles as Buildkite artifacts ==="
        buildkite-agent artifact upload "${PROFILE_OUTPUT_DIR}/*.nsys-rep" 2>&1 || \
            echo "WARNING: Failed to upload profiles (non-fatal)"
    else
        echo "=== nsys profiles saved to ${PROFILE_OUTPUT_DIR}/ (no buildkite-agent for upload) ==="
        ls -lh "${PROFILE_OUTPUT_DIR}"/*.nsys-rep 2>/dev/null || true
    fi
}

if should_profile; then
    echo "=== nsys profiling ENABLED for '${BUILDKITE_LABEL}' ==="

    export VLLM_NVTX_SCOPES_FOR_PROFILING=1
    export VLLM_ENABLE_LAYERWISE_NVTX_TRACING=1
    export VLLM_WORKER_MULTIPROC_METHOD=spawn

    mkdir -p "${PROFILE_OUTPUT_DIR}"

    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    PROFILE_PATH="${PROFILE_OUTPUT_DIR}/${PROFILE_NAME}_${TIMESTAMP}"

    echo "Profile output: ${PROFILE_PATH}.nsys-rep"
    echo "Profiled command: $*"

    nsys profile \
        -o "${PROFILE_PATH}" \
        --force-overwrite=true \
        --trace=cuda,nvtx,osrt \
        --trace-fork-before-exec=true \
        --cuda-graph-trace=node \
        "$@"
    TEST_EXIT=$?

    upload_profiles

    exit ${TEST_EXIT}
else
    exec "$@"
fi
