#!/usr/bin/env bash
# Build the ROCm CI test image or wheel artifact.
#
# When base selection changes, build the full image so downstream ROCm tests
# validate the selected base -> ci_base -> CI image chain.

set -euo pipefail

ZEN_CPU_DEPENDENCIES=(
    ".buildkite/image_build/image_build.yaml"
    ".buildkite/image_build/image_build_zen_cpu.sh"
    ".buildkite/scripts/rocm/build-test-image.sh"
    "docker/Dockerfile.zen"
    "docker/Dockerfile.cpu"
    "requirements/cpu.txt"
    "requirements/build/cpu.txt"
    "requirements/build/rust.txt"
    "requirements/test/cpu.txt"
    "requirements/test/cuda.in"
    "requirements/lint.txt"
    "rust/"
    "rust-toolchain.toml"
    "build_rust.sh"
    "tools/build_rust.py"
    "use_existing_torch.py"
    "csrc/cpu/"
    "cmake/cpu_extension.cmake"
)

metadata_get() {
    local key="$1"
    if command -v buildkite-agent >/dev/null 2>&1; then
        buildkite-agent meta-data get "${key}" 2>/dev/null || true
    fi
}

get_changed_files() {
    local merge_base=""

    if [[ "${BUILDKITE_BRANCH:-}" == "main" ]]; then
        git diff --name-only --diff-filter=ACMDR HEAD~1 HEAD 2>/dev/null || return 1
        return 0
    fi

    merge_base="$(git merge-base origin/main HEAD 2>/dev/null || true)"
    if [[ -z "${merge_base}" ]]; then
        return 1
    fi

    git diff --name-only --diff-filter=ACMDR "${merge_base}" HEAD 2>/dev/null
}

should_build_zen_cpu_image() {
    local mode="${VLLM_ZEN_CPU_BUILD:-auto}"
    local changed_files=()
    local file=""
    local dependency=""

    if [[ "${mode}" == "0" ]]; then
        echo "Skipping Zen CPU validation because VLLM_ZEN_CPU_BUILD=0"
        return 1
    fi

    if [[ "${mode}" == "1" ]]; then
        echo "Running Zen CPU validation because VLLM_ZEN_CPU_BUILD=1"
        return 0
    fi

    if ! mapfile -t changed_files < <(get_changed_files); then
        echo "Could not determine changed files; running Zen CPU validation defensively"
        return 0
    fi

    for file in "${changed_files[@]}"; do
        for dependency in "${ZEN_CPU_DEPENDENCIES[@]}"; do
            if [[ "${file}" == "${dependency}" || "${file}" == "${dependency}"* ]]; then
                echo "Detected Zen CPU build input change: ${file}"
                return 0
            fi
        done
    done

    echo "No Zen CPU build inputs changed; skipping Zen CPU validation"
    return 1
}

maybe_build_zen_cpu_image() {
    if ! should_build_zen_cpu_image; then
        return 0
    fi

    echo "--- :docker: Building Zen CPU validation image on amd-cpu"
    .buildkite/image_build/image_build_zen_cpu.sh "${BUILDKITE_COMMIT}"
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
    export "${env_name?}"
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
        rocm-base-image BASE_IMAGE "ROCm base handoff"; then
        if [[ "${BUILDKITE:-false}" == "true" ]]; then
            echo "Required ROCm base handoff metadata is missing or invalid" >&2
            return 1
        fi
        echo "No ROCm base handoff metadata found; using the local default"
    fi

    if [[ "$(metadata_get rocm-base-refresh)" == "1" ]]; then
        echo "The selected ROCm base differs from the current stable base"
        base_refreshed=1
    fi

    if [[ "${ROCM_CI_ARTIFACT_ONLY:-0}" == "1" && "${base_refreshed}" != "1" ]]; then
        echo "ROCM_CI_ARTIFACT_ONLY=1; building ROCm wheel artifact only"
        IMAGE_TAG="" bash .buildkite/scripts/ci-bake-rocm.sh test-rocm-ci-with-artifacts
        maybe_build_zen_cpu_image
        return
    fi

    bash .buildkite/scripts/ci-bake-rocm.sh test-rocm-ci-with-wheel
    maybe_build_zen_cpu_image
}

main "$@"
