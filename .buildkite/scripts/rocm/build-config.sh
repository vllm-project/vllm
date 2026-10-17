#!/usr/bin/env bash
# Shared Dockerfile selection for the normal AMD CI build chain.
# CI_ROCM_DOCKERFILE_BASE and CI_ROCM_DOCKERFILE accept repository-relative paths.
# For Rock, set CI_ROCM_DOCKERFILE_BASE=docker/Dockerfile.rock_base and
# CI_ROCM_DOCKERFILE=docker/Dockerfile.rock.
# Custom recipes must preserve the CI targets and artifact layout. Additional
# COPY inputs also need the corresponding *_CONTENT_FILES overrides.

rocm_base_dockerfile() {
    local default="docker/Dockerfile.rocm_base"
    printf '%s\n' "${CI_ROCM_DOCKERFILE_BASE:-${ROCM_BASE_DOCKERFILE:-${default}}}"
}

rocm_ci_dockerfile() {
    local default="docker/Dockerfile.rocm"
    printf '%s\n' "${CI_ROCM_DOCKERFILE:-${CI_BASE_DOCKERFILE:-${default}}}"
}

using_custom_rocm_dockerfiles() {
    [[ "$(rocm_base_dockerfile)" != "docker/Dockerfile.rocm_base" \
        || "$(rocm_ci_dockerfile)" != "docker/Dockerfile.rocm" ]]
}

uses_standard_rocm_configuration() {
    local name=""

    using_custom_rocm_dockerfiles && return 1
    [[ "${TORCH_NIGHTLY:-0}" != 1 \
        && "${ROCM_BASE_PUSH_STABLE_TAG-1}" == 1 \
        && "${CI_BASE_PUSH_STABLE_TAG-1}" == 1 \
        && "${VLLM_BAKE_FILE:-docker/docker-bake-rocm.hcl}" == docker/docker-bake-rocm.hcl \
        && "${CI_HCL_SOURCE:-${CI_HCL_FILE:-docker/ci-rocm.hcl}}" == docker/ci-rocm.hcl ]] \
        || return 1
    # The pipeline explicitly supplies this standard CI architecture list.
    # BASE_IMAGE and CI_BASE_IMAGE are validated digest handoffs, not overrides
    # here. Cache settings and build parallelism do not select a custom stack.
    if [[ -v PYTORCH_ROCM_ARCH \
        && "${PYTORCH_ROCM_ARCH}" != 'gfx90a;gfx942;gfx950' ]]; then
        return 1
    fi
    # Keep this list aligned with ci-infra's AMD promotion upload gate. Empty
    # values also veto promotion: the base builder forwards explicitly empty
    # ARG overrides instead of restoring the checked-in Dockerfile defaults.
    for name in \
        ROCM_SDK_VERSION TORCH_VERSION TORCHVISION_VERSION TORCHAUDIO_VERSION \
        TRITON_VERSION TRITON_BRANCH TRITON_REPO \
        PYTORCH_BRANCH PYTORCH_REPO PYTORCH_VISION_BRANCH PYTORCH_VISION_REPO \
        PYTORCH_AUDIO_BRANCH PYTORCH_AUDIO_REPO \
        FA_BRANCH FA_REPO AITER_BRANCH AITER_REPO AITER_ROCM_ARCH \
        MORI_BRANCH MORI_REPO ROCM_SYSTEMS_REPO ROCM_RUNTIME_COMMIT \
        ROCPROFILER_SDK_COMMIT ROCPROFILER_SDK_PR_7796 ROCPROFILER_SDK_PR_7924 \
        ROCM_RELEASE_WHEELS_MULTIARCH_URL ROCM_NIGHTLY_WHEELS_MULTIARCH_URL \
        PYTHON_VERSION SITE_PACKAGES ARG_PYTORCH_ROCM_ARCH \
        NIXL_BRANCH NIXL_REPO UCX_BRANCH UCX_REPO \
        ROCSHMEM_BRANCH ROCSHMEM_REPO DEEPEP_BRANCH DEEPEP_REPO \
        DEEPEP_ROCM_ARCH DEEPEP_NIC ROCM_TRITON_KERNELS_COMMIT \
        LMCACHE_REPO LMCACHE_REF LMCACHE_VERSION LMCACHE_ROCM_ARCH INSTALL_LMCACHE \
        NIC_BACKEND AINIC_VERSION UBUNTU_CODENAME VLLM_REPO COMMON_WORKDIR \
        ROCM_BASE_CONTENT_FILES ROCM_BASE_CONTENT_ARGS ROCM_BASE_BUILD_ARGS \
        ROCM_BASE_METADATA_VERSION CI_BASE_CONTENT_FILES CI_BASE_CONTENT_ARGS \
        CI_BASE_DOCKERFILE_STAGES CI_BASE_METADATA_VERSION CI_BASE_CONTENT_HASH \
        ROCM_CSRC_CONTENT_FILES ROCM_CSRC_CONTENT_ARGS ROCM_CSRC_DOCKERFILE_STAGES \
        ROCM_RUST_CONTENT_FILES ROCM_RUST_CONTENT_ARGS ROCM_RUST_DOCKERFILE_STAGES; do
        if [[ -v "${name}" ]]; then
            return 1
        fi
    done
    return 0
}

publish_rocm_standard_configuration() {
    local key="$1"
    local standard="$2"
    local previous=""

    [[ "${BAKE_PRINT_ONLY:-0}" != 1 ]] || return 0
    if ! command -v buildkite-agent >/dev/null 2>&1; then
        [[ "${BUILDKITE:-false}" != true ]] && return 0
        echo "buildkite-agent is required to publish ROCm configuration provenance" >&2
        return 1
    fi
    previous=$(buildkite-agent meta-data get "${key}" --default "" 2>/dev/null) \
        || previous=0
    # Once any attempt uses custom inputs, later retries cannot bless the
    # same build for stable publication. Unknown provenance on retry is also
    # conservative; a first attempt may start without an existing key.
    if [[ "${previous}" == 0 \
        || ( -n "${previous}" && "${previous}" != 1 ) \
        || ( "${BUILDKITE_RETRY_COUNT:-0}" != 0 && "${previous}" != 1 ) ]]; then
        standard=0
    fi
    buildkite-agent meta-data set "${key}" "${standard}"
}

rocm_dockerfile_stages() {
    awk '
        toupper($1) == "FROM" {
            for (idx = 2; idx < NF; idx++) {
                if (tolower($idx) == "as") print tolower($(idx + 1))
            }
        }
    ' "$1"
}

validate_rocm_dockerfile() {
    local dockerfile="$1"
    local stage=""
    local stages=""
    shift

    case "${dockerfile}" in
        /*|../*|*/../*|*/..|*[[:space:]]*)
            echo "ROCm Dockerfile must be a repository-relative path without '..' or whitespace: ${dockerfile}" >&2
            return 2
            ;;
    esac
    if [[ ! -f "${dockerfile}" ]]; then
        echo "ROCm Dockerfile not found: ${dockerfile}" >&2
        return 2
    fi
    if (($#)) && ! awk '
        toupper($1) == "FROM" {
            named = 0
            for (idx = 2; idx < NF; idx++) {
                if (tolower($idx) == "as") named = 1
            }
            if (!named) exit 1
        }
    ' "${dockerfile}"; then
        echo "ROCm Dockerfile must name every build stage: ${dockerfile}" >&2
        return 2
    fi
    stages=$(rocm_dockerfile_stages "${dockerfile}")
    for stage in "$@"; do
        if ! grep -Fxq -- "${stage}" <<< "${stages}"; then
            echo "ROCm Dockerfile ${dockerfile} is missing required stage: ${stage}" >&2
            return 2
        fi
    done
}

configure_rocm_build() {
    ROCM_BASE_DOCKERFILE=$(rocm_base_dockerfile)
    CI_BASE_DOCKERFILE=$(rocm_ci_dockerfile)
    export ROCM_BASE_DOCKERFILE CI_BASE_DOCKERFILE
    using_custom_rocm_dockerfiles || return 0

    # Custom stacks may supply dependencies without separate build stages.
    export ROCM_DEP_CACHE_EXPORT_MODE=never

    # Preserve the standard runtime tags, which are unique to each build,
    # while preventing experimental builds from promoting stable images.
    export ROCM_BASE_PUSH_STABLE_TAG=0
    export CI_BASE_PUSH_STABLE_TAG=0
    echo "Custom ROCm Dockerfiles: ${ROCM_BASE_DOCKERFILE} and ${CI_BASE_DOCKERFILE}"
}
