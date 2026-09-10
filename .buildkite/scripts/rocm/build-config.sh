#!/usr/bin/env bash
# Shared Dockerfile selection for the normal AMD CI build chain.

configure_rocm_build() {
    case "${VLLM_USE_ROCK:-0}" in
        0) return 0 ;;
        1) ;;
        *)
            echo "VLLM_USE_ROCK must be 0 or 1" >&2
            return 2
            ;;
    esac

    export ROCM_BASE_DOCKERFILE="docker/Dockerfile.rock_base"
    export CI_BASE_DOCKERFILE="docker/Dockerfile.rock"
    export CI_BASE_DOCKERFILE_STAGES="base rust_toolchain_input_0 rust-toolchain-input rust-toolchain build_nixl lmcache_source build_lmcache build_deepep mori_base ci_base"

    # Rock supplies ROCShmem in the SDK, without a build_rocshmem stage.
    export ROCM_DEP_CACHE_EXPORT_MODE=never
    unset ROCSHMEM_BRANCH ROCSHMEM_CACHE_KEY DEEPEP_CACHE_KEY

    # Preserve the standard runtime tags, which are unique to each build,
    # while preventing experimental builds from promoting stable images.
    export ROCM_BASE_PUSH_STABLE_TAG=0
    export CI_BASE_PUSH_STABLE_TAG=0
    echo "VLLM_USE_ROCK=1: using ${ROCM_BASE_DOCKERFILE} and ${CI_BASE_DOCKERFILE}"
}

configure_rocm_build
