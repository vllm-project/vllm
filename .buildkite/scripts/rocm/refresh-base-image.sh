#!/usr/bin/env bash
# Build and publish a fresh ROCm base image when Dockerfile.rocm_base changes.
#
# Normal AMD CI builds should not pay for this path. The script no-ops unless
# docker/Dockerfile.rocm_base changed relative to the PR base, the previous
# main commit, or ROCM_BASE_REFRESH_FORCE=1 is set.

set -euo pipefail

DOCKERFILE="${ROCM_BASE_DOCKERFILE:-docker/Dockerfile.rocm_base}"
BASE_REPO="${ROCM_BASE_IMAGE_REPO:-rocm/vllm-dev}"
CI_IMAGE_REPO="${ROCM_CI_IMAGE_REPO:-rocm/vllm-ci}"
BUILDER_NAME="${ROCM_BASE_BUILDER_NAME:-vllm-rocm-base-builder}"

metadata_set() {
    local key="$1"
    local value="$2"

    if command -v buildkite-agent >/dev/null 2>&1; then
        buildkite-agent meta-data set "${key}" "${value}" || true
    fi
}

clean_docker_tag() {
    local input="$1"
    echo "${input}" | sed 's/[^a-zA-Z0-9._-]/_/g' | cut -c1-128
}

tag_component() {
    local input="$1"
    local max_chars="${2:-24}"

    clean_docker_tag "${input:-unknown}" | cut -c1-"${max_chars}"
}

extract_arg_default() {
    local arg_name="$1"

    sed -n -E "s/^[[:space:]]*ARG[[:space:]]+${arg_name}=\"?([^\"[:space:]]+)\"?.*/\\1/p" \
        "${DOCKERFILE}" | head -1
}

rocm_version_from_base_image() {
    local base_image="$1"
    local version=""

    version="$(sed -n -E 's/.*:([0-9]+\.[0-9]+(\.[0-9]+)?)-.*/\1/p' <<<"${base_image}")"
    tag_component "${version:-${base_image}}" 16
}

git_diff_changed_base() {
    local range="$1"
    [[ -n "$(git diff --name-only "${range}" -- "${DOCKERFILE}" 2>/dev/null)" ]]
}

rocm_base_changed() {
    local base_branch="${BUILDKITE_PULL_REQUEST_BASE_BRANCH:-main}"
    local base_ref="refs/remotes/origin/${base_branch}"
    local merge_base=""

    if [[ "${ROCM_BASE_REFRESH_SKIP:-0}" == "1" ]]; then
        echo "ROCM_BASE_REFRESH_SKIP=1 set; skipping ROCm base refresh"
        return 1
    fi

    if [[ "${ROCM_BASE_REFRESH_FORCE:-0}" == "1" ]]; then
        echo "ROCM_BASE_REFRESH_FORCE=1 set; refreshing ROCm base image"
        return 0
    fi

    if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
        echo "Not in a git checkout; skipping ROCm base refresh unless forced"
        return 1
    fi

    if [[ "${BUILDKITE_PULL_REQUEST:-false}" != "false" ]]; then
        git fetch --no-tags --depth=200 origin \
            "+refs/heads/${base_branch}:${base_ref}" >/dev/null 2>&1 || true
        merge_base=$(git merge-base HEAD "${base_ref}" 2>/dev/null || true)
        if [[ -n "${merge_base}" ]] && git_diff_changed_base "${merge_base}...HEAD"; then
            return 0
        fi
    elif git rev-parse --verify HEAD~1 >/dev/null 2>&1; then
        if git_diff_changed_base "HEAD~1..HEAD"; then
            return 0
        fi
    fi

    return 1
}

should_push_stable_tag() {
    if [[ "${BUILDKITE_PULL_REQUEST:-false}" != "false" ]]; then
        return 1
    fi

    if [[ "${ROCM_BASE_PUSH_STABLE_TAG:-}" == "1" ]]; then
        return 0
    fi
    if [[ "${ROCM_BASE_PUSH_STABLE_TAG:-}" == "0" ]]; then
        return 1
    fi

    [[ "${BUILDKITE_PULL_REQUEST:-false}" == "false" \
        && "${BUILDKITE_BRANCH:-}" == "${ROCM_BASE_STABLE_BRANCH:-main}" ]]
}

setup_builder() {
    echo "--- :buildkite: Setting up buildx builder for ROCm base"
    if docker buildx inspect "${BUILDER_NAME}" >/dev/null 2>&1; then
        docker buildx use "${BUILDER_NAME}"
    else
        docker buildx create --name "${BUILDER_NAME}" --driver docker-container --use
    fi
    docker buildx inspect --bootstrap
}

compute_base_content_hash() {
    local use_sccache="$1"

    {
        printf 'dockerfile:%s\n' "${DOCKERFILE}"
        sha256sum "${DOCKERFILE}"
        printf 'arg:USE_SCCACHE=%s\n' "${use_sccache}"
    } | sha256sum | cut -d' ' -f1
}

build_base_image() {
    local use_sccache="${ROCM_BASE_USE_SCCACHE:-${USE_SCCACHE:-0}}"
    local base_hash=""
    local short_hash=""
    local build_date=""
    local base_image_arg=""
    local rocm_version=""
    local triton_arg=""
    local pytorch_arg=""
    local pytorch_vision_arg=""
    local pytorch_audio_arg=""
    local fa_arg=""
    local aiter_arg=""
    local mori_arg=""
    local pytorch_branch=""
    local aiter_branch=""
    local dependency_summary=""
    local descriptor=""
    local ci_descriptor=""
    local descriptive_tag=""
    local stable_tag="${BASE_REPO}:base"
    local ci_descriptive_tag=""
    local -a tags=()
    local -a no_cache_args=()
    local -a sccache_args=()

    if [[ ! -f "${DOCKERFILE}" ]]; then
        echo "Error: ROCm base Dockerfile not found: ${DOCKERFILE}" >&2
        exit 1
    fi

    base_hash=$(compute_base_content_hash "${use_sccache}")
    short_hash="${base_hash:0:16}"
    build_date="${ROCM_BASE_TAG_DATE:-$(date -u +%Y%m%d)}"
    base_image_arg="$(extract_arg_default BASE_IMAGE)"
    rocm_version="$(rocm_version_from_base_image "${base_image_arg}")"
    triton_arg="$(extract_arg_default TRITON_BRANCH)"
    pytorch_arg="$(extract_arg_default PYTORCH_BRANCH)"
    pytorch_vision_arg="$(extract_arg_default PYTORCH_VISION_BRANCH)"
    pytorch_audio_arg="$(extract_arg_default PYTORCH_AUDIO_BRANCH)"
    fa_arg="$(extract_arg_default FA_BRANCH)"
    aiter_arg="$(extract_arg_default AITER_BRANCH)"
    mori_arg="$(extract_arg_default MORI_BRANCH)"
    pytorch_branch="$(tag_component "${pytorch_arg}" 16)"
    aiter_branch="$(tag_component "${aiter_arg}" 24)"
    dependency_summary="base=${base_image_arg},rocm=${rocm_version},pytorch=${pytorch_arg},torchvision=${pytorch_vision_arg},torchaudio=${pytorch_audio_arg},triton=${triton_arg},flash-attn=${fa_arg},aiter=${aiter_arg},mori=${mori_arg}"
    descriptor="$(clean_docker_tag "base_custom_aiter_${aiter_branch}_torch_${pytorch_branch}_${build_date}_${short_hash}")"
    ci_descriptor="$(clean_docker_tag "ci_custom_aiter_${aiter_branch}_torch_${pytorch_branch}_${build_date}_${short_hash}")"

    descriptive_tag="${BASE_REPO}:${descriptor}"
    ci_descriptive_tag="${CI_IMAGE_REPO}:${ci_descriptor}"

    tags=(-t "${descriptive_tag}")
    if should_push_stable_tag; then
        tags+=(-t "${stable_tag}")
        metadata_set "rocm-base-push-stable-tag" "1"
    else
        metadata_set "rocm-base-push-stable-tag" "0"
    fi

    if [[ "${ROCM_BASE_NO_CACHE:-1}" == "1" ]]; then
        no_cache_args=(--no-cache)
    fi

    for env_name in \
        SCCACHE_DOWNLOAD_URL \
        SCCACHE_ENDPOINT \
        SCCACHE_BUCKET_NAME \
        SCCACHE_REGION_NAME \
        SCCACHE_S3_NO_CREDENTIALS; do
        if [[ -n "${!env_name:-}" ]]; then
            sccache_args+=(--build-arg "${env_name}=${!env_name}")
        fi
    done

    echo "--- :docker: Building ROCm base image"
    echo "Dockerfile: ${DOCKERFILE}"
    echo "Descriptive tag: ${descriptive_tag}"
    echo "Stable tag: ${stable_tag} ($(should_push_stable_tag && echo enabled || echo disabled))"
    echo "Dependency summary: ${dependency_summary}"
    echo "USE_SCCACHE: ${use_sccache}"

    docker buildx build \
        "${no_cache_args[@]}" \
        --pull \
        --progress plain \
        --file "${DOCKERFILE}" \
        --build-arg "USE_SCCACHE=${use_sccache}" \
        "${sccache_args[@]}" \
        --label "org.opencontainers.image.source=https://github.com/vllm-project/vllm" \
        --label "org.opencontainers.image.vendor=vLLM" \
        --label "org.opencontainers.image.title=vLLM ROCm base" \
        --label "org.opencontainers.image.revision=${BUILDKITE_COMMIT:-}" \
        --label "vllm.rocm_base.content_hash=${base_hash}" \
        --label "vllm.rocm_base.descriptor=${descriptor}" \
        --label "vllm.rocm_base.dependency_summary=${dependency_summary}" \
        --label "vllm.rocm_base.base_image=${base_image_arg}" \
        --label "vllm.rocm_base.rocm_version=${rocm_version}" \
        --label "vllm.rocm_base.triton_branch=${triton_arg}" \
        --label "vllm.rocm_base.pytorch_branch=${pytorch_arg}" \
        --label "vllm.rocm_base.pytorch_vision_branch=${pytorch_vision_arg}" \
        --label "vllm.rocm_base.pytorch_audio_branch=${pytorch_audio_arg}" \
        --label "vllm.rocm_base.flash_attention_branch=${fa_arg}" \
        --label "vllm.rocm_base.aiter_branch=${aiter_arg}" \
        --label "vllm.rocm_base.mori_branch=${mori_arg}" \
        "${tags[@]}" \
        --push \
        .

    docker buildx imagetools inspect "${descriptive_tag}" >/dev/null

    metadata_set "rocm-base-refresh" "1"
    metadata_set "rocm-base-image" "${descriptive_tag}"
    metadata_set "rocm-base-image-descriptive" "${descriptive_tag}"
    metadata_set "rocm-base-image-stable" "${stable_tag}"
    metadata_set "rocm-base-content-hash" "${base_hash}"
    metadata_set "rocm-base-descriptor" "${descriptor}"
    metadata_set "rocm-base-dependency-summary" "${dependency_summary}"
    metadata_set "rocm-ci-image-descriptive" "${ci_descriptive_tag}"

    echo "--- :white_check_mark: ROCm base image published"
    echo "Use BASE_IMAGE=${descriptive_tag} for downstream ROCm CI builds"
}

main() {
    metadata_set "rocm-base-refresh" "0"

    if ! rocm_base_changed; then
        echo "ROCm base Dockerfile did not change; skipping base image refresh"
        return 0
    fi

    setup_builder
    build_base_image
}

main "$@"
