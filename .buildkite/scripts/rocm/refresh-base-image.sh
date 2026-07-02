#!/usr/bin/env bash
# Build and publish a fresh ROCm base image when Dockerfile.rocm_base changes.
#
# Normal AMD CI builds should not pay for this path. The script no-ops unless
# docker/Dockerfile.rocm_base changed relative to the branch base, the previous
# main commit, or ROCM_BASE_REFRESH_FORCE=1 is set.

set -euo pipefail

DOCKERFILE="${ROCM_BASE_DOCKERFILE:-docker/Dockerfile.rocm_base}"
BASE_REPO="${ROCM_BASE_IMAGE_REPO:-rocm/vllm-dev}"
CI_IMAGE_REPO="${ROCM_CI_IMAGE_REPO:-rocm/vllm-ci}"
BUILDER_NAME="${ROCM_BASE_BUILDER_NAME:-vllm-rocm-base-builder}"
DEFAULT_ROCM_BASE_METADATA_VERSION="1"
DEFAULT_ROCM_BASE_CONTENT_FILES="${DOCKERFILE}"
DEFAULT_ROCM_BASE_CONTENT_ARGS="BASE_IMAGE TRITON_BRANCH TRITON_REPO PYTORCH_BRANCH PYTORCH_REPO PYTORCH_VISION_BRANCH PYTORCH_VISION_REPO PYTORCH_AUDIO_BRANCH PYTORCH_AUDIO_REPO FA_BRANCH FA_REPO AITER_BRANCH AITER_REPO MORI_BRANCH MORI_REPO PYTORCH_ROCM_ARCH PYTHON_VERSION USE_SCCACHE"

metadata_set() {
    local key="$1"
    local value="$2"

    [[ -n "${value}" ]] || return 0
    if command -v buildkite-agent >/dev/null 2>&1; then
        buildkite-agent meta-data set "${key}" "${value}" || true
    fi
}

compute_content_hash() {
    local path=""
    local file=""

    for path in "$@"; do
        if [[ -d "${path}" ]]; then
            while IFS= read -r -d '' file; do
                printf 'file:%s\n' "${file}"
                sha256sum "${file}"
            done < <(find "${path}" -type f -print0 | sort -z)
        elif [[ -f "${path}" ]]; then
            printf 'file:%s\n' "${path}"
            sha256sum "${path}"
        else
            printf 'missing:%s\n' "${path}"
        fi
    done | sha256sum | cut -d' ' -f1
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

resolve_image_digest() {
    local image_ref="$1"

    docker buildx imagetools inspect "${image_ref}" 2>/dev/null \
        | sed -n -E 's/^Digest:[[:space:]]+//p' \
        | head -1 || true
}

resolve_rocm_base_arg_value() {
    local arg_name="$1"
    local use_sccache="$2"

    case "${arg_name}" in
        USE_SCCACHE)
            printf '%s\n' "${use_sccache}"
            ;;
        *)
            extract_arg_default "${arg_name}"
            ;;
    esac
}

hash_rocm_base_arg_values() {
    local use_sccache="$1"
    local base_image_digest="$2"
    local arg_name=""
    local arg_value=""
    shift 2 || true

    for arg_name in "$@"; do
        [[ -n "${arg_name}" ]] || continue
        arg_value=$(resolve_rocm_base_arg_value "${arg_name}" "${use_sccache}")
        printf 'arg:%s=%s\n' "${arg_name}" "${arg_value:-<empty>}"
        if [[ "${arg_name}" == "BASE_IMAGE" && -n "${arg_value}" ]]; then
            printf 'arg:%s.digest=%s\n' "${arg_name}" "${base_image_digest:-unknown}"
        fi
    done
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
        echo "Checking ROCm base Dockerfile changes against PR base: ${base_ref}"
        if [[ -n "${merge_base}" ]] && git_diff_changed_base "${merge_base}...HEAD"; then
            return 0
        fi
    elif [[ "${BUILDKITE_BRANCH:-}" == "${ROCM_BASE_STABLE_BRANCH:-main}" ]] \
        && git rev-parse --verify HEAD~1 >/dev/null 2>&1; then
        echo "Checking ROCm base Dockerfile changes against previous ${ROCM_BASE_STABLE_BRANCH:-main} commit"
        if git_diff_changed_base "HEAD~1..HEAD"; then
            return 0
        fi
    else
        git fetch --no-tags --depth=200 origin \
            "+refs/heads/${base_branch}:${base_ref}" >/dev/null 2>&1 || true
        merge_base=$(git merge-base HEAD "${base_ref}" 2>/dev/null || true)
        echo "Checking ROCm base Dockerfile changes against branch base: ${base_ref}"
        if [[ -n "${merge_base}" ]] && git_diff_changed_base "${merge_base}...HEAD"; then
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
    local base_image_digest="$2"
    local content_files="${ROCM_BASE_CONTENT_FILES:-${DEFAULT_ROCM_BASE_CONTENT_FILES}}"
    local content_args="${ROCM_BASE_CONTENT_ARGS:-${DEFAULT_ROCM_BASE_CONTENT_ARGS}}"
    local -a content_paths=()
    local -a content_arg_names=()

    read -r -a content_paths <<< "${content_files}"
    read -r -a content_arg_names <<< "${content_args}"

    {
        printf 'content-files-hash:%s\n' "$(compute_content_hash "${content_paths[@]}")"
        printf 'dockerfile:%s\n' "${DOCKERFILE}"
        printf 'resolved-build-args:\n'
        hash_rocm_base_arg_values \
            "${use_sccache}" "${base_image_digest}" "${content_arg_names[@]}"
    } | sha256sum | cut -d' ' -f1
}

build_base_image() {
    local use_sccache="${ROCM_BASE_USE_SCCACHE:-${USE_SCCACHE:-0}}"
    local base_hash=""
    local build_date=""
    local build_suffix=""
    local base_image_arg=""
    local base_image_digest=""
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
    local content_files="${ROCM_BASE_CONTENT_FILES:-${DEFAULT_ROCM_BASE_CONTENT_FILES}}"
    local content_args="${ROCM_BASE_CONTENT_ARGS:-${DEFAULT_ROCM_BASE_CONTENT_ARGS}}"
    local content_files_hash=""
    local metadata_version="${ROCM_BASE_METADATA_VERSION:-${DEFAULT_ROCM_BASE_METADATA_VERSION}}"
    local git_branch="${BUILDKITE_BRANCH:-${VLLM_BRANCH:-}}"
    local -a tags=()
    local -a no_cache_args=()
    local -a sccache_args=()
    local -a content_paths=()

    if [[ ! -f "${DOCKERFILE}" ]]; then
        echo "Error: ROCm base Dockerfile not found: ${DOCKERFILE}" >&2
        exit 1
    fi

    build_date="${ROCM_BASE_TAG_DATE:-$(date -u +%Y%m%d)}"
    if [[ -n "${BUILDKITE_BUILD_NUMBER:-}" ]]; then
        build_suffix="_bk_${BUILDKITE_BUILD_NUMBER}"
    fi
    base_image_arg="$(extract_arg_default BASE_IMAGE)"
    base_image_digest="$(resolve_image_digest "${base_image_arg}")"
    read -r -a content_paths <<< "${content_files}"
    content_files_hash="$(compute_content_hash "${content_paths[@]}")"
    base_hash=$(compute_base_content_hash "${use_sccache}" "${base_image_digest}")
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
    descriptor="$(clean_docker_tag "base_custom_aiter_${aiter_branch}_torch_${pytorch_branch}_${build_date}${build_suffix}")"
    ci_descriptor="$(clean_docker_tag "ci_custom_aiter_${aiter_branch}_torch_${pytorch_branch}_${build_date}${build_suffix}")"

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
    echo "Content hash: ${base_hash}"
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
        --label "vllm.rocm_base.metadata_version=${metadata_version}" \
        --label "vllm.rocm_base.content_hash=${base_hash}" \
        --label "vllm.rocm_base.content_files_hash=${content_files_hash}" \
        --label "vllm.rocm_base.content_files=${content_files}" \
        --label "vllm.rocm_base.content_args=${content_args}" \
        --label "vllm.rocm_base.dockerfile=${DOCKERFILE}" \
        --label "vllm.rocm_base.image.descriptive=${descriptive_tag}" \
        --label "vllm.rocm_base.image.stable=${stable_tag}" \
        --label "vllm.rocm_base.image.ci_descriptive=${ci_descriptive_tag}" \
        --label "vllm.rocm_base.git_commit=${BUILDKITE_COMMIT:-}" \
        --label "vllm.rocm_base.git_branch=${git_branch}" \
        --label "vllm.rocm_base.stable_branch=${ROCM_BASE_STABLE_BRANCH:-main}" \
        --label "vllm.rocm_base.descriptor=${descriptor}" \
        --label "vllm.rocm_base.ci_descriptor=${ci_descriptor}" \
        --label "vllm.rocm_base.dependency_summary=${dependency_summary}" \
        --label "vllm.rocm_base.base_image=${base_image_arg}" \
        --label "vllm.rocm_base.base_image_digest=${base_image_digest}" \
        --label "vllm.rocm_base.rocm_version=${rocm_version}" \
        --label "vllm.rocm_base.triton_branch=${triton_arg}" \
        --label "vllm.rocm_base.pytorch_branch=${pytorch_arg}" \
        --label "vllm.rocm_base.pytorch_vision_branch=${pytorch_vision_arg}" \
        --label "vllm.rocm_base.pytorch_audio_branch=${pytorch_audio_arg}" \
        --label "vllm.rocm_base.flash_attention_branch=${fa_arg}" \
        --label "vllm.rocm_base.aiter_branch=${aiter_arg}" \
        --label "vllm.rocm_base.mori_branch=${mori_arg}" \
        --label "vllm.rocm_base.use_sccache=${use_sccache}" \
        "${tags[@]}" \
        --push \
        .

    docker buildx imagetools inspect "${descriptive_tag}" >/dev/null

    metadata_set "rocm-base-refresh" "1"
    metadata_set "rocm-base-image" "${descriptive_tag}"
    metadata_set "rocm-base-image-descriptive" "${descriptive_tag}"
    metadata_set "rocm-base-image-stable" "${stable_tag}"
    metadata_set "rocm-base-image-ci-descriptive" "${ci_descriptive_tag}"
    metadata_set "rocm-base-metadata-version" "${metadata_version}"
    metadata_set "rocm-base-content-hash" "${base_hash}"
    metadata_set "rocm-base-content-files-hash" "${content_files_hash}"
    metadata_set "rocm-base-content-files" "${content_files}"
    metadata_set "rocm-base-content-args" "${content_args}"
    metadata_set "rocm-base-base-image-digest" "${base_image_digest}"
    metadata_set "rocm-base-dockerfile" "${DOCKERFILE}"
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
