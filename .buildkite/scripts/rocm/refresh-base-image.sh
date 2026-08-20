#!/usr/bin/env bash
# Select an immutable ROCm base image, building only on an exact registry miss.

set -euo pipefail

DOCKERFILE="${ROCM_BASE_DOCKERFILE:-docker/Dockerfile.rocm_base}"
BASE_REPO="${ROCM_BASE_IMAGE_REPO:-rocm/vllm-dev}"
CACHE_REPO="${ROCM_BASE_CACHE_REPO:-${DOCKERHUB_CACHE_REPO:-rocm/vllm-ci-cache}}"
BUILDER_NAME="${ROCM_BASE_BUILDER_NAME:-vllm-rocm-base-builder}"
DEFAULT_ROCM_BASE_METADATA_VERSION="2"
DEFAULT_ROCM_BASE_CONTENT_FILES="${DOCKERFILE}"
DEFAULT_ROCM_BASE_CONTENT_ARGS="BASE_IMAGE TRITON_BRANCH TRITON_REPO PYTORCH_BRANCH PYTORCH_REPO PYTORCH_VISION_BRANCH PYTORCH_VISION_REPO PYTORCH_AUDIO_BRANCH PYTORCH_AUDIO_REPO FA_BRANCH FA_REPO AITER_BRANCH AITER_REPO MORI_BRANCH MORI_REPO PYTORCH_ROCM_ARCH PYTHON_VERSION USE_SCCACHE SCCACHE_DOWNLOAD_URL SCCACHE_BUCKET_NAME SCCACHE_REGION_NAME SCCACHE_S3_NO_CREDENTIALS"
DEFAULT_ROCM_BASE_BUILD_ARGS="${DEFAULT_ROCM_BASE_CONTENT_ARGS} SCCACHE_ENDPOINT"

ROCM_BASE_LAYER_CACHE_REF=""
ROCM_BASE_TRUSTED_LAYER_CACHE_REF="${CACHE_REPO}:rocm-base-main"
declare -a ROCM_BASE_CACHE_ARGS=()

metadata_set() {
    local key="$1"
    local value="$2"

    [[ -n "${value}" ]] || return 0
    if command -v buildkite-agent >/dev/null 2>&1; then
        buildkite-agent meta-data set "${key}" "${value}"
    elif [[ "${BUILDKITE:-false}" == "true" ]]; then
        echo "buildkite-agent is required to publish ROCm base metadata" >&2
        return 1
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

normalize_repo_slug() {
    local repo_slug="${1:-}"

    repo_slug="${repo_slug%/}"
    repo_slug="${repo_slug%.git}"
    repo_slug="${repo_slug#https://github.com/}"
    repo_slug="${repo_slug#http://github.com/}"
    repo_slug="${repo_slug#ssh://git@github.com/}"
    repo_slug="${repo_slug#git@github.com:}"
    repo_slug="${repo_slug#github.com/}"
    printf '%s\n' "${repo_slug}"
}

is_trusted_main_build() {
    local actual_repo=""
    local stable_branch="${ROCM_BASE_STABLE_BRANCH:-${CI_BASE_STABLE_BRANCH:-main}}"
    local stable_repo="${ROCM_BASE_STABLE_REPO_SLUG:-${CI_BASE_STABLE_REPO_SLUG:-vllm-project/vllm}}"
    local trusted_repo=""

    [[ "${BUILDKITE:-false}" == "true" ]] || return 1
    [[ "${BUILDKITE_PULL_REQUEST:-false}" == "false" ]] || return 1
    [[ "${BUILDKITE_BRANCH:-}" == "${stable_branch}" ]] || return 1
    actual_repo=$(normalize_repo_slug "${BUILDKITE_REPO:-}")
    trusted_repo=$(normalize_repo_slug "${stable_repo}")
    [[ -n "${actual_repo}" && "${actual_repo}" == "${trusted_repo}" ]]
}

rocm_base_layer_cache_scope() {
    local pull_request="${BUILDKITE_PULL_REQUEST:-false}"
    local branch="${BUILDKITE_PULL_REQUEST_HEAD_BRANCH:-${BUILDKITE_BRANCH:-local}}"
    local identity=""
    local repo_slug=""

    if is_trusted_main_build; then
        printf 'main\n'
        return 0
    fi
    if [[ "${pull_request}" != "false" && -n "${pull_request}" ]]; then
        repo_slug=$(normalize_repo_slug "${BUILDKITE_REPO:-local}")
        identity=$(printf '%s\n' "${repo_slug:-local}" | sha256sum | cut -c1-12)
        printf 'pr-%s-%s\n' \
            "$(tag_component "${pull_request}" 32)" "${identity}"
        return 0
    fi

    identity=$(printf '%s\n%s\n' "${BUILDKITE_REPO:-local}" "${branch}" \
        | sha256sum | cut -c1-12)
    printf 'preview-%s-%s\n' "$(tag_component "${branch}" 24)" "${identity}"
}

configure_rocm_base_layer_cache() {
    local scope=""

    ROCM_BASE_CACHE_ARGS=()
    if [[ "${ROCM_BASE_NO_CACHE:-0}" == "1" ]]; then
        ROCM_BASE_CACHE_ARGS+=(--no-cache)
        ROCM_BASE_LAYER_CACHE_REF="disabled"
        return 0
    fi

    scope=$(rocm_base_layer_cache_scope)
    ROCM_BASE_LAYER_CACHE_REF="${CACHE_REPO}:rocm-base-${scope}"
    ROCM_BASE_CACHE_ARGS+=(
        --cache-from "type=registry,ref=${ROCM_BASE_LAYER_CACHE_REF}"
    )
    if [[ "${ROCM_BASE_LAYER_CACHE_REF}" != \
        "${ROCM_BASE_TRUSTED_LAYER_CACHE_REF}" ]]; then
        ROCM_BASE_CACHE_ARGS+=(
            --cache-from "type=registry,ref=${ROCM_BASE_TRUSTED_LAYER_CACHE_REF}"
        )
    fi
    ROCM_BASE_CACHE_ARGS+=(
        --cache-to \
        "type=registry,ref=${ROCM_BASE_LAYER_CACHE_REF},mode=max,ignore-error=true"
    )
}

extract_arg_default() {
    local arg_name="$1"

    sed -n -E "s/^[[:space:]]*ARG[[:space:]]+${arg_name}=\"?([^\"[:space:]]+)\"?.*/\\1/p" \
        "${DOCKERFILE}" | head -1
}

# 0: found; 1: confirmed missing; 2: registry/parsing failure.
resolve_image_digest() {
    local image_ref="$1"
    local attempts="${ROCM_IMAGE_DIGEST_ATTEMPTS:-4}"
    local delay_secs="${ROCM_IMAGE_DIGEST_RETRY_DELAY:-2}"
    local attempt=0
    local output=""
    local digest=""
    local status=0

    if [[ "${image_ref}" =~ @(sha256:[0-9a-f]{64})$ ]]; then
        printf '%s\n' "${BASH_REMATCH[1]}"
        return 0
    fi
    if [[ ! "${attempts}" =~ ^[1-9][0-9]*$ \
        || ! "${delay_secs}" =~ ^[0-9]+$ ]]; then
        echo "Invalid image digest retry configuration" >&2
        return 2
    fi

    for ((attempt = 1; attempt <= attempts; attempt++)); do
        status=0
        output=$(docker buildx imagetools inspect "${image_ref}" 2>&1) || status=$?
        digest=$(awk '$1 == "Digest:" { print $2; exit }' <<< "${output}")
        if ((status == 0)) && [[ "${digest}" =~ ^sha256:[0-9a-f]{64}$ ]]; then
            printf '%s\n' "${digest}"
            return 0
        fi
        if ((status != 0)) && { grep -Eqi \
            'manifest[ _]unknown|name[ _]unknown|no such manifest|unexpected status from (HEAD|GET) request to https?://[^[:space:]]+/v2/[^[:space:]]+/manifests/[^[:space:]]+:[[:space:]]*404([^0-9]|$)' \
            <<< "${output}" \
            || grep -Fqix -- "ERROR: ${image_ref}: not found" <<< "${output}" \
            || grep -Fqix -- "ERROR: docker.io/${image_ref#docker.io/}: not found" \
                <<< "${output}"; }; then
            return 1
        fi
        if ((attempt < attempts)); then
            printf \
                'Image digest lookup failed for %s (%d/%d, status %d); retrying\n' \
                "${image_ref}" "${attempt}" "${attempts}" "${status}" >&2
            sleep "${delay_secs}"
        fi
    done

    printf 'Failed to resolve digest for %s (status %d)\n%s\n' \
        "${image_ref}" "${status}" "${output:-<no output>}" >&2
    return 2
}

canonical_pinned_image_ref() {
    local image_ref="$1"
    local digest="$2"
    local repository="${image_ref%@*}"
    local last_component="${repository##*/}"

    [[ "${digest}" =~ ^sha256:[0-9a-f]{64}$ ]] || return 1
    if [[ "${last_component}" == *:* ]]; then
        repository="${repository%:*}"
    fi
    [[ -n "${repository}" ]] || return 1
    printf '%s@%s\n' "${repository}" "${digest}"
}

trusted_base_content_ref() {
    local base_hash="$1"

    printf '%s:base-%s\n' "${BASE_REPO}" "${base_hash}"
}

scoped_base_content_ref() {
    local base_hash="$1"
    local scope=""

    scope=$(tag_component "$(rocm_base_layer_cache_scope)" 40)
    printf '%s:base-%s-%s\n' "${BASE_REPO}" "${scope}" "${base_hash}"
}

find_matching_base_content_ref() {
    local expected_hash="$1"
    local expected_version="$2"
    local image_ref=""
    local digest=""
    local immutable_ref=""
    local label_values=""
    local remote_hash=""
    local remote_version=""
    local attempts="${ROCM_IMAGE_DIGEST_ATTEMPTS:-4}"
    local delay_secs="${ROCM_IMAGE_DIGEST_RETRY_DELAY:-2}"
    local attempt=0
    local digest_status=0
    local inspect_status=0
    shift 2

    if [[ ! "${attempts}" =~ ^[1-9][0-9]*$ \
        || ! "${delay_secs}" =~ ^[0-9]+$ ]]; then
        echo "Invalid image identity retry configuration" >&2
        return 3
    fi

    for image_ref in "$@"; do
        digest_status=0
        digest=$(resolve_image_digest "${image_ref}") || digest_status=$?
        case "${digest_status}" in
            0) immutable_ref="${image_ref}@${digest}" ;;
            1) continue ;;
            *) return 3 ;;
        esac
        label_values=""
        for ((attempt = 1; attempt <= attempts; attempt++)); do
            inspect_status=0
            label_values=$(docker buildx imagetools inspect "${immutable_ref}" \
                --format '{{ index .Image.Config.Labels "vllm.rocm_base.content_hash" }} {{ index .Image.Config.Labels "vllm.rocm_base.metadata_version" }}' \
                2>/dev/null) || inspect_status=$?
            remote_hash=""
            remote_version=""
            if ((inspect_status == 0)) && [[ -n "${label_values}" ]]; then
                read -r remote_hash remote_version _ <<< "${label_values}"
                if [[ "${remote_hash}" == "${expected_hash}" \
                    && "${remote_version}" == "${expected_version}" ]]; then
                    printf '%s\n' "${immutable_ref}"
                    return 0
                fi
            fi
            if ((attempt < attempts)); then
                printf \
                    'ROCm base identity lookup incomplete or mismatched for %s (%d/%d); retrying\n' \
                    "${immutable_ref}" "${attempt}" "${attempts}" >&2
                sleep "${delay_secs}"
            fi
        done
        if ((inspect_status == 0)) \
            && [[ -n "${remote_hash}" && -n "${remote_version}" ]]; then
            echo "ROCm base content ref has unexpected identity: ${immutable_ref}" >&2
            echo "  expected hash/version: ${expected_hash} / ${expected_version}" >&2
            echo "  found hash/version:    ${remote_hash} / ${remote_version}" >&2
            return 2
        fi
        echo "Failed to read complete ROCm base identity labels: ${immutable_ref}" >&2
        return 3
    done
    return 1
}

resolve_rocm_base_arg_value() {
    local arg_name="$1"
    local use_sccache="$2"

    case "${arg_name}" in
        USE_SCCACHE)
            printf '%s\n' "${use_sccache}"
            ;;
        *)
            if [[ -v "${arg_name}" ]]; then
                printf '%s\n' "${!arg_name}"
            else
                extract_arg_default "${arg_name}"
            fi
            ;;
    esac
}

hash_rocm_base_arg_values() {
    local use_sccache="$1"
    local pinned_base_image="$2"
    local arg_name=""
    local arg_value=""
    shift 2 || true

    for arg_name in "$@"; do
        [[ -n "${arg_name}" ]] || continue
        if [[ "${arg_name}" == "BASE_IMAGE" ]]; then
            arg_value="${pinned_base_image}"
        else
            arg_value=$(resolve_rocm_base_arg_value "${arg_name}" "${use_sccache}")
        fi
        printf 'arg:%s=%s\n' "${arg_name}" "${arg_value:-<empty>}"
    done
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
    local pinned_base_image="$2"
    local metadata_version="${3:-${ROCM_BASE_METADATA_VERSION:-${DEFAULT_ROCM_BASE_METADATA_VERSION}}}"
    local content_files="${ROCM_BASE_CONTENT_FILES:-${DEFAULT_ROCM_BASE_CONTENT_FILES}}"
    local content_args="${ROCM_BASE_CONTENT_ARGS:-${DEFAULT_ROCM_BASE_CONTENT_ARGS}}"
    local -a content_paths=()
    local -a content_arg_names=()

    read -r -a content_paths <<< "${content_files}"
    read -r -a content_arg_names <<< "${content_args}"

    {
        printf 'metadata-version:%s\n' "${metadata_version}"
        printf 'content-files-hash:%s\n' "$(compute_content_hash "${content_paths[@]}")"
        printf 'dockerfile:%s\n' "${DOCKERFILE}"
        printf 'resolved-build-args:\n'
        hash_rocm_base_arg_values \
            "${use_sccache}" "${pinned_base_image}" "${content_arg_names[@]}"
    } | sha256sum | cut -d' ' -f1
}

build_base_image() {
    local use_sccache="${ROCM_BASE_USE_SCCACHE:-${USE_SCCACHE:-0}}"
    local base_hash=""
    local base_image_arg=""
    local base_image_digest=""
    local pinned_base_image=""
    local rocm_version=""
    local triton_arg=""
    local pytorch_arg=""
    local pytorch_vision_arg=""
    local pytorch_audio_arg=""
    local fa_arg=""
    local aiter_arg=""
    local mori_arg=""
    local python_version_arg=""
    local pytorch_rocm_arch_arg=""
    local dependency_summary=""
    local stable_tag="${BASE_REPO}:base"
    local trusted_content_tag=""
    local scoped_content_tag=""
    local writable_content_tag=""
    local build_ref=""
    local immutable_ref=""
    local reuse_status=0
    local cache_hit=0
    local build_required=0
    local content_files="${ROCM_BASE_CONTENT_FILES:-${DEFAULT_ROCM_BASE_CONTENT_FILES}}"
    local content_files_hash=""
    local metadata_version="${ROCM_BASE_METADATA_VERSION:-${DEFAULT_ROCM_BASE_METADATA_VERSION}}"
    local build_arg_names_config="${ROCM_BASE_BUILD_ARGS:-${DEFAULT_ROCM_BASE_BUILD_ARGS}}"
    local arg_name=""
    local arg_value=""
    local -a build_args=()
    local -a build_arg_names=()
    local -a content_paths=()

    if [[ ! -f "${DOCKERFILE}" ]]; then
        echo "Error: ROCm base Dockerfile not found: ${DOCKERFILE}" >&2
        exit 1
    fi
    if [[ ! "${metadata_version}" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]{0,15}$ ]]; then
        echo "Invalid ROCm base metadata version: ${metadata_version}" >&2
        return 1
    fi
    if [[ "${use_sccache}" != "0" && "${use_sccache}" != "1" ]]; then
        echo "ROCm base USE_SCCACHE must be 0 or 1: ${use_sccache}" >&2
        return 1
    fi

    base_image_arg="$(resolve_rocm_base_arg_value BASE_IMAGE "${use_sccache}")"
    if ! base_image_digest=$(resolve_image_digest "${base_image_arg}"); then
        echo "Failed to resolve ROCm base input image: ${base_image_arg}" >&2
        return 1
    fi
    if ! pinned_base_image=$(canonical_pinned_image_ref \
        "${base_image_arg}" "${base_image_digest}"); then
        echo "Failed to pin ROCm base input image: ${base_image_arg}" >&2
        return 1
    fi
    read -r -a content_paths <<< "${content_files}"
    content_files_hash="$(compute_content_hash "${content_paths[@]}")"
    base_hash=$(compute_base_content_hash \
        "${use_sccache}" "${pinned_base_image}" "${metadata_version}")
    rocm_version="$(tag_component "${base_image_digest}" 16)"
    triton_arg="$(resolve_rocm_base_arg_value TRITON_BRANCH "${use_sccache}")"
    pytorch_arg="$(resolve_rocm_base_arg_value PYTORCH_BRANCH "${use_sccache}")"
    pytorch_vision_arg="$(resolve_rocm_base_arg_value PYTORCH_VISION_BRANCH "${use_sccache}")"
    pytorch_audio_arg="$(resolve_rocm_base_arg_value PYTORCH_AUDIO_BRANCH "${use_sccache}")"
    fa_arg="$(resolve_rocm_base_arg_value FA_BRANCH "${use_sccache}")"
    aiter_arg="$(resolve_rocm_base_arg_value AITER_BRANCH "${use_sccache}")"
    mori_arg="$(resolve_rocm_base_arg_value MORI_BRANCH "${use_sccache}")"
    python_version_arg="$(resolve_rocm_base_arg_value PYTHON_VERSION "${use_sccache}")"
    pytorch_rocm_arch_arg="$(resolve_rocm_base_arg_value PYTORCH_ROCM_ARCH "${use_sccache}")"
    dependency_summary="base=${pinned_base_image},rocm=${rocm_version},python=${python_version_arg},pytorch=${pytorch_arg},torchvision=${pytorch_vision_arg},torchaudio=${pytorch_audio_arg},triton=${triton_arg},flash-attn=${fa_arg},aiter=${aiter_arg},mori=${mori_arg},pytorch-rocm-arch=${pytorch_rocm_arch_arg}"
    trusted_content_tag=$(trusted_base_content_ref "${base_hash}")
    scoped_content_tag=$(scoped_base_content_ref "${base_hash}")
    # Preview writes share an exact-content namespace so stacked PRs can reuse
    # identical images. They may import the trusted ref, but never overwrite it.
    if is_trusted_main_build; then
        scoped_content_tag="${trusted_content_tag}"
        writable_content_tag="${trusted_content_tag}"
    else
        writable_content_tag="${scoped_content_tag}"
    fi

    configure_rocm_base_layer_cache
    read -r -a build_arg_names <<< "${build_arg_names_config}"
    for arg_name in "${build_arg_names[@]}"; do
        [[ -n "${arg_name}" ]] || continue
        if [[ "${arg_name}" == "BASE_IMAGE" ]]; then
            arg_value="${pinned_base_image}"
        else
            arg_value=$(resolve_rocm_base_arg_value "${arg_name}" "${use_sccache}")
        fi
        build_args+=(--build-arg "${arg_name}=${arg_value}")
    done

    echo "--- :docker: Preparing ROCm base image"
    echo "Dockerfile: ${DOCKERFILE}"
    echo "Trusted content tag: ${trusted_content_tag}"
    echo "Writable content tag: ${writable_content_tag}"
    echo "Stable tag: ${stable_tag} (promotion deferred until smoke passes)"
    echo "Content hash: ${base_hash}"
    echo "Dependency summary: ${dependency_summary}"
    echo "USE_SCCACHE: ${use_sccache}"
    echo "BuildKit layer cache: ${ROCM_BASE_LAYER_CACHE_REF}"

    if [[ "${ROCM_BASE_REFRESH_FORCE:-0}" != "1" \
        && "${ROCM_BASE_NO_CACHE:-0}" != "1" ]]; then
        reuse_status=0
        if [[ "${trusted_content_tag}" == "${scoped_content_tag}" ]]; then
            immutable_ref=$(find_matching_base_content_ref \
                "${base_hash}" "${metadata_version}" \
                "${trusted_content_tag}") || reuse_status=$?
        else
            immutable_ref=$(find_matching_base_content_ref \
                "${base_hash}" "${metadata_version}" \
                "${trusted_content_tag}" "${scoped_content_tag}") \
                || reuse_status=$?
        fi
        if [[ ${reuse_status} -gt 1 ]]; then
            return 1
        fi
    else
        reuse_status=1
        echo "Forced/no-cache ROCm base build; bypassing content image reuse"
    fi

    if [[ ${reuse_status} -eq 0 && -n "${immutable_ref}" ]]; then
        cache_hit=1
        echo "Reusing ROCm base image with matching content: ${immutable_ref}"
    else
        if [[ "${ROCM_BASE_REFRESH_SKIP:-0}" == "1" ]]; then
            echo "ROCM_BASE_REFRESH_SKIP=1 but no exact ROCm base image exists" >&2
            return 1
        fi
        build_required=1
        metadata_set "rocm-base-built-in-build" "1"
        build_ref="${writable_content_tag}"
        echo "No reusable ROCm base content image found; building ${build_ref}"
        setup_builder
        docker buildx build \
            "${ROCM_BASE_CACHE_ARGS[@]}" \
            --pull \
            --provenance=false \
            --progress "${BUILDKIT_PROGRESS:-plain}" \
            --file "${DOCKERFILE}" \
            "${build_args[@]}" \
            --label "org.opencontainers.image.source=https://github.com/vllm-project/vllm" \
            --label "org.opencontainers.image.vendor=vLLM" \
            --label "org.opencontainers.image.title=vLLM ROCm base" \
            --label "vllm.rocm_base.metadata_version=${metadata_version}" \
            --label "vllm.rocm_base.content_hash=${base_hash}" \
            --label "vllm.rocm_base.content_files_hash=${content_files_hash}" \
            --label "vllm.rocm_base.dockerfile=${DOCKERFILE}" \
            --label "vllm.rocm_base.dependency_summary=${dependency_summary}" \
            --label "vllm.rocm_base.base_image=${pinned_base_image}" \
            --label "vllm.rocm_base.base_image_digest=${base_image_digest}" \
            --label "vllm.rocm_base.dependency.rocm=${rocm_version}" \
            --label "vllm.rocm_base.dependency.python=${python_version_arg}" \
            --label "vllm.rocm_base.dependency.pytorch=${pytorch_arg}" \
            --label "vllm.rocm_base.dependency.torchvision=${pytorch_vision_arg}" \
            --label "vllm.rocm_base.dependency.torchaudio=${pytorch_audio_arg}" \
            --label "vllm.rocm_base.dependency.triton=${triton_arg}" \
            --label "vllm.rocm_base.dependency.flash_attention=${fa_arg}" \
            --label "vllm.rocm_base.dependency.aiter=${aiter_arg}" \
            --label "vllm.rocm_base.dependency.mori=${mori_arg}" \
            --label "vllm.rocm_base.pytorch_rocm_arch=${pytorch_rocm_arch_arg}" \
            -t "${build_ref}" \
            --push \
            .
        if ! immutable_ref=$(find_matching_base_content_ref \
            "${base_hash}" "${metadata_version}" "${build_ref}"); then
            echo "Published ROCm base image failed identity validation: ${build_ref}" >&2
            return 1
        fi
    fi

    metadata_set "rocm-base-image" "${immutable_ref}"
    metadata_set "rocm-base-content-hash" "${base_hash}"
    metadata_set "rocm-base-image-content" "${immutable_ref%@*}"
    metadata_set "rocm-base-image-stable" "${stable_tag}"
    # Do not overwrite a build latch on retry. If an earlier attempt did not
    # publish an outcome, the missing key makes the consumer build fully.
    if ((build_required == 0)) \
        && [[ "${BUILDKITE_RETRY_COUNT:-0}" == "0" ]]; then
        metadata_set "rocm-base-built-in-build" "0"
    fi
    metadata_set "rocm-base-build-required" "${build_required}"
    metadata_set "rocm-base-cache-hit" "${cache_hit}"

    echo "--- :white_check_mark: ROCm base image published"
    echo "Use BASE_IMAGE=${immutable_ref} for downstream ROCm CI builds"
}

main() {
    build_base_image
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    main "$@"
fi
