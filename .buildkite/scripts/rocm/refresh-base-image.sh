#!/usr/bin/env bash
# Select the content-addressed ROCm base for this checkout, building it only
# when the exact content ref is absent.

set -euo pipefail

DOCKERFILE="${ROCM_BASE_DOCKERFILE:-docker/Dockerfile.rocm_base}"
BASE_REPO="${ROCM_BASE_IMAGE_REPO:-rocm/vllm-dev}"
CACHE_REPO="${ROCM_BASE_CACHE_REPO:-${DOCKERHUB_CACHE_REPO:-rocm/vllm-ci-cache}}"
BUILDER_NAME="${ROCM_BASE_BUILDER_NAME:-vllm-rocm-base-builder}"
DEFAULT_ROCM_BASE_METADATA_VERSION="2"
DEFAULT_ROCM_BASE_CONTENT_FILES="${DOCKERFILE}"

ROCM_BASE_LAYER_CACHE_REF=""
ROCM_BASE_TRUSTED_LAYER_CACHE_REF="${CACHE_REPO}:rocm-base-main"
ROCM_BASE_STABLE_TAG_UPDATED=0
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
    local trusted_repo=""

    [[ "${BUILDKITE:-false}" == "true" ]] || return 1
    [[ "${BUILDKITE_PULL_REQUEST:-false}" == "false" ]] || return 1
    [[ "${BUILDKITE_BRANCH:-}" == "${ROCM_BASE_STABLE_BRANCH:-main}" ]] || return 1
    actual_repo=$(normalize_repo_slug "${BUILDKITE_REPO:-}")
    trusted_repo=$(normalize_repo_slug \
        "${ROCM_BASE_STABLE_REPO_SLUG:-vllm-project/vllm}")
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
        return 1
    fi

    for ((attempt = 1; attempt <= attempts; attempt++)); do
        status=0
        output=$(docker buildx imagetools inspect "${image_ref}" 2>&1) || status=$?
        digest=$(awk '$1 == "Digest:" { print $2; exit }' <<< "${output}")
        if ((status == 0)) && [[ "${digest}" =~ ^sha256:[0-9a-f]{64}$ ]]; then
            printf '%s\n' "${digest}"
            return 0
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
    return 1
}

trusted_base_content_ref() {
    local base_hash="$1"
    local metadata_version="$2"

    printf '%s:base-v%s-%s\n' "${BASE_REPO}" "${metadata_version}" "${base_hash}"
}

scoped_base_content_ref() {
    local base_hash="$1"
    local metadata_version="$2"

    printf '%s:base-v%s-preview-%s\n' \
        "${BASE_REPO}" "${metadata_version}" "${base_hash}"
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
    local inspect_status=0
    shift 2

    if [[ ! "${attempts}" =~ ^[1-9][0-9]*$ \
        || ! "${delay_secs}" =~ ^[0-9]+$ ]]; then
        echo "Invalid image identity retry configuration" >&2
        return 1
    fi

    for image_ref in "$@"; do
        if ! digest=$(resolve_image_digest "${image_ref}"); then
            continue
        fi
        immutable_ref="${image_ref}@${digest}"
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
            # A content-addressed ref with the wrong identity is corrupt. The
            # stable alias is allowed to describe older content and is only a
            # migration/recovery candidate.
            if [[ "${image_ref}" != "${BASE_REPO}:base" ]]; then
                return 2
            fi
        fi
        echo "Failed to read complete ROCm base identity labels: ${immutable_ref}" >&2
    done
    return 1
}

rocm_version_from_base_image() {
    local base_image="$1"
    local version=""

    version="$(sed -n -E 's/.*:([0-9]+\.[0-9]+(\.[0-9]+)?)-.*/\1/p' <<<"${base_image}")"
    tag_component "${version:-${base_image}}" 16
}

should_push_stable_tag() {
    if [[ "${BUILDKITE_PULL_REQUEST:-false}" != "false" ]]; then
        return 1
    fi

    if [[ "${ROCM_BASE_PUSH_STABLE_TAG:-}" == "0" ]]; then
        return 1
    fi

    is_trusted_main_build
}

trusted_main_tip_matches_build() {
    local branch="${ROCM_BASE_STABLE_BRANCH:-main}"
    local build_commit="${BUILDKITE_COMMIT:-}"
    local remote_tip=""

    is_trusted_main_build || return 1
    if [[ ! "${build_commit}" =~ ^[0-9a-fA-F]{40}$ ]]; then
        echo "Skipping ROCm stable tag: Buildkite commit is missing or invalid" >&2
        return 1
    fi
    remote_tip=$(git ls-remote --exit-code "${BUILDKITE_REPO}" \
        "refs/heads/${branch}" 2>/dev/null | awk 'NR == 1 { print $1 }')
    if [[ ! "${remote_tip}" =~ ^[0-9a-fA-F]{40}$ ]]; then
        echo "Skipping ROCm stable tag: could not resolve remote ${branch} tip" >&2
        return 1
    fi
    if [[ "${remote_tip,,}" != "${build_commit,,}" ]]; then
        echo "Skipping ROCm stable tag: ${branch} changed after ${build_commit} (${remote_tip})" >&2
        return 1
    fi
    return 0
}

promote_stable_base_tag() {
    local source_ref="$1"
    local stable_tag="$2"

    ROCM_BASE_STABLE_TAG_UPDATED=0
    should_push_stable_tag && trusted_main_tip_matches_build || return 0

    docker buildx imagetools create --prefer-index=false \
        -t "${stable_tag}" "${source_ref}"
    ROCM_BASE_STABLE_TAG_UPDATED=1
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
    local sccache_arg=""
    local sccache_value=""
    local -a content_paths=()

    read -r -a content_paths <<< "${content_files}"

    {
        printf 'content-files-hash:%s\n' "$(compute_content_hash "${content_paths[@]}")"
        printf 'dockerfile:%s\n' "${DOCKERFILE}"
        printf 'base-image-digest:%s\n' "${base_image_digest}"
        printf 'use-sccache:%s\n' "${use_sccache}"
        if [[ "${use_sccache}" == "1" ]]; then
            # These values can change the installed binary or final image
            # configuration. SCCACHE_ENDPOINT is transport-only and therefore
            # deliberately excluded from image identity.
            for sccache_arg in \
                SCCACHE_DOWNLOAD_URL \
                SCCACHE_BUCKET_NAME \
                SCCACHE_REGION_NAME \
                SCCACHE_S3_NO_CREDENTIALS; do
                sccache_value="${!sccache_arg:-}"
                if [[ -z "${sccache_value}" ]]; then
                    sccache_value=$(extract_arg_default "${sccache_arg}")
                fi
                printf 'arg:%s=%s\n' \
                    "${sccache_arg}" "${sccache_value:-<empty>}"
            done
        fi
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
    local selected_digest=""
    local stable_digest=""
    local reuse_status=0
    local content_files="${ROCM_BASE_CONTENT_FILES:-${DEFAULT_ROCM_BASE_CONTENT_FILES}}"
    local content_files_hash=""
    local metadata_version="${ROCM_BASE_METADATA_VERSION:-${DEFAULT_ROCM_BASE_METADATA_VERSION}}"
    local -a sccache_args=()
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

    base_image_arg="$(extract_arg_default BASE_IMAGE)"
    if ! base_image_digest=$(resolve_image_digest "${base_image_arg}"); then
        echo "Failed to resolve ROCm base input image: ${base_image_arg}" >&2
        return 1
    fi
    pinned_base_image="${base_image_arg%@*}@${base_image_digest}"
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
    python_version_arg="$(extract_arg_default PYTHON_VERSION)"
    pytorch_rocm_arch_arg="$(extract_arg_default PYTORCH_ROCM_ARCH)"
    dependency_summary="base=${base_image_arg},rocm=${rocm_version},python=${python_version_arg},pytorch=${pytorch_arg},torchvision=${pytorch_vision_arg},torchaudio=${pytorch_audio_arg},triton=${triton_arg},flash-attn=${fa_arg},aiter=${aiter_arg},mori=${mori_arg},pytorch-rocm-arch=${pytorch_rocm_arch_arg}"
    trusted_content_tag=$(trusted_base_content_ref "${base_hash}" "${metadata_version}")
    scoped_content_tag=$(scoped_base_content_ref "${base_hash}" "${metadata_version}")
    # Preview writes share an exact-content namespace so stacked PRs can reuse
    # identical images. They may import the trusted ref, but never overwrite it.
    if is_trusted_main_build; then
        scoped_content_tag="${trusted_content_tag}"
        writable_content_tag="${trusted_content_tag}"
    else
        writable_content_tag="${scoped_content_tag}"
    fi

    configure_rocm_base_layer_cache

    if [[ "${use_sccache}" == "1" ]]; then
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
    fi

    echo "--- :docker: Preparing ROCm base image"
    echo "Dockerfile: ${DOCKERFILE}"
    echo "Trusted content tag: ${trusted_content_tag}"
    echo "Writable content tag: ${writable_content_tag}"
    echo "Stable tag: ${stable_tag} ($(should_push_stable_tag && echo enabled || echo disabled))"
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
                "${trusted_content_tag}" "${stable_tag}") || reuse_status=$?
        else
            immutable_ref=$(find_matching_base_content_ref \
                "${base_hash}" "${metadata_version}" \
                "${trusted_content_tag}" "${scoped_content_tag}" \
                "${stable_tag}") \
                || reuse_status=$?
        fi
    else
        reuse_status=1
        echo "Forced/no-cache ROCm base build; bypassing content image reuse"
    fi

    if ((reuse_status > 1)); then
        return "${reuse_status}"
    fi

    if [[ ${reuse_status} -eq 0 && -n "${immutable_ref}" ]]; then
        echo "Reusing ROCm base image with matching content: ${immutable_ref}"
        if [[ "${immutable_ref%@*}" == "${stable_tag}" \
            && "${writable_content_tag}" != "${stable_tag}" ]]; then
            echo "Backfilling ROCm base content tag: ${writable_content_tag}"
            docker buildx imagetools create --prefer-index=false \
                -t "${writable_content_tag}" "${immutable_ref}"
            if ! immutable_ref=$(find_matching_base_content_ref \
                "${base_hash}" "${metadata_version}" \
                "${writable_content_tag}"); then
                echo "Backfilled ROCm base content tag failed identity validation" >&2
                return 1
            fi
        fi
    else
        build_ref="${writable_content_tag}"
        echo "No reusable ROCm base content image found; building ${build_ref}"
        setup_builder
        docker buildx build \
            "${ROCM_BASE_CACHE_ARGS[@]}" \
            --pull \
            --provenance=false \
            --progress "${BUILDKIT_PROGRESS:-plain}" \
            --file "${DOCKERFILE}" \
            --build-arg "BASE_IMAGE=${pinned_base_image}" \
            --build-arg "USE_SCCACHE=${use_sccache}" \
            "${sccache_args[@]}" \
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

    selected_digest="${immutable_ref##*@}"
    metadata_set "rocm-base-image" "${immutable_ref}"
    metadata_set "rocm-base-push-stable-tag" "0"
    stable_digest=$(resolve_image_digest "${stable_tag}" 2>/dev/null || true)
    if [[ -n "${stable_digest}" && "${stable_digest}" == "${selected_digest}" ]]; then
        echo "--- :white_check_mark: Existing ROCm base is current"
        echo "Stable ${stable_tag} already resolves to ${selected_digest}"
        return 0
    fi

    promote_stable_base_tag "${immutable_ref}" "${stable_tag}"

    metadata_set "rocm-base-push-stable-tag" "${ROCM_BASE_STABLE_TAG_UPDATED}"
    metadata_set "rocm-base-refresh" "1"

    echo "--- :white_check_mark: ROCm base image published"
    echo "Use BASE_IMAGE=${immutable_ref} for downstream ROCm CI builds"
}

main() {
    metadata_set "rocm-base-refresh" "0"

    if [[ "${ROCM_BASE_REFRESH_SKIP:-0}" == "1" ]]; then
        local stable_ref="${BASE_REPO}:base"
        local stable_digest=""

        if ! stable_digest=$(resolve_image_digest "${stable_ref}"); then
            echo "Could not pin explicitly skipped ROCm base: ${stable_ref}" >&2
            return 1
        fi
        metadata_set "rocm-base-image" "${stable_ref}@${stable_digest}"
        metadata_set "rocm-base-push-stable-tag" "0"
        echo "ROCM_BASE_REFRESH_SKIP=1; pinned current stable base at ${stable_digest}"
        return 0
    fi
    build_base_image
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    main "$@"
fi
