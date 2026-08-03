#!/usr/bin/env bash
# Select an immutable ROCm base image, building it only when no exact image exists.

set -euo pipefail

DOCKERFILE="${ROCM_BASE_DOCKERFILE:-docker/Dockerfile.rocm_base}"
BASE_REPO="${ROCM_BASE_IMAGE_REPO:-rocm/vllm-dev}"
CACHE_REPO="${ROCM_BASE_CACHE_REPO:-${DOCKERHUB_CACHE_REPO:-rocm/vllm-ci-cache}}"
BUILDER_NAME="${ROCM_BASE_BUILDER_NAME:-vllm-rocm-base-builder}"
DEFAULT_ROCM_BASE_METADATA_VERSION="2"
DEFAULT_ROCM_BASE_CONTENT_ARGS="BASE_IMAGE TRITON_BRANCH TRITON_REPO PYTORCH_BRANCH PYTORCH_REPO PYTORCH_VISION_BRANCH PYTORCH_VISION_REPO PYTORCH_AUDIO_BRANCH PYTORCH_AUDIO_REPO FA_BRANCH FA_REPO AITER_BRANCH AITER_REPO MORI_BRANCH MORI_REPO PYTORCH_ROCM_ARCH PYTHON_VERSION USE_SCCACHE SCCACHE_DOWNLOAD_URL SCCACHE_ENDPOINT SCCACHE_BUCKET_NAME SCCACHE_REGION_NAME SCCACHE_S3_NO_CREDENTIALS"

declare -A ROCM_BASE_ARG_VALUES=()

metadata_set() {
    local key="$1"
    local value="$2"

    [[ -n "${value}" ]] || return 0
    if command -v buildkite-agent >/dev/null 2>&1; then
        buildkite-agent meta-data set "${key}" "${value}"
    elif [[ "${BUILDKITE:-false}" == "true" ]]; then
        echo "buildkite-agent not found; cannot publish ${key}" >&2
        return 1
    fi
}

clean_docker_tag() {
    local input="$1"
    printf '%s\n' "${input}" | sed 's/[^a-zA-Z0-9._-]/_/g' | cut -c1-128
}

tag_component() {
    clean_docker_tag "${1:-unknown}" | cut -c1-"${2:-24}"
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

extract_arg_default() {
    local arg_name="$1"

    sed -n -E \
        "s/^[[:space:]]*ARG[[:space:]]+${arg_name}=\"?([^\"[:space:]]+)\"?.*/\\1/p" \
        "${DOCKERFILE}" | head -1
}

validate_arg_name() {
    [[ "$1" =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]]
}

resolve_rocm_base_arg_value() {
    local arg_name="$1"
    local use_sccache="$2"

    validate_arg_name "${arg_name}" || {
        echo "Invalid ROCm base build argument name: ${arg_name}" >&2
        return 1
    }
    if [[ "${arg_name}" == "USE_SCCACHE" ]]; then
        printf '%s\n' "${use_sccache}"
    elif [[ -v "${arg_name}" ]]; then
        printf '%s\n' "${!arg_name}"
    else
        extract_arg_default "${arg_name}"
    fi
}

resolve_image_digest() {
    local image_ref="$1"
    local attempts="${ROCM_BASE_IMAGE_DIGEST_ATTEMPTS:-4}"
    local delay_secs="${ROCM_BASE_IMAGE_DIGEST_RETRY_DELAY:-2}"
    local output=""
    local digest=""
    local status=0
    local attempt=0

    if [[ "${image_ref}" =~ @(sha256:[0-9a-f]{64})$ ]]; then
        printf '%s\n' "${BASH_REMATCH[1]}"
        return 0
    fi
    if [[ ! "${attempts}" =~ ^[1-9][0-9]*$ || ! "${delay_secs}" =~ ^[0-9]+$ ]]; then
        echo "Invalid ROCm base digest retry configuration" >&2
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
            echo "Image digest lookup failed for ${image_ref} (${attempt}/${attempts}); retrying" >&2
            sleep "${delay_secs}"
        fi
    done

    printf 'Failed to resolve digest for %s (status %d)\n%s\n' \
        "${image_ref}" "${status}" "${output:-<no output>}" >&2
    return 1
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

rocm_base_scope() {
    local pull_request="${BUILDKITE_PULL_REQUEST:-false}"
    local branch="${BUILDKITE_PULL_REQUEST_HEAD_BRANCH:-${BUILDKITE_BRANCH:-local}}"
    local identity=""
    local repo_slug=""

    if is_trusted_main_build; then
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

compute_base_input_hash() {
    local content_args="${ROCM_BASE_CONTENT_ARGS:-${DEFAULT_ROCM_BASE_CONTENT_ARGS}}"
    local arg_name=""

    {
        printf 'schema:%s\n' "${ROCM_BASE_METADATA_VERSION:-${DEFAULT_ROCM_BASE_METADATA_VERSION}}"
        printf 'dockerfile:%s\n' "${DOCKERFILE}"
        sha256sum "${DOCKERFILE}"
        printf 'parent:%s\n' "${ROCM_BASE_PARENT_PINNED}"
        for arg_name in ${content_args}; do
            if [[ "${arg_name}" == "BASE_IMAGE" ]]; then
                printf 'arg:%s=%s\n' "${arg_name}" "${ROCM_BASE_PARENT_PINNED}"
            else
                printf 'arg:%s=%s\n' \
                    "${arg_name}" "${ROCM_BASE_ARG_VALUES[${arg_name}]:-<empty>}"
            fi
        done
    } | sha256sum | cut -d' ' -f1
}

prepare_base_inputs() {
    local content_args="${ROCM_BASE_CONTENT_ARGS:-${DEFAULT_ROCM_BASE_CONTENT_ARGS}}"
    local arg_name=""
    local scope=""
    local canonical_suffix=""

    [[ -f "${DOCKERFILE}" ]] || {
        echo "ROCm base Dockerfile not found: ${DOCKERFILE}" >&2
        return 1
    }
    ROCM_BASE_USE_SCCACHE_EFFECTIVE="${ROCM_BASE_USE_SCCACHE:-${USE_SCCACHE:-0}}"
    for arg_name in ${content_args}; do
        ROCM_BASE_ARG_VALUES["${arg_name}"]=$(resolve_rocm_base_arg_value \
            "${arg_name}" "${ROCM_BASE_USE_SCCACHE_EFFECTIVE}")
    done

    ROCM_BASE_PARENT_REF="${ROCM_BASE_ARG_VALUES[BASE_IMAGE]:-}"
    [[ -n "${ROCM_BASE_PARENT_REF}" ]] || {
        echo "BASE_IMAGE has no effective value in ${DOCKERFILE}" >&2
        return 1
    }
    ROCM_BASE_PARENT_DIGEST=$(resolve_image_digest "${ROCM_BASE_PARENT_REF}")
    ROCM_BASE_PARENT_PINNED=$(canonical_pinned_image_ref \
        "${ROCM_BASE_PARENT_REF}" "${ROCM_BASE_PARENT_DIGEST}")
    ROCM_BASE_INPUT_HASH=$(compute_base_input_hash)
    [[ "${ROCM_BASE_INPUT_HASH}" =~ ^[0-9a-f]{64}$ ]] || return 1

    scope=$(rocm_base_scope)
    canonical_suffix="input-${ROCM_BASE_INPUT_HASH}"
    if [[ -n "${scope}" ]]; then
        canonical_suffix="${scope}-${canonical_suffix}"
    fi
    ROCM_BASE_CANONICAL_TAG="${BASE_REPO}:base-${canonical_suffix}"
    ROCM_BASE_TRUSTED_TAG="${BASE_REPO}:base-input-${ROCM_BASE_INPUT_HASH}"
    ROCM_BASE_STABLE_TAG="${BASE_REPO}:base"

    if [[ -n "${scope}" ]]; then
        ROCM_BASE_LAYER_CACHE_SCOPE="${scope}"
    else
        ROCM_BASE_LAYER_CACHE_SCOPE="main"
    fi
    ROCM_BASE_LAYER_CACHE_REF="${CACHE_REPO}:rocm-base-${ROCM_BASE_LAYER_CACHE_SCOPE}"
    ROCM_BASE_TRUSTED_LAYER_CACHE_REF="${CACHE_REPO}:rocm-base-main"
}

remote_image_exists() {
    local image_ref="$1"
    local attempts="${ROCM_BASE_IMAGE_LOOKUP_ATTEMPTS:-4}"
    local delay_secs="${ROCM_BASE_IMAGE_LOOKUP_RETRY_DELAY:-2}"
    local output=""
    local status=0
    local attempt=0

    if [[ ! "${attempts}" =~ ^[1-9][0-9]*$ || ! "${delay_secs}" =~ ^[0-9]+$ ]]; then
        echo "Invalid ROCm base image lookup retry configuration" >&2
        return 2
    fi
    for ((attempt = 1; attempt <= attempts; attempt++)); do
        status=0
        output=$(docker buildx imagetools inspect "${image_ref}" 2>&1) || status=$?
        if ((status == 0)); then
            return 0
        fi
        if grep -Eqi \
            'manifest unknown|name unknown|no such manifest|(^|[^0-9])404([^0-9]|$)|(^|[[:space:]])[^[:space:]]+/[^[:space:]]+:[^[:space:]]+:[[:space:]]+not found([[:space:]]|$)' \
            <<< "${output}"; then
            return 1
        fi
        if ((attempt < attempts)); then
            echo "Image lookup failed for ${image_ref} (${attempt}/${attempts}); retrying" >&2
            sleep "${delay_secs}"
        fi
    done

    printf 'Registry lookup failed for %s (status %d)\n%s\n' \
        "${image_ref}" "${status}" "${output:-<no output>}" >&2
    return 2
}

get_remote_image_label() {
    local image_ref="$1"
    local label_key="$2"
    local template=""
    local attempts="${ROCM_BASE_LABEL_LOOKUP_ATTEMPTS:-4}"
    local delay_secs="${ROCM_BASE_LABEL_LOOKUP_RETRY_DELAY:-2}"
    local output=""
    local status=0
    local attempt=0

    [[ "${label_key}" =~ ^[A-Za-z0-9_.-]+$ ]] || return 1
    [[ "${attempts}" =~ ^[1-9][0-9]*$ && "${delay_secs}" =~ ^[0-9]+$ ]] \
        || return 2
    template="{{with .Image.Config.Labels}}{{index . \"${label_key}\"}}{{end}}"
    for ((attempt = 1; attempt <= attempts; attempt++)); do
        status=0
        output=$(docker buildx imagetools inspect \
            "${image_ref}" --format "${template}" 2>&1) || status=$?
        if ((status == 0)); then
            printf '%s\n' "${output}"
            return 0
        fi
        if ((attempt < attempts)); then
            sleep "${delay_secs}"
        fi
    done
    echo "Failed to inspect ${label_key} on ${image_ref}" >&2
    return 2
}

remote_rocm_base_matches() {
    local image_ref="$1"
    local expected_hash="$2"
    local metadata_version=""
    local input_hash=""

    metadata_version=$(get_remote_image_label \
        "${image_ref}" "vllm.rocm_base.metadata_version") || return 2
    input_hash=$(get_remote_image_label \
        "${image_ref}" "vllm.rocm_base.input_hash") || return 2
    [[ "${metadata_version}" == \
        "${ROCM_BASE_METADATA_VERSION:-${DEFAULT_ROCM_BASE_METADATA_VERSION}}" \
        && "${input_hash}" == "${expected_hash}" ]]
}

select_cached_base_image() {
    local candidate=""
    local candidate_digest=""
    local candidate_pinned=""
    local exists_status=0
    local match_status=0
    local previous=""

    SELECTED_BASE_REF=""
    ROCM_BASE_CACHE_SOURCE=""

    for candidate in "${ROCM_BASE_CANONICAL_TAG}" "${ROCM_BASE_TRUSTED_TAG}"; do
        [[ -n "${candidate}" ]] || continue
        [[ "${candidate}" != "${previous}" ]] || continue
        previous="${candidate}"
        exists_status=0
        remote_image_exists "${candidate}" || exists_status=$?
        case "${exists_status}" in
            0) ;;
            1) continue ;;
            *) return 3 ;;
        esac
        candidate_digest=$(resolve_image_digest "${candidate}") || return 3
        candidate_pinned=$(canonical_pinned_image_ref \
            "${candidate}" "${candidate_digest}") || return 3
        match_status=0
        remote_rocm_base_matches \
            "${candidate_pinned}" "${ROCM_BASE_INPUT_HASH}" || match_status=$?
        case "${match_status}" in
            0) ;;
            1)
                echo "ROCm base tag has incompatible metadata: ${candidate}" >&2
                return 2
                ;;
            *) return 3 ;;
        esac
        SELECTED_BASE_REF="${candidate_pinned}"
        if [[ "${candidate}" == "${ROCM_BASE_CANONICAL_TAG}" ]]; then
            ROCM_BASE_CACHE_SOURCE="scope"
        else
            ROCM_BASE_CACHE_SOURCE="trusted-main"
        fi
        return 0
    done

    return 1
}

setup_builder() {
    echo "--- :buildkite: Setting up Buildx builder for ROCm base"
    if docker buildx inspect "${BUILDER_NAME}" >/dev/null 2>&1; then
        docker buildx use "${BUILDER_NAME}"
    else
        docker buildx create --name "${BUILDER_NAME}" --driver docker-container --use
    fi
    docker buildx inspect --bootstrap
}

build_base_image() {
    local content_args="${ROCM_BASE_CONTENT_ARGS:-${DEFAULT_ROCM_BASE_CONTENT_ARGS}}"
    local arg_name=""
    local arg_value=""
    local content_files_hash=""
    local -a build_args=()
    local -a cache_args=()

    for arg_name in ${content_args}; do
        if [[ "${arg_name}" == "BASE_IMAGE" ]]; then
            arg_value="${ROCM_BASE_PARENT_PINNED}"
        else
            arg_value="${ROCM_BASE_ARG_VALUES[${arg_name}]:-}"
        fi
        build_args+=(--build-arg "${arg_name}=${arg_value}")
    done

    if [[ "${ROCM_BASE_NO_CACHE:-0}" == "1" ]]; then
        cache_args+=(--no-cache)
    else
        cache_args+=(--cache-from "type=registry,ref=${ROCM_BASE_LAYER_CACHE_REF}")
        if [[ "${ROCM_BASE_LAYER_CACHE_REF}" != \
            "${ROCM_BASE_TRUSTED_LAYER_CACHE_REF}" ]]; then
            cache_args+=(--cache-from \
                "type=registry,ref=${ROCM_BASE_TRUSTED_LAYER_CACHE_REF}")
        fi
        cache_args+=(--cache-to \
            "type=registry,ref=${ROCM_BASE_LAYER_CACHE_REF},mode=max,ignore-error=true")
    fi
    content_files_hash=$(sha256sum "${DOCKERFILE}" | cut -d' ' -f1)

    echo "--- :docker: Building input-addressed ROCm base image"
    echo "Canonical tag: ${ROCM_BASE_CANONICAL_TAG}"
    echo "Input hash: ${ROCM_BASE_INPUT_HASH}"
    echo "Pinned parent: ${ROCM_BASE_PARENT_PINNED}"
    echo "Layer cache: ${ROCM_BASE_LAYER_CACHE_REF}"

    docker buildx build \
        "${cache_args[@]}" \
        --pull \
        --progress "${BUILDKIT_PROGRESS:-plain}" \
        --file "${DOCKERFILE}" \
        "${build_args[@]}" \
        --label "org.opencontainers.image.source=https://github.com/vllm-project/vllm" \
        --label "org.opencontainers.image.vendor=vLLM" \
        --label "org.opencontainers.image.title=vLLM ROCm base" \
        --label "vllm.rocm_base.metadata_version=${ROCM_BASE_METADATA_VERSION:-${DEFAULT_ROCM_BASE_METADATA_VERSION}}" \
        --label "vllm.rocm_base.input_hash=${ROCM_BASE_INPUT_HASH}" \
        --label "vllm.rocm_base.content_files_hash=${content_files_hash}" \
        --label "vllm.rocm_base.dockerfile=${DOCKERFILE}" \
        --label "vllm.rocm_base.base_image=${ROCM_BASE_PARENT_PINNED}" \
        --label "vllm.rocm_base.base_image_digest=${ROCM_BASE_PARENT_DIGEST}" \
        --label "vllm.rocm_base.image.canonical=${ROCM_BASE_CANONICAL_TAG}" \
        --label "vllm.rocm_base.image.stable=${ROCM_BASE_STABLE_TAG}" \
        --tag "${ROCM_BASE_CANONICAL_TAG}" \
        --push \
        .

    if ! remote_image_exists "${ROCM_BASE_CANONICAL_TAG}" \
        || ! remote_rocm_base_matches \
            "${ROCM_BASE_CANONICAL_TAG}" "${ROCM_BASE_INPUT_HASH}"; then
        echo "Published ROCm base image failed validation" >&2
        return 1
    fi
    SELECTED_BASE_REF="${ROCM_BASE_CANONICAL_TAG}"
    ROCM_BASE_CACHE_SOURCE="built"
}

publish_base_handoff() {
    local cache_hit="$1"
    local build_required="$2"
    local digest=""
    local handoff_ref=""

    digest=$(resolve_image_digest "${SELECTED_BASE_REF}")
    handoff_ref=$(canonical_pinned_image_ref "${SELECTED_BASE_REF}" "${digest}")

    metadata_set "rocm-base-image" "${handoff_ref}"
    metadata_set "rocm-base-input-hash" "${ROCM_BASE_INPUT_HASH}"
    metadata_set "rocm-base-canonical-tag" "${ROCM_BASE_CANONICAL_TAG}"
    metadata_set "rocm-base-stable-tag" "${ROCM_BASE_STABLE_TAG}"
    metadata_set "rocm-base-cache-hit" "${cache_hit}"
    metadata_set "rocm-base-build-required" "${build_required}"

    echo "--- :white_check_mark: Selected immutable ROCm base"
    echo "Image: ${handoff_ref}"
    echo "Cache source: ${ROCM_BASE_CACHE_SOURCE}"
}

main() {
    local cache_hit=0
    local build_required=0
    local lookup_status=0

    prepare_base_inputs
    if [[ "${1:-}" == "--print-input-hash" ]]; then
        printf '%s\n' "${ROCM_BASE_INPUT_HASH}"
        return 0
    fi

    select_cached_base_image || lookup_status=$?
    case "${lookup_status}" in
        0)
            cache_hit=1
            ;;
        1)
            if [[ "${ROCM_BASE_REFRESH_SKIP:-0}" == "1" ]]; then
                echo "Exact ROCm base image is missing and rebuilding is disabled" >&2
                return 1
            fi
            build_required=1
            setup_builder
            build_base_image
            ;;
        *)
            echo "ROCm base image selection failed" >&2
            return 1
            ;;
    esac
    publish_base_handoff "${cache_hit}" "${build_required}"
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    main "$@"
fi
