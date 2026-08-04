#!/usr/bin/env bash
# Select an immutable ROCm base image, building only on an exact registry miss.

set -euo pipefail

DOCKERFILE="${ROCM_BASE_DOCKERFILE:-docker/Dockerfile.rocm_base}"
BASE_REPO="${ROCM_BASE_IMAGE_REPO:-rocm/vllm-dev}"
CACHE_REPO="${ROCM_BASE_CACHE_REPO:-${DOCKERHUB_CACHE_REPO:-rocm/vllm-ci-cache}}"
BUILDER_NAME="${ROCM_BASE_BUILDER_NAME:-vllm-rocm-base-builder}"
DEFAULT_ROCM_BASE_METADATA_VERSION="2"
DEFAULT_ROCM_BASE_CONTENT_FILES="${DOCKERFILE}"
DEFAULT_ROCM_BASE_CONTENT_ARGS="BASE_IMAGE TRITON_BRANCH TRITON_REPO PYTORCH_BRANCH PYTORCH_REPO PYTORCH_VISION_BRANCH PYTORCH_VISION_REPO PYTORCH_AUDIO_BRANCH PYTORCH_AUDIO_REPO FA_BRANCH FA_REPO AITER_BRANCH AITER_REPO MORI_BRANCH MORI_REPO PYTORCH_ROCM_ARCH PYTHON_VERSION USE_SCCACHE SCCACHE_DOWNLOAD_URL SCCACHE_ENDPOINT SCCACHE_BUCKET_NAME SCCACHE_REGION_NAME SCCACHE_S3_NO_CREDENTIALS"

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

remote_image_exists() {
    local image_ref="$1"
    local attempts="${ROCM_BASE_IMAGE_LOOKUP_ATTEMPTS:-4}"
    local delay_secs="${ROCM_BASE_IMAGE_LOOKUP_RETRY_DELAY:-2}"
    local output=""
    local status=0
    local attempt=0

    if [[ ! "${attempts}" =~ ^[1-9][0-9]*$ \
        || ! "${delay_secs}" =~ ^[0-9]+$ ]]; then
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

trusted_base_content_ref() {
    local base_hash="$1"
    local metadata_version="$2"

    printf '%s:base-v%s-%s\n' "${BASE_REPO}" "${metadata_version}" "${base_hash}"
}

scoped_base_content_ref() {
    local base_hash="$1"
    local metadata_version="$2"
    local scope=""

    scope=$(tag_component "$(rocm_base_layer_cache_scope)" 40)
    printf '%s:base-v%s-%s-%s\n' \
        "${BASE_REPO}" "${metadata_version}" "${scope}" "${base_hash}"
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
    local exists_status=0
    local inspect_status=0
    shift 2

    if [[ ! "${attempts}" =~ ^[1-9][0-9]*$ \
        || ! "${delay_secs}" =~ ^[0-9]+$ ]]; then
        echo "Invalid image identity retry configuration" >&2
        return 3
    fi

    for image_ref in "$@"; do
        exists_status=0
        remote_image_exists "${image_ref}" || exists_status=$?
        case "${exists_status}" in
            0) ;;
            1) continue ;;
            *) return 3 ;;
        esac
        if ! digest=$(resolve_image_digest "${image_ref}"); then
            return 3
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

rocm_version_from_base_image() {
    local base_image="$1"
    local version=""

    version="$(sed -n -E 's/.*:([0-9]+\.[0-9]+(\.[0-9]+)?)-.*/\1/p' <<<"${base_image}")"
    tag_component "${version:-${base_image}}" 16
}

git_diff_changed_base() {
    local range="$1"
    local changed_files=""
    local status=0

    changed_files=$(git diff --name-only "${range}" -- "${DOCKERFILE}" 2>/dev/null) \
        || status=$?
    if ((status != 0)); then
        echo "Unable to compare ROCm base inputs over ${range}" >&2
        return 2
    fi
    [[ -n "${changed_files}" ]]
}

short_git_ref() {
    local ref="$1"

    git rev-parse --short "${ref}" 2>/dev/null || printf '%s\n' "${ref}"
}

extract_arg_default_from_ref() {
    local ref="$1"
    local arg_name="$2"
    local content=""

    content="$(git show "${ref}:${DOCKERFILE}" 2>/dev/null || true)"
    sed -n -E "s/^[[:space:]]*ARG[[:space:]]+${arg_name}=\"?([^\"[:space:]]+)\"?.*/\\1/p" \
        <<<"${content}" | head -1
}

log_arg_default_changes() {
    local old_ref="$1"
    local new_ref="$2"
    local content_args="${ROCM_BASE_CONTENT_ARGS:-${DEFAULT_ROCM_BASE_CONTENT_ARGS}}"
    local arg_name=""
    local old_value=""
    local new_value=""
    local changed=0

    echo "Changed ROCm base ARG defaults:"
    for arg_name in ${content_args}; do
        old_value="$(extract_arg_default_from_ref "${old_ref}" "${arg_name}")"
        new_value="$(extract_arg_default_from_ref "${new_ref}" "${arg_name}")"
        if [[ "${old_value}" != "${new_value}" ]]; then
            echo "  - ${arg_name}: ${old_value:-<unset>} -> ${new_value:-<unset>}"
            changed=1
        fi
    done

    if [[ "${changed}" == "0" ]]; then
        echo "  - none detected; Dockerfile instructions changed outside tracked ARG defaults"
    fi
}

log_arg_line_diff() {
    local range="$1"
    local arg_diff=""

    arg_diff="$(
        git diff --unified=0 "${range}" -- "${DOCKERFILE}" 2>/dev/null \
            | awk '/^[+-][[:space:]]*ARG[[:space:]]/ && $0 !~ /^(---|\+\+\+)/ { print "  " $0 }' \
            || true
    )"

    if [[ -n "${arg_diff}" ]]; then
        echo "Changed Dockerfile ARG lines:"
        printf '%s\n' "${arg_diff}"
    fi
}

log_rocm_base_change_check() {
    local context="$1"
    local range="$2"
    local old_ref="$3"
    local old_short=""
    local head_short=""

    old_short="$(short_git_ref "${old_ref}")"
    head_short="$(short_git_ref HEAD)"

    echo "--- :mag: ROCm base refresh check"
    echo "Context: ${context}"
    echo "Dockerfile: ${DOCKERFILE}"
    echo "Base revision: ${old_short}"
    echo "Head revision: ${head_short}"
    echo "Git diff range: ${range}"
}

log_rocm_base_rebuild_reason() {
    local context="$1"
    local range="$2"
    local old_ref="$3"
    local changed_files=""

    log_rocm_base_change_check "${context}" "${range}" "${old_ref}"

    changed_files="$(git diff --name-only "${range}" -- "${DOCKERFILE}" 2>/dev/null || true)"
    echo "Changed files:"
    if [[ -n "${changed_files}" ]]; then
        printf '  - %s\n' "${changed_files//$'\n'/$'\n  - '}"
    else
        echo "  - ${DOCKERFILE}"
    fi
    log_arg_default_changes "${old_ref}" HEAD
    log_arg_line_diff "${range}"
    echo "Decision: rebuilding ROCm base image because ${DOCKERFILE} changed."
}

rocm_base_changed_in_range() {
    local context="$1"
    local range="$2"
    local old_ref="$3"
    local diff_status=0

    git_diff_changed_base "${range}" || diff_status=$?
    case "${diff_status}" in
        0)
            log_rocm_base_rebuild_reason "${context}" "${range}" "${old_ref}"
            return 0
            ;;
        1)
            log_rocm_base_change_check "${context}" "${range}" "${old_ref}"
            echo "Decision: ROCm base refresh not required; ${DOCKERFILE} is unchanged."
            return 1
            ;;
        *)
            echo "ROCm base refresh check failed; refusing to treat the base as unchanged" >&2
            return 2
            ;;
    esac
}

find_base_merge_base() {
    local base_branch="$1"
    local base_ref="$2"
    local initial_depth="${ROCM_BASE_DIFF_FETCH_DEPTH:-200}"
    local deepen_by="${ROCM_BASE_DIFF_FETCH_DEEPEN:-1000}"
    local merge_base=""

    if [[ ! "${initial_depth}" =~ ^[1-9][0-9]*$ \
        || ! "${deepen_by}" =~ ^[1-9][0-9]*$ ]]; then
        echo "Invalid ROCm base diff fetch depth configuration" >&2
        return 2
    fi
    if ! git fetch --no-tags --depth="${initial_depth}" origin \
        "+refs/heads/${base_branch}:${base_ref}" >/dev/null 2>&1; then
        echo "Unable to fetch base branch ${base_branch} for ROCm base comparison" >&2
        return 2
    fi

    merge_base=$(git merge-base HEAD "${base_ref}" 2>/dev/null || true)
    if [[ -n "${merge_base}" ]]; then
        printf '%s\n' "${merge_base}"
        return 0
    fi

    if git rev-parse --is-shallow-repository 2>/dev/null | grep -qx true; then
        echo "Deepening checkout by ${deepen_by} commits for ROCm base comparison" >&2
        if ! git fetch --no-tags --deepen="${deepen_by}" origin \
            "+refs/heads/${base_branch}:${base_ref}" >/dev/null 2>&1; then
            echo "Unable to deepen checkout for ROCm base comparison" >&2
            return 2
        fi
        merge_base=$(git merge-base HEAD "${base_ref}" 2>/dev/null || true)
    fi

    if [[ -z "${merge_base}" ]]; then
        echo "Unable to determine merge base with ${base_ref} after bounded fetch/deepen" >&2
        return 2
    fi
    printf '%s\n' "${merge_base}"
}

trusted_stable_base_is_current() {
    local use_sccache="${ROCM_BASE_USE_SCCACHE:-${USE_SCCACHE:-0}}"
    local base_image_arg=""
    local base_image_digest=""
    local pinned_base_image=""
    local expected_hash=""
    local metadata_version="${ROCM_BASE_METADATA_VERSION:-${DEFAULT_ROCM_BASE_METADATA_VERSION}}"
    local stable_ref="${BASE_REPO}:base"

    if [[ ! -f "${DOCKERFILE}" ]]; then
        echo "Cannot validate stable ROCm base without ${DOCKERFILE}" >&2
        return 2
    fi
    base_image_arg=$(extract_arg_default BASE_IMAGE)
    if [[ -z "${base_image_arg}" ]] \
        || ! base_image_digest=$(resolve_image_digest "${base_image_arg}"); then
        echo "Could not resolve the ROCm base parent while checking ${stable_ref}" >&2
        return 2
    fi
    pinned_base_image=$(canonical_pinned_image_ref \
        "${base_image_arg}" "${base_image_digest}") || return 2
    if ! expected_hash=$(compute_base_content_hash \
        "${use_sccache}" "${pinned_base_image}"); then
        echo "Could not calculate the expected stable ROCm base identity" >&2
        return 2
    fi

    if find_matching_base_content_ref \
        "${expected_hash}" "${metadata_version}" "${stable_ref}" >/dev/null; then
        echo "Trusted stable ROCm base already matches current inputs"
        return 0
    fi
    echo "Trusted stable ROCm base is missing or stale; refreshing it"
    return 1
}

rocm_base_changed() {
    local base_branch="${BUILDKITE_PULL_REQUEST_BASE_BRANCH:-main}"
    local base_ref="refs/remotes/origin/${base_branch}"
    local context=""
    local range=""
    local old_ref=""
    local merge_base=""
    local change_status=0
    local stable_status=0

    if [[ "${ROCM_BASE_REFRESH_SKIP:-0}" == "1" ]]; then
        echo "ROCM_BASE_REFRESH_SKIP=1 set; skipping ROCm base refresh"
        return 1
    fi

    if [[ "${ROCM_BASE_REFRESH_FORCE:-0}" == "1" ]]; then
        echo "ROCM_BASE_REFRESH_FORCE=1 set; refreshing ROCm base image"
        return 0
    fi

    if [[ "${ROCM_BASE_REFRESH_DIFF_UNAVAILABLE:-0}" == "1" ]]; then
        echo "ROCM_BASE_REFRESH_DIFF_UNAVAILABLE=1 set; refreshing ROCm base image"
        return 0
    fi

    if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
        echo "Not in a git checkout; cannot safely determine ROCm base changes" >&2
        return 2
    fi

    if [[ "${BUILDKITE_BRANCH:-}" == "${ROCM_BASE_STABLE_BRANCH:-main}" \
        && "${BUILDKITE_PULL_REQUEST:-false}" == "false" ]]; then
        if ! git rev-parse --verify HEAD~1 >/dev/null 2>&1; then
            if ! git fetch --no-tags --deepen="${ROCM_BASE_DIFF_FETCH_DEPTH:-200}" \
                origin >/dev/null 2>&1 \
                || ! git rev-parse --verify HEAD~1 >/dev/null 2>&1; then
                echo "Unable to establish previous stable-branch commit; refusing to skip ROCm base refresh" >&2
                return 2
            fi
        fi
        context="stable branch build; comparing against previous ${ROCM_BASE_STABLE_BRANCH:-main} commit"
        range="HEAD~1..HEAD"
        old_ref="HEAD~1"
    else
        if ! merge_base=$(find_base_merge_base "${base_branch}" "${base_ref}"); then
            echo "Unable to establish base comparison; refusing to skip ROCm base refresh" >&2
            return 2
        fi
        if [[ "${BUILDKITE_PULL_REQUEST:-false}" != "false" ]]; then
            context="pull request build against ${base_ref}"
        else
            context="branch build against ${base_ref}"
        fi
        range="${merge_base}...HEAD"
        old_ref="${merge_base}"
    fi

    rocm_base_changed_in_range "${context}" "${range}" "${old_ref}" \
        || change_status=$?
    if ((change_status != 1)); then
        return "${change_status}"
    fi

    # A failed build at commit N must be repaired by N+1 even when N+1 does
    # not touch the Dockerfile. On trusted main, cheaply validate the published
    # stable identity before accepting an unchanged git diff.
    if is_trusted_main_build; then
        trusted_stable_base_is_current || stable_status=$?
        case "${stable_status}" in
            0)
                return 1
                ;;
            1)
                return 0
                ;;
            *)
                return "${stable_status}"
                ;;
        esac
    fi
    return 1
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
        if git fetch --no-tags --depth=1 "${BUILDKITE_REPO}" "${remote_tip}" \
            >/dev/null 2>&1 \
            && git diff --quiet "${build_commit}" "${remote_tip}" -- "${DOCKERFILE}"; then
            echo "Current ${branch} tip has the same ROCm base inputs; publishing completed build"
            return 0
        fi
        echo "Skipping ROCm stable tag: ${branch} changed after ${build_commit} (${remote_tip})" >&2
        return 1
    fi
    return 0
}

tag_base_image_aliases() {
    local source_ref="$1"
    local descriptive_tag="$2"
    local stable_tag="$3"
    local -a tags=(-t "${descriptive_tag}")

    ROCM_BASE_STABLE_TAG_UPDATED=0
    if should_push_stable_tag && trusted_main_tip_matches_build; then
        tags+=(-t "${stable_tag}")
        # shellcheck disable=SC2034  # Read by sourced legacy callers/tests.
        ROCM_BASE_STABLE_TAG_UPDATED=1
    fi

    docker buildx imagetools create --prefer-index=false \
        "${tags[@]}" "${source_ref}"
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
    local content_args="${ROCM_BASE_CONTENT_ARGS:-${DEFAULT_ROCM_BASE_CONTENT_ARGS}}"
    local content_files_hash=""
    local metadata_version="${ROCM_BASE_METADATA_VERSION:-${DEFAULT_ROCM_BASE_METADATA_VERSION}}"
    local arg_name=""
    local arg_value=""
    local -a build_args=()
    local -a content_paths=()
    local -a content_arg_names=()

    if [[ ! -f "${DOCKERFILE}" ]]; then
        echo "Error: ROCm base Dockerfile not found: ${DOCKERFILE}" >&2
        exit 1
    fi
    if [[ ! "${metadata_version}" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]{0,15}$ ]]; then
        echo "Invalid ROCm base metadata version: ${metadata_version}" >&2
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
    base_hash=$(compute_base_content_hash "${use_sccache}" "${pinned_base_image}")
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
    trusted_content_tag=$(trusted_base_content_ref "${base_hash}" "${metadata_version}")
    scoped_content_tag=$(scoped_base_content_ref "${base_hash}" "${metadata_version}")
    if is_trusted_main_build; then
        scoped_content_tag="${trusted_content_tag}"
        writable_content_tag="${trusted_content_tag}"
    else
        writable_content_tag="${scoped_content_tag}"
    fi

    configure_rocm_base_layer_cache
    read -r -a content_arg_names <<< "${content_args}"
    for arg_name in "${content_arg_names[@]}"; do
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
