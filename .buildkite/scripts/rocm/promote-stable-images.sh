#!/usr/bin/env bash
# Promote immutable ROCm candidates only from the latest trusted main build.

set -euo pipefail

readonly TRUSTED_REPO="vllm-project/vllm"
readonly STABLE_BRANCH="main"
readonly IMAGE_REPO="rocm/vllm-dev"

normalize_repo() {
    local repo="${1:-}"
    repo="${repo%/}"
    repo="${repo%.git}"
    repo="${repo#git@github.com:}"
    repo="${repo#ssh://git@github.com/}"
    repo="${repo#https://github.com/}"
    repo="${repo#http://github.com/}"
    printf '%s\n' "${repo}"
}

is_trusted_main() {
    [[ "${BUILDKITE:-false}" == true \
        && "${BUILDKITE_PULL_REQUEST:-false}" == false \
        && "${BUILDKITE_BRANCH:-}" == \
            "${ROCM_BASE_STABLE_BRANCH:-${STABLE_BRANCH}}" \
        && "$(normalize_repo "${BUILDKITE_REPO:-}")" == \
            "$(normalize_repo \
                "${ROCM_BASE_STABLE_REPO_SLUG:-${TRUSTED_REPO}}")" ]]
}

metadata_required() {
    local value=""
    if ! command -v buildkite-agent >/dev/null 2>&1 \
        || ! value=$(buildkite-agent meta-data get "$1" 2>/dev/null) \
        || [[ -z "${value}" ]]; then
        echo "Required ROCm promotion metadata is missing: $1" >&2
        return 1
    fi
    printf '%s\n' "${value}"
}

is_pinned() {
    [[ "${1:-}" =~ ^[^[:space:]@]+@sha256:[0-9a-f]{64}$ ]]
}

is_safe_git_path() {
    local path="${1:-}"

    [[ "${path}" =~ ^[a-zA-Z0-9._/+@-]+$ \
        && "${path}" != /* && "${path}" != -* \
        && "${path}" != ".." && "${path}" != ../* \
        && "${path}" != */../* && "${path}" != */.. ]]
}

repository_of() {
    local repo="${1%@*}"
    [[ "${repo##*/}" == *:* ]] && repo="${repo%:*}"
    [[ -n "${repo}" ]] || return 1
    printf '%s\n' "${repo}"
}

is_missing() {
    grep -Eqi \
        'manifest unknown|name unknown|no such manifest|(^|[^0-9])404([^0-9]|$)|(^|[[:space:]])[^[:space:]]+/[^[:space:]]+:[^[:space:]]+:[[:space:]]+not found([[:space:]]|$)' \
        <<< "$1"
}

# 0: found; 1: confirmed missing; 2: registry/parsing failure.
lookup_digest() {
    local ref="$1"
    local attempts="${ROCM_PROMOTION_LOOKUP_ATTEMPTS:-4}"
    local delay="${ROCM_PROMOTION_LOOKUP_RETRY_DELAY:-2}"
    local output="" digest="" status=0 attempt=0

    [[ "${attempts}" =~ ^[1-9][0-9]*$ && "${delay}" =~ ^[0-9]+$ ]] \
        || { echo "Invalid promotion retry configuration" >&2; return 2; }
    for ((attempt = 1; attempt <= attempts; attempt++)); do
        status=0
        output=$(docker buildx imagetools inspect "${ref}" 2>&1) || status=$?
        digest=$(awk '$1 == "Digest:" { print $2; exit }' <<< "${output}")
        if ((status == 0)) && [[ "${digest}" =~ ^sha256:[0-9a-f]{64}$ ]]; then
            printf '%s\n' "${digest}"
            return 0
        fi
        ((status != 0)) && is_missing "${output}" && return 1
        ((attempt == attempts)) || sleep "${delay}"
    done
    printf 'Registry lookup failed for %s (status %d)\n%s\n' \
        "${ref}" "${status}" "${output:-<no output>}" >&2
    return 2
}

inspect_labels() {
    local ref="$1" format="$2"
    local attempts="${ROCM_PROMOTION_LOOKUP_ATTEMPTS:-4}"
    local delay="${ROCM_PROMOTION_LOOKUP_RETRY_DELAY:-2}"
    local output="" attempt=0

    [[ "${attempts}" =~ ^[1-9][0-9]*$ && "${delay}" =~ ^[0-9]+$ ]] \
        || return 1
    for ((attempt = 1; attempt <= attempts; attempt++)); do
        if output=$(docker buildx imagetools inspect \
            "${ref}" --format "${format}" 2>/dev/null) \
            && [[ -n "${output}" ]]; then
            printf '%s\n' "${output}"
            return 0
        fi
        ((attempt == attempts)) || sleep "${delay}"
    done
    echo "Could not inspect promotion labels: ${ref}" >&2
    return 1
}

validate_candidates() {
    local base="$1" ci="$2" parent="$3" repo="$4" ci_stable="$5"
    local base_hash_metadata="$6" base_content="$7" base_stable="$8"
    local ci_content="$9" ci_build="${10}"
    local base_values="" ci_values="" base_digest="" ci_digest=""
    local base_hash="" base_version="" base_parent_ref="" base_parent=""
    local base_file="" ci_hash="" ci_version="" ci_parent_ref=""
    local ci_parent="" ci_files=""
    local expected_base_version="${ROCM_BASE_METADATA_VERSION:-2}"
    local expected_ci_version="${CI_BASE_METADATA_VERSION:-3}"
    local base_format='{{ index .Image.Config.Labels "vllm.rocm_base.content_hash" }}|{{ index .Image.Config.Labels "vllm.rocm_base.metadata_version" }}|{{ index .Image.Config.Labels "vllm.rocm_base.base_image" }}|{{ index .Image.Config.Labels "vllm.rocm_base.base_image_digest" }}|{{ index .Image.Config.Labels "vllm.rocm_base.dockerfile" }}'
    local ci_format='{{ index .Image.Config.Labels "vllm.ci_base.content_hash" }}|{{ index .Image.Config.Labels "vllm.ci_base.metadata_version" }}|{{ index .Image.Config.Labels "vllm.rocm.base_image" }}|{{ index .Image.Config.Labels "vllm.rocm.base_image_digest" }}|{{ index .Image.Config.Labels "vllm.ci_base.content_files_hash" }}'

    if ! is_pinned "${base}" || ! is_pinned "${ci}" \
        || ! is_pinned "${parent}" \
        || [[ ! "${expected_base_version}" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]{0,15}$ ]] \
        || [[ ! "${expected_ci_version}" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]{0,15}$ ]]; then
        echo "Promotion candidates or metadata versions are invalid" >&2
        return 1
    fi
    if [[ "${parent}" != "${base}" \
        || "${base_content}" != "${base%@*}" \
        || "${base_stable}" != "${repo}:base" \
        || "${ci_content}" != "${ci%@*}" \
        || "${ci_build}" != \
            "${ci_stable}-build-${BUILDKITE_BUILD_ID}" ]]; then
        echo "Promotion handoff refs are inconsistent" >&2
        return 1
    fi
    base_digest=$(lookup_digest "${base}") || return 1
    ci_digest=$(lookup_digest "${ci}") || return 1
    [[ "$(lookup_digest "${ci_build}")" == "${ci_digest}" ]] || return 1
    base_values=$(inspect_labels "${base}" "${base_format}") || return 1
    ci_values=$(inspect_labels "${ci}" "${ci_format}") || return 1
    IFS='|' read -r \
        base_hash base_version base_parent_ref base_parent base_file \
        <<< "${base_values}"
    IFS='|' read -r \
        ci_hash ci_version ci_parent_ref ci_parent ci_files <<< "${ci_values}"

    if ! is_pinned "${base_parent_ref}" || [[ \
        "${base_digest}" != "${base##*@}" \
        || "${ci_digest}" != "${ci##*@}" \
        || ! "${base_hash}" =~ ^[0-9a-f]{64}$ \
        || "${base_hash}" != "${base_hash_metadata}" \
        || "${base_version}" != "${expected_base_version}" \
        || ! "${base_parent}" =~ ^sha256:[0-9a-f]{64}$ \
        || "${base_parent_ref##*@}" != "${base_parent}" \
        || "${base_file}" != \
            "${ROCM_BASE_DOCKERFILE:-docker/Dockerfile.rocm_base}" \
        || "${base%@*}" != \
            "${repo}:base-v${base_version}-${base_hash}" \
        || ! "${ci_hash}" =~ ^[0-9a-f]{64}$ \
        || "${ci_version}" != "${expected_ci_version}" \
        || ! "${ci_files}" =~ ^[0-9a-f]{64}$ \
        || "${ci_parent_ref}" != "${ci_parent}" \
        || "${ci_parent}" != "${base##*@}" \
        || "${ci%@*}" != \
            "${ci_stable}-v${ci_version}-${ci_hash}" ]]; then
        echo "ROCm promotion candidate identity or parent relation is invalid" >&2
        return 1
    fi
}

validate_smoke() {
    local required="$1" ref="$2" smoked="$3" smoked_ref="$4"
    local expected="${ROCM_CI_IMAGE_REPO:-rocm/vllm-ci}:build-${BUILDKITE_BUILD_ID}"
    local revision="" status=0

    if [[ "${required}" != 1 || "${smoked}" != 1 \
        || "${ref}" != "${expected}" || "${smoked_ref}" != "${expected}" ]]; then
        echo "Promotion requires the successful build-scoped smoke test" >&2
        return 1
    fi
    lookup_digest "${ref}" >/dev/null || status=$?
    ((status == 0)) || return 1
    revision=$(inspect_labels "${ref}" \
        '{{ index .Image.Config.Labels "org.opencontainers.image.revision" }}') \
        || return 1
    [[ "${revision,,}" == "${BUILDKITE_COMMIT,,}" ]] \
        || { echo "Smoked image revision does not match this build" >&2; return 1; }
}

validate_checked_in_parent() {
    local base="$1" values="" parent_digest="" dockerfile=""
    local parent_ref="" checked_in_digest=""
    local format='{{ index .Image.Config.Labels "vllm.rocm_base.base_image_digest" }}|{{ index .Image.Config.Labels "vllm.rocm_base.dockerfile" }}'

    values=$(inspect_labels "${base}" "${format}") || return 1
    IFS='|' read -r parent_digest dockerfile <<< "${values}"
    if [[ ! "${parent_digest}" =~ ^sha256:[0-9a-f]{64}$ ]] \
        || ! is_safe_git_path "${dockerfile}"; then
        echo "Invalid ROCm base parent metadata" >&2
        return 1
    fi
    parent_ref=$(git show "${BUILDKITE_COMMIT}:${dockerfile}" \
        | sed -n -E \
            's/^[[:space:]]*ARG[[:space:]]+BASE_IMAGE="?([^"[:space:]]+)"?.*/\1/p' \
        | head -1)
    if [[ -z "${parent_ref}" ]]; then
        echo "Could not resolve BASE_IMAGE from ${dockerfile}" >&2
        return 1
    fi
    checked_in_digest=$(lookup_digest "${parent_ref}") || return 1
    if [[ "${checked_in_digest}" != "${parent_digest}" ]]; then
        echo "ROCm base parent does not match the checked-in Dockerfile" >&2
        return 1
    fi
}

recheck_main() {
    local branch="${ROCM_BASE_STABLE_BRANCH:-${STABLE_BRANCH}}" tip=""
    tip=$(git ls-remote --exit-code "${BUILDKITE_REPO}" \
        "refs/heads/${branch}" 2>/dev/null | awk 'NR == 1 { print $1 }')
    [[ "${tip}" =~ ^[0-9a-fA-F]{40}$ ]] \
        || { echo "Could not resolve latest ${branch}" >&2; return 1; }
    if [[ "${tip,,}" != "${BUILDKITE_COMMIT,,}" ]]; then
        echo "Skipping promotion: this build is no longer latest ${branch}"
        return 2
    fi
}

rollback_aliases() {
    local aliases_name="$1" previous_name="$2"
    local -n aliases_ref="${aliases_name}" previous_ref="${previous_name}"
    local source="" actual="" failed=0 i=0

    echo "Restoring previous stable ROCm aliases" >&2
    for i in "${!aliases_ref[@]}"; do
        if [[ -z "${previous_ref[i]}" ]]; then
            echo "No previous value existed for ${aliases_ref[i]}" >&2
            continue
        fi
        source="$(repository_of "${aliases_ref[i]}")@${previous_ref[i]}"
        if ! docker buildx imagetools create --prefer-index=false \
            -t "${aliases_ref[i]}" "${source}"; then
            failed=1
            continue
        fi
        actual=$(lookup_digest "${aliases_ref[i]}") || { failed=1; continue; }
        [[ "${actual}" == "${previous_ref[i]}" ]] || failed=1
    done
    return "${failed}"
}

main() {
    local repo="${ROCM_BASE_IMAGE_REPO:-${IMAGE_REPO}}"
    local ci_stable="${CI_BASE_IMAGE_TAG:-${repo}:ci_base}"
    local ci_version="${CI_BASE_METADATA_VERSION:-3}"
    local ci_versioned="${CI_BASE_STABLE_CACHE_REF:-${ci_stable}-v${ci_version}}"
    local base="" base_hash="" base_content="" base_stable=""
    local ci="" ci_content="" ci_build="" parent=""
    local required="" smoke_ref="" smoked="" smoked_ref=""
    local status=0 failed=0 needs_write=0 actual="" i=0
    local -a aliases=("${repo}:base" "${ci_stable}" "${ci_versioned}")
    local -a previous=("" "" "") candidates=()

    is_trusted_main \
        || { echo "Skipping stable ROCm promotion outside trusted main"; return 0; }
    [[ "${BUILDKITE_COMMIT:-}" =~ ^[0-9a-fA-F]{40}$ \
        && -n "${BUILDKITE_BUILD_ID:-}" ]] \
        || { echo "Promotion requires a full commit and build ID" >&2; return 1; }
    command -v docker >/dev/null
    command -v git >/dev/null

    base=$(metadata_required rocm-base-image)
    base_hash=$(metadata_required rocm-base-content-hash)
    base_content=$(metadata_required rocm-base-image-content)
    base_stable=$(metadata_required rocm-base-image-stable)
    ci=$(metadata_required rocm-ci-base-image)
    ci_content=$(metadata_required rocm-ci-base-image-content)
    ci_build=$(metadata_required rocm-ci-base-image-build)
    parent=$(metadata_required rocm-ci-base-parent-image)
    required=$(metadata_required rocm-ci-image-smoke-required)
    smoke_ref=$(metadata_required rocm-ci-image-smoke-ref)
    smoked=$(metadata_required rocm-ci-image-smoked)
    smoked_ref=$(metadata_required rocm-ci-image-smoked-ref)
    candidates=("${base}" "${ci}" "${ci}")

    validate_candidates \
        "${base}" "${ci}" "${parent}" "${repo}" "${ci_stable}" \
        "${base_hash}" "${base_content}" "${base_stable}" \
        "${ci_content}" "${ci_build}"
    validate_smoke "${required}" "${smoke_ref}" "${smoked}" "${smoked_ref}"
    validate_checked_in_parent "${base}"

    for i in "${!aliases[@]}"; do
        status=0
        previous[i]=$(lookup_digest "${aliases[i]}") || status=$?
        case "${status}" in
            0) [[ "${previous[i]}" == "${candidates[i]##*@}" ]] || needs_write=1 ;;
            1) previous[i]=""; needs_write=1 ;;
            *) echo "Could not snapshot ${aliases[i]}" >&2; return 1 ;;
        esac
    done
    ((needs_write == 1)) \
        || { echo "Stable ROCm aliases already match the candidates"; return 0; }
    # v3 is initially absent; use the legacy ci_base value as its rollback
    # point so a first-time partial promotion still restores one generation.
    [[ -n "${previous[2]}" ]] || previous[2]="${previous[1]}"

    status=0
    recheck_main || status=$?
    case "${status}" in 0) ;; 2) return 0 ;; *) return 1 ;; esac

    docker buildx imagetools create --prefer-index=false \
        -t "${aliases[0]}" "${base}" || failed=1
    if ((failed == 0)); then
        docker buildx imagetools create --prefer-index=false \
            -t "${aliases[1]}" -t "${aliases[2]}" "${ci}" || failed=1
    fi
    if ((failed == 0)); then
        for i in "${!aliases[@]}"; do
            actual=$(lookup_digest "${aliases[i]}") || { failed=1; break; }
            [[ "${actual}" == "${candidates[i]##*@}" ]] \
                || { failed=1; break; }
        done
    fi
    if ((failed)); then
        echo "Stable ROCm promotion failed; rolling back all aliases" >&2
        rollback_aliases aliases previous \
            || echo "One or more aliases could not be restored" >&2
        return 1
    fi
    echo "Stable ROCm aliases now reference the validated candidates"
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    main "$@"
fi
