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
    repo="${repo#github.com/}"
    printf '%s\n' "${repo}"
}

is_trusted_main() {
    local stable_branch="${ROCM_BASE_STABLE_BRANCH:-${CI_BASE_STABLE_BRANCH:-${STABLE_BRANCH}}}"
    local stable_repo="${ROCM_BASE_STABLE_REPO_SLUG:-${CI_BASE_STABLE_REPO_SLUG:-${TRUSTED_REPO}}}"
    [[ "${BUILDKITE:-false}" == true \
        && "${BUILDKITE_PULL_REQUEST:-false}" == false \
        && "${BUILDKITE_BRANCH:-}" == "${stable_branch}" \
        && "$(normalize_repo "${BUILDKITE_REPO:-}")" == \
            "$(normalize_repo "${stable_repo}")" ]]
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

is_tagged() {
    [[ "${1:-}" =~ ^[^[:space:]@]+:[A-Za-z0-9_][A-Za-z0-9_.-]{0,127}$ ]]
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
    local ref="$1" output="$2"
    grep -Eqi \
        'manifest[ _]unknown|name[ _]unknown|no such manifest|unexpected status from (HEAD|GET) request to https?://[^[:space:]]+/v2/[^[:space:]]+/manifests/[^[:space:]]+:[[:space:]]*404([^0-9]|$)' \
        <<< "${output}" \
        || grep -Fqix -- "ERROR: ${ref}: not found" <<< "${output}" \
        || grep -Fqix -- \
            "ERROR: docker.io/${ref#docker.io/}: not found" <<< "${output}"
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
        ((status != 0)) && is_missing "${ref}" "${output}" && return 1
        ((attempt == attempts)) || sleep "${delay}"
    done
    printf 'Registry lookup failed for %s (status %d)\n%s\n' \
        "${ref}" "${status}" "${output:-<no output>}" >&2
    return 2
}

retag_and_verify() {
    local target="$1" source="$2"
    docker buildx imagetools create --prefer-index=false \
        -t "${target}" "${source}" || return 1
    [[ "$(lookup_digest "${target}")" == "${source##*@}" ]]
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
    local base_values="" ci_values=""
    local base_hash="" base_version="" base_parent_ref="" base_parent=""
    local base_file="" ci_hash="" ci_version="" ci_parent_ref=""
    local ci_parent="" ci_files=""
    local checked_in_parent="" checked_in_parent_digest=""
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
    [[ "$(lookup_digest "${ci_build}")" == "${ci##*@}" ]] || return 1
    base_values=$(inspect_labels "${base}" "${base_format}") || return 1
    ci_values=$(inspect_labels "${ci}" "${ci_format}") || return 1
    IFS='|' read -r \
        base_hash base_version base_parent_ref base_parent base_file \
        <<< "${base_values}"
    IFS='|' read -r \
        ci_hash ci_version ci_parent_ref ci_parent ci_files <<< "${ci_values}"
    if ! is_pinned "${base_parent_ref}" || [[ \
        ! "${base_hash}" =~ ^[0-9a-f]{64}$ \
        || "${base_hash}" != "${base_hash_metadata}" \
        || "${base_version}" != "${expected_base_version}" \
        || ! "${base_parent}" =~ ^sha256:[0-9a-f]{64}$ \
        || "${base_parent_ref##*@}" != "${base_parent}" \
        || "${base_file}" != \
            "${ROCM_BASE_DOCKERFILE:-docker/Dockerfile.rocm_base}" \
        || "${base%@*}" != \
            "${repo}:base-${base_hash}" \
        || ! "${ci_hash}" =~ ^[0-9a-f]{64}$ \
        || "${ci_version}" != "${expected_ci_version}" \
        || ! "${ci_files}" =~ ^[0-9a-f]{64}$ \
        || "${ci_parent_ref}" != "${ci_parent}" \
        || "${ci_parent}" != "${base##*@}" \
        || "${ci%@*}" != \
            "${ci_stable}-${ci_hash}" ]]; then
        echo "ROCm promotion candidate identity or parent relation is invalid" >&2
        return 1
    fi
    if ! is_safe_git_path "${base_file}"; then
        echo "Invalid ROCm base parent metadata" >&2
        return 1
    fi
    checked_in_parent=$(git show "${BUILDKITE_COMMIT}:${base_file}" \
        | sed -n -E \
            's/^[[:space:]]*ARG[[:space:]]+BASE_IMAGE="?([^"[:space:]]+)"?.*/\1/p' \
        | head -1)
    if [[ -z "${checked_in_parent}" ]]; then
        echo "Could not resolve BASE_IMAGE from ${base_file}" >&2
        return 1
    fi
    checked_in_parent_digest=$(lookup_digest "${checked_in_parent}") || return 1
    if [[ "${checked_in_parent_digest}" != "${base_parent}" ]]; then
        echo "ROCm base parent does not match the checked-in Dockerfile" >&2
        return 1
    fi
}

validate_smoke() {
    local required="$1" ref="$2" smoked="$3" smoked_ref="$4" ci="$5"
    local expected="${ROCM_CI_IMAGE_REPO:-rocm/vllm-ci}:build-${BUILDKITE_BUILD_ID}"
    local expected_digest="" smoke_values="" revision="" smoke_ci=""
    if [[ "${required}" != 1 || "${smoked}" != 1 \
        || "${ref}" != "${expected}" ]]; then
        echo "Promotion requires the successful build-scoped smoke test" >&2
        return 1
    fi
    if ! is_pinned "${smoked_ref}" || [[ \
        "$(repository_of "${smoked_ref}")" != \
        "$(repository_of "${expected}")" ]]; then
        echo "Promotion requires the successful build-scoped smoke test" >&2
        return 1
    fi
    expected_digest=$(lookup_digest "${ref}") || return 1
    if [[ "${smoked_ref##*@}" != "${expected_digest}" ]]; then
        echo "Build-scoped smoke image changed after validation" >&2
        return 1
    fi
    smoke_values=$(inspect_labels "${smoked_ref}" \
        '{{ index .Image.Config.Labels "org.opencontainers.image.revision" }}|{{ index .Image.Config.Labels "vllm.rocm.ci_base_image" }}') \
        || return 1
    IFS='|' read -r revision smoke_ci <<< "${smoke_values}"
    if [[ "${revision,,}" != "${BUILDKITE_COMMIT,,}" \
        || "${smoke_ci}" != "${ci}" ]]; then
        echo "Smoked image revision or ci_base parent does not match this build" >&2
        return 1
    fi
}

recheck_main() {
    local branch="${ROCM_BASE_STABLE_BRANCH:-${CI_BASE_STABLE_BRANCH:-${STABLE_BRANCH}}}" tip=""
    tip=$(git ls-remote --exit-code "${BUILDKITE_REPO}" \
        "refs/heads/${branch}" 2>/dev/null | awk 'NR == 1 { print $1 }')
    [[ "${tip}" =~ ^[0-9a-fA-F]{40}$ ]] \
        || { echo "Could not resolve latest ${branch}" >&2; return 1; }
    if [[ "${tip,,}" != "${BUILDKITE_COMMIT,,}" ]]; then
        echo "Skipping stable promotion: this build is no longer latest ${branch}"
        return 2
    fi
}

main() {
    local repo="${ROCM_BASE_IMAGE_REPO:-${IMAGE_REPO}}"
    local ci_stable=""
    local base="" base_hash="" base_content="" base_stable=""
    local ci="" ci_content="" ci_build="" parent=""
    local required="" smoke_ref="" smoked="" smoked_ref=""
    local status=0 failed=0 transaction_active=0
    local actual="" source="" stable_parent="" i=0
    local -a aliases=() candidates=() previous=("" "" "" "")
    local -a needs_write=(0 0 0 0)
    rollback_transaction() {
        local rollback_failed=0 rollback_i=0
        ((transaction_active == 1)) || return 0
        transaction_active=0
        echo "Restoring previous stable ROCm aliases" >&2
        for rollback_i in 0 1; do
            source="$(repository_of "${aliases[rollback_i]}")@${previous[rollback_i]}"
            retag_and_verify "${aliases[rollback_i]}" "${source}" \
                || rollback_failed=1
        done
        ((rollback_failed == 0)) \
            || echo "One or more aliases could not be restored" >&2
        return 0
    }
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
    ci_stable=$(metadata_required rocm-ci-base-image-stable)
    parent=$(metadata_required rocm-ci-base-parent-image)
    required=$(metadata_required rocm-ci-image-smoke-required)
    smoke_ref=$(metadata_required rocm-ci-image-smoke-ref)
    smoked=$(metadata_required rocm-ci-image-smoked)
    smoked_ref=$(metadata_required rocm-ci-image-smoked-ref)
    aliases=("${repo}:base" "${ci_stable}"
        "${ci_stable}-${BUILDKITE_COMMIT}"
        "${ROCM_CI_IMAGE_REPO:-rocm/vllm-ci}:${BUILDKITE_COMMIT}")
    candidates=("${base}" "${ci}" "${ci}" "${smoked_ref}")
    if ! is_tagged "${aliases[1]}" || ! is_tagged "${aliases[2]}" \
        || ! is_tagged "${aliases[3]}" \
        || [[ "$(repository_of "${aliases[1]}")" != "${repo}" ]]; then
        echo "ROCm promotion aliases are invalid" >&2
        return 1
    fi
    validate_candidates \
        "${base}" "${ci}" "${parent}" "${repo}" "${ci_stable}" \
        "${base_hash}" "${base_content}" "${base_stable}" \
        "${ci_content}" "${ci_build}"
    validate_smoke \
        "${required}" "${smoke_ref}" "${smoked}" "${smoked_ref}" "${ci}"
    for i in 0 1 2 3; do
        status=0
        previous[i]=$(lookup_digest "${aliases[i]}") || status=$?
        case "${status}" in
            0)
                [[ "${previous[i]}" == "${candidates[i]##*@}" ]] \
                    || needs_write[i]=1
                ;;
            1)
                if ((i < 2)); then
                    echo "Cannot safely promote without an existing rollback value for ${aliases[i]}" >&2
                    return 1
                fi
                needs_write[i]=1
                ;;
            *) echo "Could not inspect ${aliases[i]}" >&2; return 1 ;;
        esac
    done
    stable_parent=$(inspect_labels "${repo}@${previous[1]}" \
        '{{ index .Image.Config.Labels "vllm.rocm.base_image_digest" }}') \
        || return 1
    if [[ ! "${stable_parent}" =~ ^sha256:[0-9a-f]{64}$ ]]; then
        echo "Stable ci_base has an invalid base parent" >&2
        return 1
    fi
    if [[ "${previous[0]}" != "${stable_parent}" ]]; then
        echo "Repairing interrupted ROCm stable alias update"
        if ! retag_and_verify "${aliases[0]}" "${repo}@${stable_parent}"; then
            echo "Could not restore coherence between stable ROCm aliases" >&2
            return 1
        fi
        actual=$(lookup_digest "${aliases[1]}") || return 1
        if [[ "${actual}" != "${previous[1]}" ]]; then
            echo "Stable ci_base changed during alias repair" >&2
            return 1
        fi
        previous[0]="${stable_parent}"
    fi
    [[ "${previous[0]}" == "${candidates[0]##*@}" ]] \
        && needs_write[0]=0 || needs_write[0]=1
    ((needs_write[0] || needs_write[1] || needs_write[2] || needs_write[3])) \
        || { echo "Stable and compatibility aliases are already current"; return 0; }
    if ((needs_write[0] || needs_write[1])); then
        status=0
        recheck_main || status=$?
        case "${status}" in
            0) ;;
            2) needs_write[0]=0; needs_write[1]=0 ;;
            *) return 1 ;;
        esac
    fi
    if ((needs_write[0] || needs_write[1])); then
        transaction_active=1
        trap 'status=$?; trap - EXIT INT TERM; rollback_transaction; exit "${status}"' EXIT
        trap 'trap - EXIT INT TERM; rollback_transaction; exit 130' INT
        trap 'trap - EXIT INT TERM; rollback_transaction; exit 143' TERM
        for i in 0 1; do
            if ((needs_write[i])); then
                docker buildx imagetools create --prefer-index=false \
                    -t "${aliases[i]}" "${candidates[i]}" \
                    || { failed=1; break; }
            fi
        done
        if ((failed == 0)); then
            for i in 0 1; do
                actual=$(lookup_digest "${aliases[i]}") || { failed=1; break; }
                [[ "${actual}" == "${candidates[i]##*@}" ]] \
                    || { failed=1; break; }
            done
        fi
        if ((failed)); then
            echo "Stable ROCm promotion failed; rolling back stable aliases" >&2
            rollback_transaction
            trap - EXIT INT TERM
            return 1
        fi
        transaction_active=0
        trap - EXIT INT TERM
    fi
    # Commit-scoped aliases are independent of the mutable stable aliases.
    # A partial failure is retryable without disturbing the stable aliases.
    for i in 2 3; do
        ((needs_write[i])) || continue
        if ! retag_and_verify "${aliases[i]}" "${candidates[i]}"; then
            echo "Could not publish compatibility alias ${aliases[i]}" >&2
            return 1
        fi
    done
    echo "ROCm image alias update completed"
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    main "$@"
fi
