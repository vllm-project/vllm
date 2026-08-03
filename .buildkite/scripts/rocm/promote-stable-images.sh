#!/usr/bin/env bash
# Promote immutable ROCm image candidates after serialized freshness checks.

set -euo pipefail

readonly DEFAULT_TRUSTED_REPO_SLUG="vllm-project/vllm"
readonly DEFAULT_STABLE_BRANCH="main"
readonly DEFAULT_BASE_IMAGE_REPO="rocm/vllm-dev"
readonly DEFAULT_CI_BASE_STABLE_TAG="rocm/vllm-dev:ci_base"
readonly LATEST_MAIN_REF="refs/remotes/rocm-promotion/latest-main"

metadata_get_required() {
    local key="$1"
    local value=""

    if ! command -v buildkite-agent >/dev/null 2>&1 \
        || ! value="$(buildkite-agent meta-data get "${key}" 2>/dev/null)" \
        || [[ -z "${value}" ]]; then
        echo "Required ROCm image handoff metadata is missing: ${key}" >&2
        return 1
    fi
    printf '%s\n' "${value}"
}

normalize_repo_slug() {
    local repo="${1:-}"

    repo="${repo%/}"
    repo="${repo%.git}"
    repo="${repo#git@github.com:}"
    repo="${repo#ssh://git@github.com/}"
    repo="${repo#https://github.com/}"
    repo="${repo#http://github.com/}"
    printf '%s\n' "${repo}"
}

is_trusted_main_build() {
    local actual_repo=""
    local trusted_repo=""

    [[ "${BUILDKITE:-false}" == "true" ]] || return 1
    [[ "${BUILDKITE_PULL_REQUEST:-false}" == "false" ]] || return 1
    [[ "${BUILDKITE_BRANCH:-}" == "${ROCM_BASE_STABLE_BRANCH:-${DEFAULT_STABLE_BRANCH}}" ]] \
        || return 1

    actual_repo="$(normalize_repo_slug "${BUILDKITE_REPO:-}")"
    trusted_repo="$(normalize_repo_slug \
        "${ROCM_BASE_STABLE_REPO_SLUG:-${DEFAULT_TRUSTED_REPO_SLUG}}")"
    [[ -n "${actual_repo}" && "${actual_repo}" == "${trusted_repo}" ]]
}

is_digest_pinned_image() {
    local image_ref="${1:-}"
    [[ "${image_ref}" =~ ^[^[:space:]@]+@sha256:[0-9a-f]{64}$ ]]
}

canonical_digest_ref() {
    local image_ref="$1"
    local repository="${image_ref%@*}"
    local last_component="${repository##*/}"

    is_digest_pinned_image "${image_ref}" || return 1
    if [[ "${last_component}" == *:* ]]; then
        repository="${repository%:*}"
    fi
    printf '%s@%s\n' "${repository}" "${image_ref##*@}"
}

is_safe_git_path() {
    local path="${1:-}"

    [[ "${path}" =~ ^[a-zA-Z0-9._/+@-]+$ ]] \
        && [[ "${path}" != /* && "${path}" != -* ]] \
        && [[ "${path}" != ".." && "${path}" != ../* \
            && "${path}" != */../* && "${path}" != */.. ]]
}

inspect_image_config() {
    local image_ref="$1"
    local config=""

    if ! config="$(docker buildx imagetools inspect \
        "${image_ref}" --format '{{json .Image}}')" \
        || ! jq -e '.config.Labels | type == "object"' >/dev/null <<<"${config}"; then
        echo "Could not inspect image metadata: ${image_ref}" >&2
        return 1
    fi
    printf '%s\n' "${config}"
}

image_label_required() {
    local config="$1"
    local key="$2"
    local value=""

    if ! value="$(jq -er --arg key "${key}" \
        '.config.Labels[$key] // empty' <<<"${config}")" \
        || [[ -z "${value}" ]]; then
        echo "Required candidate image label is missing: ${key}" >&2
        return 1
    fi
    printf '%s\n' "${value}"
}

lookup_manifest_digest() {
    local image_ref="$1"
    local attempts="${ROCM_PROMOTION_IMAGE_LOOKUP_ATTEMPTS:-4}"
    local delay_secs="${ROCM_PROMOTION_IMAGE_LOOKUP_RETRY_DELAY:-2}"
    local digest=""
    local output=""
    local status=0
    local attempt=0

    if [[ ! "${attempts}" =~ ^[1-9][0-9]*$ || ! "${delay_secs}" =~ ^[0-9]+$ ]]; then
        echo "Invalid promotion image lookup retry configuration" >&2
        return 2
    fi
    for ((attempt = 1; attempt <= attempts; attempt++)); do
        status=0
        output=$(docker buildx imagetools inspect \
            "${image_ref}" --format '{{json .Manifest.Digest}}' 2>&1) \
            || status=$?
        if ((status == 0)); then
            digest=$(jq -er 'select(type == "string")' <<< "${output}") || true
            if [[ "${digest}" =~ ^sha256:[0-9a-f]{64}$ ]]; then
                printf '%s\n' "${digest}"
                return 0
            fi
        elif grep -Eqi \
            'manifest unknown|name unknown|no such manifest|(^|[^0-9])404([^0-9]|$)|(^|[[:space:]])[^[:space:]]+/[^[:space:]]+:[^[:space:]]+:[[:space:]]+not found([[:space:]]|$)' \
            <<< "${output}"; then
            return 1
        fi
        if ((attempt < attempts)); then
            echo "Image digest lookup failed for ${image_ref} (${attempt}/${attempts}); retrying" >&2
            sleep "${delay_secs}"
        fi
    done

    printf 'Registry digest lookup failed for %s (status %d)\n%s\n' \
        "${image_ref}" "${status}" "${output:-<no output>}" >&2
    return 2
}

inspect_manifest_digest() {
    local image_ref="$1"
    local digest=""

    if ! digest=$(lookup_manifest_digest "${image_ref}"); then
        echo "Could not resolve image digest: ${image_ref}" >&2
        return 1
    fi
    printf '%s\n' "${digest}"
}

fetch_latest_main() {
    local branch="${ROCM_BASE_STABLE_BRANCH:-${DEFAULT_STABLE_BRANCH}}"

    git check-ref-format --branch "${branch}" >/dev/null
    git fetch --no-tags --force --depth=1 "${BUILDKITE_REPO}" \
        "+refs/heads/${branch}:${LATEST_MAIN_REF}"
    git rev-parse --verify "${LATEST_MAIN_REF}^{commit}" >/dev/null
}

latest_base_parent_digest() {
    local dockerfile="$1"
    local parent_ref=""

    parent_ref="$(git show "${LATEST_MAIN_REF}:${dockerfile}" \
        | sed -n -E 's/^[[:space:]]*ARG[[:space:]]+BASE_IMAGE="?([^"[:space:]]+)"?.*/\1/p' \
        | head -1)"
    if [[ -z "${parent_ref}" ]]; then
        echo "Could not resolve BASE_IMAGE from latest main: ${dockerfile}" >&2
        return 1
    fi
    inspect_manifest_digest "${parent_ref}"
}

promote_tag() {
    local candidate="$1"
    local stable_tag="$2"
    local candidate_digest="${candidate##*@}"
    local promoted_digest=""

    echo "Promoting ${stable_tag} from ${candidate}"
    if ! docker buildx imagetools create \
        --prefer-index=false \
        -t "${stable_tag}" \
        "${candidate}"; then
        echo "Failed to update stable image tag: ${stable_tag}" >&2
        return 1
    fi
    promoted_digest="$(inspect_manifest_digest "${stable_tag}")"
    if [[ "${promoted_digest}" != "${candidate_digest}" ]]; then
        echo "Promoted tag does not resolve to the candidate digest: ${stable_tag}" >&2
        return 1
    fi
}

restore_stable_tags() {
    local previous_base="$1"
    local base_stable_tag="$2"
    local previous_ci_base="$3"
    local ci_stable_tag="$4"
    local restore_failed=0

    echo "Restoring the previous stable ROCm image aliases" >&2
    if [[ -n "${previous_base}" ]]; then
        promote_tag "${previous_base}" "${base_stable_tag}" || restore_failed=1
    else
        echo "No previous base alias existed; leaving any newly created alias for the retry" >&2
    fi
    if [[ -n "${previous_ci_base}" ]]; then
        promote_tag "${previous_ci_base}" "${ci_stable_tag}" || restore_failed=1
    else
        echo "No previous ci_base alias existed; leaving any newly created alias for the retry" >&2
    fi
    return "${restore_failed}"
}

main() {
    local base_candidate=""
    local base_input_hash=""
    local base_canonical_tag=""
    local base_stable_tag=""
    local ci_candidate=""
    local base_config=""
    local ci_config=""
    local base_dockerfile=""
    local base_parent_image=""
    local base_parent_digest=""
    local ci_content_hash=""
    local ci_parent_image=""
    local ci_stable_tag="${CI_BASE_IMAGE_TAG:-${DEFAULT_CI_BASE_STABLE_TAG}}"
    local latest_main_commit=""
    local previous_base_digest=""
    local previous_base_ref=""
    local previous_base_status=0
    local previous_ci_digest=""
    local previous_ci_ref=""
    local previous_ci_status=0
    local smoke_required=""
    local smoked=""

    if ! is_trusted_main_build; then
        echo "Skipping stable ROCm image promotion outside a trusted main-branch build"
        return 0
    fi
    command -v docker >/dev/null
    command -v git >/dev/null
    command -v jq >/dev/null

    smoke_required="$(metadata_get_required rocm-ci-image-smoke-required)"
    smoked="$(metadata_get_required rocm-ci-image-smoked)"
    [[ "${smoke_required}" == "1" && "${smoked}" == "1" ]] || {
        echo "Stable ROCm promotion requires a successful commit-image smoke test" >&2
        return 1
    }

    base_candidate="$(metadata_get_required rocm-base-image)"
    base_input_hash="$(metadata_get_required rocm-base-input-hash)"
    base_canonical_tag="$(metadata_get_required rocm-base-canonical-tag)"
    base_stable_tag="$(metadata_get_required rocm-base-stable-tag)"
    ci_candidate="$(metadata_get_required rocm-ci-base-image)"

    is_digest_pinned_image "${base_candidate}" \
        || { echo "Invalid ROCm base promotion candidate: ${base_candidate}" >&2; return 1; }
    is_digest_pinned_image "${ci_candidate}" \
        || { echo "Invalid ROCm ci_base promotion candidate: ${ci_candidate}" >&2; return 1; }
    [[ "${base_input_hash}" =~ ^[0-9a-f]{64}$ ]] \
        || { echo "Invalid ROCm base input hash: ${base_input_hash}" >&2; return 1; }
    [[ "${base_stable_tag}" == "${ROCM_BASE_IMAGE_REPO:-${DEFAULT_BASE_IMAGE_REPO}}:base" ]] \
        || { echo "Unexpected ROCm base stable tag: ${base_stable_tag}" >&2; return 1; }
    [[ "${base_canonical_tag}" == "${base_stable_tag%:base}:base-input-${base_input_hash}" \
        && "${base_candidate%@*}" == "${base_stable_tag%:base}" ]] \
        || { echo "ROCm base candidate/canonical tag mismatch" >&2; return 1; }

    base_config="$(inspect_image_config "${base_candidate}")"
    ci_config="$(inspect_image_config "${ci_candidate}")"

    [[ "$(image_label_required "${base_config}" vllm.rocm_base.metadata_version)" == "${ROCM_BASE_METADATA_VERSION:-2}" \
        && "$(image_label_required "${base_config}" vllm.rocm_base.input_hash)" == "${base_input_hash}" \
        && "$(image_label_required "${base_config}" vllm.rocm_base.image.canonical)" == "${base_canonical_tag}" \
        && "$(image_label_required "${base_config}" vllm.rocm_base.image.stable)" == "${base_stable_tag}" ]] \
        || { echo "ROCm base candidate labels do not match the handoff" >&2; return 1; }
    base_dockerfile="$(image_label_required "${base_config}" vllm.rocm_base.dockerfile)"
    base_parent_image="$(image_label_required "${base_config}" vllm.rocm_base.base_image)"
    base_parent_digest="$(image_label_required "${base_config}" vllm.rocm_base.base_image_digest)"
    [[ "${base_dockerfile}" == "${ROCM_BASE_DOCKERFILE:-docker/Dockerfile.rocm_base}" \
        && "${base_parent_digest}" =~ ^sha256:[0-9a-f]{64}$ \
        && "$(canonical_digest_ref "${base_parent_image}")" == *"@${base_parent_digest}" ]] \
        || { echo "Invalid ROCm base candidate input labels" >&2; return 1; }

    ci_content_hash="$(image_label_required "${ci_config}" vllm.ci_base.content_hash)"
    [[ "${ci_content_hash}" =~ ^[0-9a-f]{64}$ \
        && "${ci_candidate%@*}" == "${ci_stable_tag}-${ci_content_hash}" \
        && "$(image_label_required "${ci_config}" vllm.ci_base.image.content)" == "${ci_candidate%@*}" \
        && "$(image_label_required "${ci_config}" vllm.ci_base.metadata_version)" == "${CI_BASE_METADATA_VERSION:-2}" ]] \
        || { echo "Invalid ROCm ci_base candidate identity labels" >&2; return 1; }
    ci_parent_image="$(image_label_required "${ci_config}" vllm.rocm.base_image)"
    [[ "$(canonical_digest_ref "${ci_parent_image}")" == "$(canonical_digest_ref "${base_candidate}")" \
        && "$(image_label_required "${ci_config}" vllm.rocm.base_image_digest)" == "${base_candidate##*@}" ]] \
        || { echo "ROCm ci_base candidate was not built from the selected base" >&2; return 1; }

    is_safe_git_path "${base_dockerfile}" \
        || { echo "Unsafe ROCm base Dockerfile path: ${base_dockerfile}" >&2; return 1; }

    fetch_latest_main
    latest_main_commit="$(git rev-parse --verify "${LATEST_MAIN_REF}^{commit}")"
    if [[ "${BUILDKITE_COMMIT}" != "${latest_main_commit}" ]]; then
        echo "Skipping stable promotion: build commit is no longer latest main"
        echo "  build commit:       ${BUILDKITE_COMMIT}"
        echo "  latest main commit: ${latest_main_commit}"
        return 0
    fi
    [[ "$(latest_base_parent_digest "${base_dockerfile}")" == "${base_parent_digest}" ]] \
        || { echo "Skipping stable promotion: latest main changed the ROCm base parent digest"; return 0; }

    [[ "$(inspect_manifest_digest "${base_candidate}")" == "${base_candidate##*@}" \
        && "$(inspect_manifest_digest "${ci_candidate}")" == "${ci_candidate##*@}" ]] \
        || { echo "Candidate digest validation failed" >&2; return 1; }

    previous_base_digest=$(lookup_manifest_digest "${base_stable_tag}") \
        || previous_base_status=$?
    previous_ci_digest=$(lookup_manifest_digest "${ci_stable_tag}") \
        || previous_ci_status=$?
    if ((previous_base_status > 1 || previous_ci_status > 1)); then
        echo "Could not inspect the existing stable ROCm image aliases" >&2
        return 1
    fi
    if ((previous_base_status == 1)); then
        echo "Stable ROCm base alias is absent; promotion will bootstrap it"
    else
        previous_base_ref="$(canonical_digest_ref \
            "${base_stable_tag}@${previous_base_digest}")"
    fi
    if ((previous_ci_status == 1)); then
        echo "Stable ROCm ci_base alias is absent; promotion will bootstrap it"
    else
        previous_ci_ref="$(canonical_digest_ref \
            "${ci_stable_tag}@${previous_ci_digest}")"
    fi
    if ((previous_base_status == 0 && previous_ci_status == 0)) \
        && [[ "${previous_base_digest}" == "${base_candidate##*@}" \
        && "${previous_ci_digest}" == "${ci_candidate##*@}" ]]; then
        echo "Stable ROCm image aliases already match the selected candidates"
        return 0
    fi

    if ! promote_tag "${base_candidate}" "${base_stable_tag}" \
        || ! promote_tag "${ci_candidate}" "${ci_stable_tag}"; then
        echo "Stable ROCm image promotion failed; attempting rollback" >&2
        if ! restore_stable_tags \
            "${previous_base_ref}" "${base_stable_tag}" \
            "${previous_ci_ref}" "${ci_stable_tag}"; then
            echo "Failed to restore one or more stable ROCm image aliases" >&2
        fi
        return 1
    fi
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    main "$@"
fi
