#!/usr/bin/env bash

# Callers provide list_content_files so each image flow can mirror its own
# Docker build context exclusions.

hash_content_file() {
    local file="$1"
    local mode="644"
    local raw_mode=""

    if [[ -L "${file}" ]]; then
        printf 'symlink:%s\n' "${file}"
        printf 'target:%s\n' "$(readlink "${file}")"
        return 0
    fi

    raw_mode=$(stat -c '%a' "${file}")
    if (((8#${raw_mode} & 0111) != 0)); then
        mode="755"
    fi
    printf 'file:%s\n' "${file}"
    printf 'mode:%s\n' "${mode}"
    sha256sum "${file}"
}

hash_content_directory() {
    local path="$1"
    local file=""
    local hash_status=0
    local list_fd=""
    local list_pid=""
    local list_status=0

    exec {list_fd}< <(list_content_files "${path}")
    list_pid=$!
    while IFS= read -r -d '' -u "${list_fd}" file; do
        if ((hash_status == 0)); then
            hash_content_file "${file}" || hash_status=$?
        fi
    done
    exec {list_fd}<&-

    wait "${list_pid}" || list_status=$?
    if ((list_status != 0)); then
        echo "Error: failed to enumerate content files under ${path}" >&2
        return "${list_status}"
    fi
    if ((hash_status != 0)); then
        echo "Error: failed to hash a content file under ${path}" >&2
        return "${hash_status}"
    fi
}

compute_content_hash() {
    local path=""

    for path in "$@"; do
        if [[ -L "${path}" ]]; then
            hash_content_file "${path}" || return $?
        elif [[ -d "${path}" ]]; then
            hash_content_directory "${path}" || return $?
        elif [[ -f "${path}" ]]; then
            hash_content_file "${path}" || return $?
        else
            printf 'missing:%s\n' "${path}"
        fi
    done | sha256sum | cut -d' ' -f1
}

resolve_image_digest() {
    local image_ref="$1"
    local attempts="${ROCM_IMAGE_DIGEST_ATTEMPTS:-4}"
    local initial_delay_secs="${ROCM_IMAGE_DIGEST_RETRY_DELAY:-2}"
    local max_delay_secs="${ROCM_IMAGE_DIGEST_RETRY_MAX_DELAY:-8}"
    local attempt=0
    local delay_secs=0
    local digest=""
    local inspect_output=""
    local inspect_status=0

    if [[ "${image_ref}" =~ @(sha256:[0-9a-f]{64})$ ]]; then
        printf '%s\n' "${BASH_REMATCH[1]}"
        return 0
    fi

    if [[ "${BASE_IMAGE_DIGEST_CACHE_READY:-0}" == "1" ]] \
        && [[ "${image_ref}" == "${BASE_IMAGE_DIGEST_CACHE_REF:-}" ]]; then
        printf '%s\n' "${BASE_IMAGE_DIGEST_CACHE_VALUE:-}"
        return 0
    fi

    if [[ ! "${attempts}" =~ ^[1-9][0-9]*$ ]]; then
        echo "Error: ROCM_IMAGE_DIGEST_ATTEMPTS must be a positive integer" >&2
        return 1
    fi
    if [[ ! "${initial_delay_secs}" =~ ^[0-9]+$ ]]; then
        echo "Error: ROCM_IMAGE_DIGEST_RETRY_DELAY must be a non-negative integer" >&2
        return 1
    fi
    if [[ ! "${max_delay_secs}" =~ ^[0-9]+$ ]]; then
        echo "Error: ROCM_IMAGE_DIGEST_RETRY_MAX_DELAY must be a non-negative integer" >&2
        return 1
    fi

    delay_secs="${initial_delay_secs}"
    if ((delay_secs > max_delay_secs)); then
        delay_secs="${max_delay_secs}"
    fi

    for ((attempt = 1; attempt <= attempts; attempt++)); do
        inspect_output=""
        inspect_status=0
        if inspect_output=$(docker buildx imagetools inspect "${image_ref}" 2>&1); then
            inspect_status=0
        else
            inspect_status=$?
        fi
        digest=$(
            sed -n -E 's/^Digest:[[:space:]]+//p' <<< "${inspect_output}" \
                | head -1 || true
        )
        if ((inspect_status == 0)) \
            && [[ "${digest}" =~ ^sha256:[0-9a-f]{64}$ ]]; then
            printf '%s\n' "${digest}"
            return 0
        fi

        if ((attempt < attempts)); then
            printf \
                'Warning: image digest lookup for %s failed (attempt %d/%d, exit status %d); retrying in %ss\n' \
                "${image_ref}" "${attempt}" "${attempts}" \
                "${inspect_status}" "${delay_secs}" >&2
            sleep "${delay_secs}"
            delay_secs=$((delay_secs * 2))
            if ((delay_secs > max_delay_secs)); then
                delay_secs="${max_delay_secs}"
            fi
        fi
    done

    printf \
        'Error: failed to resolve image digest for %s after %d attempts (last exit status %d)\n' \
        "${image_ref}" "${attempts}" "${inspect_status}" >&2
    if [[ -n "${inspect_output}" ]]; then
        echo "Last docker buildx imagetools inspect output:" >&2
        printf '%s\n' "${inspect_output}" >&2
    else
        echo "docker buildx imagetools inspect produced no output" >&2
    fi
    return 1
}
