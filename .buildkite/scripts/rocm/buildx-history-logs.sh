#!/usr/bin/env bash
# Helpers for saving Buildx build records as compressed plain-text logs.

_vllm_buildx_sanitize_component() {
    local component="${1:-}"

    component="${component//[^[:alnum:]._-]/_}"
    if [[ -z "${component}" || "${component}" == "." || "${component}" == ".." ]]; then
        component="build"
    fi
    printf '%s\n' "${component}"
}

_vllm_buildx_log_dir() {
    local step_name="${1:-}"
    local safe_step=""

    safe_step="$(_vllm_buildx_sanitize_component "${step_name}")"
    printf '%s/%s\n' \
        "${BUILDX_HISTORY_LOG_ROOT:-build/buildkit-logs}" \
        "${safe_step}"
}

_vllm_buildx_list_history_refs() {
    docker buildx history ls \
        --no-trunc \
        --format '{{.FullRef}}'
}

# Create a collision-safe metadata path to pass to buildx --metadata-file.
buildx_history_metadata_file() {
    local step_name="${1:-}"
    local output_dir=""
    local metadata_file=""

    output_dir="$(_vllm_buildx_log_dir "${step_name}")"
    if ! mkdir -p -- "${output_dir}"; then
        echo "WARNING: could not create Buildx history directory: ${output_dir}" >&2
        return 1
    fi
    if ! metadata_file="$(mktemp "${output_dir}/metadata.XXXXXX")"; then
        echo "WARNING: could not create a Buildx metadata file in ${output_dir}" >&2
        return 1
    fi
    printf '%s\n' "${metadata_file}"
}

# Record existing build IDs so a failed multi-target bake can later identify
# every history record created by that invocation.
snapshot_buildx_history_refs() {
    local metadata_file="${1:-}"
    local snapshot_file="${metadata_file}.history-before"
    local snapshot_refs=""
    local snapshot_tmp=""

    [[ -n "${metadata_file}" ]] || return 1
    if ! snapshot_tmp="$(mktemp "${snapshot_file}.XXXXXX")"; then
        echo "WARNING: could not create a Buildx history snapshot" >&2
        return 1
    fi
    if ! snapshot_refs=$(_vllm_buildx_list_history_refs 2>/dev/null); then
        rm -f -- "${snapshot_tmp}"
        echo "WARNING: could not snapshot existing Buildx history" >&2
        return 1
    fi
    if ! LC_ALL=C sort -u <<< "${snapshot_refs}" > "${snapshot_tmp}"; then
        rm -f -- "${snapshot_tmp}"
        echo "WARNING: could not write the Buildx history snapshot" >&2
        return 1
    fi
    if ! mv -- "${snapshot_tmp}" "${snapshot_file}"; then
        rm -f -- "${snapshot_tmp}"
        echo "WARNING: could not finalize the Buildx history snapshot" >&2
        return 1
    fi
}

_vllm_buildx_metadata_refs() {
    local metadata_file="${1:-}"
    local refs=""

    [[ -s "${metadata_file}" ]] || return 0

    if command -v jq >/dev/null 2>&1; then
        if refs="$(
            jq -r \
                '.. | objects | .["buildx.build.ref"]? |
                 select(type == "string" and length > 0)' \
                "${metadata_file}" 2>/dev/null
        )"; then
            printf '%s\n' "${refs}" | awk 'NF && !seen[$0]++'
            return 0
        fi
    fi

    if refs="$(
        grep -oE \
            '"buildx[.]build[.]ref"[[:space:]]*:[[:space:]]*"[^"]+"' \
            "${metadata_file}" 2>/dev/null
    )"; then
        printf '%s\n' "${refs}" \
            | sed -E 's/^.*:[[:space:]]*"([^"]+)"$/\1/' \
            | awk 'NF && !seen[$0]++'
    fi
}

_vllm_buildx_new_history_refs() {
    local metadata_file="${1:-}"
    local snapshot_file="${metadata_file}.history-before"
    local current_refs=""
    local history_ref=""
    local -A existing_refs=()

    [[ -f "${snapshot_file}" ]] || return 1
    while IFS= read -r history_ref; do
        [[ -n "${history_ref}" ]] || continue
        existing_refs["${history_ref}"]=1
    done < "${snapshot_file}"

    if ! current_refs=$(_vllm_buildx_list_history_refs 2>/dev/null); then
        return 1
    fi
    while IFS= read -r history_ref; do
        [[ -n "${history_ref}" ]] || continue
        if [[ -z "${existing_refs[${history_ref}]:-}" ]]; then
            printf '%s\n' "${history_ref}"
        fi
    done <<< "${current_refs}"
}

_vllm_capture_buildx_history_ref() {
    local build_ref="$1"
    local output_dir="$2"
    local index="$3"
    local builder_name=""
    local ref_id=""
    local safe_ref=""
    local log_tmp=""
    local log_path=""
    local -a builder_args=()
    local -a pipeline_status=()

    ref_id="${build_ref##*/}"
    if [[ "${build_ref}" == */*/* ]]; then
        builder_name="${build_ref%%/*}"
        builder_args=(--builder "${builder_name}")
    fi
    safe_ref="$(_vllm_buildx_sanitize_component "${ref_id}")"
    if ! log_tmp="$(
        mktemp "${output_dir}/$(printf '%03d' "${index}")-${safe_ref}.XXXXXX"
    )"; then
        echo "WARNING: could not create a Buildx history log in ${output_dir}" >&2
        return 1
    fi

    if docker buildx "${builder_args[@]}" history logs --progress=plain "${ref_id}" 2>&1 \
        | gzip -c > "${log_tmp}"; then
        pipeline_status=("${PIPESTATUS[@]}")
    else
        pipeline_status=("${PIPESTATUS[@]}")
    fi

    if [[ "${pipeline_status[0]:-1}" -ne 0 || "${pipeline_status[1]:-1}" -ne 0 ]]; then
        rm -f -- "${log_tmp}"
        echo "WARNING: could not export Buildx history log for ${build_ref}" >&2
        return 1
    fi

    log_path="${log_tmp}.log.gz"
    if ! mv -- "${log_tmp}" "${log_path}"; then
        rm -f -- "${log_tmp}"
        echo "WARNING: could not finalize Buildx history log: ${log_path}" >&2
        return 1
    fi
    echo "Saved Buildx history log for ${build_ref}: ${log_path}"
}

# Capture logs without masking the build status supplied by the caller.
capture_buildx_history_logs() {
    if [[ "$#" -ne 3 ]]; then
        echo "Usage: capture_buildx_history_logs METADATA_FILE STEP_NAME BUILD_RC" >&2
        return 2
    fi

    local metadata_file="$1"
    local step_name="$2"
    local build_rc="$3"
    local output_dir=""
    local new_refs=""
    local build_ref=""
    local index=0
    local -a build_refs=()
    local -a new_build_refs=()

    if ! [[ "${build_rc}" =~ ^[0-9]+$ ]] || ((build_rc > 255)); then
        echo "WARNING: invalid Buildx build exit status: ${build_rc}" >&2
        return 2
    fi

    output_dir="$(_vllm_buildx_log_dir "${step_name}")"
    if ! mkdir -p -- "${output_dir}"; then
        echo "WARNING: could not create Buildx history directory: ${output_dir}" >&2
        return "${build_rc}"
    fi
    if ! command -v docker >/dev/null 2>&1; then
        echo "WARNING: Docker is unavailable; skipping Buildx history logs" >&2
        return "${build_rc}"
    fi
    if ! command -v gzip >/dev/null 2>&1; then
        echo "WARNING: gzip is unavailable; skipping Buildx history logs" >&2
        return "${build_rc}"
    fi

    mapfile -t build_refs < <(_vllm_buildx_metadata_refs "${metadata_file}")

    if [[ "${build_rc}" -ne 0 ]]; then
        if [[ -f "${metadata_file}.history-before" ]]; then
            if new_refs=$(_vllm_buildx_new_history_refs "${metadata_file}"); then
                if [[ -n "${new_refs}" ]]; then
                    mapfile -t new_build_refs <<< "${new_refs}"
                    build_refs+=("${new_build_refs[@]}")
                    echo "Build failed; exporting all new Buildx history records." >&2
                fi
            else
                echo "WARNING: could not compare Buildx history for ${step_name}" >&2
            fi
        elif [[ "${#build_refs[@]}" -eq 0 ]]; then
            echo "WARNING: no pre-build history snapshot is available for ${step_name}" >&2
        fi
    fi
    mapfile -t build_refs < <(
        printf '%s\n' "${build_refs[@]}" | awk 'NF && !seen[$0]++'
    )

    if [[ "${#build_refs[@]}" -eq 0 ]]; then
        echo "WARNING: no Buildx history records were available for ${step_name}" >&2
        return "${build_rc}"
    fi

    for build_ref in "${build_refs[@]}"; do
        ((index += 1))
        _vllm_capture_buildx_history_ref \
            "${build_ref}" "${output_dir}" "${index}" || true
    done

    return "${build_rc}"
}
