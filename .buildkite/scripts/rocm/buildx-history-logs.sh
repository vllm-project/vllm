#!/usr/bin/env bash
# Save Buildx build records as compressed plain-text logs.

_buildx_safe_name() {
    local name="${1//[^[:alnum:]._-]/_}"
    [[ -n "${name}" && "${name}" != "." && "${name}" != ".." ]] || name=build
    printf '%s\n' "${name}"
}

_buildx_log_dir() {
    printf '%s/%s\n' "${BUILDX_HISTORY_LOG_ROOT:-build/buildkit-logs}" \
        "$(_buildx_safe_name "${1:-}")"
}

_buildx_history_refs() {
    docker buildx history ls --no-trunc --format '{{.FullRef}}'
}

buildx_history_metadata_file() {
    local output_dir
    output_dir="$(_buildx_log_dir "${1:-}")"
    mkdir -p -- "${output_dir}" || {
        echo "WARNING: could not create ${output_dir}" >&2
        return 1
    }
    mktemp "${output_dir}/metadata.XXXXXX"
}

snapshot_buildx_history_refs() {
    local snapshot="${1}.history-before"
    if ! _buildx_history_refs > "${snapshot}" 2>/dev/null; then
        rm -f -- "${snapshot}"
        echo "WARNING: could not snapshot Buildx history" >&2
        return 1
    fi
}

_buildx_metadata_refs() {
    [[ -s "$1" ]] || return 0
    grep -oE '"buildx[.]build[.]ref"[[:space:]]*:[[:space:]]*"[^"]+"' \
        "$1" 2>/dev/null \
        | sed -E 's/^.*:[[:space:]]*"([^"]+)"$/\1/' || true
}

_export_buildx_log() {
    local build_ref="$1"
    local output_dir="$2"
    local index="$3"
    local ref_id="${build_ref##*/}"
    local builder="${build_ref%%/*}"
    local log_path
    local -a builder_args=()
    local -a statuses=()

    [[ "${build_ref}" != */*/* ]] || builder_args=(--builder "${builder}")
    log_path=$(mktemp "${output_dir}/$(printf '%03d' "${index}")-$(
        _buildx_safe_name "${ref_id}"
    ).XXXXXX.log.gz") || return 1

    if docker buildx "${builder_args[@]}" history logs \
        --progress=plain "${ref_id}" 2>&1 | gzip -c > "${log_path}"; then
        statuses=("${PIPESTATUS[@]}")
    else
        statuses=("${PIPESTATUS[@]}")
    fi
    if ((statuses[0] != 0 || statuses[1] != 0)); then
        rm -f -- "${log_path}"
        echo "WARNING: could not export Buildx history log for ${build_ref}" >&2
        return 1
    fi
    echo "Saved Buildx history log for ${build_ref}: ${log_path}"
}

capture_buildx_history_logs() {
    local metadata_file="$1" step_name="$2" build_rc="$3"
    local output_dir refs ref index=0 exported=0
    local -A before=() seen=()
    local -a build_refs=()

    output_dir="$(_buildx_log_dir "${step_name}")"
    mkdir -p -- "${output_dir}" || return "${build_rc}"
    while IFS= read -r ref; do
        [[ -z "${ref}" || -n "${seen[${ref}]:-}" ]] || {
            build_refs+=("${ref}")
            seen["${ref}"]=1
        }
    done < <(_buildx_metadata_refs "${metadata_file}")
    if [[ -f "${metadata_file}.history-before" ]]; then
        while IFS= read -r ref; do
            [[ -z "${ref}" ]] || before["${ref}"]=1
        done < "${metadata_file}.history-before"
        if refs=$(_buildx_history_refs 2>/dev/null); then
            while IFS= read -r ref; do
                [[ -z "${ref}" || -n "${before[${ref}]:-}" \
                    || -n "${seen[${ref}]:-}" ]] || {
                    build_refs+=("${ref}")
                    seen["${ref}"]=1
                }
            done <<< "${refs}"
        fi
    fi
    for ref in "${build_refs[@]}"; do
        ((index += 1))
        if _export_buildx_log "${ref}" "${output_dir}" "${index}"; then
            ((exported += 1))
        fi
    done
    if ((exported > 0)) && command -v buildkite-agent >/dev/null 2>&1; then
        buildkite-agent artifact upload "${output_dir}/*.log.gz" \
            || echo "WARNING: could not upload Buildx history logs" >&2
    fi
    rm -f -- "${metadata_file}" "${metadata_file}.history-before"
    return "${build_rc}"
}
