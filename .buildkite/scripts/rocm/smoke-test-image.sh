#!/usr/bin/env bash
# Fast structural smoke test for the full ROCm CI image.

set -euo pipefail

metadata_get() {
    if command -v buildkite-agent >/dev/null 2>&1; then
        buildkite-agent meta-data get "$1" 2>/dev/null || true
    fi
}

metadata_set() {
    if command -v buildkite-agent >/dev/null 2>&1; then
        buildkite-agent meta-data set "$1" "$2"
    elif [[ "${BUILDKITE:-false}" == "true" ]]; then
        echo "buildkite-agent not found; cannot publish $1" >&2
        return 1
    fi
}

resolve_image_digest() {
    local image_ref="$1"
    local attempts="${ROCM_IMAGE_DIGEST_ATTEMPTS:-4}"
    local delay_secs="${ROCM_IMAGE_DIGEST_RETRY_DELAY:-2}"
    local output=""
    local digest=""
    local status=0
    local attempt=0

    if [[ "${image_ref}" =~ @(sha256:[0-9a-f]{64})$ ]]; then
        printf '%s\n' "${BASH_REMATCH[1]}"
        return 0
    fi
    if [[ ! "${attempts}" =~ ^[1-9][0-9]*$ \
        || ! "${delay_secs}" =~ ^[0-9]+$ ]]; then
        echo "Invalid ROCm smoke image digest retry configuration" >&2
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
        ((attempt == attempts)) || sleep "${delay_secs}"
    done
    printf 'Failed to resolve smoke image digest for %s (status %d)\n%s\n' \
        "${image_ref}" "${status}" "${output:-<no output>}" >&2
    return 1
}

pin_image_ref() {
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

main() {
    local image_ref="${VLLM_CI_SMOKE_IMAGE:-}"
    local image_digest=""
    local pinned_image=""
    local post_smoke_digest=""
    local required_ref=""
    local smoke_required=""

    if [[ "${BUILDKITE:-false}" == "true" ]]; then
        smoke_required="$(metadata_get rocm-ci-image-smoke-required)"
        case "${smoke_required}" in
            0)
                echo "Artifact-only ROCm build; no commit image to smoke-test"
                return 0
                ;;
            1) ;;
            *)
                echo "Required ROCm image smoke policy metadata is missing" >&2
                return 1
                ;;
        esac
        required_ref="$(metadata_get rocm-ci-image-smoke-ref)"
        if [[ -z "${image_ref}" ]]; then
            image_ref="${required_ref}"
        fi
        if [[ -z "${required_ref}" || "${image_ref}" != "${required_ref}" ]]; then
            echo "ROCm smoke image does not match its build-scoped handoff" >&2
            return 1
        fi
    elif [[ -z "${image_ref}" ]]; then
        image_ref="rocm/vllm-ci:${BUILDKITE_COMMIT:?set VLLM_CI_SMOKE_IMAGE or BUILDKITE_COMMIT}"
    fi

    image_digest=$(resolve_image_digest "${image_ref}") || return 1
    pinned_image=$(pin_image_ref "${image_ref}" "${image_digest}") || return 1

    docker run --rm --network=none --entrypoint /bin/bash "${pinned_image}" -ec '
  if [ ! -d /vllm-workspace ]; then echo Missing directory: /vllm-workspace >&2; exit 1; fi
  if [ ! -d /vllm-workspace/tests ]; then echo Missing directory: /vllm-workspace/tests >&2; exit 1; fi
  if [ ! -d /vllm-workspace/src/vllm ]; then echo Missing directory: /vllm-workspace/src/vllm >&2; exit 1; fi
  if [ ! -x /vllm-workspace/src/vllm/vllm-rs ]; then echo Missing executable: /vllm-workspace/src/vllm/vllm-rs >&2; exit 1; fi

  command -v python3
  command -v uv
  command -v pytest

  if ! command -v amd-smi >/dev/null 2>&1 && ! command -v rocminfo >/dev/null 2>&1; then
    echo No ROCm CLI found in image >&2
    exit 1
  fi

  python3 - <<PY
import torch
import vllm

print(torch.__version__)
print(vllm.__version__)
PY

  echo AMD image smoke OK
'

    post_smoke_digest=$(resolve_image_digest "${image_ref}") || return 1
    if [[ "${post_smoke_digest}" != "${image_digest}" ]]; then
        echo "ROCm smoke image tag changed while the smoke test was running" >&2
        return 1
    fi

    metadata_set rocm-ci-image-smoked-ref "${pinned_image}"
    metadata_set rocm-ci-image-smoked 1
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    main "$@"
fi
