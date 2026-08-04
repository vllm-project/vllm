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

main() {
    local image_ref="${VLLM_CI_SMOKE_IMAGE:-}"
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

    docker run --rm --network=none --entrypoint /bin/bash "${image_ref}" -ec '
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

    metadata_set rocm-ci-image-smoked-ref "${image_ref}"
    metadata_set rocm-ci-image-smoked 1
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    main "$@"
fi
