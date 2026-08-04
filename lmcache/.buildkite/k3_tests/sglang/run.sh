#!/usr/bin/env bash
# SGLang + LMCache MP integration test entrypoint for K8s pods.
# Sources the SGLang-specific env setup, then dispatches to scripts/.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

source .buildkite/k3_harness/setup-sglang-env.sh

chmod +x "${SCRIPT_DIR}"/scripts/*.sh

case "${1:-}" in
    correctness)
        exec bash "${SCRIPT_DIR}/scripts/run-correctness.sh"
        ;;
    perf)
        exec bash "${SCRIPT_DIR}/scripts/run-perf.sh"
        ;;
    *)
        echo "Usage: $0 {correctness|perf}" >&2
        exit 2
        ;;
esac
