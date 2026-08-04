#!/usr/bin/env bash
# Correctness test entrypoint for K8s pods.
# Thin wrapper: sets up environment, then delegates to scripts/.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

cd "${REPO_ROOT}"

# ── Environment setup ────────────────────────────────────────
source .buildkite/k3_harness/setup-env.sh
uv pip install aiohttp tqdm pandas huggingface_hub

# ── Ensure all scripts are executable ────────────────────────
chmod +x "${SCRIPT_DIR}"/scripts/*.sh

# ── Run the actual test logic ────────────────────────────────
# Retry up to 3 times. vLLM's `VLLM_BATCH_INVARIANT=1` guarantee
# currently does not hold across process restarts in the vllm nightlies
# we pin to (reproduced locally: two back-to-back vanilla vllm-only runs
# produce ~55 different outputs out of 100). Phase 4's bitwise
# comparison between the base-vllm run and the vllm+LMCache run is
# bimodal (0 diff or ~50 diff) depending on which cuBLAS/kernel plans
# the two separate processes happen to pick. Retrying lets a lucky
# scheduling pass; a real LMCache regression will persistently fail.
MAX_ATTEMPTS="${CORRECTNESS_MAX_ATTEMPTS:-3}"
for attempt in $(seq 1 "${MAX_ATTEMPTS}"); do
    echo "=== Correctness attempt ${attempt}/${MAX_ATTEMPTS} ==="
    if bash "${SCRIPT_DIR}/scripts/run-correctness.sh"; then
        exit 0
    fi
    echo "[INFO] Attempt ${attempt} failed."
done
echo "[FAIL] Correctness persistently failed after ${MAX_ATTEMPTS} attempts." >&2
exit 1
