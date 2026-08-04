#!/usr/bin/env bash
# Integration test entrypoint for K8s pods.
# Thin wrapper: sets up environment, then delegates to scripts/.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

cd "${REPO_ROOT}"

# ── Environment setup ────────────────────────────────────────
source .buildkite/k3_harness/setup-env.sh
uv pip install aiohttp

# ── Ensure all scripts are executable ────────────────────────
chmod +x "${SCRIPT_DIR}"/scripts/*.sh

# ── Run the actual test logic ────────────────────────────────
exec bash "${SCRIPT_DIR}/scripts/run-integration.sh" "$@"
