#!/usr/bin/env bash
# Comprehensive integration test entrypoint for K8s pods.
# Usage: run.sh <config.yaml>
# Thin wrapper: sets up environment, then delegates to scripts/.
set -euo pipefail

CFG_NAME="${1:?Usage: $0 <config.yaml>}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

cd "${REPO_ROOT}"

# ── Environment setup ────────────────────────────────────────
source .buildkite/k3_harness/setup-env.sh

# Install test utilities (yq for YAML parsing, jq for JSON, openai/pandas/matplotlib for benchmarks)
uv pip install yq jq openai pandas matplotlib 2>/dev/null || true

# ── Ensure all scripts are executable ────────────────────────
chmod +x "${SCRIPT_DIR}"/scripts/*.sh

# ── Run the actual test logic ────────────────────────────────
exec bash "${SCRIPT_DIR}/scripts/run-single-config.sh" "$CFG_NAME"
