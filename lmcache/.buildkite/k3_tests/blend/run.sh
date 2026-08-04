#!/usr/bin/env bash
 
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

cd "${REPO_ROOT}"

# ── Private plugin coordinates (kept out of this public repo) ──
# GH_TOKEN (clone auth) and CB_PLUGIN_REPO (private owner/name): use an exported
# env var if set, else the Buildkite secret. Missing values surface the error in
# setup-blend-env.sh.
if command -v buildkite-agent >/dev/null 2>&1; then
    if [ -z "${GH_TOKEN:-}" ]; then
        GH_TOKEN="$(buildkite-agent secret get BLEND_TM_PAT 2>/dev/null || true)"
        export GH_TOKEN
    fi
    if [ -z "${CB_PLUGIN_REPO:-}" ]; then
        CB_PLUGIN_REPO="$(buildkite-agent secret get CB_PLUGIN_REPO 2>/dev/null || true)"
        export CB_PLUGIN_REPO
    fi
fi

# ── Environment setup ────────────────────────────────────────
source .buildkite/k3_harness/setup-blend-env.sh

# ── Ensure all scripts are executable ────────────────────────
chmod +x "${SCRIPT_DIR}"/scripts/*.sh

# ── Run the actual test logic ────────────────────────────────
exec bash "${SCRIPT_DIR}/scripts/run-compat.sh" "$@"
