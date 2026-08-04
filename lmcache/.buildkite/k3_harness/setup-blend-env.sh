#!/usr/bin/env bash
# Per-job environment setup for CacheBlend-plugin compatibility tests.
 
set -euo pipefail

trap 'echo "ERROR: setup-blend-env.sh failed at line $LINENO (exit code $?)" >&2' ERR

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

# 1. Full env: vLLM nightly + LMCache(PR) from source (+ c_ops for this GPU).
#    setup-env.sh already runs the GPU health pre-check.
source "${REPO_ROOT}/.buildkite/k3_harness/setup-env.sh"

# setup-env.sh installed its own ERR trap; restore ours for accurate attribution.
trap 'echo "ERROR: setup-blend-env.sh failed at line $LINENO (exit code $?)" >&2' ERR

# 2. Plugin-harness dep not in the LMCache/vLLM dependency closure.
echo "--- :python: Installing cacheblend-plugin harness deps"
uv pip install httpx

# 3. Pull the cacheblend-plugin. Coordinates come from the Buildkite pipeline
#    env; CB_PLUGIN_REPO is required (set it in the pipeline env so the repo
#    owner/name is not hardcoded here). REF/DIR have sensible fallbacks.
CB_PLUGIN_REPO="${CB_PLUGIN_REPO:?CB_PLUGIN_REPO not set — set the plugin repo (owner/name) in the Buildkite pipeline env}"
CB_PLUGIN_REF="${CB_PLUGIN_REF:-main}"
CB_PLUGIN_DIR="${CB_PLUGIN_DIR:-/tmp/cb-plugin}"

echo "--- :arrow_down: Pulling ${CB_PLUGIN_REPO}@${CB_PLUGIN_REF}"
_cb_tok="${GH_TOKEN:-${GITHUB_TOKEN:-}}"
echo "    GH_TOKEN present: $([ -n "${_cb_tok}" ] && echo yes || echo NO)"
if [ -z "${_cb_tok}" ]; then
    echo "ERROR: GH_TOKEN/GITHUB_TOKEN not set. ${CB_PLUGIN_REPO} is private — set a" >&2
    echo "       fine-grained PAT (resource owner = the repo's org, Contents: read)" >&2
    echo "       as GH_TOKEN in the Buildkite pipeline env." >&2
    exit 1
fi
_cb_url="https://x-access-token:${_cb_tok}@github.com/${CB_PLUGIN_REPO}.git"
# Isolate from the Buildkite agent's git config: its checkout credential is a
# GitHub App token scoped to LMCache (injected via the GLOBAL git config —
# insteadOf / extraheader / credential.helper) and 403s on the plugin repo.
# Ignoring global+system config forces git to use ONLY our GH_TOKEN from the URL.
_cb_git=(env GIT_CONFIG_GLOBAL=/dev/null GIT_CONFIG_SYSTEM=/dev/null
         git -c credential.helper=)
rm -rf "${CB_PLUGIN_DIR}"
# --branch handles a branch/tag; a commit SHA needs a full clone + checkout.
if ! "${_cb_git[@]}" clone --depth 1 --branch "${CB_PLUGIN_REF}" "${_cb_url}" "${CB_PLUGIN_DIR}" 2>/dev/null; then
    "${_cb_git[@]}" clone "${_cb_url}" "${CB_PLUGIN_DIR}"
    git -C "${CB_PLUGIN_DIR}" checkout "${CB_PLUGIN_REF}"
fi
echo "    cacheblend-plugin @ $(git -C "${CB_PLUGIN_DIR}" rev-parse --short HEAD) (ref=${CB_PLUGIN_REF})"

# 4. Install the plugin editable; --no-deps keeps the env's torch/vllm/lmcache,
#    --no-build-isolation because the plugin is pure-Python (no compile step).
echo "--- :python: Installing cacheblend-plugin (editable)"
git config --global --add safe.directory "${CB_PLUGIN_DIR}" 2>/dev/null || true
uv pip install -e "${CB_PLUGIN_DIR}" --no-deps --no-build-isolation

# 5. Import/registration smoke so a broken install fails here, not 180s into a
#    server-boot timeout inside the harness.
python -c "import lmcache_cacheblend; print('cacheblend-plugin imported OK')"

export CB_PLUGIN_DIR CB_PLUGIN_REF
echo "--- :white_check_mark: CacheBlend env ready (plugin @ ${CB_PLUGIN_REF}, dir=${CB_PLUGIN_DIR})"
