#!/usr/bin/env bash
# Per-job environment setup for jobs that DON'T need vLLM (e.g. unit tests).
# Installs LMCache from source on top of the ci-base image, which already
# has torch + requirements/cuda.txt + build.txt baked in. Much faster than
# setup-env.sh since it skips the vLLM nightly install entirely.
set -euo pipefail

trap 'echo "ERROR: setup-lmcache-only-env.sh failed at line $LINENO (exit code $?)" >&2' ERR

# ── GPU health pre-check ────────────────────────────────────
# Fail fast if GPUs are occupied by stale host processes.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "${REPO_ROOT}/.buildkite/k3_tests/common_scripts/helpers.sh"
check_gpu_health 80

echo "--- :python: Installing LMCache from source (no vLLM)"
# Skip setuptools_scm git describe; the repo carries non-PEP-440 tags
# (nightly, nightly-cu13) that crash the newer vcs_versioning backend.
export SETUPTOOLS_SCM_PRETEND_VERSION_FOR_LMCACHE="${SETUPTOOLS_SCM_PRETEND_VERSION_FOR_LMCACHE:-0.0.0+ci}"
uv pip install -e . --no-build-isolation

echo "--- :white_check_mark: Environment ready (LMCache only, no vLLM)"
python -c "import lmcache; print('LMCache installed from source')"
