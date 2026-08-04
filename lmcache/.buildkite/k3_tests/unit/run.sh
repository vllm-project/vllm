#!/usr/bin/env bash
# Unit test entrypoint for K8s pods.
# Installs LMCache (no vLLM) + test deps, then runs pytest with coverage.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

cd "${REPO_ROOT}"

# ── Per-build scratch dir ────────────────────────────────────
# /scratch is a shared hostPath mount (see pipeline.yml) backed by ext4/xfs
# on local NVMe — the only place GDS (cuFile) works in this pod (overlayfs
# /tmp fails with CU_FILE_IO_NOT_SUPPORTED). Give this build its own
# subdirectory so concurrent pods don't collide, and clean it up on exit.
# Direct subdir rather than K8s subPathExpr since the latter's bind mount
# breaks cuFile's fs-type detection.
#
# LMCACHE_TEST_TMPDIR is consumed by the handful of test files that need
# GDS-capable scratch (test_gds_backend, test_cache_engine, the xpu
# benchmarks). Leaving TMPDIR unset means /tmp stays pod-internal overlay
# for pip/uv/build caches — those don't need direct I/O.
BUILD_TAG="${BUILDKITE_BUILD_ID:-manual-$$}"
export LMCACHE_TEST_TMPDIR="/scratch/bk-${BUILD_TAG}"
mkdir -p "${LMCACHE_TEST_TMPDIR}"
trap 'rm -rf "${LMCACHE_TEST_TMPDIR}" 2>/dev/null || true' EXIT

# ── Environment setup ────────────────────────────────────────
source .buildkite/k3_harness/setup-lmcache-only-env.sh
uv pip install -r requirements/test.txt

# ── Run unit tests with coverage ─────────────────────────────
LMCACHE_TRACK_USAGE="false" \
pytest --maxfail=1 --cov=lmcache \
    --cov-report term --cov-report=html:coverage-test \
    --cov-report=xml:coverage-test.xml --html=durations/test.html \
    --ignore=tests/disagg --ignore=tests/v1/test_pos_kernels.py \
    --ignore=tests/skipped \
    --ignore=tests/v1/storage_backend/test_eic.py

cat << EOF | buildkite-agent annotate --style "info"
  Read the <a href="artifact://coverage-test/index.html">uploaded coverage report</a>
EOF
