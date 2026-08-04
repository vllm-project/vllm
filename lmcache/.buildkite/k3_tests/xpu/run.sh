#!/usr/bin/env bash
# XPU smoke test entrypoint.
# Runs directly inside the agent-stack-k8s job pod.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

log() {
  echo "[intel-xpu-unit-tests] $*"
}

fail() {
  echo "[intel-xpu-unit-tests] ERROR: $*" >&2
  exit 1
}

[ -f /opt/intel/oneapi/setvars.sh ] || fail "/opt/intel/oneapi/setvars.sh not found"
# shellcheck disable=SC1091
log "enable Intel XPU runtime environment"
source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1 || true

log "checking torch.xpu and xpu h/w availability"
python - <<'PY'
import torch

assert hasattr(torch, "xpu") and torch.xpu.is_available(), "Intel XPU not available in pod"
print("torch.xpu.is_available() = True")
PY

cd "${REPO_ROOT}"
source "${REPO_ROOT}/.buildkite/k3_harness/setup-lmcache-only-env.sh"

log "installing job dependencies"
uv pip install -r requirements/common.txt -r requirements/test.txt

discover_xpu_tests() {
  python - <<'PY'
from pathlib import Path

root = Path(".")
# Temporary allowlist of XPU cases that are known to run in the current
# vLLM-based XPU image.
#
# Future plan:
#   1. Replace this allowlist with denylist + blocklist and run full test
#      discovery by default.
#   2. denylist: CUDA-only tests that should never run in XPU jobs.
#   3. blocklist: tests that are expected to fail on current XPU runtime and
#      are not ready yet.
#   4. As tests become ready, shrink blocklist until XPU jobs run the full set.
allowlist = {
  "tests/benchmarks/test_*.py",
  "tests/test_*.py",
  "tests/cli/**/test_*.py",
  "tests/disagg/test_*.py",
  "tests/v1/mp_coordinator/**/test_*.py",
  "tests/v1/mp_observability/**/test_*.py",
  "tests/v1/multiprocess/**/test_*.py",
  "tests/v1/distributed/**/test_*.py",
  "tests/v1/test_xpu_connector.py",
  "tests/v1/test_torch_ops.py",
}
selected: set[str] = set()

for pattern in allowlist:
    for path in root.glob(pattern):
        if path.is_file():
            selected.add(path.as_posix())

for rel in sorted(selected):
    print(rel)
PY
}

mapfile -t XPU_TEST_FILES < <(discover_xpu_tests)
if [ "${#XPU_TEST_FILES[@]}" -eq 0 ]; then
  fail "no XPU-related test files found under tests"
fi

log "discovered ${#XPU_TEST_FILES[@]} XPU-related test files"
printf '  %s\n' "${XPU_TEST_FILES[@]}"

PYTEST_ARGS=(-q --maxfail=1 -m "not cuda")
if [ -n "${TEST_SELECTOR:-}" ]; then
  PYTEST_ARGS+=(-k "${TEST_SELECTOR}")
fi

log "running XPU-related tests"
pytest "${PYTEST_ARGS[@]}" "${XPU_TEST_FILES[@]}"
log "xpu smoke test finished successfully"

