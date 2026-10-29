#!/bin/bash
# Usage (inside a throwaway container of the target image): run_tests.sh stock|patched
set -u
MODE="$1"
P=$(python3 -c "import vllm,os;print(os.path.dirname(vllm.__file__))")
python3 -c "import pytest" 2>/dev/null || pip install -q pytest >/dev/null 2>&1
if [ "$MODE" = "patched" ]; then
  cp /work/patched/cpu_manager.py "$P/v1/kv_offload/cpu/manager.py"
  cp /work/patched/tiering_manager.py "$P/v1/kv_offload/tiering/manager.py"
fi
cd /work/upstream_tests || exit 2
echo "=== $MODE: regression tests"
python3 -m pytest -q -p no:cacheprovider test_step_hit_protection.py 2>&1 | tail -15
if [ "$MODE" = "patched" ]; then
  echo "=== $MODE: upstream v0.32.0 suites"
  python3 -m pytest -q -p no:cacheprovider test_tiering_offloading.py cpu/test_manager.py 2>&1 | tail -8
fi
