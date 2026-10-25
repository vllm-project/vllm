#!/bin/bash
# Usage (inside a throwaway CPU-only container of the base image, repo
# overlay dir mounted at /work, upstream v0.32.0 tests at /work/upstream_tests):
#   run_tests.sh stock|patched
#
# Host-side driver (no GPU, does not touch the running prod container):
#   docker run --rm --entrypoint bash \
#     -v <repo>/myia_vllm/configs/docker/overlay_v0320_fsadmit:/work \
#     -v <scratch>/upstream_tests_v0320:/work/upstream_tests \
#     vllm-openai-v0320-tierpin:v1 /work/tests/run_tests.sh patched
#
# upstream_tests_v0320/ holds the three suites at tag v0.32.0 (fetch with):
#   for f in tiering/test_tiering_offloading.py tiering/test_fs_tier.py cpu/test_manager.py; do
#     mkdir -p "$(dirname "upstream_tests_v0320/$f")"
#     curl -fsSL "https://raw.githubusercontent.com/vllm-project/vllm/v0.32.0/tests/v1/kv_offload/$f" -o "upstream_tests_v0320/$f"
#   done
set -u
MODE="$1"
P=$(python3 -c "import vllm,os;print(os.path.dirname(vllm.__file__))")
python3 -c "import pytest" 2>/dev/null || pip install -q pytest >/dev/null 2>&1
if [ "$MODE" = "patched" ]; then
  cp /work/vllm/v1/kv_offload/tiering/manager.py "$P/v1/kv_offload/tiering/manager.py"
fi

echo "=== $MODE: admission regression tests (default env, filter ON at 2)"
cd /work/tests || exit 2
python3 -m pytest -q -p no:cacheprovider test_fs_admission.py 2>&1 | tail -12

if [ "$MODE" = "patched" ]; then
  echo "=== $MODE: upstream v0.32.0 tiering suite (default env, CPU-medium tiers)"
  cd /work/upstream_tests || exit 2
  python3 -m pytest -q -p no:cacheprovider tiering/test_tiering_offloading.py 2>&1 | tail -6

  echo "=== $MODE: upstream v0.32.0 fs-tier + cpu-manager suites (filter OFF = rollback mode must be byte-identical)"
  MYIA_FS_ADMIT_MIN_HITS=0 python3 -m pytest -q -p no:cacheprovider tiering/test_fs_tier.py 2>&1 | tail -6
  MYIA_FS_ADMIT_MIN_HITS=0 python3 -m pytest -q -p no:cacheprovider cpu/test_manager.py 2>&1 | tail -6

  echo "=== $MODE: upstream fs-tier events test under the DEFAULT filter (expected: this one test fails — the filter is the point)"
  python3 -m pytest -q -p no:cacheprovider tiering/test_fs_tier.py -k test_cascade_store_emits_fs_event_through_tiering_manager 2>&1 | tail -6
fi
