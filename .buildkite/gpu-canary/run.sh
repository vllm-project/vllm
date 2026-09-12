#!/bin/bash
set -euo pipefail
CANARY_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export CI_INFRA_OTEL_DIR="${CANARY_ROOT}/.buildkite/scripts/ci-otel"
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
unset CI_INFRA_OTEL_RUNTIME_DIR CI_INFRA_OTEL_SPOOL_DIR _CI_INFRA_OTEL_INITIALIZED
cp "${CANARY_ROOT}/.buildkite/gpu-canary/test_gpu_activity.py" /tmp/test_gpu_activity.py
cp "${CANARY_ROOT}/.buildkite/gpu-canary/benchmark.py" /tmp/gpu_benchmark.py
cd /tmp
uv venv --python "$(command -v python3)" --system-site-packages /tmp/gpu-canary-venv
export PATH="/tmp/gpu-canary-venv/bin:$PATH"
sha256sum "${CI_INFRA_OTEL_DIR}/ci_gpu.py" "${CI_INFRA_OTEL_DIR}/ci_otel.py" "${CI_INFRA_OTEL_DIR}/ci_otel.sh"
nvidia-smi --query-gpu=index,uuid,name,utilization.gpu,memory.used,memory.total,mig.mode.current --format=csv,noheader,nounits
. "${CI_INFRA_OTEL_DIR}/ci_otel.sh"
ci_otel_run 1 'Controlled idle/load/retained-memory pytest cases' pytest -v -s /tmp/test_gpu_activity.py
/tmp/gpu-canary-venv/bin/python /tmp/gpu_benchmark.py calibrate
index=1
for mode in 0 1 1 0 1 0; do
  index=$((index + 1))
  export CI_INFRA_GPU_SAMPLING=$mode
  ci_otel_run "$index" "CUDA benchmark sampling=$mode" /tmp/gpu-canary-venv/bin/python /tmp/gpu_benchmark.py "$mode"
done
export CI_INFRA_GPU_SAMPLING=1
set +e
ci_otel_run 8 'Intentional exit 7 verifies failure preservation' sh -c 'sleep 3; exit 7'
status=$?
set -e
[ "$status" -eq 7 ] || exit 1
printf 'PASS: wrapped failure preserved exit %s\n' "$status"
