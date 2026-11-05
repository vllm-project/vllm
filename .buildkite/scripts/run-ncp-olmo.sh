#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
set -Eeuo pipefail

if [[ ${1:-} != --bounded ]]; then
    exec timeout --signal=TERM --kill-after=30s 60m bash "$0" --bounded "$@"
fi
shift
REPO=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$REPO"
PYTHON=${NCP_TEST_PYTHON:-$(uv python find)}
# Preserve the venv interpreter symlink: resolving it selects the base Python.
PYTHON=$(cd -- "$(dirname -- "$PYTHON")" && pwd)/$(basename -- "$PYTHON")
test -x "$PYTHON"
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export VLLM_USE_V2_MODEL_RUNNER=1 VLLM_BATCH_INVARIANT=1 VLLM_GPU_SYNC_CHECK=error
export VLLM_USE_FLASHINFER_SAMPLER=0
export NCP_OLMO_TEST_MODEL=${NCP_OLMO_TEST_MODEL:-ArchSpace-Collection/NCP_ArchPreview_dolma3_8.9B_Stage2_v1}
HF_ENV=${NCP_OLMO_HF_ENV_DIR:-$REPO/.ncp-hf}
if [[ ! -x "$HF_ENV/bin/python" ]]; then
    uv venv "$HF_ENV" --python "$PYTHON"
fi
env -u UV_OVERRIDE uv pip install --python "$HF_ENV/bin/python" \
    --require-hashes --torch-backend cu130 -r requirements/test/ncp_hf.txt
uv pip check --python "$HF_ENV/bin/python"
export NCP_OLMO_HF_PYTHON
NCP_OLMO_HF_PYTHON=$(cd -- "$HF_ENV/bin" && pwd)/python
"$PYTHON" - <<'PY'
import json
import os
import subprocess
from importlib.metadata import version

packages = ("torch", "torchvision", "numpy", "tokenizers", "safetensors")
command = (
    "import json; from importlib.metadata import version, distributions; "
    "assert not any(d.metadata['Name'].lower() == 'vllm' for d in distributions()); "
    f"print(json.dumps({{k:version(k) for k in {packages!r}}}))"
)
actual = json.loads(subprocess.check_output(
    [os.environ["NCP_OLMO_HF_PYTHON"], "-I", "-c", command], text=True
))
assert actual == {k: version(k) for k in packages}, actual
print("NCP_REFERENCE_ENVIRONMENT_OK", actual, flush=True)
PY
if [[ ${1:-} == --setup-only ]]; then
    exit 0
fi

mkdir -p tests/artifacts/ncp-olmo
RESULT=${NCP_OLMO_CI_RESULT_DIR:-$(mktemp -d "$REPO/tests/artifacts/ncp-olmo/run.XXXXXXXX")}
mkdir -p "$RESULT"
RESULT=$(realpath "$RESULT")
record_exit() {
    local code=$?
    printf '%s\n' "$code" > "$RESULT/exit-status.txt"
    if [[ ${BUILDKITE:-false} == true ]]; then
        (cd "$RESULT" && buildkite-agent artifact upload '**/*') || code=1
    fi
    exit "$code"
}
trap record_exit EXIT
export TMPDIR
TMPDIR=$(mktemp -d /tmp/ncp-ci.XXXXXXXX)
export VLLM_RPC_BASE_PATH=$TMPDIR
export PATH="$(dirname "$PYTHON"):$PATH"
PYTHON_DIR=$(dirname "$PYTHON")
# Use the selected environment's standard launcher, including in CI discovery.
test "$("$PYTHON" -c 'import sys; print(sys.prefix, sys.version)')" = \
    "$("$PYTHON_DIR/python3" -c 'import sys; print(sys.prefix, sys.version)')"
uv pip freeze --python "$PYTHON" > "$RESULT/vllm-environment.txt"
uv pip freeze --python "$NCP_OLMO_HF_PYTHON" > "$RESULT/hf-environment.txt"
export NCP_OLMO_CI_JOURNAL=$RESULT/initialization-events.jsonl
"$PYTHON_DIR/python3" -m pytest -v -s -p tests.models.ncp_olmo.ci_guard \
    tests/models/test_initialization.py -k NCPOlmo3ForCausalLM \
    --junitxml="$RESULT/initialization-junit.xml"
export NCP_OLMO_CI_JOURNAL=$RESULT/model-events.jsonl
"$PYTHON_DIR/python3" -m pytest -v -s -p tests.models.ncp_olmo.ci_guard \
    tests/models/ncp_olmo \
    tests/models/language/generation/test_ncp_olmo.py \
    --basetemp="$RESULT/pytest" --junitxml="$RESULT/model-junit.xml"
printf 'NCP_OLMO_CI_OK model=%s\n' "$NCP_OLMO_TEST_MODEL" | tee "$RESULT/status.txt"
