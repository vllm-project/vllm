#!/usr/bin/env bash
set -euo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
engine=${1:?Usage: run.sh ENGINE MODE [--port PORT] [--output DIR]}
case "$engine" in vllm|sglang) ;; *) echo 'Engine must be vllm or sglang' >&2; exit 2 ;; esac
# Each engine needs its own environment because their torch versions differ.
engine_python=${ENGINE_PYTHON:-"$root/.deps/$engine/bin/python"}
exec canhazgpu run --gpus 1 -- "$engine_python" "$root/benchmarks/dflash_4b/run.py" "$@"
