#!/usr/bin/env bash

set -euo pipefail

mode="${1:?usage: benchmark_rdna4_all_reduce.sh correctness|benchmark|full-benchmark|profile}"
lock_dir=/tmp/team_gpu_locks
mkdir -p "$lock_dir"

# GEAK's outer gpu_lock owns GPU 0. A collective needs the remaining physical
# GPUs as one indivisible resource, so hold their standard GEAK locks too.
exec 201>"$lock_dir/gpu_1.lock"
exec 202>"$lock_dir/gpu_2.lock"
exec 203>"$lock_dir/gpu_3.lock"
flock -x -w 1200 201
flock -x -w 1200 202
flock -x -w 1200 203

export HIP_VISIBLE_DEVICES=0,1,2,3
export NCCL_PROTO=Simple
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"

python_bin="$PWD/.venv/bin/python"
bench="$PWD/benchmarks/kernels/benchmark_rdna4_all_reduce.py"

tp2_sizes="8KiB,32KiB,64KiB,64.015625KiB,128KiB,256KiB,512KiB,1MiB,2MiB,8MiB,32MiB,128MiB"
tp4_sizes="8KiB,32KiB,64KiB,128KiB,192KiB,192.015625KiB,256KiB,384KiB,512KiB,768KiB,1023.9375KiB,1MiB,2MiB,4MiB,8MiB,16MiB,32MiB,48MiB,64MiB,128MiB"

case "$mode" in
  correctness)
    common=(--providers "pynccl,routed" --executions "eager,graph" --iterations 1 --warmup 1 --samples 1)
    ;;
  benchmark)
    common=(--providers "pynccl,routed" --executions "eager,graph" --warmup 10 --samples 3)
    ;;
  full-benchmark)
    common=(--providers "pynccl,routed" --executions "eager,graph" --warmup 20 --samples 7)
    ;;
  profile)
    # One streaming case keeps profiler traces compact while exposing the P2P
    # transport's copy/reduce/synchronization structure.
    exec "$python_bin" "$bench" --world-size 4 --sizes 64MiB \
      --providers pynccl,routed --executions graph --iterations 20 \
      --warmup 5 --samples 1
    ;;
  *)
    echo "unknown mode: $mode" >&2
    exit 2
    ;;
esac

"$python_bin" "$bench" --world-size 2 --sizes "$tp2_sizes" "${common[@]}"
"$python_bin" "$bench" --world-size 4 --sizes "$tp4_sizes" "${common[@]}"
