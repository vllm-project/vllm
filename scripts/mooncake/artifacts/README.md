# Kimi 1P/1D Manual Bring-Up

This directory contains helper scripts for a 1 prefill node plus 1 DEP8 decode instance. The prefill uses 1 node x 4 GPUs. Decode uses 2 nodes x 4 GPUs.

The scripts assume these default repo paths:

```bash
VLLM_REPO=/home/${USER}/vllm
VLLM_BENCH_REPO=/home/${USER}/vllm
ROUTER_REPO=/home/${USER}/router
```

Override them if your checkout names differ:

```bash
export VLLM_REPO=/home/${USER}/vllm-mooncake
export VLLM_BENCH_REPO=/home/${USER}/vllm-dao
export ROUTER_REPO=/home/${USER}/router-internal
```

## Environment

Set the runtime environment before launching:

```bash
export PATH="/usr/local/cuda-13.1/bin:${PATH}"
export CUDA_HOME="/usr/local/cuda-13.1"
export LD_LIBRARY_PATH="/usr/local/cuda-13.1/lib64:${LD_LIBRARY_PATH:-}"

export HF_HOME="/home/${USER}/hf-models"
export HF_HUB_OFFLINE=true
export MOONCAKE_CONFIG_PATH="$(pwd)/mooncake_config.json"
```

Common optional overrides:

```bash
export MODEL="nvidia/Kimi-K2.5-NVFP4"
export PORT=8000
```

## Start P/D

Submit the 3-node Slurm job:

```bash
launch_1p1d_slurm.sh
```

Useful Slurm overrides:

```bash
PARTITION=batch
TIME_LIMIT=3:00:00
TOTAL_NODES=3
GPUS_PER_NODE=4
JOB_NAME=kimi-1p1d
LOG_DIR=$(pwd)/logs/my-run
```

The launcher writes:

```bash
${LOG_DIR}/nodes.env
${LOG_DIR}/prefill-<node>.log
${LOG_DIR}/decode-<node0>.log
${LOG_DIR}/decode-<node1>.log
${LOG_DIR}/slurm-<jobid>.out
```

Check status:

```bash
squeue -j <jobid>
tail -f ${LOG_DIR}/slurm-<jobid>.out
tail -f ${LOG_DIR}/prefill-<node>.log
tail -f ${LOG_DIR}/decode-<node0>.log
tail -f ${LOG_DIR}/decode-<node1>.log
```

## Start Router

Run the router separately from the P/D launcher, from a shell on the router node. For the default layout, that is the prefill node from `${LOG_DIR}/nodes.env`.

Use a separate shell:

```bash
export LOG_DIR=logs/<run-dir>
start_1p1d_router.sh \
  > "${LOG_DIR}/router.log" 2>&1
```

Or run it in the background:

```bash
export LOG_DIR=logs/<run-dir>
start_1p1d_router.sh \
  > "${LOG_DIR}/router.log" 2>&1 &
```

Router defaults:

```bash
ROUTER_PORT=8100
PROMETHEUS_PORT=29000
POLICY=round_robin
LOG_LEVEL=info
HOST=0.0.0.0
INTRA_NODE_DATA_PARALLEL_SIZE=1
```

If not using `${LOG_DIR}/nodes.env`, provide endpoints explicitly:

```bash
export PREFILL_URLS="http://<prefill-node>:8000"
export DECODE_URLS="http://<decode-node0>:8000,http://<decode-node1>:8000"
```

Check router health and metrics:

```bash
curl -fsS http://127.0.0.1:8100/health
curl -fsS http://127.0.0.1:29000/metrics
```

The router log mostly shows startup. Per-request activity is easiest to confirm through Prometheus metrics such as `vllm_router_pd_requests_total`.

## Run Load Test

Run the load test separately from a shell on the router node. It sends traffic to the router at `http://127.0.0.1:8100` by default.

Use a separate shell:

```bash
export LOG_DIR=logs/<run-dir>
run_1p1d_load_test.sh
```

Or run it in the background:

```bash
export LOG_DIR=logs/<run-dir>
run_1p1d_load_test.sh \
  > "${LOG_DIR}/perf/load_test.shell.log" 2>&1 &
```

Load-test defaults:

```bash
ROUTER_URL=http://127.0.0.1:8100
NUM_PROMPTS=75
MAX_CONCURRENCY=75
MULTI_TURN_NUM_TURNS=30
RANDOM_PREFIX_LEN=20000
RANDOM_INPUT_LEN=10000
PER_TURN_INPUT_LEN=2048
LIMIT_MIN_TOKENS=900
LIMIT_MAX_TOKENS=900
```

The load test writes:

```bash
${LOG_DIR}/perf/multi_turn_random.json
${LOG_DIR}/perf/multi_turn_stats.json
${LOG_DIR}/perf/load_test.log
```

For a small smoke test:

```bash
LOG_DIR=logs/<run-dir> \
NUM_PROMPTS=2 \
MAX_CONCURRENCY=2 \
run_1p1d_load_test.sh
```

## Stop

Cancel the Slurm job to stop P/D:

```bash
scancel <jobid>
```

If the router was launched in the background from your shell, stop that shell job or kill the router process:

```bash
jobs
kill %<job-number>
```
