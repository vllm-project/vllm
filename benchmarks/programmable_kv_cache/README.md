# Programmable KV cache benchmarks

This experimental implementation follows [RFC #57103](https://github.com/vllm-project/vllm/issues/57103).
The [interface reference](../../vllm/v1/kv_hints/README.md) describes native actions, lifecycle semantics, resource limits and supported layouts.
Run the commands below from the repository root using the project virtual environment.
Replace model, dataset and GPU arguments with resources available on your machine.
Outputs under `benchmarks/programmable_kv_cache/results/` are ignored by Git.

## Serving and functional validation

Start a local server with prefix caching and native control enabled:

```bash
.venv/bin/python -m vllm.entrypoints.openai.api_server \
  --model /path/to/model \
  --served-model-name kv-validation \
  --tensor-parallel-size 2 \
  --max-model-len 4096 \
  --enable-prefix-caching \
  --enable-prompt-tokens-details \
  --kv-control-config '{"engine_id":"kv-validation"}' \
  --host 127.0.0.1 --port 18083
```

To test CPU transfers, add this connector configuration:

```bash
--kv-transfer-config '{"kv_connector":"OffloadingConnector","kv_role":"kv_both","kv_connector_extra_config":{"cpu_bytes_to_use":8589934592,"blocks_per_chunk":1}}'
```

For Mooncake transfers, install the matching wheel from `requirements/kv_connectors.txt`, start a Store master and save its client configuration:

```bash
.venv/bin/mooncake_master \
  --rpc_address=127.0.0.1 --rpc_port=19151 \
  --metrics_host=127.0.0.1 --metrics_port=19153 \
  --enable_metric_reporting=false
```

```json
{
  "metadata_server": "P2PHANDSHAKE",
  "master_server_address": "127.0.0.1:19151",
  "protocol": "tcp",
  "device_name": "",
  "global_segment_size": "8GB",
  "local_buffer_size": "256MB"
}
```

Set `MOONCAKE_CONFIG_PATH` to that file before starting the server, and add:

```bash
--kv-transfer-config '{"kv_connector":"MooncakeStoreConnector","kv_role":"kv_both","kv_connector_extra_config":{"enable_lookup":false,"enable_save":false,"save_decode_cache":false}}'
```

Disabling automatic connector lookup and save isolates the effects of explicit COPY/MOVE operations.
Native Mooncake transfers require PCP=DCP=1 and targets aligned to every cache group's supported boundaries.
CPU transfers require one block per chunk. Query `/v1/kv/capabilities` for the active layout and transfer directions.

Exercise attached and independent operations, actual cache reuse and priority:

```bash
.venv/bin/python -m benchmarks.programmable_kv_cache.http_control \
  --base-url http://127.0.0.1:18083 \
  --engine-id kv-validation \
  --dataset /path/to/gsm8k \
  --output /tmp/kv-control.json
```

The dataset is a downloaded GSM8K repository containing `main/*.parquet`.
Use `--tier cpu` for the CPU connector, `--evaluation-samples 128` for the quality comparison, or `--destination-url` for a compatible second engine sharing the Store.
For `http_control --policy`, replace the server's control configuration with:

```bash
--kv-control-config '{"engine_id":"kv-validation","policies":[{"name":"retention","implementation":"retention"},{"name":"movement","implementation":"tool_wait","options":{"tier":"mooncake"}}]}'
```

A completed signal receipt still requires collecting its separately identified derived operations.

## Repeated rollout comparisons

`verl_rollout.py` uses verl's `ToolAgentLoop`, executes tools and records generated tokens, logprobs, response masks, rewards and control outcomes.
It requires an importable verl checkout and its Agent Loop dependencies.

`run_rollout_matrix.py` starts and cleans up servers, a Mooncake Store and GPU sampling processes.
It checks port ownership, warms the server, resets caches between runs, and records commands, source hashes, workload fingerprints and timing data.
Without `--trace`, it collects a live trace before the timed fixed-work replays:

```bash
.venv/bin/python -m benchmarks.programmable_kv_cache.run_rollout_matrix \
  --model /path/to/model --dataset /path/to/gsm8k \
  --gpus 0,1 --gpu-memory-utilization 0.5 \
  --prompts 64 --samples 4 --families 32 --fewshot 48 \
  --concurrency 64 --max-num-seqs 64 \
  --arms native,idle,noop,policy --repeats 3 \
  --output-dir /tmp/kv-comparison
```

| Mode or option | Purpose |
| --- | --- |
| `native` | Native automatic prefix caching |
| `idle` / `noop` | Controller enabled; controller plus connector enabled; neither sends hints |
| `priority` / `movement` / `composed` | Application-issued priority, transfer, or combined commands |
| `policy` | Signal-driven retention, movement and quota policies |
| `--policy-components quota` | Isolate scheduling without a movement connector |
| `--policy-components retention` | Isolate retention; can be combined with quota |
| `--trace FILE` | Replay recorded inputs, output lengths and pauses |
| `--live-after-replay` / `--live-only` | Run actual tool loops separately from replay measurements |
| `--prediction-mode adaptive` | Estimate pauses from completed tool batches |
| `--prediction-mode fixed --prediction-seconds N` | Use a fixed pause estimate |
| `--oracle` | Use recorded future pauses during replay; label this diagnostic separately |
| `--prefetch-on-admission` | Prepare queued sessions when selected by the scheduler |
| `--movement-pressure-only` | Disable the movement pause-cost filter for an ablation |
| `--cache-gib N` | Set an explicit GPU KV budget for a capacity diagnostic |
| `--profile` | Collect CPU/GPU profiles separately from throughput runs; requires `yappi` |

Use `--help` for policy thresholds, waiting budgets, generation settings and profiling windows.
Compare configurations at the same GPU KV budget and report the additional host memory and CPU used by movement.
Use separate output directories when changing policy components or workload settings.

Fixed-work replay regenerates output using recorded input tokens and output lengths; it does not execute tools or compute rewards.
Live trajectories can differ between configurations and require separate quality comparisons.
Elapsed time includes client queueing, tool waits, control collection and transfer draining, and excludes startup and explicit warmup.
GPU occupancy audits must cover the measured window. Profiled runs are excluded from ordinary throughput comparisons.

```bash
.venv/bin/python -m benchmarks.programmable_kv_cache.analyze_rollout \
  --run native=/path/to/native.json.gz \
  --run policy=/path/to/policy.json.gz \
  --baseline native --output /tmp/kv-comparison.json
```

Repeat `--run` for repeated measurements under the same label.
The analyzer checks equal workload fingerprints and serving conditions, and reports throughput, tail latency and forecast errors.
For live quality, it separates reported rewards from complete final answers: a correct incidental number in truncated output does not establish a completed correct answer.

## Additional validation tools

| Module | Scope and required arguments |
| --- | --- |
| `evaluate` | Recomputed/APC model quality: `--model --dataset --output`; optionally `--enable-kv-control` |
| `lifecycle` | Shared branches, TTL, cancellation and resume races: `--base-url --trace --output` |
| `weight_update` | Real parameter update, invalidation and rollback: `--base-url --model --trace --output` |
| `control_overhead` | CPU-only result polling and cached guidance: `--output` |
| `run_recovery` | Managed server/Store validation: `--model --dataset --trace --output-dir`; optional `--lifecycle`, `--weight-update`, `--evaluation-samples 128` or `--cpu-evaluation-samples 128` |

Invoke these as `.venv/bin/python -m benchmarks.programmable_kv_cache.MODULE`.
`run_recovery --capacity-deferral-only` tests partial loads, budget refusal, retries and recomputation with a one-block load budget; it is a correctness diagnostic.
Weight-update validation requires the same-host model checkpoint, an idle server, `--weight-transfer-config '{"backend":"ipc"}'`, `VLLM_SERVER_DEV_MODE=1` and `VLLM_ALLOW_INSECURE_SERIALIZATION=1` on a locally bound server.
It changes real GPU parameters and restores them; it does not run an optimizer.

## Regression checks

```bash
.venv/bin/python -m pytest \
  tests/v1/test_kv_hint_control.py \
  tests/v1/test_kv_hint_state.py \
  tests/v1/engine/test_kv_control.py \
  tests/entrypoints/serve/kv_control/test_api.py \
  tests/v1/core/test_prefix_caching.py \
  tests/v1/core/test_scheduler.py \
  tests/v1/kv_connector/unit/test_mooncake_store_scheduler.py -q
```

`tests/v1/kv_connector/mooncake_integration/test_store_buffers.py` additionally validates real Store transfers across two CUDA GPUs, including missing-rank and reset behavior.
It requires the matching native Mooncake wheel and either a local master executable or `VLLM_TEST_MOONCAKE_MASTER`.
Model evaluations and hardware-dependent transfer checks should accompany changes to cache contents or execution behavior.
