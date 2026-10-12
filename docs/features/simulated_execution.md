# Simulated Execution

Simulated execution runs the real vLLM engine (scheduler, KV-cache manager,
prefix caching, chunked prefill, preemption, async scheduling and the OpenAI
frontend) while replacing the model forward pass with a timing model. It
answers questions such as "what TTFT and TPOT would this configuration give on
that hardware" without the hardware, downloading weights or running kernels,
and it lets scheduler and KV-cache behavior be exercised deterministically on
a CPU-only CI node.

## How it works

* The model is built on the `meta` device, so vLLM's own KV-cache specs,
  attention metadata and hybrid/sliding-window layouts are available without
  parameters in memory.
* The KV cache is virtual: block accounting runs through the normal path with
  the capacity given by `--kv-cache-memory-bytes`, but no tensors are
  allocated.
* Output tokens come from `SamplingParams.trace_decode_token_ids` (see
  [trace replay](../serving/online_serving/trace_replay.md)) or, for requests
  without a trace, from `fallback_token_id`; such requests end by
  `max_tokens`.
* Each step's duration comes from a *timing provider*. The provider sees a
  plain description of the step and returns how long the device would be
  busy. The time is paid when the engine core consumes the step's output, so
  the engine keeps scheduling the next step meanwhile, exactly as with a real
  GPU under async scheduling. The device is modeled as a serial resource:
  steps queue behind the previous one.

The worker runs on CUDA-alike hosts as well as the CPU platform. On a CUDA host
it binds a device for the runner's small request-state buffers and uses the
stock Triton input-preparation kernels; on the CPU platform the same kernels
need Triton's CPU backend.

## Usage

```bash
vllm serve Qwen/Qwen3-0.6B \
  --simulate-forward \
  --kv-cache-memory-bytes 8589934592 \
  --simulation-config '{"timing": "constant",
                        "timing_args": {"fixed_us": 3000,
                                        "prefill_us_per_token": 20,
                                        "decode_us_per_req": 40},
                        "log_path": "/tmp/sim_steps.jsonl"}'
```

Any client works unchanged, including `vllm bench serve`. Offline:

```python
from vllm import LLM, SamplingParams

llm = LLM(
    model="Qwen/Qwen3-0.6B",
    simulation_config={"simulate_forward": True, "timing": "constant",
                       "timing_args": {"fixed_us": 3000, "decode_us_per_req": 40}},
    kv_cache_memory_bytes=4 * 1024**3,
)
out = llm.generate(
    [{"prompt_token_ids": list(range(300))}],
    SamplingParams(trace_decode_token_ids=[501, 502, 503]),
)
```

`--kv-cache-memory-bytes` should be the KV budget of the deployment being
modeled (read it off the real server's startup log); prefix-cache hit rates
and preemption behavior depend strongly on it.

### Timing providers

| `timing` | behavior | `timing_args` |
|---|---|---|
| `zero` | every step takes no device time (default; KV/scheduler-only simulation) | none |
| `constant` | `fixed_us + prefill_us_per_token * P + decode_us_per_req * D + decode_us_per_kv_token * K` | those four, plus `host_us` |
| `table` | nearest-neighbour interpolation over recorded steps | `path`, `k`, `critical_path` |
| `replay` | exact match on the step's batch composition, nearest-neighbour fallback, reports the miss rate | `path`, `fallback` (`knn`/`zero`), `k`, `miss_fallback_s`, `critical_path` |
| `module.Class` | any class implementing `vllm.v1.worker.simulated.timing.TimingProvider` | passed as keyword arguments |

A provider receives a `StepDescriptor`: per request the tokens scheduled this
step, the KV length before it, prompt length, prefix-cache hits at admission,
draft tokens and whether it is still prefilling; plus batch totals, the number
of common-prefix blocks and the parallel configuration. It returns a
`StepCost` with `device_s` and an optional `host_s`.

### Recording a real deployment

A real deployment can write the same per-step log, measured with CUDA events,
to calibrate or validate simulated runs:

```bash
vllm serve Qwen/Qwen3-0.6B \
  --simulation-config '{"record_step_timing": true, "log_path": "/tmp/real_steps.jsonl"}'
```

Feeding that file to the `replay` provider (`"timing": "replay",
"timing_args": {"path": "/tmp/real_steps.jsonl"}`) reproduces the recorded
step times whenever the simulated scheduler forms the same batches; the
reported miss rate measures how far the two runs diverged. The `table`
provider interpolates the same records for workloads the recording did not
cover.

Both default to `critical_path: "max"`: a recorded step costs
`max(device_s, host_s)`. Steps whose host work outlasts the device work
(typically large prefills, whose kernel launches and sampling take longer on
the CPU than the kernels take on the GPU) are host-bound on the real
deployment, and pricing them by device time alone underestimates TTFT. Use
`"device"` to price by device time only.

On Qwen3-0.6B / H200 with `vllm bench serve` (ISL 512, OSL 64, 10 rps,
64 prompts), replaying a recording of the same workload reproduces the real
run within 2% on mean TTFT, TPOT and E2E latency and within 7% on P99 TTFT.

## Limitations

* Single rank: tensor, pipeline and data parallelism are rejected. The
  parallel configuration is passed to the provider for modeling purposes only.
* No speculative decoding, LoRA, KV transfer or pooling models.
* Time is real time: a simulated run takes as long as the deployment it
  models. An offline virtual clock for fast sweeps is planned.
* Prompt and output token ids are not meaningful; only their counts are.
