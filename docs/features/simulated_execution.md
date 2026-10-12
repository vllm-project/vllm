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
* Each step's cost comes from a *timing provider*. The provider sees a plain
  description of the step and returns how long the device would be busy
  (`device_s`) and, optionally, how long the host spends inside the step's
  `execute_model`/`sample_tokens` calls (`host_s`: input preparation, kernel
  launches, sampling). The host time holds the engine thread, as the real
  calls do; the device time runs on a serial device queue and is paid when
  the engine core consumes the step's output, so the engine keeps scheduling
  the next step meanwhile, exactly as with a real GPU under async scheduling.
  A step completes when both its device work and its host work are done. The
  rest of the host loop (scheduling, output processing, the frontend) is the
  real code and takes its real time.

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
| `table` | nearest-neighbour interpolation over recorded steps | `path`, `k` |
| `replay` | exact match on the step's batch composition, nearest-neighbour fallback, reports the miss rate | `path`, `fallback` (`knn`/`zero`), `k`, `miss_fallback_s` |
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

Both providers return the recorded device and host times. Recording both
matters: large prefills are typically host-bound (their kernel launches and
sampling take longer on the CPU than the kernels take on the GPU), and pricing
them by device time alone underestimates TTFT. The recorded host time spans
the engine thread from `execute_model` to the end of `sample_tokens`, so it
also contains whatever the engine core does between the two calls (e.g. the
structured-output grammar bitmask) and any wait on a full kernel-launch queue;
replaying it under a different scheduler configuration attributes that time
to the step regardless.

On Qwen3-0.6B / H200 with `vllm bench serve` (ISL 512, OSL 64, 10 rps,
64 prompts), replaying a recording of the same workload reproduces the real
run within 2% on mean, median and P99 TTFT, TPOT and ITL.

### Offline runs in simulated time

`vllm.v1.worker.simulated.offline` drives an in-process engine with a timed
arrival schedule and timestamps every output token, so a workload can be run
without waiting for it:

```python
from vllm.engine.arg_utils import EngineArgs
from vllm.sampling_params import SamplingParams
from vllm.v1.worker.simulated.offline import TimedRequest, simulate

requests = [
    TimedRequest(prompt_token_ids=list(range(512)),
                 sampling_params=SamplingParams(max_tokens=128, ignore_eos=True),
                 arrival_time=0.05 * i)
    for i in range(200)
]
result = simulate(
    EngineArgs(
        model="Qwen/Qwen3-0.6B",
        kv_cache_memory_bytes=8 * 1024**3,
        simulation_config={"simulate_forward": True, "clock": "fast_forward",
                           "timing": "replay",
                           "timing_args": {"path": "/tmp/real_steps.jsonl"}},
    ),
    requests,
)
print(result.summary())  # TTFT/TPOT/ITL/E2E percentiles, as vllm bench serve
```

`simulation_config.clock` selects the time base:

| `clock` | waits (device time, idle gaps between arrivals) | engine host loop (scheduling, output processing) | use |
|---|---|---|---|
| `real` (default) | slept | real time | serving; the wall-clock baseline |
| `fast_forward` | skipped | real time | sweeps: same latencies as `real` in a fraction of the time |
| `virtual` | skipped | free | deterministic runs (CI, scheduler research); underestimates host-bound workloads |

Under every clock a step's predicted `host_s` holds the engine thread. The
non-real clocks need an engine built by `make_engine()` (which `simulate()`
calls), so that the engine core runs in the driver's process. A run takes as long as the engine's host
work, roughly 1 ms per step, so the speedup over real time is the step time
of the modeled deployment divided by that. `make_engine()` and `run()` also drive a
real (non-simulated) engine on the real clock, which is how the
reference for a comparison is produced.
[examples/features/simulated_execution/sweep_offline.py](../../examples/features/simulated_execution/sweep_offline.py)
sweeps `max_num_seqs` this way.

## Limitations

* Single rank: tensor, pipeline and data parallelism are rejected. The
  parallel configuration is passed to the provider for modeling purposes only.
* No speculative decoding, LoRA, KV transfer or pooling models.
* The `fast_forward` and `virtual` clocks are offline-only and single-process;
  served simulations run in real time.
* Prompt and output token ids are not meaningful; only their counts are.
