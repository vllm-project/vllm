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
need Triton's CPU backend (see [Running without a GPU](#running-without-a-gpu)).

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

### Speculative decoding

With a `speculative_config`, the scheduler schedules draft tokens as usual and
the simulated runner decides how many are accepted per request and step. The
draft model is built on the meta device like the target (its KV cache counts
against the budget), and no draft is ever proposed: the runner reports
placeholder drafts, as the real runner does under async scheduling.

`spec_acceptance_rates` gives the probability that the draft at each position
is accepted given that all earlier ones were; the last value applies to later
positions. A timing provider that sets `StepCost.num_accepted` overrides it.
A step log recorded with speculative decoding stores the accepted count of
every request, and passing its path instead of a list estimates the rates from
it:

```python
simulation_config={
    "simulate_forward": True,
    "timing": "replay",
    "timing_args": {"path": "/tmp/real_steps.jsonl"},
    "spec_acceptance_rates": "/tmp/real_steps.jsonl",
}
```

Recorded on Qwen3-8B with an EAGLE3 speculator (3 drafts) on H200 and
replayed this way under the `fast_forward` clock, a Poisson workload of 100
MMLU-Pro prompts at 4 requests/s reproduces mean TPOT, TTFT and E2E latency
within 2.5% and tokens per step within 3%, in 5 s instead of 30 s. Acceptance
is sampled independently per step, so the spread of per-request acceptance
(and with it P99 TPOT) is narrower than on the real deployment. Trace replay
is not available with speculative decoding; outputs are fallback tokens.

### Parallelism

Tensor, pipeline and data parallelism run as in the real deployment: one
simulated worker per rank, each building its shard of the model on the meta
device, so per-rank KV-cache specs (and with them the KV capacity) are
vLLM's own. `--kv-cache-memory-bytes` is per rank. No collective runs during a
step; every rank derives the same timing from the same scheduler output.

* **Tensor parallelism**: the ranks run in lockstep and agree on every step's
  time.
* **Pipeline parallelism**: no hidden states pass between stages. `device_s`
  is the step's time through the whole pipeline and each of the balanced
  stages is busy for an equal share, so consecutive microbatches overlap as on
  the real pipeline. A step recorded under pipeline parallelism is timed on
  the last stage from its arrival there, which includes the wait for the
  earlier stages, i.e. the same through-pipeline time. When the real
  pipeline is saturated the last stage is still busy with the previous
  microbatch as the next one arrives, so its recorded time shrinks toward one
  stage's; record such workloads with a recording at lower load or a
  per-stage model.
* **Data parallelism**: dense models run independent engines; DP rank
  i > 0 writes its step log to `<log_path>.dp<i>`. MoE models run their DP
  ranks in lockstep (their forward passes meet in all-to-all collectives), so
  a step's cost depends on every rank's batch. Where the real runner
  synchronizes the ranks (every step that runs the model and every dummy
  batch of an idle rank), the simulated ranks, and the recording ranks of a
  real deployment, gather their step descriptors into one joint step. The
  joint step is priced once, starts on the latest rank, and is what DP rank 0
  logs; the other ranks write no log. Accepted draft counts are not recorded
  for joint steps.

Only the output rank (last pipeline stage, first tensor-parallel rank) writes
the step log, for simulated and recorded runs alike. On CUDA hosts each rank
binds its own device, as in the deployment being modeled; a simulated rank
needs only a few MB of device memory.

On Qwen3-8B / H200 with a Poisson workload (ISL 512, OSL 64, 8 req/s, 96
requests), replaying a recording of the same configuration through the
offline driver on the real clock gives:

| | TTFT mean | TPOT mean | E2E mean |
|---|---|---|---|
| TP=2, real / simulated (ms) | 19.94 / 20.09 | 3.990 / 3.996 | 271.3 / 271.9 |
| PP=2, real / simulated (ms) | 18.81 / 18.30 | 5.881 / 5.915 | 389.3 / 391.0 |

With PP=2 saturated (256 requests at 64 req/s) the simulation is about 10%
optimistic (TPOT 10.06 vs 11.12 ms), as the recorded through-pipeline times
shrink (see above). Qwen3-VL-30B-A3B-Instruct-FP8 (MoE, text only) with DP=2
and expert parallelism under `vllm serve` (ISL 512, OSL 64, 16 req/s, 128
requests) reproduces mean TPOT within 2% and mean TTFT within 9%.

### Running without a GPU

A CPU build of vLLM (`VLLM_TARGET_DEVICE=cpu`, which installs Triton's CPU
backend from `requirements/cpu.txt`) runs simulations on any x86-64 host:

```bash
TRITON_CPU_BACKEND=1 vllm serve Qwen/Qwen3-235B-A22B --simulate-forward \
  --tensor-parallel-size 8 --pipeline-parallel-size 2 \
  --kv-cache-memory-bytes 68719476736 \
  --simulation-config '{"timing": "replay", "timing_args": {"path": "steps.jsonl"}}'
```

Ranks need no devices there, so a layout larger than the host's GPU count
(16 ranks above) runs on one CPU host. Simulated workers run only the small
input-preparation kernels, so instead of each getting its own NUMA node they
share the host with at most four OpenMP threads each. A simulation keeps the settings of the GPU
deployment it models rather than the CPU platform's: async scheduling stays
on and the in-process executor (which the `fast_forward` and `virtual`
clocks need) stays available. The one exception is the KV block size: the CPU
attention backend needs a multiple of 32, so the default is 32 instead of the
GPU default of 16, which coarsens prefix-cache matching. Pass `--block-size`
to match the deployment when its backend uses a larger one; MLA models on
CPUs without AMX keep the CPU backend's required size. The CPU build needs
Python 3.11 or newer for triton-cpu.

## Limitations

* Ranks cannot share a device on CUDA hosts, so simulating a deployment there
  needs as many devices as it has ranks (not their memory); the CPU platform
  has no such limit.
* On the CPU platform the KV block size must be a multiple of 32.
* Pipeline stages are assumed balanced.
* No LoRA, KV transfer or pooling models. Speculative decoding without
  adaptive verification only, and no diffusion models.
* The `fast_forward` and `virtual` clocks are offline-only and need a single
  process (one rank); served and multi-rank simulations run in real time.
* Prompt and output token ids are not meaningful; only their counts are.
