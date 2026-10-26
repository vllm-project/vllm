# Simulated Execution

Simulated execution runs the real vLLM engine (scheduler, KV cache manager,
prefix caching, chunked prefill, preemption, async scheduling and the OpenAI
frontend) but replaces the model forward pass with a timing model. It predicts
a deployment's TTFT, TPOT and throughput without its hardware or weights, and
exercises scheduler and KV cache behavior deterministically on CPU-only hosts.

## How it works

* The model is built on the `meta` device, so KV cache specs, attention
  metadata and hybrid layouts are vLLM's own, with no parameters in memory.
* The KV cache is virtual: blocks are accounted as usual within
  `--kv-cache-memory-bytes`, but no tensors are allocated.
* Output tokens come from `SamplingParams.trace_decode_token_ids` (see
  [trace replay](../serving/online_serving/trace_replay.md)) or, without a
  trace, are `fallback_token_id`; such requests end by `max_tokens`.
* A *timing provider* prices each step: its device time (`device_s`) runs on
  a modeled device queue, overlapping the next step's scheduling as under
  async scheduling, and its optional host time (`host_s`, spent in
  `execute_model` and `sample_tokens`) holds the engine thread. The rest of
  the engine loop is the real code and takes its real time.

Simulation runs on CUDA-alike hosts and on the CPU platform (see
[Running without a GPU](#running-without-a-gpu)).

## Usage

```bash
vllm serve Qwen/Qwen3-0.6B \
  --simulate-forward \
  --kv-cache-memory-bytes 8589934592 \
  --simulation-config '{"timing": "linear",
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
    simulation_config={"simulate_forward": True, "timing": "linear",
                       "timing_args": {"fixed_us": 3000, "decode_us_per_req": 40}},
    kv_cache_memory_bytes=4 * 1024**3,
)
out = llm.generate(
    [{"prompt_token_ids": list(range(300))}],
    SamplingParams(trace_decode_token_ids=[501, 502, 503]),
)
```

Set `--kv-cache-memory-bytes` to the KV budget of the modeled deployment, as
its startup log reports it: prefix-cache hit rates and preemption depend on it.

### Timing providers

| `timing` | behavior | `timing_args` |
| --- | --- | --- |
| `zero` | steps take no device time (default) | none |
| `linear` | `fixed_us + prefill_us_per_token * P + decode_us_per_req * D + decode_us_per_kv_token * K` | those four, plus `host_us` or `host_fit_log` |
| `table` | nearest-neighbor interpolation over a step log | `path`, `k` |
| `replay` | exact match on the step's batch composition, else `fallback` | `path`, `fallback` (`knn`/`zero`), `k`, `miss_fallback_s` |
| `module.Class` | a `vllm.v1.worker.simulated.timing.TimingProvider` | its keyword arguments |

A provider receives a `StepDescriptor`: per request, the tokens scheduled in
the step, the KV length before it, the prompt length, prefix-cache hits at
admission, draft tokens and whether it is prefilling; plus batch totals,
common-prefix blocks and the parallel configuration. It returns a `StepCost`.

`table` and `replay` read a recorded step log (see below). A `table` can also
be written by hand, as JSONL records of
`{"num_reqs", "num_tokens", "sum_past_kv", "device_s"[, "host_s"]}`: the
step's requests and scheduled tokens, and the sum of their KV lengths before
the step.

`HostTimeModel` prices host time as `fixed_us + per_req_us * R +
per_prefill_token_us * P`, fitted to a step log (`HostTimeModel.from_log`, or
`host_fit_log` for `linear`), for providers that model only the device. Under
async scheduling host time matters only where it exceeds device time, mostly
in prefills, so it mainly affects TTFT. Fit it to a recording of the same
parallel layout and speculative decoding setup, made under async scheduling;
terms the recording does not determine stay zero.

### Recording a real deployment

A real deployment writes the same step log, timed with CUDA events:

```bash
vllm serve Qwen/Qwen3-0.6B \
  --simulation-config '{"record_step_timing": true, "log_path": "/tmp/real_steps.jsonl"}'
```

`replay` with this log reproduces the recorded step times wherever the
simulated scheduler forms the same batches; `table` interpolates them for
other workloads. Both return the recorded device and host times. Host time
matters: large prefills are often host-bound, and pricing them by device time
alone underestimates TTFT. The recorded host time spans the engine thread
from `execute_model` to the end of `sample_tokens`, so it includes engine core
work between the two calls, such as structured-output bitmasks.

### Offline runs in simulated time

`vllm.v1.worker.simulated.offline` drives an in-process engine with timed
arrivals of requests and multi-turn sessions, and timestamps every output
token:

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
print(result.summary())  # as vllm bench serve reports them
```

`simulation_config.clock` selects the time base:

| `clock` | waits (device time, idle gaps) | engine host loop | use |
| --- | --- | --- | --- |
| `real` (default) | slept | real time | serving |
| `fast_forward` | skipped | real time | sweeps: the latencies of `real`, faster |
| `virtual` | skipped | free | deterministic runs; underestimates host-bound workloads |

Predicted `host_s` holds the engine thread under every clock. The non-real
clocks need an engine from `make_engine()` (which `simulate()` uses), and a
run then takes as long as the engine's host work. `make_engine()` and `run()`
also drive a real engine on the real clock, to produce a reference run.
[`vllm bench sweep simulate`](../benchmarking/sweeps.md#simulated-sweep) runs a
`vllm bench sweep serve` experiment this way.

Only the engine core runs, so `frontend_latency_s` (of `run()` and
`simulate()`) adds the server's latency outside it (HTTP, API server, IPC) to
each request's arrival.

A `TimedSession` is a multi-turn conversation: each `Turn` appends its tokens
to the previous turn's prompt and output, and arrives `think_time` seconds
after the previous turn finished, so later turns reuse the conversation's KV
cache. Results have one timing per turn, in input order. Prompts are
concatenated token ids, so chat-template tokens belong in each turn's
`new_token_ids`, and `max_model_len` must fit a whole conversation.

```python
from vllm.v1.worker.simulated.offline import TimedSession, Turn

params = SamplingParams(max_tokens=128, ignore_eos=True)
session = TimedSession(
    turns=[Turn(first_prompt_ids, params)]
    + [Turn(user_ids, params, think_time=3.0) for user_ids in follow_ups],
    arrival_time=0.0,
)
```

### Speculative decoding

The scheduler schedules draft tokens as usual, and the simulated runner
decides how many are accepted. The draft model is built on the meta device
too, and its KV cache counts against the budget.

`spec_acceptance_rates` gives the probability that the draft at each position
is accepted given that all earlier ones were; the last value repeats. A step
log recorded with speculative decoding holds every request's accepted count,
and its path estimates the rates instead:

```python
simulation_config={
    "simulate_forward": True,
    "timing": "replay",
    "timing_args": {"path": "/tmp/real_steps.jsonl"},
    "spec_acceptance_rates": "/tmp/real_steps.jsonl",
}
```

A provider's `StepCost.num_accepted` overrides the rates. Acceptance is
sampled independently per step, so per-request acceptance, and with it P99
TPOT, varies less than on a real deployment. Trace replay is not available
with speculative decoding.

### Parallelism

Tensor, pipeline, data and context (prefill and decode) parallelism run as
deployed: one simulated worker per rank builds its shard of the model, so
per-rank KV capacity is vLLM's own; `--kv-cache-memory-bytes` is per rank.
No collective runs in a step: every rank derives the same timing from the
same scheduler output.

* **Pipeline parallelism**: `device_s` is a step's time through the pipeline,
  split evenly over the stages, so consecutive microbatches overlap. Steps
  recorded under pipeline parallelism carry the same through-pipeline time.
  The last stage's host time includes receiving intermediate tensors, which
  can bound the step rate under load; fit host time to a recording of the
  same layout.
* **Data parallelism**: dense models run independent engines, and DP rank
  `i > 0` logs to `<log_path>.dp<i>`. MoE models run their DP ranks in
  lockstep: wherever the real runner synchronizes them, the ranks gather
  their step descriptors into one joint step, priced once and logged by DP
  rank 0. Joint steps do not record accepted drafts.

Only the output rank (last pipeline stage, first tensor-parallel rank) writes
the step log. On CUDA hosts each rank binds its own device, using a few MB of
its memory.

### KV cache offloading

CPU offloading by the `OffloadingConnector` (its CPU spec) runs with the real
scheduler side, so the CPU cache's hits, evictions and load and store jobs are
the real ones; a simulated worker replaces the copy streams. Each direction
is a serial queue, a transfer starts once the device has finished the step
running when it was submitted, and it takes `latency + bytes / bandwidth`.

`kv_transfer_timing` gives the bandwidths in 1e9 bytes/s, with optional
latencies:

```python
kv_transfer_config={
    "kv_connector": "OffloadingConnector",
    "kv_role": "kv_both",
    "kv_connector_extra_config": {"cpu_bytes_to_use": 32 << 30},
},
simulation_config={
    "simulate_forward": True,
    "timing": "replay",
    "timing_args": {"path": "/tmp/real_steps.jsonl"},
    "kv_transfer_timing": {"load_gbps": 25, "store_gbps": 25},
},
```

A step log recorded with offloading also holds the transfers completed in
each step; its path as `kv_transfer_timing` fits each direction's latency and
bandwidth by least squares. A transfer's size is its blocks' KV page size,
which overstates layouts where the real copy moves less (padded hybrid pages,
canonical-layout stores split across ranks); rates fitted to the same model
absorb this. A step that waits for stores the scheduler flushes pays the wait,
which its recorded host time also includes.

### Disaggregated prefill

`vllm.v1.worker.simulated.disaggregated` runs prefill and decode instances
connected by the `NixlConnector` (pull mode) in one process, in simulated
time:

```python
from vllm.v1.worker.simulated.disaggregated import simulate_disaggregated

result = simulate_disaggregated(
    prefill_args,  # EngineArgs with simulate_forward and a timing provider
    decode_args,
    requests,      # TimedRequests
    num_prefill=2,
    num_decode=1,
    handoff_latency_s=7e-3,
)
```

`EngineArgs` built in Python take the generic scheduler defaults rather than
`vllm serve`'s: set `max_num_batched_tokens` and `max_num_seqs` as the served
instances have them. (`vllm bench sweep simulate` applies `vllm serve`'s
defaults.)

The driver plays the proxy: a request goes to a prefill instance with
`max_tokens=1`, then to a decode instance with the `kv_transfer_params` it
returned. Instances are picked round-robin or, with
`routing="least_unfinished"`, by fewest unfinished requests;
`handoff_latency_s` is the proxy's latency between the two. The connector's
scheduler side is the real one; its worker side moves no data. A read takes
`latency + bytes / bandwidth` at `kv_transfer_timing`'s load rate, reads
queue on each instance's link, and a completed read frees the prefill
instance's blocks. Leases do not expire, since heartbeats run on the wall
clock. Each instance has one rank and its own virtual clock; the instance
furthest behind steps next.

Posting a step's reads holds a decode instance's engine thread. NIXL merges
blocks consecutive on both instances into one descriptor per KV cache
region, so posting slows once a decode instance's blocks scatter, after its
KV cache has filled once. The simulation charges `post_us_per_read` per read
plus `post_us_per_run` per run of consecutive blocks. A decode instance
recorded with the real `NixlConnector` logs its reads and postings, and its
step log as `kv_transfer_timing` fits the read rate and posting costs. Record
until its KV cache has filled, or the cost of scattered reads is
extrapolated.

#### Served instances

Simulated prefill and decode instances also run as servers, on the real
clock, with tensor parallelism, behind any proxy:

```bash
VLLM_NIXL_SIDE_CHANNEL_PORT=5600 vllm serve Qwen/Qwen3-8B --port 8100 \
  --tensor-parallel-size 2 --simulate-forward \
  --kv-transfer-config '{"kv_connector": "NixlConnector", "kv_role": "kv_producer"}' \
  --simulation-config '{"timing": "replay", "timing_args": {"path": "p.jsonl"}, "kv_transfer_timing": "d.jsonl"}'
VLLM_NIXL_SIDE_CHANNEL_PORT=5700 vllm serve Qwen/Qwen3-8B --port 8200 \
  --tensor-parallel-size 2 --simulate-forward \
  --kv-transfer-config '{"kv_connector": "NixlConnector", "kv_role": "kv_consumer"}' \
  --simulation-config '{"timing": "replay", "timing_args": {"path": "d.jsonl"}, "kv_transfer_timing": "d.jsonl"}'
```

A decode rank books its reads over ZMQ with the prefill ranks the real
connector maps it to. Each prefill rank listens on a port derived from its
instance's side-channel port and its rank, which must be free. The two
instances' tensor parallelism may differ as with the real connector, and a
request's reads from several ranks run concurrently. A read that cannot be
booked, or a layout the real connector rejects, fails its request, which the
decode instance then prefills itself. Pipeline parallelism (unsupported by
the real pull connector), decode context parallelism and hybrid (Mamba)
layouts are not modeled.

#### Push mode and pipeline parallelism

With the `NixlPushConnector`, a prefill instance with pipeline parallelism
writes into a decode instance without it. As with the real connector, the
proxy sends a request to both instances at once. Decode ranks register each
request with their paired prefill ranks in every stage and forward their
heartbeats; a prefill rank writes its share once the request is both
finished and registered, and frees the blocks when its writes end or the
lease of an unregistered request (a full prefix hit) expires. Here the
prefill instance records the transfers, so fit `kv_transfer_timing` to its
step log.

### Running without a GPU

A CPU build of vLLM (`VLLM_TARGET_DEVICE=cpu`, which installs Triton's CPU
backend) runs simulations on any x86-64 host:

```bash
TRITON_CPU_BACKEND=1 vllm serve Qwen/Qwen3-235B-A22B --simulate-forward \
  --tensor-parallel-size 8 --pipeline-parallel-size 2 \
  --kv-cache-memory-bytes 68719476736 \
  --simulation-config '{"timing": "replay", "timing_args": {"path": "steps.jsonl"}}'
```

Ranks need no devices, so a layout larger than any one GPU host runs on a
CPU host. Simulated workers share the host's cores instead of each taking a
NUMA node, and keep the modeled GPU deployment's settings (async scheduling,
the in-process executor) rather than the CPU platform's. The exception is the
KV block size, which the CPU attention backend needs to be a multiple of 32:
pass `--block-size` to match the deployment where it uses a larger one.
Layers still pick the CPU platform's attention backends and kernels, so a
model needing one it lacks (sparse MLA, decode context parallelism, NVFP4 or
MXFP4 weights, online FP8 quantization) fails to build there. The CPU build needs Python 3.11 or newer.

## Limitations

* On CUDA hosts each rank needs its own device (but little of its memory);
  the CPU platform has no such limit.
* Pipeline stages are assumed balanced.
* No LoRA, pooling or diffusion models, and speculative decoding without
  adaptive verification only.
* KV transfer is limited to CPU offloading by the `OffloadingConnector` and
  disaggregated prefill by the `NixlConnector`: in-process instances in pull
  mode with one rank each; served instances in pull mode without pipeline
  parallelism, or in push mode with it on the prefill instance.
* The `fast_forward` and `virtual` clocks need a single process with one
  rank; served and multi-rank simulations run on the real clock.
* Token ids are not meaningful; only their counts are.
