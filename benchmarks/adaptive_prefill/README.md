# Experimental adaptive prefill admission

This opt-in prototype addresses interference discussed in issue #62574. It is
not an upstream-accepted policy or proof of improved GPU performance.

## Controller

- `--adaptive-prefill` enables a separate prompt-time budget. Disabled by default.
- `--adaptive-prefill-target-ms 40` sets a soft GPU step-time target.
- `--adaptive-prefill-token-ms 0.02` is a bootstrap cost; calibrate for the model.
- `--adaptive-prefill-min-share 0.1` accrues bounded progress credits.
- `--adaptive-prefill-max-wait-ms 250` enables aged-work credit spending.

Running and waiting prompt chunks share the allowance; running decode is visited
first. Prompt cost is context-weighted, with per-request fair shares. Feedback
uses completed forward/sampling CUDA events through the existing output channel,
not a new per-step device synchronization. Outstanding prompt work is reserved
until its corresponding output arrives. Existing KV allocation, cache watermark,
token and context constraints still apply. Preempted allocations are refunded.

The cost model separates fixed step overhead, marginal decode population cost
and context-weighted prefill cost. Decode-only samples fit the first two terms;
one observed population identifies only a whole-batch baseline, not a per-request
cost. Without decode samples, differing prefill sizes can identify overhead.
The bootstrap remains in use until overhead is identified. Three valid residual
samples are needed before updating prefill cost, with a five-sample median to
reduce sensitivity to isolated compilation/timing spikes.

This is still a bootstrap heuristic, not a calibrated latency guarantee.
Mixed-step residual estimates are approximations. Timing currently
excludes input preparation, output transfer, postprocessing and CPU queueing.
The policy does not directly control request-level p99; measure it externally.
It does not learn a separate hardware/backend model or reserve additional CUDA
workspace. Fairness is request-level, not tenant-level.

With no decode currently running or outstanding, a prefill-only pipeline may
issue two steps, each with its own time allowance. This avoids sharing a single
shrinking allowance indefinitely across consecutive async steps. Per-request
fair shares remain in place to leave room for waiting prompts.

After aging, one token may exceed the soft allowance with at most one earlier
step outstanding. No additional prefill is admitted with two outstanding steps.
This bounds the escape without requiring an async pipeline to become empty. It
may increase queued latency and cannot guarantee deadlines or bounded waiting
under overload or exhausted KV capacity. Decode and minimum prefill service
cannot both be hard guarantees.

Initial supported scope: text generation, chunked prefill, FCFS, one KV group,
PP=DP=CP=1, no microbatching, speculation or KV connector. TP is permitted because a centralized
scheduler supplies one plan to the worker group. Other configurations fail at
scheduler initialization. The original speculative hybrid-model GB10 issue is
not claimed reproduced or fixed by this narrower first increment.

## Live GPU benchmark

Build/install this checkout using the project's normal GPU setup. Run baseline
and adaptive servers sequentially on the SAME GPU(s), checkpoint and commit,
with identical options except the controller flag. Example for a smaller dense
model (this is not the original two-node issue reproduction):

```bash
vllm serve Qwen/Qwen3-0.6B --port 8000 --max-model-len 32768 \
  --max-num-batched-tokens 8192 --long-prefill-token-threshold 1024 \
  --max-num-seqs 64 --enable-chunked-prefill
# For the adaptive run, append --adaptive-prefill.
python benchmarks/adaptive_prefill/serve.py --model Qwen/Qwen3-0.6B \
  --label baseline --output /tmp/prefill-baseline.json
# Restart the server with --adaptive-prefill, then run the identical workload:
python benchmarks/adaptive_prefill/serve.py --model Qwen/Qwen3-0.6B \
  --label adaptive --output /tmp/prefill-adaptive.json
python benchmarks/adaptive_prefill/compare.py \
  /tmp/prefill-baseline.json /tmp/prefill-adaptive.json
```

Use `--long-interval`, `--short-interval`, counts and delay to sweep bursts and
sustained arrivals. Keep the seed/workload equal. Set `--long-requests 0` or
`--short-requests 0` for controls. Sweep target budgets and repeat runs; record GPU
model, TP topology, full server options, commit and idle load separately.
Warmups are excluded from timing. Prompt token IDs assume the specified vocabulary
range is valid. Output lengths are fixed with `ignore_eos`; usage and completion
are checked. A timeout or broken stream must not count as a throughput success.

Metrics include short/long TTFT, output throughput, failures, long-request
completion and client streaming-event gaps. Exact token-gap p99 is null unless
every output event declares one token; it is not inferred by dividing event gaps.
Even single-token observations are client-visible timing, including transport and
buffering, not kernel-only decode timing. A finished bounded workload is evidence
of progress for that workload, not a proof of starvation freedom for all traffic.

## CPU checks

```bash
python -m unittest discover -s tests/v1/core -p 'test_adaptive_prefill*.py' -v
PYTHONPATH=. python benchmarks/adaptive_prefill/simulate.py
# With the vLLM CPU test dependencies and Triton CPU installed:
VLLM_TARGET_DEVICE=cpu CUDA_VISIBLE_DEVICES= python -m pytest \
  tests/v1/core/test_scheduler.py tests/v1/core/test_async_scheduler.py -q
# Extended randomized priority/preemption coverage (can take several minutes):
VLLM_TARGET_DEVICE=cpu CUDA_VISIBLE_DEVICES= python -m pytest \
  tests/v1/core/test_priority_scheduler_random.py -q
```

Use Python 3.11 or newer for this source snapshot. CPU scheduler tests instantiate
the real scheduler and KV managers with synthetic worker outputs, not a live
model runner. They need model configuration metadata but not model weights.
After the metadata is cached, `HF_HUB_OFFLINE=1` avoids repeated network lookups.
The scheduler tests cover enabled/disabled policy, sync/async scheduling, and
fixed-overhead feedback with a delayed result: three long prompts with a late
short arrival, and four long prompts without decode work. These use the declared
synthetic timing `14.76 + 0.53 * decode_requests + 0.07 * weighted_prefill_tokens`
milliseconds, not measured GPU performance. The remaining cases exercise the default disabled policy;
their success is not evidence that every existing mode supports the controller.

No CUDA kernels, event timing, model inference, tensor-parallel collectives or
GPU latency/throughput are validated by these CPU commands. In particular,
`test_scheduler_e2e.py` launches a model, exercises baseline/adaptive modes and
mixed-length prompt completion, and is not included in this CPU-only gate.

`simulate.py` executes a declared synthetic cost model and no GPU kernels. Its
numbers MUST NOT be described as vLLM GPU benchmark results. It exposes the
latency/throughput tradeoff and checks bounded-workload prefill completion.
