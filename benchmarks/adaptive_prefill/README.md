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

The cost model is a bootstrap heuristic, not a calibrated latency guarantee.
Mixed-step residual estimates are conservative approximations. Timing currently
excludes input preparation, output transfer, postprocessing and CPU queueing.
The policy does not directly control request-level p99; measure it externally.
It does not learn a separate hardware/backend model or reserve additional CUDA
workspace. Fairness is request-level, not tenant-level.

After aging, one token can exceed the soft time allowance if no earlier batch is
outstanding. This prevents a cost estimate from permanently reducing progress to
zero; it cannot guarantee deadlines or bounded waiting under overload or exhausted
KV capacity. Decode and minimum prefill service cannot both be hard guarantees.

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
# With the full vLLM test environment installed:
pytest tests/v1/core/test_scheduler.py -k adaptive_prefill
```

`simulate.py` executes a declared synthetic cost model and no GPU kernels. Its
numbers MUST NOT be described as vLLM GPU benchmark results. It exposes the
latency/throughput tradeoff and checks bounded-workload prefill completion.
