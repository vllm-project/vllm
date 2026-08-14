# Weight-Offload Schedule Planning

Prefetch weight offloading is controlled by `--offload-group-size`,
`--offload-num-in-group` and `--offload-prefetch-step`. The combinations
multiply quickly, and the ones that do not fit announce themselves only by
failing to start.

Two quantities decide whether a schedule is viable, and both are exact
arithmetic once the per-position weight and buffer sizes are known:

- how many bytes stay resident on the GPU, and
- how many bytes are re-copied host-to-device on every forward.

Those sizes depend on the model, its quantization, and the tensor/expert
parallel split, so only a running engine knows them. Serve once, then compare
schedules offline against that run — one launch instead of N.

## Usage

Serve with the schedule diagnostics on:

```bash
VLLM_PREFETCH_LOG_SCHEDULE=1 vllm serve ... \
  --offload-backend prefetch --offload-group-size 3 \
  --offload-num-in-group 1 --offload-prefetch-step 3 \
  --offload-params experts 2>&1 | tee server.log
```

The offloader writes one `manifest_json=` line per rank after post-init, once
quantization transforms and buffer-fallback decisions are final. Feed that log
straight in:

```bash
python -m vllm.benchmarks.weight_offload.planner server.log --rank 0
```

```text
recorded run: rank=0 schedule=3/1/3 units=20 H2D/forward=31.377 GiB
(every row is relative to that run; H2D is a volume, not a time)
     G/O/P  units   resident vs run    H2D/forward   alt
     1/1/1     61       -67.460 GiB     95.700 GiB
   61/60/1     60       -65.892 GiB     94.131 GiB
     1/1/2     61       -65.892 GiB     95.700 GiB
   31/30/1     59       -64.323 GiB     92.562 GiB    30
   61/60/2     60       -64.323 GiB     94.131 GiB
```

`resident vs run` is negative when a schedule frees more GPU memory than the
recorded one did. `alt` counts further schedules with the same outcome, folded
into that row.

Both columns are relative to a run you have already observed, which is what
makes them actionable: `1/1/1` frees 67 GiB more, and moves 95.700 / 31.377 =
3.05x the traffic over the same link. Multiply that ratio into the throughput
you measured.

Pass `--headroom-bytes` with the free memory that run had to drop schedules
that would not fit, `--json` for machine-readable output, and `--max-prefetch`
to widen the search.

## Why a delta

Everything about a deployment that does not depend on the schedule — KV cache,
non-offloadable weights, allocator slack, CUDA graph reserve — is identical
across candidates, so it cancels. Reporting against the recorded run therefore
needs no memory model of the deployment and no hand-entered totals, which is
where an absolute estimate would go wrong.

Only two terms move: the selector-matched weights a schedule evicts, and the
runtime buffers it holds. vLLM allocates `prefetch_step` slots for each
distinct pooled layout, so a wider prefetch window frees less memory even
though it offloads the same weights.

## What it does not model

Latency. Whether a transfer hides behind compute depends on measured H2D
bandwidth and per-layer compute time under real TP/EP contention, which this
tool cannot obtain and does not guess at. `H2D/forward` is reported because it
is exact — units that sit alone in a slot stay resident and cost nothing in
steady state — but it is a transfer volume, not a time.

There is deliberately no bandwidth setting. Bytes divided by bandwidth gives
the PCIe time a schedule occupies, not whether that time is hidden, and the
latter is the actual question -- answering it needs per-layer compute under
real contention. Reporting a duration would invite reading it as a latency
cost. The ratio against a run you measured carries the same information
without that trap.

Use it to discard schedules that cannot fit and to see what each one costs in
traffic. Measure the survivors.

## Limits

Buffer layouts are finalized per selected unit, so the manifest only describes
positions the recorded run actually offloaded. When a run used more than one
pooled layout, a schedule reaching positions it never selected is refused
rather than guessed at; record a run whose schedule covers them.

Schedules are rank-local. When ranks disagree the planner asks for `--rank`
instead of picking one.
