# Multi-Tenant Sleep Mode Operations

This page captures operational guidance for running many `--enable-sleep-mode`
vLLM containers on the same GPU host, learned from one production deployment.
For the feature reference and basic API surface, see
[Sleep Mode](sleep_mode.md).

!!! note "Scope of this guide"
    The recommendations below are drawn from a single deployment — 9 vLLM
    containers on a 4×24 GiB host, with continuous wake/sleep cycling over
    ~24h/day. Numbers and ratios are illustrative, not universal. Calibrate
    against your own workload before relying on any specific value.

## When sleep mode helps (and when it doesn't)

In a co-tenant fleet you generally have two kinds of workloads:

| Workload shape | Suggested strategy | Why |
|---|---|---|
| Cold-load-on-demand models that share GPUs in a mutually-exclusive **swap group** (e.g. several large generate-runners competing for the same physical VRAM) | `--enable-sleep-mode` with an external orchestrator driving `/sleep` and `/wake_up` | Single GPU process per model, ~5-15s wake time vs. minutes for full cold load, weights stay backed in host RAM |
| Always-warm sidecars (embeddings, rerankers, small generation models) that fit alongside the active heavy model | Run them as plain processes without sleep mode, or use sleep mode only as a defensive fallback | They're never the bottleneck for VRAM admission, and sleep mode on small pooling models has tradeoffs (see [Cross-cycle residual growth](#cross-cycle-residual-growth) below) |

The orchestration model that worked for us: a thin admission controller in
front of the OpenAI-compatible endpoints, sleeping the inactive set on
incoming requests for a different model in the same swap group, and waking
the requested model before forwarding the request. vLLM's `/sleep`,
`/wake_up`, and `/is_sleeping` endpoints are sufficient primitives for this
pattern.

## Sleep level 1 vs level 2 — observed VRAM cost

The reference docs describe what each level releases. In our deployment, the
residual VRAM after `/sleep?level=1` varies significantly with runner type:

| Model (runner) | Awake VRAM/GPU | L1 residual/GPU | Residual ratio |
|---|---|---|---|
| Qwen3-4B AWQ (generate) | 4500 MiB | ~100 MiB | ~2% |
| Qwen2.5-VL-3B AWQ (generate, multimodal) | 15500 MiB peak / 11500 MiB steady | ~1500 MiB | ~10% |
| GLM-OCR (generate, multimodal) | 3500 MiB | ~100 MiB | ~3% |
| BAAI/bge-m3 (pooling) | 1500 MiB | ~1300 MiB | **~87%** |
| mxbai-rerank-base-v2 (pooling) | 2500 MiB | ~2000 MiB | **~80%** |

Generate-runner models clean-release at L1 (matching the "frees 90%+" claim
in the reference docs). The pooling-runner co-tenants in our fleet retain
the bulk of their footprint at L1 — we have not fully root-caused this,
but the working hypothesis is variable-shape allocator fragmentation
interacting with the `CuMemAllocator` plugin. We log the observation here
so operators know to **calibrate L1 residual per model**, not assume "90%+
freed" universally.

Level 2 sleep discards the weights as well and is equivalent to a fresh load
on wake. In our testing, L2 on a model where L1 already retained most of the
footprint (the MoE model in our fleet) reclaimed **zero additional** GPU
memory — because what L1 had retained was the allocator's reservations, not
the weights themselves. If your L1 residual is high and L2 doesn't help,
consider stopping the container instead (see
[Co-tenancy planning](#co-tenancy-planning) below).

## Cross-cycle residual growth

We've observed that on long-running pooling-runner deployments, the
post-sleep VRAM residual grows over many wake/sleep cycles rather than
returning to a stable baseline. Over ~24h of normal traffic this added
3-6 GiB of unreclaimable memory per pooling-runner process in our fleet,
which eventually caused admission failures on co-tenants that needed that
headroom.

The behavior is tracked upstream — see
[issue #36651](https://github.com/vllm-project/vllm/issues/36651) for the
double-free and stale-error-code aspect of the cumem allocator interaction.

### Diagnostic

A simple offline probe that runs N `/sleep` ↔ `/wake` cycles and reports
per-cycle residual delta is helpful for confirming whether a given model
exhibits the growth pattern in your environment. See
[PR #45095](https://github.com/vllm-project/vllm/pull/45095) for an
in-tree probe script.

### Mitigation we use today

A nightly `docker restart` of the affected containers during a low-traffic
window reclaims the accumulated residual. We do this only for the pooling
co-tenants in our fleet; the generate-runner peers are stable across cycles
and don't need it.

```cron
# Reclaim accumulated slept-state VRAM on pooling runners.
17 4 * * *  docker restart vllm-embeddings vllm-reranker
```

For heavy generate-runners in a swap group where we want the full footprint
back (not just the post-L1 residual), we use a "stop on evict" pattern: the
orchestrator does `docker stop` on the evicted peer instead of `/sleep`,
accepting the cold-load cost on next demand (~5-7 min from page cache for a
27B-class model) in exchange for full reclaim of CUDA context + NCCL
buffers (~5 GiB per GPU above the L1 residual in our measurements).

## Detecting decode deadlocks

The standard `/health` endpoint reports process liveness — the engine is
running and the HTTP server is up. It does **not** report whether the
engine is making forward progress. We've observed a class of failure where
prefill completes at full throughput but the decode loop stalls at 0
tok/s, with repeated "new 2-rank communicator" log lines as a TP=2 PP=2
NCCL P2P interaction. The engine is "healthy" by `/health` but the model
is functionally dead.

See [issue #45094](https://github.com/vllm-project/vllm/issues/45094) for
the reproducer, and [PR #45097](https://github.com/vllm-project/vllm/pull/45097)
for a proposed `/health/decode` endpoint exposing forward-progress liveness.
[PR #45105](https://github.com/vllm-project/vllm/pull/45105) adds an opt-in
`NCCL_ASYNC_ERROR_HANDLING` preservation pathway that surfaces the
underlying NCCL hang to the worker for detection.

**Operational recommendation**: orchestrators driving sleep/wake should
probe a forward-progress endpoint, not just `/health`, when deciding
whether a model is ready to receive traffic post-wake. If your vLLM build
does not yet include `/health/decode`, a tiny periodic generation probe
(send a 1-token request, watch for a response within a budget) is a
serviceable proxy.

## Co-tenancy planning

When sizing a host that runs more than one vLLM container concurrently:

- **Account for L1 residual.** Summing each model's *awake* footprint
  understates the floor when peers are slept. Track an
  `expected_vram_mb_per_gpu` and a calibrated `sleep_l1_residual_mb` per
  model and use both in your admission decisions.
- **Account for the CUDA primary context.** Each vLLM process holds a few
  hundred MiB per GPU for the primary CUDA context, independent of weights
  and KV cache. This is paid per-process, not per-model.
- **Account for NCCL buffers** on TP > 1 or PP > 1 deployments. These are
  lazily allocated on first collective and are not released by `/sleep`.
- **Leave headroom for activation peaks.** Multimodal models in particular
  can have transient encoder-cache + cudagraph-capture peaks well above
  their steady-state awake footprint. In our deployment we saw a 3B vision
  model peak at 15.1 GiB during load (32k-token video max + cudagraph
  capture) versus an 11.5 GiB steady state. Admission should reserve for
  the peak, not the steady state, or wake will OOM under load.
- **Don't over-commit on a single GPU even if individual containers' utils
  sum to <1.0.** We've found ~10-15% headroom on each GPU is a sane budget
  for the CUDA primary context, NCCL buffers, and unanticipated peaks.

### Pipeline-parallel rank asymmetry

In a PP > 1 deployment, the last pipeline rank holds the language-model
head, the sampler, and the embedding outputs in addition to its share of
the decoder blocks. In our TP=2 PP=2 27B AWQ deployment we observed the
two PP ranks differing by ~1.5 GiB awake on each GPU. If you are
distributing PP ranks across GPUs with different free-memory budgets,
prefer to put the **heavier** rank on the GPU with more headroom.

## Notes for orchestrator authors

Some primitives that would have simplified the orchestrator we wrote:

- **Per-model VRAM telemetry.** Today we use `nvidia-smi` and attribute
  process-level VRAM to a model by container ↔ process ↔ GPU mapping.
  A vLLM-side endpoint reporting "this engine is currently holding X MiB
  in weights, Y MiB in KV cache, Z MiB in CUDA primary context, W MiB
  retained-but-unmapped via cumem" would be more accurate and remove the
  attribution layer.
- **A hard release admin endpoint** distinct from `/sleep` that frees
  everything reclaimable (allocator pools, retained VA reservations,
  NCCL buffers) without killing the process — for cases where the
  operator wants to push the residual back to baseline without paying
  full container restart cost.
- **A forward-progress watchdog with a configurable threshold** built
  into the server (a generalization of `/health/decode`), so individual
  orchestrators don't each have to implement decode-stall detection.

## Related public artifacts

For the curious, our admission controller and the calibrated per-model
residual numbers for our fleet are at
<https://github.com/intarweb/vllm-jukebox>. The numbers in this guide
trace back to that deployment.
