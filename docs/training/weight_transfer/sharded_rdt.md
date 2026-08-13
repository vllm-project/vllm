# Sharded RDT Engine

The sharded RDT engine transfers weights over [NIXL](https://github.com/ai-dynamo/nixl) using Ray Direct Transport, and it is **pull-based**: the inference workers initiate every transfer. Unlike NCCL (which broadcasts a whole parameter to everyone) each worker pulls only the *slice* it actually consumes under tensor and expert parallelism, so a large MoE model moves roughly `total_bytes / num_workers` per worker instead of `total_bytes`.

## When to Use Sharded RDT

- Very large models where broadcasting whole parameters is the bottleneck — an MoE served with expert parallelism, where each worker owns a small fraction of the experts
- Trainer and inference on **separate GPUs**, with a fabric NIXL supports (InfiniBand, RoCE, EFA)
- Trainers that are themselves sharded, including **pipeline-parallel** trainers where a rank only holds part of the model

Requirements:

- `distributed_executor_backend="ray"` — the workers must be Ray actors
- `nixl` installed in the environment shared by trainer and workers
- `is_checkpoint_format=True` (layerwise reload)
- Weight loaders that only use the supported op set (below)

## How It Works

Three ideas stack on top of each other.

### 1. Op chains: ask for the slice, not the tensor

A weight loader normally receives a full HF-format tensor and slices out the part this worker needs. The engine instead hands the loader a `LazyRDTTensor` — a zero-storage tensor built with `_make_wrapper_subclass`, so `.shape`/`.dtype`/`.size()` work without allocating. Every allowlisted view/slice/shape op returns a *new* lazy with the op appended to a recorded chain; `copy_` is the data sink.

The chain is the wire format. `("model.layers.0.w", (("narrow", (0, 512, 512), ()), ("t", (), ())))` tells the producer: take this tensor, `narrow` it, transpose it, send the result. The producer replays it with `getattr(tensor, op)(*args, **kwargs)`.

`SUPPORTED_OPS` in `sharded_rdt_common.py` is the single table both sides derive from — the consumer intercepts exactly those `torch.Tensor` methods, and the producer's `ALLOWED_OPS` is `frozenset(SUPPORTED_OPS.values())`. They must not drift: an op the consumer can record but the producer rejects bakes successfully at init and then fails at the first pull, a whole sync later. The producer's check is also a guard — replaying a chain calls arbitrary methods on live trainer tensors, so the set must stay confined to pure view / shape-only / byte-bounding operations. `to` in particular must not be allowed: it would let a replay change dtype or device.

Anything a loader does that needs real data — arithmetic, `.to()`, `.float()`, `.item()`, `.data`, bool-mask indexing — escapes the allowlist, reaches `__torch_dispatch__`, and raises `_UnsupportedLazyOp`. That is deliberate: failing loudly at init beats silently transferring the wrong bytes.

### 2. Bake once, replay every sync

Discovering the chains means running the model's loaders, which is expensive. So it happens exactly once, at `init_transfer_engine`, as a **dry run**:

1. `initialize_layerwise_reload` puts the params on meta and saves the kernel tensors.
2. `_install_recording_stamps` wraps each loadable param's *original* `weight_loader` (bypassing `online_process_loader`, so `_layerwise_process` never fires) with a stamp that sets `BakeSink.current = (leaf_module, param_name)`.
3. One `model.load_weights` pass over every name. Each `copy_` reaches `BakeSink.accept_copy`, which records a `_BakedCopy`: the op chain from the source lazy, and the `offset/shape/stride` of the destination read off the **meta** view (valid on meta — no real storage needed). Then it fires a meta `copy_`, which moves nothing but still counts against the layer's loaded numel.
4. Modules that fully loaded (copied numel ≥ `get_layer_size`) become `_BakedModule`s indexed by source name. Partial, unattributable or attention-scale modules are left out and fall back to a plain load.
5. The model is restored.

Every later sync is pure replay: no `load_weights`, no lazy dispatch, no discovery. Reconstruct each destination as `param.as_strided(shape, stride, offset)` and `copy_` the received slice in.

A `copy_` that arrives with no loader stamp cannot be attributed to a param. It is recorded as `None`, which poisons its whole module — that module then takes the plain load rather than scattering into a region nobody identified.

### 3. Gather groups, chunks, and the ring

Weights are transferred in **gather groups**: `layerwise_groups` (in `base.py`, because it defines what a group index means for any `WeightSource`) partitions the flat name list into the pre block, one group per decoder layer, then the post block. The split is by *position* relative to the first layer name, not by name class, so flattening a partition always reproduces the input order and group index *g* means the same thing on every rank and every consumer.

The group is the unit of three things at once: the trainer's gather, the consumer's free barrier, and the arena budget. Without it a whole model becomes one chunk and the receive and serve arenas balloon to the full per-worker share.

Each group's copies are then cut into one **chunk** per producer expert coordinate present — the `ep_rank` stamp of a copy is `name_ep_rank[copy.src_name]`, `-1` (replicated) first, then ascending — so a worker pulls each expert from a rank that actually holds it, under any static vLLM expert placement (`linear`, `round_robin`). Dense models carry only `-1` stamps and keep one chunk per group. Copies are atomic, so a single huge one (an untied `lm_head`) simply makes its chunk oversized — that is what `arena_presize_gb` exists to cover. A module's copies may span chunks (a FusedMoE's experts land on several producer coordinates); `materialize` fires on its first chunk and quant/kernel-copy/`reset` on its last, which makes materialize-once correct by construction rather than by a runtime counter.

Chunks stream over a ring of receive slots (`num_rdt_buffers` deep). While chunk *i*'s RDMA lands inside its `ray.get`, the producer serves chunk *i+1* into its own ring slot and a background thread scatters chunk *i-1* out of another receive slot. The reads themselves stay serialized — they share the flow's NIC, which is the bandwidth floor, not a loss.

## The packed layout

Both sides compute the same byte-exact layout independently: slices packed at 16-byte-aligned offsets in key order. The producer packs into its serve arena and returns one blob; the consumer carves dtype views back out of its receive arena to scatter from. This is the core invariant of the transport — if the two ever disagree, weights are silently wrong.

`pack_check` on both init infos exists to localize exactly that: it checksums each blob on both sides into `/tmp/rdt_profile/packcheck_{prod,cons}.jsonl` so the two streams can be diffed offline.

## Ownership and M:N routing

Producers and consumers need not be the same size, and a producer need not hold the whole model. `RdtRouter`, built identically on both sides from wire-carried data (`num_producers`, `num_consumers`, `group_owners`, `producer_ep_ranks`), answers one question: which producer serves each (gather group, `ep_rank`) pull unit for a given consumer.

- `group_owners[g]` lists the producer ranks that gather and publish group *g*. `None` means every producer owns every group — the gather-to-all layout.
- Expert ownership is two parallel stamp lists that must match: `name_ep_rank` stamps each weight name with the expert-parallel coordinate holding it (`-1` = replicated), `producer_ep_ranks` stamps each producer rank with its own coordinate. A pull for a name stamped `k >= 0` goes to a group owner whose coordinate is `k`; `-1` names match every group owner.
- Each pull unit is served by exactly **one** producer per consumer. Splitting one pull across producers only multiplies produce calls, since the consumer's own NIC bounds the pull either way.
- Freeing does NOT route: every consumer signals `free_group(g)` at every owner of *g*, exactly once per sync, and each owner counts signals against the live consumer total handed to `begin_sync` — a per-group barrier, one uniform integer, no routed targets.

Disagreement between the two sides is not a wrong number but a **hang** or a loud misroute. Three things guard it: `RdtRouter.validate()` (every group owned, coordinates cover every producer), SHA-256 digests of the metadata name sequence AND the stamp list cross-checked across trainer ranks, and the producer's `served_names` allowlist (its owned groups' replicated names plus its own coordinate's experts), which turns a misroute into a loud error instead of an unbounded wait.

A pipeline-parallel trainer declares partial ownership through `WeightSource.owned_groups()`; an expert-parallel trainer declares name-level ownership through `WeightSource.expert_ownership()`. The same requirements come with both: `metadata()` must still describe the **whole** model on every rank (only the sender's metadata reaches the consumers, so a rank describing just its own share would leave the rest silently un-transferred), iteration must yield **only** the owned groups, in metadata order, and within them a name stamped with a foreign coordinate yields `None` (the name stays in the list; only the data is absent).

## Producer lifecycle

Each trainer rank owns a `_RDTProducerServer` Ray actor sharing its GPU. One group's life:

```
engine                      server                          consumer
  begin_sync(live_count) ──►  reset counts, set the barrier target
  gather group (collective)
  publish_group(gi, ...) ──► wait while lookahead is full
                             rebuild CUDA-IPC tensors, take a slot
                                   ◄──── rdt_produce_weights_batched
                                         wait for names in cache
                                         replay chains, pack, serve
                                   ◄──── free_group(gi)   (from EVERY consumer)
                             count to live_count; on the last one:
                             drop cache entries, release the slot
  end_sync() ─────────────►  wait until nothing is in flight
```

Details that matter:

- **Freeing is a barrier keyed by group index.** Every live consumer signals every owner of a group exactly once per sync — after its last chunk of the group, or at sync start when it pulls nothing from it. The free contract is an integer; there is no cross-side name-tuple matching to get wrong.
- **Signals can arrive before their publish.** A consumer with nothing to pull for a group signals it as its plan starts. So `publish_group` completes a group whose barrier is already satisfied, rather than waiting for a signal that will never come again.
- **CUDA-IPC exports must outlive the import.** The engine holds strong refs to every gathered tensor it shared and drops them only when the server reports the group freed.
- **One IPC export per storage, not per name.** Names are described as `as_strided` view specs against a whole-storage uint8 export. The per-name rebuild this replaced cost ~32 µs/name of pure Python plus an IPC open per new storage — and IPC opens are ~9× slower again when the exporting process uses the `expandable_segments` allocator, which is why `trainer_init` warns about it.
- **`gather_lookahead` bounds trainer memory.** `publish_group` blocks while that many groups are resident; the consumers' `free_group` barrier drains it. So the gather loop self-paces to the consumers' pull rate. The default of 1 is the steady state "serve group N while gathering N+1"; raise it if the barrier ever paces the wall.
- **`warmup_nixl` breaks a startup deadlock.** Creating the NIXL agent lazily deadlocks on EFA-class fabrics: libfabric's `fi_getinfo` probes CUDA HMEM with a `cudaMalloc`/`cudaFree`, and `cudaFree` blocks behind the co-resident trainer rank's persistent NCCL kernel — which cannot finish because the collective waits on the sender rank, which waits on the worker-init RPC, which waits on this server. Creating the agent up front, while the GPU is quiet, breaks the cycle.

## The slot generation handshake

This one is worth stating precisely, because getting it wrong produced nondeterministic weight corruption.

A `torch.cuda.Event` only waits on its **last** `record()`. If the RPC thread reaches `synchronize()` for slot *s* before the background thread has *recorded* the event for every item ever queued on that slot, the wait silently binds to an earlier record and passes — and the next RDMA overwrites the slot under a pending scatter.

So the reuse guard has two stages, both required:

1. **Generation wait.** The RPC thread counts items queued per slot; the background thread counts records. A pull may only proceed once `done[slot]` has caught up with `queued[slot]`.
2. **Event synchronize.** Then wait for the recorded scatters to actually finish on the GPU.

The guard must precede `set_target_for_ref`, not just the `ray.get`: the transfer may start any time after the metadata push.

## Tuning

| Knob | Default | Effect |
| ---- | ------- | ------ |
| `num_rdt_buffers` | 2 | Ring depth on both sides. Must match, and the producer's ring must be no shallower than the consumer's — the producer-ring safety argument depends on it. |
| `arena_presize_gb` | 0 | Pre-size each arena slot. Set it to cover the largest atomic chunk. |
| `gather_lookahead` | 1 | Resident gathered groups on the trainer before the gather loop blocks; the per-group free barrier is the back-edge. |
| `pack_check` | off | Checksum every blob on both sides for offline diffing. |

Two results worth not rediscovering:

**A spare receive slot does not help.** Receive slots are both the RDMA landing zones and the recycling unit, and there are exactly as many as in-flight pulls — so chunk *j* takes the slot of chunk *j-K*, which `drain_one()` completed on the line before. Every chunk therefore pays a wait on a scatter dispatched microseconds earlier (measured 7.2 ms/pull at 235B). Adding one spare slot does remove that wait entirely, and **measured twice, the wall got worse**: 3.39 s vs 3.10 s with the old per-name pack, then 3.44 s vs 2.70 s after the pack was made cheap. A deeper receive pipeline pulls demand forward past what the trainer can supply under `gather_lookahead=2`: the serve RPC starts arriving before the group is published (producer wait 3.6 → 9.3 ms/call) and the extra concurrency inflates the pack (8.3 → 21.4 ms/call, GIL-bound in the sidecar). The binding constraint is gather **supply**, not the slot structure. Revisit only after the gather can run further ahead — e.g. freeing a gathered group once it is *packed* rather than once its RDMA completes.

**Sizing arenas once matters beyond throughput.** Ray's NIXL descriptor cache is keyed by `data_ptr` and its entries outlive their tensors, so repeated small regrowths can false-hit a recycled pointer and skip registering the new extent — surfacing as `NIXL_ERR_NOT_FOUND` at `initialize_xfer`, or worse a stale-MR write. Hence `arena_alloc_bytes`' coarse round-up, and the pre-registration of every buffer at init while the fabric is idle.

**Sync 0 runs serial.** Both sides still grow and register arenas on the first sync, and a producer-side registration churns its NIXL agent-metadata version; with pulls in flight the consumer's remote-agent cache can go stale for one of them (`createXferReq`: "no backend had the required registrations"). So the chunk pipeline runs one-deep during sync 0 and pipelines from sync 1, when registrations are at high-water.

## Measuring

The engine ships a hardcoded profiling layer (`_nixl_profile.py`) that monkeypatches Ray's NIXL transport to accumulate per-process register/transfer/deregister counters, plus `PhaseTimer` for the consumer's process phases and jsonl sinks under `/tmp/rdt_profile/`. It is benchmark scaffolding, not shipping code, and two properties are worth knowing before reading its numbers:

- `PhaseTimer.phase()` calls `stream.synchronize()` on **every** scope exit and wraps materialize/scatter/quant/kernel-copy. The instrumentation therefore serializes the very pipeline those phases exist to overlap: the split sums to `process`, but slightly inflates it against the un-instrumented path.
- `install_nixl_timing()` is installed unconditionally on both the consumer worker and the producer sidecar.

Summing overlapped sub-operations cannot equal wall time. When a pull's `produce_wait` and a background scatter genuinely run at once, adding them double-counts; the per-phase numbers localize *where* time goes, and only the wall bounds *how much*.

## Known rough edges

The producer server's concurrency has hazards that are currently prevented by protocol rather than by locks. They have not bitten in practice, but a change to free timing or actor concurrency could expose any of them:

- `_serve_rings` dict membership is guarded by `_serve_lock`, but the element writes are not, so a `reserve_serve_arena` regrow can swap the arena a concurrent serve is packing into.
- `_pack_dsts` is an unguarded shared dict. Individual dict ops are GIL-atomic, so nothing corrupts, but two threads can both build views for one key and the loser's may reference a replaced arena.
- `_cache` is written under `_cache_cond` and read without it. Safe only because the last free implies every routed consumer's pull already returned.
- All three `wait()` sites are untimed. A consumer that dies mid-pull leaves `end_sync` waiting forever with no timeout, heartbeat or liveness check; the only escape is the trainer's own `set_gather_error`, and that call's failure is suppressed.
- `shutdown` clears state without `notify_all`, orphaning any thread parked on the condition.

## Examples

- `examples/rl/rlhf_sharded_rdt.py` — minimal single node, 1 trainer + TP-2
- `examples/rl/rlhf_sharded_rdt_mn.py` — arbitrary M:N across nodes, env-driven
- `examples/rl/rlhf_sharded_rdt_qwen235b.py` — Qwen3-235B-A22B with partial (pipeline-style) ownership
- `examples/rl/rlhf_sharded_rdt_kimi.py` — 1T FP8 MoE from a raw sharded checkpoint
