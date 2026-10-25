# GDN checkpoint at radix branch points — Phase 2 design (issue #198)

Status: design study. Nothing here is deployed or committed. Read-only analysis of the
fork at `dcbe5da29e` (= upstream/main, 0 behind on 2026-10-25), same logic as the
production `v0.32.0` overlay image.

Scope: answer "can a compacted prompt hit its socle again on a hybrid GDN+attention
model, and what would it take". Code references are `file:line`.

---

## 0. Headline finding: most of the mechanism already exists

The premise written in the issue ("the only GDN state checkpoint taken is at the last
prompt boundary") is true **only of the internal checkpoint mechanism it names**
(`num_prefill_checkpoint_blocks`, `is_mamba_prefill_checkpoint_valid`, FlashInfer-only,
one per prompt). It is **not** true of the GDN states the engine actually writes: in the
configuration we run, a state is cached at **every block-aligned prefill chunk end**
(§2d), and the shared-prefix junction is already one of them. So the socle does not have
zero states — it has several, and the real defect is that **every one of them is an
LRU-evictable cached block that dies before the next replay.** What is missing is
persistence, not the checkpoint.

The synchronised upstream already ships a Marconi-style branch-point mechanism, explicitly
labelled as such:

- the "shared prefix boundary" is detected per request and stored on the request
  (`vllm/v1/request.py:203-206`, "Marconi-style");
- it is computed as the gap between the deepest attention-group hit and the reconciled
  (GDN-limited) hit (`vllm/v1/core/kv_cache_coordinator.py:927`);
- `get_computed_blocks` returns it (`vllm/v1/core/kv_cache_manager.py:321-323`);
- the scheduler already **ends a prefill chunk at the block-floored junction**
  (`vllm/v1/core/sched/scheduler.py:610-640`), which is exactly where a Mamba state block
  gets written and cached;
- every cached align-mode state block is already **offered to the KV connector** for
  tiering (`vllm/v1/core/single_type_kv_cache_manager.py:1995-2018`,
  `vllm/v1/core/kv_cache_manager.py:878-902`).

So Phase 2 in this fork is **not a greenfield checkpoint engine**. It is (a) making the
junction state *survive* long enough to be reused, and (b) — only if we want the exact
socle boundary rather than the block below it — enabling the sub-block checkpoint, which
is gated behind a backend we do not run.

The rest of this note backs that up with code, then quantifies and lists the real gaps.

---

## 1. Where the decision is taken

Three different places do three different jobs. They should stay split this way.

### 1a. Detection — `HybridKVCacheCoordinator.find_longest_cache_hit`

`vllm/v1/core/kv_cache_coordinator.py:796-931`.

It runs a fixed-point loop over the KV groups: each group either accepts the candidate
length or reduces it, until the length stops shrinking. It keeps:

- `hit_length` — the reconciled hit = **min over groups** (`:903-905`). On this model the
  GDN groups are the short pole, so this is the effective hit.
- `longest_hit_length` — **max over groups** (`:901`); the full-attention group caches
  densely and goes deeper.
- the gap: `num_uncached_common_prefix_tokens = longest_hit_length - hit_length`
  (`:927`). This is *by definition* the branch point of the issue: "attention blocks
  cached deeper than the GDN state allows".

Data available here and nowhere else: the per-group `hit_length_by_group`,
`hit_blocks_by_group`, the shared `block_pool`, and the request's `block_hashes`
(`resolve_block_hashes`). That is why detection belongs here and not in the scheduler:
the scheduler only sees the reconciled number.

### 1b. Propagation — `KVCacheManager.get_computed_blocks`

`vllm/v1/core/kv_cache_manager.py:264-326`, junction at `:321-323`:

```python
shared_prefix_boundary = (num_new_computed_tokens + num_uncached) if num_uncached else 0
```

The scheduler stores it on the request at admission (`scheduler.py:1098`, and again after
reconciliation at `:1179`).

### 1c. The chunk decision — `Scheduler._mamba_block_aligned_split`

`vllm/v1/core/sched/scheduler.py:527-640`, called from `schedule()` at `:838-841`.

- `junction = request.shared_prefix_boundary` (`:610`)
- `block_floored = start + (junction - start) // block_size * block_size` (`:612`) —
  floored to the block grid *relative to an already block-aligned `start`*, so the result
  is block-aligned.
- `junction_stop = junction if self.mamba_shared_prefix_checkpoint and junction <=
  num_prompt_tokens else block_floored` (`:616-621`)
- added to the mandatory `stops` tuple (`:636`), and `end = min(stops)` (`:639`).

`self.mamba_shared_prefix_checkpoint` is **False in production** (`scheduler.py:446-449`,
see §2c), so the stop is the block-floored junction, not the exact byte.

### 1d. The write and the cache — `MambaManager`

- The GDN state is written by the model runner at the chunk-end block column
  (`vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py:1871`
  `ssm_state[prefill_state_indices] = last_recurrent_state`), with the column chosen as
  `(computed_after + block_size - 1)//block_size - 1`
  (`vllm/v1/worker/mamba_utils.py:542`). Only the chunk-end block is real; intermediate
  blocks are **null** (`single_type_kv_cache_manager.py:1844-1866`).
- `MambaManager.cache_blocks` caches it (`single_type_kv_cache_manager.py:1973-2028`,
  base `cache_blocks` at `:477-541`, then `BlockPool.cache_full_blocks`
  `vllm/v1/core/block_pool.py:226-344`, which **skips null blocks** at `:277`).
- Sub-block (partial-tail) checkpoints go through `_cache_partial_tail_block`
  (`single_type_kv_cache_manager.py:2023-2128`).

**Verdict on the question asked:** the scheduler is the right place for *the split* (it
owns chunking and the block-alignment invariant, `scheduler.py:586-590`). The KV cache
manager is the right place for *the write and the cache*. The new Phase 2 work is in a
fourth place the issue does not name: **the eviction policy in `BlockPool`** (§2d), which
is where the junction state currently dies.

---

## 2. Mechanism

### 2a. Alignment

Production values (profile `myia_vllm/configs/docker/profiles/medium-swift15-27b-fstier.yml:143,166`):

| knob | value | source |
| --- | --- | --- |
| `block_size` | **1424** | fs block file = 26,247,168 B = 1424 × 18,432 B (`NOTES` §25/10 17:30Z) |
| `hash_block_size` (=`prefix_match_unit`) | **16** | `--prefix-match-unit 16` |
| `cache_hit_alignment_tokens` | **16** | `kv_cache_coordinator.py:698-706` (partial hits on) |
| `prefill_checkpoint_alignment` | **None** | triton backend, see §2c |
| `num_prefill_checkpoint_blocks` | **0** | idem |
| `mamba_cache_mode` | `align` | prefix caching on (`single_type_kv_cache_manager.py:1437-1445`) |
| `retention_interval` | **None** (dense) | no `--prefix-cache-retention-interval` |

`is_mamba_prefill_checkpoint_valid` (`vllm/v1/core/kv_cache_interface.py:1169-1190`) is
the gate for a checkpoint whose position is *not* the chunk end (sub-block). It requires
`query_start % hash_block_size == 0`, `checkpoint_col > initial_state_col`,
`query_start + hash_block_size <= position`, `query_start < position < query_end`, and
`(position - query_start) % checkpoint_alignment == 0`. With `checkpoint_alignment = None`
it returns False outright (`:1178-1179`).

A **block-floored** junction (our case) needs none of that: it is a normal chunk end, so
it inherits the align-mode invariant — "slot p holds the state after exactly
`(p+1) * block_size` tokens" (`scheduler.py:586-590`).

### 2b. How many extra blocks, and where they live

- **One GDN checkpoint = 3 blocks**, one per GDN group (`g0`,`g1`,`g2`; `g3` is
  attention — `NOTES` §25/10 17:30Z). Each Mamba page is padded to the attention page
  size (+0.38%), so 3 × 26.25 MB ≈ **79 MB** (the "~79 MB per snapshot" measured in prod).
- The **block-floored junction allocates nothing extra**: it reuses the chunk-end block
  the prefill was going to write anyway, and `cache_full_blocks` caches it as a full
  block. So in our configuration the "extra checkpoint" is a **retention** cost, not an
  allocation cost.
- The **sub-block junction** would need an extra reserved block column
  (`num_prefill_checkpoint_blocks`, `kv_cache_interface.py:1086`,
  `MambaSpec.max_memory_usage_bytes` `:1120-1126` adds it to the per-request budget).

Storage path today, end to end:

```text
GPUPool block (cached full block)
  └─ KVCacheManager.take_boundary_state_offloads  (kv_cache_manager.py:878-902)
       └─ OffloadingConnector  →  RAM tier (24 GiB, --kv-offloading-size)
            └─ TieringOffloadingSpec cascade  →  fs tier (TouchOnHitFsTier, 250 GiB cap)
```
Write-through means every cached state is cascaded (`myia_vllm/kv_tier/lru_fs_tier.py`,
overlay `myia_vllm/configs/docker/overlay_v0320_tierpin/`). Phase 1(a) is the filter that
would keep *only* the reversible ones; Phase 2 is what makes a per-lane socle *worth*
keeping.

### 2c. Why the existing sub-block checkpoint does not fire for us

- `enable_mamba_shared_prefix_checkpoint` defaults **False** (`vllm/config/cache.py:309-314`),
  and the flag is documented as needing "EAGLE on the Mamba group" — we run
  `--speculative-config null`, so `coordinator.eagle_group_ids` is empty and
  `KV_CACHE_MANAGER.mamba_shared_prefix_checkpoint` is False
  (`kv_cache_manager.py:186-195`).
- Even with the flag, the worker-side export needs the **FlashInfer GDN prefill** backend:
  `self._checkpoint_alignment` is only set when `gdn_prefill_backend == "flashinfer"`
  (`qwen_gdn_linear_attn.py:691-702`); `_exports_checkpoints` then gates
  `num_prefill_checkpoint_blocks=1` (`:739-760`). We run `--gdn-prefill-backend triton`
  (`profile:143`), so `num_prefill_checkpoint_blocks = 0`, `prefill_checkpoint_alignment =
  None`, `has_prefill_checkpoint_blocks = False` (`single_type_kv_cache_manager.py:1442-1445`),
  and the whole `is_mamba_prefill_checkpoint_valid` path is dead.

So the exact-junction checkpoint is a **flashinfer-only feature we cannot use today**, and
the block-floored junction is the fallback that still works on triton.

### 2d. Reclamation — the actual gap

Cached blocks are LRU-evicted: `BlockPool.free_blocks`
(`vllm/v1/core/block_pool.py:776-809`) puts cached blocks at the **tail** of the free
queue and non-cached at the front; reuse goes through
`_maybe_evict_cached_block` (`:729-750`). There is **no priority, pin, or protection** for
a high-value block. `unpin_blocks` (`:708-727`) is a *release* ("last-resort eviction
candidates"), used by the tiering connector, not a pin.

Because a GDN page is 26.25 MB (≈ the attention page) and the RAM tier turns over in
22-38 min against a ~30 min cron (`NOTES` §25/10), a junction state written at replay *n*
is very likely evicted before replay *n+1*. It is re-written at every replay, so the
socle is recomputed on every replay — exactly the symptom in the issue.

**This is the lever Phase 2 must pull: keep the junction state, do not just create it.**

---

## 3. Memory cost

Per-token and per-block figures are measured, not assumed:

| quantity | value | where it comes from |
| --- | --- | --- |
| attention KV footprint | **18,432 B/token, both TP ranks** | `NOTES` §25/10 17:30Z correction: 16 attention layers × K+V × 4 KV heads × 256 × 0.5625 B (NVFP4) |
| one block (attention) | **26,247,168 B = 26.25 MB** | fs block file size, measured (`NOTES` §25/10 17:30Z) |
| one GDN state page | **≈ 26.25 MB / page**, padded +0.38% to the attention page | boot log "mamba page padded" (`NOTES` §25/10 17:30Z) |
| one GDN checkpoint (3 groups) | **≈ 79 MB** | 3 × 26.25 MB, measured "~79 MB each, ~1.6/request" |
| GPU KV capacity | **831,723 tokens @ gpu-util 0.73** (750,747 @ 0.70) | `NOTES` §24/10 21:10-21:35Z |
| GPU KV pool size | **≈ 584 attention blocks** (606 measured on a 0.70 run) | 831,723 / 1424; `NOTES` §25/10 17:30Z |

Consequences for a 262,144-token window:

- a full window is 262,144 / 1424 = **184 attention blocks**;
- the pool is ≈ **584-606 blocks**, i.e. ~3.2 windows of attention plus state headroom;
- **one GDN checkpoint = 3 blocks = 0.5 % of the pool** (3/606);
- **N distinct socles cost 3N blocks**: the 25/10 header scan found on the order of a
  handful of distinct socles (Claude Code main agent 878/997, Roo/Zoo 36, code-review
  59+9, Agent SDK 12 — `NOTES` §25/10 17:30Z), so ~15 blocks ≈ **2.5 % of the pool** for
  ~5 socles.

So capacity is not the obstacle. The obstacle is that the pool has no way to say "these
15 blocks are worth more than the next 15 that arrive".

**What must be measured before sizing for real (state the instrument, do not guess):**

1. `prompt_tokens_cached_by_source{source=disk}` vs `{source=device}` per request, and
   `num_uncached` (= attention hit − effective hit) per request. This is exactly M2 in
   `myia_vllm/docs/fs-tier/m2-instrumentation.md`. Until M2 shows `num_uncached > 0`
   on compaction replays, the whole premise is unverified on this workload.
2. The GDN page size straight from the engine (`MambaSpec.page_size_bytes`, or the fs
   file size), not from the arithmetic above.
3. The number of *distinct retained junction states* over a day (needs a counter; today
   none exists).

---

## 4. Correctness risks

The single rule: **a Mamba block cached under a hash at token position P must hold the
recurrent state after exactly P tokens.** If it does not, a later request hits it and
produces **wrong output silently** — not a recompute. Everything else (a miss, an
eviction, a refused store) is harmless: it only costs a recompute.

Properties that must hold for a checkpoint to be valid:

1. **Position match.** The state was produced by a forward whose query range ends exactly
   at P. Align mode guarantees this because chunk ends are block-aligned
   (`scheduler.py:586-590`) and the column is `cdiv(query_end, block_size) - 1`
   (`mamba_utils.py:542`).
2. **Alignment.** For sub-block positions, `is_mamba_prefill_checkpoint_valid`
   (`kv_cache_interface.py:1169-1190`) enforces hash-block, column and checkpoint-grid
   alignment. With `checkpoint_alignment = None` (us) it refuses — so a sub-block
   checkpoint cannot be taken on triton at all.
3. **Non-null source.** Only real (non-null) blocks carry a state; `cache_full_blocks`
   skips null ones (`block_pool.py:277`). If a new mechanism cached a block that the
   forward never wrote, the hash would lie.
4. **No clobbering the running state.** A checkpoint must go to a block that is *not* the
   live running-state block, or the "valid until reused" invariant of the running block
   breaks. The sub-block path reserves a column with `cdiv(seq_len, block_size) - 2`
   (`checkpoint.py:58`) and refuses via `_needs_internal_checkpoint`
   (`single_type_kv_cache_manager.py:1655-1686`) when no writable block is free.
5. **Key exactness.** The hash is keyed at the exact token count: `cache_partial_block(...,
   num_tokens=position)` (`:2071-2078`) and `cache_full_blocks` at `(i+1)*block_size`
   (`block_pool.py:280`). The `block_hash_num_tokens == position` guard
   (`:2069-2070`) prevents re-keying a block already at the right position.

What happens on the failure modes:

- **Stale** (a block whose content is from a different position): would be served as a
  hit → **wrong output**. Mitigated by invariants 1, 3, 5. This is the risk to watch in
  any change that adds new write sites.
- **Misaligned** (sub-block position not on the grid): `is_mamba_prefill_checkpoint_valid`
  returns False → the checkpoint is simply not taken → recompute. Safe.
- **Evicted**: miss → recompute. Safe, and the current default.

So today's design already keeps a wrong checkpoint from being *created*; the Phase 2
danger is introduced only if we write a state outside a forward that ends at its key, or
retain a block whose bytes were overwritten. Any implementation must keep the five
invariants above unchanged, and the safe degradation is always "miss, recompute".

---

## 5. Minimal implementation sketch

The change is small because the mechanism exists. Two variants:

### Variant A — retain the block-floored junction state (what we can ship in the fork)

No new checkpoint computation. Add a **survivable class of cached block** to the pool and
mark junction states as such.

Files touched:

1. `vllm/v1/core/block_pool.py` — a "protected" flag on `KVCacheBlock` (or a parallel
   set) and a second free-queue class that `free_blocks`/`_maybe_evict_cached_block`
   consult, so protected blocks are evicted only when nothing else is free. ~60 lines.
2. `vllm/v1/core/single_type_kv_cache_manager.py` — in `MambaManager.cache_blocks`
   (`:1995-2018`), tag a block protected when `block_hash_num_tokens ==
   request.shared_prefix_boundary`'s block-floored position. Release the tag on
   `pop_blocks_for_free`/`_remove_blocks_in_range`. ~20 lines.
3. `vllm/config/cache.py` + `vllm/engine/arg_utils.py` — a flag
   `--mamba-pin-shared-prefix-state` (separate from the eagle-only
   `enable_mamba_shared_prefix_checkpoint`), default off. ~15 lines.
4. Tests: `tests/v1/core/<new>_test_shared_prefix_pin.py` — (i) the junction block is not
   the LRU victim, (ii) a sibling resumes from it, (iii) with the flag off, behaviour is
   byte-identical to today.

Pseudocode at the write site:

```python
# MambaManager.cache_blocks, after a full block is cached
if pin_flag and block.block_hash_num_tokens == block_floored(shared_prefix_boundary):
    block_pool.mark_protected(block)          # survives until pool pressure demands it
```

Rollback: the flag. Risk: protected blocks must still be reclaimable under full pressure,
or the pool can deadlock — protected blocks must degrade to normal cached blocks when the
free queue would otherwise starve (the `unpin_blocks` "last-resort" pattern is the right
model).

Sizing: **~1 week** for one engineer including tests and a fresh-vs-fresh measurement of
recompute-per-replay (M2). It does not need the exact socle boundary: the block below it
is ≤ 1423 tokens off, which is the same slack the current align mode already accepts.

### Variant B — exact sub-block junction

Requires enabling the FlasInfer GDN prefill backend (or porting the checkpoint export to
triton), then flipping `enable_mamba_shared_prefix_checkpoint` and making it non-eagle
gated. Files additionally touched: `qwen_gdn_linear_attn.py` (`_exports_checkpoints`),
a real change to `_cache_partial_tail_block`'s `shared_prefix_checkpoint` predicate to
fire without eagle, the worker checkpoint exporter, and the flashinfer kernel path
(`checkpoint.py`). This is a **kernel-backend migration** with a different numerics/
stability surface than the triton path we keep (triton was a stability workaround, now an
inert compile-cache hash factor — `NOTES` §24/10 21:35Z,
`qwen_gdn_linear_attn.py:691-702`).

Sizing: **one month or more**, and it depends on a backend swap we have avoided. Not the
one to start with.

### Judgement

Variant A is a **one-week change** in our fork, on files we already overlay for the fs
tier. Variant B is a month and a backend risk. Start with A, measure with M2, and only
reach for B if the 1423-token slack turns out to matter.

---

## 6. Proposed upstream issue / RFC text

> **Title:** Hybrid (Mamba/GDN + full-attention) models cannot reuse a shared socle after
> context compaction: the SSM checkpoint exists only at prompt boundaries
>
> **Body:**
>
> Hybrid models with Gated DeltaNet layers (Qwen3.5/3.6, Swift-1.5, …) hit a prefix-cache
> only at positions where a Mamba state block is cached; the effective hit is the minimum
> over KV groups (`HybridKVCacheCoordinator.find_longest_cache_hit`,
> `vllm/v1/core/kv_cache_coordinator.py:796-931`). Agentic workloads compact their context
> frequently (Claude Code lanes on a ~30 min cron). After a compaction the prompt is a
> fixed *socle* (system prompt + tool definitions + injected context) followed by a
> summary. The socle's attention blocks are cached, but no GDN state exists at the point
> where the new prompt forks, so the whole socle is recomputed on every compaction — even
> though its KV is resident. The per-token attention footprint on our 27B model is 18,432
> B/token, so this is tens of thousands of tokens of wasted prefill per turn.
>
> The repository already has the right detection (`shared_prefix_boundary`, computed as
> `longest_hit_length - hit_length` at `:927`) and the scheduler already ends a chunk there
> (`scheduler.py:610-640`). What is missing is (a) protecting that state from LRU
> eviction so it survives to the next replay, and (b) allowing the checkpoint at the
> exact (sub-block) branch point on non-FlashInfer GDN prefill backends.
> `enable_mamba_shared_prefix_checkpoint` (`config/cache.py:309`) is documented as
> EAGLE-only and the worker export requires `gdn_prefill_backend == "flashinfer"`
> (`qwen_gdn_linear_attn.py:691-702`), so a triton deployment falls back to the
> block-floored junction and still loses the state to LRU.
>
> Marconi (arXiv:2411.19379) shows that SSM states must be cached with the KV at the same
> position and admitted selectively by reuse taxonomy (shared input prefix vs continuation)
> with FLOP-aware eviction. A provenance-tagged "protected" block class for the shared-
> prefix junction, and a non-EAGLE checkpoint path, would implement exactly that.
>
> Ask: (1) is a protected/priority block class in `BlockPool` the intended mechanism, or
> should this live in the connector; (2) should the sub-block checkpoint be made
> independent of the FlashInfer prefill backend; (3) which of the open admission-policy
> RFCs (#51240 / #60000 / #51981) should carry the priority signal.

---

## 7. Open questions (the code does not settle these)

1. **Does the junction fire in production, and how often is the gap non-zero?** The code
   says the branch point is written and cached. Whether it is *observably* reused is an
   empirical question for M2 (`num_uncached`, per-request attention hit vs effective hit).
   If M2 shows the socle's attention blocks are *also* evicted (`num_uncached == 0`), the
   lever shifts from "keep the junction state" to "keep the socle at all", which is a
   different design.
2. **Why does the fs tier serve only ~2.2 % of prompt tokens if the junction state is
   cascaded to it?** The state is offered to the connector
   (`take_boundary_state_offloads`), which writes through to the fs. The gap between
   "written" and "served" is not explained by the code read here; it needs the M1 mtime
   diff per group plus the offloading connector's partial-tail lookup
   (`vllm/distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py:1159-1199`).
3. **Does the block-floored junction differ per replay?** If the socle boundary shifts by
   a few tokens between replays, the block-floored position may shift by a block, so a
   pinned state at replay *n* would not be hit at replay *n+1*. Whether the socle is
   byte-stable across a lane's compactions is a hub-side question (claudish#640 / M3).
4. **Is "protected" the right primitive, or is a tier hint better?** Protecting in GPU
   costs pool; keeping it only in the RAM/fs tier costs a load but no GPU pool. The
   arithmetic in §3 says either fits; which one wins is a measurement, not a code read.
