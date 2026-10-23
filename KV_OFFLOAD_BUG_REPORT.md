# KV Offload / NIXL Bug Report

Date: 2026-10-22
Base commit: `87b804f09d` ("[Rust Frontend] Match reasoning delimiters by token identity (#62892)")

## Summary

This report covers five bugs in KV-cache offloading (the OffloadingConnector,
tiering manager and P2P secondary tier) and in the NIXL connector's
heterogeneous block-size path. Each one has a regression test in the
workspace (`git diff 87b804f09d`), in three files:

- `tests/v1/kv_connector/unit/offloading_connector/test_scheduler.py`
- `tests/v1/kv_connector/unit/test_nixl_connector_hma.py`
- `tests/v1/kv_offload/tiering/p2p/test_manager.py`

**Result: 5 of 5 confirmed.** Each bug was checked four ways:

1. The test was run here and fails.
2. The failure shows the stated bug, not a setup error or a wrong assertion.
3. The test reaches the bug the way ordinary use does, and asserts a behaviour
   the code promises (docstring, comment, docs or caller contract).
4. No comment, doc or commit marks the behaviour as intended.

None was rejected, but review changed two claims:

- **B1:** the report said the second reset kills the EngineCore. It does not.
  `EngineCoreProc._invoke_utility_method` (`vllm/v1/engine/core.py:1836`)
  catches the exception and returns it as a failed utility call. Severity
  lowered from high to medium.
- **B4:** it needs one request whose matching prefix is larger than the whole
  CPU staging tier. Severity lowered from high to medium.

**Upstream status** (searched 2026-10-22; see [Upstream search record](#upstream-search-record)):

- **B1** and **B3** already have open upstream fix PRs (#59099 and #59746). Do
  not file them again. Review or test those PRs instead.
- **B4** has the same root cause as open issue #62931 ("a secondary-tier hit
  larger than the CPU tier loads nothing"). That issue's open fix PR #62932
  caps a load at what the CPU tier can stage, and likely fixes B4 too. B4's
  symptom differs (endless deferral rather than a zero hit), so add it to
  #62931 as a comment instead of filing a new issue. Neither #62932 nor the
  related #50014 was run against the B4 test.
- **B2** and **B5** have no matching upstream issue or PR. Both are candidates
  for a new issue. A B5 fix would conflict with open PR #61667.

## Overview

| ID | Title | Component | Location | Reported | Reviewed | Upstream | Found by |
|----|-------|-----------|----------|----------|----------|----------|----------|
| B1 | Second connector reset with no step in between trips an assert | OffloadingConnector scheduler | `vllm/distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py:2199` | high | **medium** | Covered by #59099 (open) | Opus |
| B2 | P2P tier keeps `has_work()` True forever; pause/sleep/shutdown never complete | P2P secondary tier | `vllm/v1/kv_offload/tiering/p2p/manager.py:648` | high | **high** | None found · **fixed on this branch** | Opus |
| B3 | Heterogeneous block sizes misalign transfers after a prefix hit | NIXL worker | `vllm/distributed/kv_transfer/kv_connector/v1/nixl/base_worker.py:3601` | high | **high** | Covered by #59746 (open) | Codex |
| B4 | Prefix lookup defers forever when the prefix exceeds CPU staging capacity | OffloadingConnector scheduler + tiering | `vllm/distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py:779` | high | **medium** | Same root cause as #62931; likely fixed by #62932 (open, not verified) | Codex |
| B5 | P2P `drain_jobs` skips parked stores; reset then asserts | P2P secondary tier | `vllm/v1/kv_offload/tiering/p2p/manager.py:670` | medium | **medium** | None found | Codex |

Severity scale: *critical* means data loss, a security breach, or failure of
the main purpose for most users. *High* means wrong behaviour on a main path.
*Medium* means a real, ordinary bug with contained impact.

---

## B1: Second OffloadingConnector reset with no scheduler step in between trips an AssertionError

**Status:** Confirmed · **Severity:** medium (reported high)

### What's wrong

Two connector cache resets in a row, with no engine step between them, make the
second one fail an internal assertion. The second reset returns an error to the
caller, while a single reset works.

### Trigger

An OffloadingConnector setup (for example `--kv-offloading-size N`) in which
store jobs are in flight at the first reset, followed by two connector resets
with no engine step between them:

- `llm.sleep(level=1)` (which calls `_reset_caches(reset_connector=True)`)
  followed by `llm.reset_prefix_cache(reset_connector=True)`; or
- two back-to-back `reset_prefix_cache(reset_connector=True)` calls.

EngineCore handles every queued utility call before it steps, so two resets in
a row with no step between them is an ordinary sequence.

### Expected vs actual

- **Expected:** each reset succeeds and returns a bool. The pending flush set
  is kept and delivered to workers on the next step.
  `Scheduler.reset_connector_cache` is documented for unconditional cascade
  resets, for example after a weight update.
- **Actual:** the second `OffloadingConnectorScheduler.reset_cache()` fails
  `assert not self._current_batch_jobs_to_flush`.

### Root cause

`_reset_cache()` does two things that leave nothing to clear the flush set:

1. It adds every in-flight job ID to `_current_batch_jobs_to_flush` and keeps
   the set on purpose (see the comment at `scheduler.py:2237`), so that workers
   receive the flush IDs.
2. It clears `_jobs`.

Only `build_connector_meta()` clears the flush set (`scheduler.py:1995`), and
that runs during a schedule step. With `_jobs` empty,
`has_pending_push_work()` (`scheduler.py:2005`) returns False, so an idle
engine never steps. The next `reset_cache()` then fails the guard at
`scheduler.py:2199`.

### Impact (corrected)

The engine survives. `EngineCoreProc._invoke_utility_method` catches the
exception, logs it, and returns `Call to ... method failed` to the client. The
assert fires before any state changes, so the cache is simply not reset.
Another reset after one engine step succeeds. The undelivered-flush half
remains: if no step ever runs, the flush IDs are never delivered.

### Evidence

```
.venv/bin/python -m pytest -q tests/v1/kv_connector/unit/offloading_connector/test_scheduler.py::test_B1
```

```
>       assert not self._current_batch_jobs_to_flush
E       AssertionError
vllm/distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py:2199: AssertionError
```

The test builds a real `OffloadingConnectorScheduler` on a `CPUOffloadingSpec`
and schedules one request so that a store job exists. It then resets twice and
checks that the next `build_connector_meta` still carries the original job IDs
in `jobs_to_flush`.

### Why it isn't intended

The comment at `scheduler.py:2237` explains why the flush set is kept. It does
not say a second reset must be refused. Open upstream PR #59099 treats this
exact crash as a bug.

### Upstream status

**Covered by an open PR. Do not file again.**

- vllm-project/vllm#59099 (open, by sohom-cs, last updated 2026-09-30):
  *[Bugfix][KV Offload] Keep stepping until a reset_cache flush is delivered,
  and make reset_cache re-entrant*. Its description covers this bug, both
  halves: the flush set that is never delivered, and the AssertionError on a
  second reset. Its fix makes `has_pending_push_work()` return True while
  `_current_batch_jobs_to_flush` is non-empty, and removes the re-entrancy
  assert.

Related, not fixes:

- vllm-project/vllm#57810 (open): *[Core][KVConnector] Keep the offloaded KV
  tier across pause/sleep*. It adds an opt-out
  (`sleep(..., clear_connector_cache=False)`) that skips the connector reset
  on pause and sleep. That would avoid one way of reaching B1, but the default
  still resets.
- vllm-project/vllm#45268 (open issue): *EngineDeadError after first L1
  sleep/wake cycle with --kv-offloading-backend native*. Not B1. The issue
  thread traces it to an unfenced memory unmap during sleep (fixed by #45490)
  and a separate Mamba-align scheduler assert.

### Fix direction (untested)

Have `has_pending_push_work()` return True while `_current_batch_jobs_to_flush`
is non-empty, so the engine steps and delivers the flush. Make
`reset_cache()` accept a set that is still populated and merge into it.

---

## B2: P2P secondary tier keeps `has_work()` permanently True, so engine shutdown, pause and sleep never complete gracefully

**Status:** Confirmed · **Severity:** high · **Fixed on this branch**

Fixed in `vllm/v1/kv_offload/tiering/p2p/manager.py` (a real predicate over
`_finished_jobs`, `_failed_serve_ctxs`, `_unbound_stores` and
`session.has_pending_work`) plus `vllm/v1/kv_offload/tiering/manager.py`, which
keeps the engine stepping for an opted-in tier while no live tier-poll thread
services it and, in a multi-tier setup, while a peer lookup could promote out
of another tier (only a scheduler step submits and completes that promotion).
Tests: `tests/v1/kv_offload/tiering/p2p/test_manager.py` and
`tests/v1/kv_offload/tiering/test_tier_poll_thread.py`; the docs note both
behaviours.

### What's wrong

With a P2P secondary tier configured, the engine never reports itself as
idle. Pause and sleep requests hang, and graceful shutdown never reaches
resource teardown (NIXL agent and P2P session cleanup). The parent process
eventually force-kills it after the grace period.

### Trigger

The documented P2P/tiering setup (`docs/features/kv_offloading_usage.md`,
"P2P (Including P/D)"): an OffloadingConnector with `TieringOffloadingSpec` and
`secondary_tiers: [{type: p2p}]`. Then do any of:

- `POST /sleep` or `pause_generation`;
- stop the server normally (Ctrl-C or SIGTERM).

### Expected vs actual

- **Expected:** with no requests and no P2P transfers, a pause completes
  (`_pause_complete()` is True) and the pause Future resolves. On shutdown the
  busy loop exits and resource teardown runs.
- **Actual:** `_pause_complete()` is always False. Pause Futures never resolve,
  and `_handle_shutdown` never returns False.

### Root cause

`P2PSecondaryTierManager.has_pending_work()` (`p2p/manager.py:648`) returns
True unconditionally. The value travels up this chain:

```
P2PSecondaryTierManager.has_pending_work()      -> True
TieringOffloadingManager.has_pending_work()     (tiering/manager.py:1109, any(tier...))
OffloadingConnectorScheduler.has_pending_push_work()  (scheduler.py:2005)
OffloadingConnector.has_pending_push_work()     (offloading_connector.py:207)
Scheduler.has_requests()                        (core/sched/scheduler.py:2923)
EngineCore.has_work()                           (engine/core.py:1596)
```

Three places in the engine depend on `has_work()` becoming False:

- `EngineCoreProc._pause_complete()` (`core.py:2227`) is `not self.has_work()`.
- Idle callbacks, which resolve pause Futures, only run inside
  `while not self.has_work()` in `_process_input_queue` (`core.py:1640`).
- `_handle_shutdown` (`core.py:1729`) exits the busy loop only when
  `not self.has_work()`.

### Evidence

```
.venv/bin/python -m pytest -q tests/v1/kv_offload/tiering/p2p/test_manager.py::TestHasPendingWork::test_B2_idle_manager_does_not_keep_engine_busy
```

```
>       assert core._pause_complete() is True
E       assert False is True
E        +  where False = _pause_complete()
```

The test connects the real objects: a P2P manager, `TieringOffloadingManager`,
`OffloadingConnectorScheduler`, `OffloadingConnector`, `Scheduler` and
`EngineCoreProc`. The scheduler is paused (`PAUSED_NEW`) with no requests and
no jobs, and `_pause_complete()` should then be True.

### Why it isn't intended

Returning True always is deliberate, but only to keep peer polling going. The
code comment, the class docstring (`p2p/manager.py:225`) and the existing tests
in `TestHasPendingWork` all justify it as the way `_control.poll()` and
`session.poll()` keep running. Nothing exempts pause or shutdown:

- The `pause_scheduler` docstring says the Future completes after abort or
  drain.
- `_handle_shutdown` says it exits "when no work remaining".

P2P sets `tolerates_off_thread_polling = True`, so the tier-poll thread
(`tiering/manager.py:308`, default interval 1 ms) already services peers
between steps. The hang is a side effect nobody handled, not a designed
behaviour.

### Upstream status

**No matching issue or PR found.** This is a candidate for a new issue.

Context:

- vllm-project/vllm#58168 (merged): *[KV Offload] Expose
  OffloadingManager.lock()/unlock() and service the tiering control plane off
  the scheduler thread*. It added the tier-poll thread that now services P2P
  peers between steps. That largely removes the original reason for returning
  True always, which was that only the engine tick drove peer polling.
- vllm-project/vllm#51476 (open RFC): *DP pause/sleep correctness*. It notes
  that the pause Future resolves off `has_work()`, the check B2 keeps True.
  The RFC does not mention connectors that report work forever.

### Fix (this branch)

`has_pending_work()` reports real work (sessions with outstanding transfers,
parked unbound stores, finished jobs awaiting hand-off, lookups owing
`on_request_finished`), leaving idle peer polling to the tier-poll thread. The
tiering manager keeps the engine stepping when no live poll thread services the
tier (`tier_poll_interval_s <= 0`), and when a second secondary tier exists: a
peer lookup served off-thread can promote a chunk out of that tier, and only an
engine step submits and completes that promotion. A failing poll thread backs
off instead of exiting, so it always remains a live thread -- which matters
because an idle engine parks and only a step would otherwise service the tier.

### Open

The multi-tier keep-alive restores the pre-fix busyness for a `[fs, p2p]`-style
setup, so pause and graceful shutdown still do not complete there. That is the
same root problem: nothing wakes a parked engine when the tier-poll thread
changes manager state. Settling it needs an engine wake path (see RFC #51476 and
the poller PRs it tracks), which is outside B2.

---

## B3: NIXL heterogeneous block sizes misalign transfers after a prefix hit

**Status:** Confirmed · **Severity:** high

### What's wrong

In NIXL prefill/decode disaggregation with a smaller prefill block size than
decode block size, a partial prefix-cache hit on the decoder makes the transfer
pair each decoder block with the wrong prefill block. The decoder receives KV
from earlier token positions than the ones it needs. The KV is silently
corrupted, so outputs are wrong and nothing raises an error.

### Trigger

The documented configuration in `docs/features/nixl_connector_compatibility.md`
(footnote 3: non-hybrid model, P block size < D block size, "Block IDs are
remapped automatically"), with prefix caching on D, and a prompt whose uncached
suffix leaves D's last allocated block partly filled.

### Worked example

160-token prompt, P block size 16, D block size 64, ratio 4. D has a 64-token
prefix hit.

- Remote (P) blocks: `[0..9]`, covering tokens 0–160.
- D allocates 2 new blocks `[1, 2]` for tokens 64–192. The last one is only
  half filled.
- `_map_block_ids_for_block_size_ratio` expands them to sub-blocks `[4..11]`.
  That is 8 sub-blocks, including 2 padding sub-blocks (10 and 11) beyond token
  160. 8 is not more than the 10 remote blocks, so the front clip does nothing.
- `_apply_prefix_caching` then back-trims the remote list to the last 8 blocks:
  `[2..9]`.
- The pairing becomes local `[4..11]` ↔ remote `[2..9]`. Local sub-block 4,
  which holds token 64, receives remote block 2, which holds tokens 32–48.

The correct pairing is local `[4..9]` ↔ remote `[4..9]` (tokens 64–160).

### Expected vs actual

- **Expected:** skip the cached remote prefix and transfer only the remaining
  prompt tokens, excluding allocation padding, so the KV lands at the right
  token offsets.
- **Actual:** padding sub-blocks inflate the decode count, so the back-trim
  starts too early in the remote list and every pair is off by the padding
  amount.

### Root cause

The two normalisations clip from opposite ends:

- `_map_block_ids_for_block_size_ratio` (`base_worker.py:3571`) front-clips
  the expanded local list to `len(remote_group)`. That only works if local and
  remote are aligned at the start, which is false after a prefix hit.
- `_apply_prefix_caching` (`base_worker.py:3759`) back-trims the remote list to
  `len(decode)`, and that count includes D's allocation padding.

The pull path calls them in that order (`pull_worker.py:432–495`). The push
path does the same (`push_worker.py:676–690`).

### Evidence

```
.venv/bin/python -m pytest -q tests/v1/kv_connector/unit/test_nixl_connector_hma.py::test_B3_heterogeneous_blocks_transfer_only_uncached_suffix
```

```
>       assert decode_ids == [list(range(4, 10))]
E       assert [[4, 5, 6, 7, 8, 9, ...]] == [[4, 5, 6, 7, 8, 9]]
E         At index 0 diff: [4, 5, 6, 7, 8, 9, 10, 11] != [4, 5, 6, 7, 8, 9]
```

The test binds the real worker methods (`get_mapped_blocks`,
`_map_block_ids_for_block_size_ratio`, `_apply_prefix_caching`) to a
full-attention, non-Mamba worker and calls them in the same order and with the
same arguments the pull path uses.

### Why it isn't intended

`_apply_prefix_caching` promises that "only the uncomputed suffix is
transferred". The docs list P < D block sizes as supported with automatic
remapping. No test or comment covers a block-size ratio above 1 combined with a
prefix hit; the existing ratio tests use no prefix hit.

### Upstream status

**Covered by an open PR. Do not file again.**

- vllm-project/vllm#59746 (open, last updated 2026-10-20): *[Bugfix][NIXL]
  Pair P/D blocks by the last token with different block sizes*. Its overview
  describes this bug: with different P/D block sizes, NIXL pull pairs the
  block lists by position, so KV lands in the wrong blocks when the lists
  start or end at different tokens. It claims D receives every page into the
  right block "also after a partial prefix hit on D". Its GPU evals show
  wrong output without the fix (Qwen3-0.6B, 16/64 block sizes, 512-token hit
  on D: 2 of 4 identical without the PR, 4 of 4 with it). It also fixes a
  related D crash on `assert num_decode_blocks <= len(prefill_group)`.

Related, not duplicates:

- vllm-project/vllm#51527 (open draft): *[KV Connector][NIXL] Support prefix
  caching with heterogeneous P/D logical block sizes*. It aligns full-attention
  transfers by token offset, but only for hybrid Mamba/MLA deployments whose
  kernel block sizes match. It overlaps with #59746.
- vllm-project/vllm#61160 (open issue): *Cross-request context contamination
  via shared-prefix KV reuse in PD disaggregation with automatic prefix
  caching*. Same family of flaw: end-aligned pairing with D allocating one
  block fewer, so a decode block keeps another request's KV. The trigger is
  different, though: equal block sizes (64) with sliding-window replay on
  DeepSeek-V4.1. Not B3, but a fix for one should be checked against the
  other.

### Fix direction (untested)

Clip D's expanded sub-block list to the true number of uncomputed remote
blocks, dropping the allocation padding, before the back-trim. This could also
be done by aligning both lists on the suffix in one place.

---

## B4: Filesystem prefix lookup livelocks when the matching prefix exceeds CPU staging capacity

**Status:** Confirmed · **Severity:** medium (reported high)

### What's wrong

With a tiered offload setup, for example the fs tier staged through CPU, a
request whose disk-cached prefix is larger than the CPU staging tier is
deferred on every scheduler step and never makes progress. Each promotion from
disk evicts an earlier promoted chunk, so the prefix is never fully present in
CPU at once.

### Trigger

`TieringOffloadingSpec` with a secondary tier (fs), default uncapped loading
(`max_load_tokens` unset). Repeat a disk-cached prompt after its GPU blocks
have been evicted, where the matching prefix is larger than the configured CPU
staging budget and promotions complete between lookup retries.

### Expected vs actual

- **Expected:** the request makes progress. It loads the part of the prefix
  that fits in the CPU tier, or falls back to recomputing the rest.
- **Actual:** `_maximal_prefix_lookup` returns None on every step, so the
  request is deferred forever.

### Root cause

1. `TieringOffloadingManager.lookup` (`tiering/manager.py:584`) handles a
   primary miss with a secondary hit by calling `_initiate_promotion`, which
   calls `primary_tier.prepare_write`. That can evict a chunk that is already
   a HIT, because a HIT is not pinned until `prepare_load`. The lookup returns
   `HIT_PENDING`.
2. `_maximal_prefix_lookup` (`scheduler.py:749`) sets `defer_lookup` when any
   chunk is `HIT_PENDING` and returns None (`scheduler.py:779`).
3. `_lookup_complete_chunks` (`scheduler.py:825`) passes the deferral up. The
   scheduler skips the request and retries on the next step. Nothing caps the
   number of retries.
4. On the next step the promoted chunk is a HIT, but the next missing chunk's
   promotion evicts it. The cycle repeats.

### Evidence

```
.venv/bin/python -m pytest -q tests/v1/kv_connector/unit/offloading_connector/test_scheduler.py::test_B4_prefix_larger_than_primary_tier_does_not_defer_forever
```

```
E       AssertionError: [None, None, None, None]
```

The test uses the real `CPUPrimaryTierOffloadingManager` (capacity 1 chunk),
the real `TieringOffloadingManager`, and a synchronous
`ExampleSecondaryTierManager` holding a 2-chunk prefix. It runs four
lookup/`on_schedule_end` rounds and expects at least one non-deferred result.

### Why it isn't intended

The comment at `tiering/manager.py:706` says a full primary tier means the
caller "should treat the chunk as unavailable rather than retrying
indefinitely". Here the scheduler retries indefinitely.
`docs/features/kv_offloading_usage.md` presents larger secondary tiers as
on-demand reuse, not as a way to stall requests.

### Impact rationale

It is a real liveness bug, but it needs one request whose matching prefix is
larger than the whole CPU staging tier, which a sized deployment avoids. That
is why it is rated medium rather than high.

### Upstream status

**Same root cause as an open issue, with an open fix PR. Not verified, so
add B4 to the existing issue instead of filing a new one.**

- vllm-project/vllm#62931 (open issue, opened 2026-10-21): *Tiered KV offload:
  a secondary-tier hit larger than the CPU tier loads nothing*. Its root cause
  matches B4's: "Nothing bounds one request by the CPU tier's size." It
  expects the same behaviour: "load the longest prefix that the CPU tier can
  stage ... and recompute the rest". The symptom differs, though:
  - **#62931:** two KV cache groups. The first group's promotions fill the CPU
    tier, and the second group's first promotion fails. The lookup returns
    `MISS`, so the request gets a zero hit and recomputes everything.
  - **B4:** one group, with promotions completing between retries. The
    promotions keep evicting each other, so the lookup defers forever instead
    of returning a zero hit.
- vllm-project/vllm#62932 (open, last updated 2026-10-22): *[KV Offload] Bound
  staged loads by primary tier capacity*. It fixes #62931 by capping one load
  at the number of chunks the CPU tier can stage across all groups (computed
  at startup, `TieringOffloadingSpec` only). It also adds
  `staged_load_timeout_s` (default 300 s), after which a lookup shortened by
  the cap that is still deferred falls back to recompute.
  - In the B4 test the cap would be 1 chunk, so the lookup would request only
    chunk 1, which should then resolve as a 1-chunk hit. Even without that,
    the timeout would end the endless deferral.
  - This is inferred from reading the PR description. The PR was not applied
    and the B4 test was not run against it.
- vllm-project/vllm#50014 (open, last updated 2026-10-22): *[Bugfix][KV
  Offload] Protect unread promotions from eviction by other speculative
  promotions*. Another possible fix, by a different mechanism. Its current
  diff adds a `PinUnreadPromotionsPolicy` (the default) that pins a promotion
  when it lands, via `prepare_load`, until the chunk is read or its request
  finishes.
  - In the B4 test chunk 2's promotion would then fail to allocate (MISS), so
    `_maximal_prefix_lookup` would return a 1-chunk partial hit.
  - This is inferred from the diff, not run.
  - It fixes vllm-project/vllm#49902 (open issue): *Tiered KV offload promotes
    every waiting request (which fills primary DRAM pool)*. That issue
    describes the same promote → evict → re-promote cycle across many waiting
    requests, where B4 shows it within one request.

Related, not fixes:

- vllm-project/vllm#49829 (open issue): *`HIT_PENDING` has no deadline; a
  stalled write can defer requests until the client timeout*. It is the same
  missing cap on deferral, triggered by a stalled write instead of an eviction
  cycle. Its fix PR, vllm-project/vllm#53087 (open), turns a HIT_PENDING
  outstanding past `hit_pending_timeout_s` (default 5 s) into a MISS.
  - That might also end B4's cycle.
  - But B4's chunks change from HIT_PENDING to HIT to evicted, rather than
    staying HIT_PENDING, so it is unclear whether the deadline ever fires.
    Not verified.
- vllm-project/vllm#62615 (open): *Pin batch lookup HITs before
  secondary-tier promotions*. The same mechanism (a promotion evicts an
  unpinned primary HIT), but in P2P producer batch lookups (`LookupMsg`), not
  in the scheduler's lookup.
- vllm-project/vllm#62795 (open): *Flush a request's promotions before
  finalizing its tiers*, and vllm-project/vllm#62929 (open): *Hold promotions
  after a cache reset until jobs report*. Nearby changes to the promotion
  lifecycle.

### Fix direction (untested)

When a lookup would defer, either:

- return the prefix that is ready now and recompute the rest; or
- refuse promotions that would evict a chunk already counted as a hit for the
  same request, or cap a request's promotions to primary capacity, so the
  "treat as unavailable" path in `_initiate_promotion` applies.

---

## B5: P2P cache reset overlooks stores waiting for their first fetch, tripping an AssertionError

**Status:** Confirmed · **Severity:** medium

### What's wrong

On a P2P prefiller, a cache reset while stores are parked waiting for the
decoder's first fetch fails an internal assertion instead of finishing the
reset.

### Trigger

`POST /reset_prefix_cache?reset_external=true`, or a sleep or weight-update
reset, on a P2P prefiller after it has completed CPU stores whose cascade jobs
are parked waiting for the decoder's first `FetchMsg`. In the normal
prefiller-first flow, the prefiller finishes before the decoder fetches, so
this state is common.

### Expected vs actual

- **Expected:** `drain_jobs` resolves every submitted store, failing the parked
  ones because no peer has asked for them, so that
  `TieringOffloadingManager.reset_cache` can proceed. Failing that, the reset
  should return an unsuccessful result instead of asserting.
- **Actual:** `drain_jobs` returns at once. The parked store stays in
  `_unbound_stores` with no result, and the reset fails
  `assert not self._jobs`.

### Root cause

1. `complete_store` cascades the store to secondary tiers via
   `create_store_job`, which calls `_register_job`. That adds the job to
   `TieringOffloadingManager._jobs` (`tiering/manager.py:436`, `:1001`).
2. `P2PSecondaryTierManager.submit_store` (`p2p/manager.py:503`) parks the
   batch in `_unbound_stores` because no session has bound that
   `kv_request_id` yet (`p2p/manager.py:562`).
3. `TieringOffloadingManager.reset_cache` (`tiering/manager.py:1132`) calls
   each tier's `drain_jobs()`. The P2P version (`p2p/manager.py:657–680`) only
   checks `s.has_pending_work for s in self._sessions.values()`, so it ignores
   `_unbound_stores` and returns.
4. `_process_finished_jobs()` gets no completion for the parked job, and
   `assert not self._jobs` fails (`tiering/manager.py:1151`).

### Evidence

```
.venv/bin/python -m pytest -q tests/v1/kv_offload/tiering/p2p/test_manager.py::TestDrainJobs::test_B5_drain_jobs_finishes_unbound_stores
```

```
>       assert mgr._unbound_stores == {}
E       AssertionError: assert {'B5': [_Unbo...3.804885708)]} == {}
E         Left contains 1 more item:
E         {'B5': [_UnboundStoreBatch(job_id=5, keys=[b'key1'], block_ids=[0], ...)]}
```

The test submits a store for a remote-decoder request with no bound session,
calls `drain_jobs()`, and expects the parked store to be resolved as
`JobResult(job_id=5, success=False)`.

### Why it isn't intended

The `drain_jobs` docstring says it will "Block until every submitted load/store
job has completed or failed". The parked store pins primary-tier slots, and the
reset is about to invalidate them. `shutdown()` (`p2p/manager.py:1084–1091`)
already fails parked stores for the same reason, so the drain is simply missing
that step.

### Upstream status

**No matching issue or PR found.** This is a candidate for a new issue.

A fix would collide with:

- vllm-project/vllm#61667 (open, last updated 2026-10-16): *[Bugfix][KV
  Offload][P2P] Quarantine consumer slots until writes settle*. It rewrites
  `TieringOffloadingManager.reset_cache` (retries the drain, rejects the reset
  for quarantined slots) and P2P `drain_jobs` (raises `UnsafeTransferError` on
  unsafe failures). It does not touch `_unbound_stores`, so B5 remains on that
  branch. A B5 fix should be based on it, or coordinated with its author.

Context: vllm-project/vllm#59790 (merged) changed how the producer keeps store
pins until NIXL transfers stop. It does not cover parked stores either.

### Fix direction (untested)

In `drain_jobs`, after the session drain loop, fail and clear every entry in
`_unbound_stores` the same way `shutdown()` does, so the parent manager
receives a completion for each registered job. Waiting for those stores
instead would block until `_unbound_store_timeout_s`, because no peer has asked
for them.

---

## Reproducing all five

Environment: macOS CPU-only host with the existing `.venv` (editable install
with precompiled artifacts). All five tests carry `@pytest.mark.cpu_test` and
need no GPU, NIXL/UCX or model downloads.

```bash
.venv/bin/python -m pytest -q \
  tests/v1/kv_connector/unit/offloading_connector/test_scheduler.py::test_B1 \
  tests/v1/kv_offload/tiering/p2p/test_manager.py::TestHasPendingWork::test_B2_idle_manager_does_not_keep_engine_busy \
  tests/v1/kv_connector/unit/test_nixl_connector_hma.py::test_B3_heterogeneous_blocks_transfer_only_uncached_suffix \
  tests/v1/kv_connector/unit/offloading_connector/test_scheduler.py::test_B4_prefix_larger_than_primary_tier_does_not_defer_forever \
  tests/v1/kv_offload/tiering/p2p/test_manager.py::TestDrainJobs::test_B5_drain_jobs_finishes_unbound_stores
```

Observed on 2026-10-22: `5 failed in 0.80s`. Each test failed for the reason
documented above.

## Upstream search record

Searched on 2026-10-22 with read-only `gh` queries against
`vllm-project/vllm`:

```bash
gh pr list --repo vllm-project/vllm --state open --search "<query>"
gh issue list --repo vllm-project/vllm --state open --search "<query>"
```

Some queries were repeated with `--state all` to find merged or closed
context. The open PRs and issues found above were then read with `gh pr view`,
`gh pr diff` and `gh issue view`.

| Bug | Queries |
|-----|---------|
| B1 | `reset_cache offloading`, `reset_prefix_cache offloading assert`, `_current_batch_jobs_to_flush`, `reset_connector offload`, `reset_cache re-entrant` |
| B2 | `p2p has_pending_work`, `P2P shutdown hang`, `has_pending_push_work shutdown`, `P2PSecondaryTierManager`, `pause hang offload`, `sleep hang kv connector`, `p2p tier`, `has_work shutdown connector`; with `--state all`: `has_pending_work`, `pause_generation hang`, `pause never completes`, `engine never idle`, `sleep never returns`, `shutdown SIGKILL offload`, `tier-poll thread` |
| B3 | `heterogeneous block size prefix`, `block_size_ratio prefix caching`, `nixl block size prefix cache`, `hetero block size`, `_map_block_ids_for_block_size_ratio`, `_apply_prefix_caching`, `nixl different block size` |
| B4 | `promotion evicts`, `HIT_PENDING defer`, `promotion primary full`, `promotion livelock` (issues), `defer_lookup` (PRs; no results). #50014, #62615 and #49829 also came up in the B1/B2/B5 searches |
| B5 | `unbound_stores`, `drain_jobs p2p`, `p2p reset_cache`, `p2p reset prefix cache`, `tiering reset_cache assert`, `unbound store`, `drain_jobs` |

GitHub search is keyword-based, so a matching item that uses different words
could still exist. Before filing B2 or B5, or commenting on #62931 for B4,
also run the duplicate-work checks in `AGENTS.md`. Where an open PR above
overlaps, comment on it rather than opening a new one.
