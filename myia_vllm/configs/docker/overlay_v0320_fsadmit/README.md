# overlay_v0320_fsadmit — fs KV tier write admission (issue jsboige/vllm#198, Phase 1a)

**Status: built for evaluation, NOT deployed.** Deploying means pointing the
fstier profile at this image + a user GO (see “Deployment” below). Nothing in
this overlay touches a file that is currently in production.

## What it does

Upstream tiering is write-through: every store that reaches the RAM tier is
cascaded to every secondary tier (`vllm/v1/kv_offload/tiering/manager.py`,
header principle 1; the cascade loop lives in `complete_store`). On our
production workload (engine start 2026-10-25 09:04Z, ~8.9 h) that wrote
**427.5 GB** to the fs tier — equal to the GPU→RAM stores, i.e. every RAM
store cascaded — of which **9.3 % was ever read back** (issue #198 for the
full numbers). This overlay adds a **frequency admission filter** for
STORAGE-medium (fs/SSD) secondary tiers, after the same idea as:

- **NVIDIA Dynamo KVBM** disk-offload filter (on by default,
  `DYN_KVBM_DISABLE_DISK_OFFLOAD_FILTER`): CPU→disk only for blocks with
  frequency ≥ 2;
- **SGLang HiCache** `--hicache-write-policy write_through_selective` (back
  up only radix nodes whose hit count passes a threshold).

A chunk is cascaded to the fs tier only once it has been **found (lookup
HIT) in the primary (RAM) tier at least N times** (N = `MYIA_FS_ADMIT_MIN_HITS`,
default 2). The storing request’s own walk sees MISS/absent, so the count
starts accruing with the first *reusing* request — one-shot content (the
majority of GDN state snapshots, cold prefixes) never reaches the disk.

RAM is never filtered: the primary tier and every non-STORAGE secondary tier
keep exact upstream behaviour.

## The one patched file, and how it composes with the tierpin overlay

This overlay replaces **`vllm/v1/kv_offload/tiering/manager.py` only**. The
production image (`vllm-openai-v0320-tierpin:v1`, in service since 2026-10-24)
already carries a 2-file fix for the tiering race `Chunk … not found in cache`
(`overlay_v0320_tierpin/`, PR #176): `cpu/manager.py` (the `extra_protected`
parameter) and this same `tiering/manager.py` (`_step_hit_keys`).

Composition is by construction, not by merge: the file shipped here **is the
tierpin version plus the admission filter** (copied from
`overlay_v0320_tierpin/vllm/v1/kv_offload/tiering/manager.py`, then patched).
`PATCH.diff` in this directory is diffed **against the tierpin file** — it
shows only the fsadmit delta. `cpu/manager.py` is inherited untouched from
the base image layer. The Dockerfile `FROM vllm-openai-v0320-tierpin:v1` and
its build-time assert checks **both** patches are present in the single
composed file (`_step_hit_keys` AND `_record_fs_admit_hit`), so this overlay
can never silently revert the race fix. Remove the whole overlay (repoint the
profile at `vllm-openai-v0320-tierpin:v1`) and you are back to exactly
today’s production image.

## What changes vs upstream — file:line evidence

All line numbers refer to the patched file in this directory
(`vllm/v1/kv_offload/tiering/manager.py`, 1290 lines). The upstream/tierpin
line numbers are given as `base:NNN` (tierpin file).

| Change | Where | What |
| --- | --- | --- |
| Header note | `manager.py:12-18` | principle 1 amended: STORAGE cascades are gated |
| Config readers | `manager.py:67-116` | env switches, defaults, tracker cap 250 000 |
| Init state | `manager.py:269-300` | `_fs_admit_min_hits`, `_fs_admit_hits` (per-key hit counter), `_fs_admitted_keys`, GDN flag + one-time logs |
| Hit counting | `manager.py:678` (base:451 area, `lookup()` primary-HIT branch) | every primary-tier lookup HIT calls `_record_fs_admit_hit` |
| Threshold crossing | `manager.py:501-560` | on the Nth hit, `_cascade_key_to_storage_tiers` submits a 1-key store job to every STORAGE tier (chunk is resident+ready at that moment; `create_store_job` pins it, base:797) |
| Cascade gate | `manager.py:1010-1022` (base:783-786, `complete_store()`) | per-tier: STORAGE tiers receive only `_filter_fs_admitted(keys)` — keys with counter ≥ N or already admitted |
| GDN stub | `manager.py:597-640` | `_gdn_latest_allows` — documented no-op, see below |

Two hooks that are deliberately NOT touched:

- `_cascade_existing_chunks_to_request_level_tiers` (base:692) — the
  request-level path serves tiers that explicitly asked for the complete KV
  context of a request; filtering there would break their contract. Our fs
  tier is CHUNK_LEVEL, so this path never feeds it.
- `create_store_job` itself (base:797) — kept unfiltered so P2P/external
  store jobs (`serve_external_requests`) behave exactly as upstream.

### What counts as a “hit”

Only a **ready HIT in the primary (RAM) tier observed by `lookup()`** — the
canonical “found in an upper tier” event. Not counted: the `prepare_load()`
that follows the same HIT (same use, not a second one), `touch()` (recency
only, and dead in v0.32.0), secondary-tier hits (a chunk already on disk
needs no disk admission), and HIT_PENDING. The counter is per content key
(`OffloadKey`), survives RAM eviction and `reset_cache()` — it is a fact
about the content, like the fs files themselves.

### Effective semantics on our production fs tier (read this)

The threshold-crossing cascade submits **without** an existence probe,
because the real `FileSystemTierManager.lookup` is asynchronous
(`FsAsyncLookupManager`: a fresh probe returns RETRY until the next step’s
flush) — probing would defer the write to a re-store that may never come for
a chunk that stays hot in RAM. Consequences:

- A **hot socle** reaches the fs tier within minutes (2 reusing requests),
  without waiting for a RAM eviction.
- The cost of no-probe: a key that is already on disk and re-crosses after a
  restart is **rewritten once with identical bytes** (content-addressed
  file, temp + `os.replace`). Bounded at one rewrite per key per process
  lifetime by `_fs_admitted_keys`; measured scale: a restart re-crosses the
  active conversations’ chains once, order of GB against a 47 GB/h write
  stream — and restarts are rare.
- The fs tier’s store path is idempotent, so a duplicate submission is never
  an error.

Residual cost of the filter itself (accepted by design in issue #198): a
chunk that is evicted from RAM *before* ever crossing the threshold is not on
disk, so its next consumer recomputes instead of promoting from fs. This is
the “fewer fs reads” side of the trade — the acceptance gate (disk-served
prompt-token share ≥ 80 % of the 2.2 % baseline) must be checked on real
traffic after deploy; the tuning knob is N (`1` admits on first re-sighting,
`0` is write-through).

## Config switches

| Variable | Default | Meaning |
| --- | --- | --- |
| `MYIA_FS_ADMIT_MIN_HITS` | `2` | cascade to STORAGE tiers only after ≥ N primary-tier lookup hits. **`0` = filter disabled = exact upstream write-through** (rollback). Read once per manager construction — a change needs a container recreate, like `MYIA_FS_TOUCH`. Non-int or negative values fall back to 2 / 0 respectively, with a log line. |
| `MYIA_FS_ADMIT_GDN_LATEST` | `0` (off) | see the stub below. Setting it only logs a warning; it changes no behaviour. |

Bookkeeping bound: counters + admitted set are capped at 250 000 entries
(`_FS_ADMIT_TRACKER_MAX`, ~10⁵ distinct keys/day measured); overflow clears
both, which only biases admission toward *fewer* fs writes until counters
rebuild.

## Rollback

One variable, no image swap:

```bash
MYIA_FS_ADMIT_MIN_HITS=0   # in the profile environment + container recreate
```

With 0, every code path added by this overlay short-circuits (`_record_fs_admit_hit`
returns immediately; the `complete_store` gate is skipped) and behaviour is
byte-identical to the tierpin image, including fs-tier KV events (verified by
running the upstream v0.32.0 fs-tier suite in that mode). Full-image rollback =
repoint the profile at `vllm-openai-v0320-tierpin:v1`.

## Why it is safe

- A refused store means the chunk is simply **not on the fs tier**; any later
  consumer that misses everywhere recomputes the prefix on GPU. The engine’s
  correctness never depends on a cache being populated — this is the same
  contract as a pruned fs file or a failed load under
  `kv_load_failure_policy=recompute`, both of which production already
  exercises. No assertion is added on the read path; no eviction behaviour
  changes; RAM eviction, promotions and backpressure are untouched.
- The filter cannot refuse a *load*: it only sits in front of
  `submit_store` for STORAGE tiers.
- Failure containment: the crossing path uses the very same
  `create_store_job` + `submit_store` as the write-through cascade (same
  ref-counting, same job tracking, same completion path), so a failure modes
  review of one is a review of the other.

## GDN latest-state rule (Phase 1b) — documented stub, NOT implemented

`MYIA_FS_ADMIT_GDN_LATEST` exists, warns, and does nothing. Implementing the
rule at this layer would require two facts the tiering manager does not have
(analysis in the code, `manager.py:597-640`):

1. **Which KV group indices are GDN state groups.** The group index is
   recoverable from the key (`get_offload_group_idx`), but the
   group→attention-type mapping lives in `OffloadingConfig` / the
   scheduler-side connector, never in the tiering manager.
2. **Conversation-chain identity.** Keys are opaque content hashes: two
   snapshots of one conversation cannot be related here, and `ReqContext`
   carries only `req_id` plus per-request token positions
   (`_offload_key_positions`), not which prior request a prompt continues.
   “Snapshot K_new supersedes K_old of the same chain” is simply not
   computable from what this layer sees.

Where the rule belongs: the **scheduler-side offloading connector**, which
owns both facts. Concretely, upstream’s in-flight `TieringAdmissionPolicy`
(RFC vllm-project/vllm#51240; framework PR vllm-project/vllm#60000 — open,
not merged, checked 2026-10-25) is fed `should_admit(keys, tier_idx,
is_promotion)` plus per-request lifecycle hooks (`on_admitted`,
`on_completed`, `reset`) and is constructed where request identity exists; a
`GdnLatestStatePolicy` instantiated with the request’s KV group layout and
chain identity is the natural shape. Phase 2 of issue #198 (GDN checkpoints
at radix branch points, Marconi-style) is the other viable home, since it
gives state snapshots a position-tagged identity in the prefix tree.

Note the frequency filter already removes most of the GDN waste indirectly: a
one-shot end-of-request snapshot (written, conversation moves on) has 0
upper-tier sightings at store time and is never re-stored, so it never
reaches fs. What the stub leaves on the table are snapshots of *active*
conversations that got evicted from RAM and recomputed at least once after
two sightings — the M1/M2 measurements of issue #198 Phase 0 quantify how
much that is before any further work.

## Where upstream #60000 replaces this overlay

PR vllm-project/vllm#60000 (“[KV Offload] Add admission policy framework and
backpressure adapter”, open as of 2026-10-25) moves the existing per-tier
backpressure detectors behind one `TieringAdmissionPolicy` and hooks it in
**`create_store_job`** — `should_admit(keys, tier_idx, is_promotion)` runs
before any primary-tier pinning or job-ID allocation, a rejected call returns
`None` and `complete_store` skips the tier. When it merges (and reaches a
release we run):

1. The `complete_store` gate here (per-key filtering before
   `create_store_job`) is replaced by a `FrequencyAdmissionPolicy` — its
   state is exactly `_fs_admit_hits` / `_fs_admitted_keys`; its
   `should_admit` implements `_filter_fs_admitted`. Note #60000’s hook is
   all-or-nothing per job, so either the policy gets the per-key reduction
   (propose upstream: `should_admit` returning an admitted subset, or the
   manager splitting the batch) or our per-key pre-filter stays at the
   `complete_store` call site, which #60000 does not remove.
2. The threshold-crossing trigger (the `lookup()` hook) has no equivalent in
   #60000 — it needs job submission, so it stays manager-side; propose it
   upstream as a policy hook (`on_upper_tier_hit`) driven from the same
   place.
3. The whole overlay (fsadmit + its tierpin base) is dropped as soon as a
   stock release carries both the tiering-race fix (#60729 case 2 /
   #54914 / our #62615) and the admission framework.

## Tests

`tests/test_fs_admission.py` (same fixtures/style as the tierpin overlay’s
regression tests and upstream `tests/v1/kv_offload/tiering/`): a fake
STORAGE-medium secondary tier + the upstream example (CPU-medium) tier.

- filter off (`0`): first store cascades to every tier — exact write-through;
- filter on: first sighting not admitted (RAM + CPU tier still get
  everything); crossing on the 2nd HIT submits exactly one 1-key fs job;
  re-store after RAM eviction is admitted by the persistent counter; an
  fs→RAM promotion is not a sighting but post-promotion hits re-admit;
  GDN flag = no-op; tracker overflow clears.

`tests/run_tests.sh stock|patched` runs them in a throwaway CPU-only
container of the tierpin base image (driver command in the script header; no
`--gpus`, nothing shared with the running prod container), then, patched
only, the upstream v0.32.0 suites: `test_tiering_offloading.py` at default
env (CPU-medium tiers — unaffected by the filter) and `test_fs_tier.py` +
`cpu/test_manager.py` with `MYIA_FS_ADMIT_MIN_HITS=0` (the rollback mode must
stay byte-identical to upstream, fs events included). It also runs the one
upstream test that legitimately changes under the default filter
(`test_cascade_store_emits_fs_event_through_tiering_manager`) to show it
failing for the intended reason.

Last full run: see the bottom of this file.

## Build (evaluation only)

```bash
docker build -t vllm-openai-v0320-fsadmit:v1 myia_vllm/configs/docker/overlay_v0320_fsadmit
```

## Deployment (requires user GO — not done by this overlay)

Add to `medium-swift15-27b-fstier.yml` `environment:` (or `.env`):

```yaml
- MYIA_FS_ADMIT_MIN_HITS=2       # 0 = rollback to write-through
```

and point the profile at the image (`VLLM_IMAGE_SWIFT15` or the `image:`
line), then recreate the engine. Acceptance per issue #198: fs bytes written
per prompt token ÷ ≥ 5 vs the 1.7 KB baseline; disk-served prompt-token share
≥ 80 % of 2.2 %; no new assertion. Watch `vllm:kv_offload_tiering_write_bytes`
(tier=fs) for the write drop.

---

## Test run record

2026-10-25, throwaway CPU-only containers of the tierpin base image (repo
overlay dir + fetched v0.32.0 suites mounted; no `--gpus`, nothing shared
with the running prod stack, which stayed `Up (healthy)` on
`vllm-openai-v0320-tierpin:v1` throughout):

- **patched** (run_tests.sh patched, overlay manager copied into
  site-packages):
  - admission regression tests, default env (filter ON at 2): **7/7 passed**;
  - upstream v0.32.0 `tiering/test_tiering_offloading.py`, default env:
    **62/62 passed** (CPU-medium tiers are unfiltered — no regression);
  - upstream v0.32.0 `tiering/test_fs_tier.py` with
    `MYIA_FS_ADMIT_MIN_HITS=0`: **44/44 passed** — rollback mode is
    byte-identical to upstream, fs events included;
  - upstream v0.32.0 `cpu/test_manager.py` with the filter off:
    **51/51 passed** (untouched file, control run);
  - `test_cascade_store_emits_fs_event_through_tiering_manager` under the
    default filter: **fails as intended** (fresh keys, 0 hits → no cascade →
    no fs event; that write is exactly what the filter removes).
- **stock** (same tests against the unpatched tierpin base): 0/7 — the
  filter-on tests fail (attributes absent / first store cascaded), which is
  the behaviour delta this overlay carries.
- Image build `vllm-openai-v0320-fsadmit:v1` from this directory: OK, and
  its build-time assert (`_step_hit_keys` AND `_record_fs_admit_hit` present,
  `extra_protected` in `cpu/manager.py`) verified again by importing from the
  built image. The image is NOT referenced by any profile.
