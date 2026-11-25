# Decode owner invariants

The decode role has no request read path. It retains the regular owner,
publication watermark, leases and replica resolver. Membership marks an
instance as `full` or `decode`; decode resolvers accept only `full` members.
All rank-local members must agree on role.

## Free-row ordering

This pressure gate controls eviction preference. The separate admission gate
in [shared admission budget](#shared-admission-budget) controls new read grants.

`BlockPool.free_blocks` invokes `eviction_order_policy.classify` once for the
owning pool's newly unreferenced cached rows. The optional policy also receives
`on_reuse` when `touch` takes a free row, independently of the lease guard.
These eviction-order hints cannot affect correctness. A release-time pressure
gate controls redundant-row placement. Each `finish_step` inspects at most
K leading free-queue rows through the public iterator, where K is
the fixed scan bound of 256 rows. Finding K rows without hashes closes the
gate: classification returns all false without querying placement hints or
recording the release, so cached rows keep their normal LRU position. A fresh
policy starts closed and cannot reorder before its first maintenance step.

Finding fewer than K leading empty rows opens the gate. Redundant rows released
from then on prepend to the standard queue front; other cached rows are appended.
Newer release batches precede older ones, preserving order within each batch.
Rows released while the gate was closed retain their LRU positions even after
pressure begins. Explicit `evict_first` bypasses classification and retirement.
Full-role schedulers leave the policy unset, costing one `None` check in
`free_blocks` and in the zero-reference branch of `touch`.

Eviction order is a best-effort preference. While K leading empty rows keep the
gate closed, cached releases cannot jump ahead of them; a decode pool with spare
initial capacity therefore retains its cached rows. With the gate open, a cached
row may precede up to K-1 sampled leading empty rows. A single allocation crossing
from empty rows into cached rows evicts its cached remainder in plain LRU order.
Hints can lag: evicting the last GPU copy costs a later miss, not wrong data.

An open gate is rechecked at every release, so unhashed releases or returned
withheld rows within a step close it before a redundant row can precede K or
more leading empty rows. Empty rows outside the leading prefix are not covered
by the K-1 bound. The queue has no decode-specific boundary or generation.
Invalidating a free row clears its hash without moving it unless a lease requires
withholding it. Successful reset preserves queue order and closes the gate;
the next maintenance step recomputes it from the empty rows.

The decode policy owns free-period records and an ordered candidate ring
containing only rows classified redundant. Each open-gate release creates a record
and prepends its candidate batch. `on_reuse` invalidates the record when
`touch` takes a free row; publication and retirement invalidate it through the
existing guard notifications. A later release is classified afresh, while
`unpin_blocks` returns the touched row to the tail without making it a candidate.
Successful reset clears records through the guard's reset scope; refused reset
preserves them.

Each maintenance pass queries at most 256 candidates. It checks
record identity, queue membership, key and zero refcount before moving a withdrawn
row to the tail with O(1) `remove` and `append`. Unchanged candidates rotate to
the end of the ring; withdrawn rows leave it permanently for that free period.
No-copy rows stay in place. The gate probe inspects at most K queue nodes per
step and per open-gate release, in addition to this bounded candidate withdrawal.
Pools without the policy keep their existing behavior.

The replica's aggregate query uses try-locks for the snapshot and shards. Lock
contention returns no result; the whole batch is placed conservatively. It does
not truncate a release batch or defer part of it into a newer batch. Retirement
keeps the candidate ring unchanged on contention or a missing member snapshot.

## Shared admission budget

Let `P` be the number of unique budgeted rows and `R` the remaining admission
reserve. Native owner state maintains `P + R <= protected_cap_blocks` under its
mutex. A row is charged from its first grant until it has neither pins nor an
absorbed/withheld pool reference. Locally used rows are conservatively charged.
Aliases and overlapping grants do not charge the same row twice.

Immediate grants check the window, reference cap, gate and shared budget before
adding pins. Deferred decisions check the same budget at absorb, alongside the
existing limit on free-row touches. Absorb resets the old rate window atomically
and computes a new window no greater than half the remaining protected budget
or the usual free-headroom limit. This leaves room for the fast path while
conservatively charging both pending pins and reserve.

Rows returned as release actions remain charged until `complete_absorb`, after
the scheduler applies touch/free/return actions. A concurrent re-grant keeps the
existing charge. Window recomputation uses the latest budget and never revokes
existing protection. No allocation interleaves with applying an absorb.
Withheld rows remain covered by reserve and are not reused.

Gates read the pool's already-reserved free count and live request statuses.
They affect only new grants. Finishing a request requests one coalesced
maintenance step so an idle engine can refresh its gate after physical free.
Pool operations stay on the scheduler thread; the decode request path performs
no ACQUIRE/CANCEL RPC.

Subscription frames advertise remaining protected capacity (zero when gated).
Readers recompute coordinator-aligned plans within that bound. The budget check
also queues deferred acquires that exceed the fast-path reserve budget. A closed
gate has a separate reply; it does not cause reader busy waiting. Remote-KV,
grammar and streaming waiters do not count as decode compute pressure.

## PD replay storage

MoRIIO's explicit `pd_replay_retention` option enables request-scoped absolute
block ranges. A range is the minimal contiguous suffix covering the normal PD
window and each manager's required tails at `replay_boundaries`. Dense segment
tails are disabled for this calculation. Storage, attention reach and prefix-hit
requirements remain separate.

The producer holds storage through chunked prefill and completion-ACK delayed
free. It exports only computed prompt blocks and validates each range has no
nulls. The consumer validates geometry/hash/speculation settings, allocates the
same range, and pairs by absolute position, excluding fully local blocks and
whole trailing recompute/speculative blocks. Copying a partial valid block
retains the existing block-granular transfer semantics.

The consumer records allocation generations against the actual local scheduler
output object. Core reconciles load errors before lifting the extra constraint;
free/pop discard old generations. Worker metadata and the default PD wire
format are unchanged when disabled. Confirmed failures only lift the extra
constraint: core retains ownership of error recovery and physical references.
Existing synchronous HMA READ failure recovery remains fail-stop.

Capacity planning includes `max(1, H) + A - 1` tokens plus normal re-prefill
retention and block rounding. Bounded-replay groups need only the normal READ's
recomputed-last-token margin. This storage-only parameter is excluded from the
KV compatibility representation and never expands attention or hit masks.
