# Peer L1 ownership

The connector's directory contains full-block prefix hashes and physical KV rows.
Placement replicas are hints; the owner revalidates complete keys and its epoch
before granting a read. KV compatibility includes geometry and hash settings.

## Publication

A cached block is initially pending. The worker records completion after all
step writers, including speculative draft writes, and reports consecutive
completed fences. The scheduler publishes only keys covered by that watermark.
Invalidation removes a pending key or retires a published row before reuse.
Only finalized full blocks may be shared.

## Immediate grants and admission

The native owner can grant rows between scheduler steps. A retired pinned free
row is withheld rather than reused. If `W` is the interval's initial reserve and
`h` rows have been withheld, the remaining reserve is `W - h`. Allocating `a`
rows and withholding `d` changes admissible capacity by exactly `a`:

```text
(F - a - d) - (W - h - d) = F - (W - h) - a
```

The new-pin window bounds all rows that could still become withheld before
absorb. Absorbed pins hold pool references. Retiring a pinned in-use row adopts
its pool reference before the local reference can be freed.

At absorb, the window is bounded by `pin_window` and free headroom. It need not
reserve the configured maximum in a busy pool. Owner pins never block eviction
or force the owner to wait for a reader.

## Deferred grants

An oversized request queues keys without pinning rows. The scheduler snapshots
candidate rows and their refcounts, then commits eligible grants under the
native mutex. Entries arriving after the snapshot wait for another absorb.
Free-row touches are limited to half the remaining free headroom after immediate
touches. No allocation may interleave between snapshot and applying these pool
references.

Pins become absorbed before a committed result is visible. A concurrent release
dirties the row for a later absorb, preserving the reference even if release
precedes application of the returned touch. Reset refuses pending entries as
well as active pins. Queue capacity is bounded by total key positions.

An idle owner receives an engine wakeup after enqueue. The callback requests a
step only; all pool operations stay on the scheduler thread. Uncommitted entries
survive until an eligible absorb or reader cancellation. A late reader consumes
a decided result before applying its busy deadline to an undecided result.

## Completion and failures

IPC batches share a copy stream and one completion callback releases all grants
for the step. A possible failed load retains ownership under fail-stop; there is
no timer that can release a possibly active IPC read.

Leased async grants retain their original lease clock. Queueing, setup and the GPU fence
must fit within half that lease. The first possible post starts the transfer
budget, capped by the same bound; subsequent segments do not renew it.

An unsubmitted failure is cancelled and released. A possible post releases only
when drained. Otherwise the worker reports both failed and finished, and the
scheduler takes an extra reference on each target row before recomputation,
abort or normal finish can free the request's references. Quarantine belongs to
the transfer identity, independently of the request, until drain or process exit.
An elapsed lease or failed status alone is not drain proof.

Discovery validates epoch, geometry and hash settings before selecting an owner.
