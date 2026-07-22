# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Helpers shared by the sharded-RDT consumer (worker) and producer (trainer)
engines.

Both sides must agree on the M:N producer/consumer binding, the arena byte
sizing, the greedy byte-balanced split, the gather-group partition, and the
op-chain allowlist. Keeping them here — imported by both
``sharded_rdt_engine`` (consumer) and ``sharded_rdt_trainer`` (producer) —
makes that agreement a single source of truth rather than two copies that can
silently drift.
"""

# Op chains the consumer's baked plan may request the producer to replay on a
# cached tensor. The producer refuses any op outside this set so a misbehaving
# or spoofed consumer cannot invoke arbitrary methods on trainer tensors.
ALLOWED_OPS = frozenset(
    (
        "narrow",
        "view",
        "reshape",
        "transpose",
        "permute",
        "contiguous",
        "squeeze",
        "unsqueeze",
        "__getitem__",
        "to",
        "chunk",
        "split",
        "select",
        "flatten",
        "unbind",
    )
)


def assign_producer_indices(
    num_producers: int, num_consumers: int, consumer_idx: int
) -> list[int]:
    """Producers (global indices) that consumer ``consumer_idx`` binds."""
    p = max(1, num_producers)
    c = max(1, num_consumers)
    if p >= c:
        return list(range(consumer_idx * p // c, (consumer_idx + 1) * p // c))
    return [consumer_idx * p // c]


def count_consumers(num_producers: int, num_consumers: int, producer_idx: int) -> int:
    """Number of consumers that bind producer ``producer_idx`` (its free target)."""
    p = max(1, num_producers)
    c = max(1, num_consumers)
    if p >= c:
        return 1
    return sum(1 for ci in range(c) if ci * p // c == producer_idx)


def arena_alloc_bytes(nbytes: int, presize: int = 0) -> int:
    """Size a NIXL arena / ring slot for ``nbytes``, rounded up so the buffer is
    allocated ONCE and never regrows: the max of the request, an optional
    ``presize`` floor, and a coarse 256MB round-up. Sizing once matters beyond
    perf — Ray's NIXL desc cache is keyed by ``data_ptr`` and its entries outlive
    their tensors, so repeated small regrowths can false-hit a recycled pointer
    and skip registering the new extent (see ``arena_presize_gb``). Shared by both
    sides (consumer receive arenas + producer serve rings)."""
    return max(nbytes, presize, -(-nbytes // (256 << 20)) * (256 << 20))


def greedy_run_starts(weights: list[int], n: int) -> list[int]:
    """Greedy contiguous byte-balanced partition of ``weights`` into at most
    ``n`` runs; returns the START index of each run (the first is always 0).
    Walks left to right, accumulating into the current run and cutting before an
    item that would push the run past the ``ceil(total/n)`` target — never
    emitting more than ``n`` runs. An item heavier than the target simply makes
    its run oversized (accepted). Shared by the gather-group -> chunk split
    (``_chunk_group_scatters``) and the M:N per-pull producer split
    (``_split_chunk_pull``); both are the same greedy cut over different weights."""
    total = sum(weights)
    target = -(-total // max(1, n))  # ceil
    starts = [0]
    cur = 0
    for i, w in enumerate(weights):
        if i > 0 and cur + w > target and len(starts) < n:
            starts.append(i)
            cur = 0
        cur += w
    return starts


def layerwise_groups(names: list[str]) -> list[list[str]]:
    """Partition flat parameter names into pre / per-decoder-layer / post gather
    groups (keys on ``model.layers.<N>.``). The group is the unit of gathering,
    freeing, AND the packed pull's chunk budget — without it a whole model
    becomes one chunk and the receive/serve arenas balloon to the full
    per-worker share."""
    pre: list[str] = []
    layers: dict[int, list[str]] = {}
    post: list[str] = []
    seen = False
    for n in names:
        if n.startswith("model.layers."):
            seen = True
            idx = int(n[len("model.layers.") :].split(".", 1)[0])
            layers.setdefault(idx, []).append(n)
        elif not seen:
            pre.append(n)
        else:
            post.append(n)
    groups: list[list[str]] = []
    if pre:
        groups.append(pre)
    for i in sorted(layers):
        groups.append(layers[i])
    if post:
        groups.append(post)
    return groups
