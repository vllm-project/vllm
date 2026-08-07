# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Helpers shared by the sharded-RDT consumer (worker) and producer (trainer)
engines.

Both sides must agree on the M:N producer/consumer binding and per-group
routing (``RdtRouter``), the arena byte sizing, the greedy byte-balanced
split, and the op-chain allowlist. Keeping them here — imported by both
``sharded_rdt_engine`` (consumer) and ``sharded_rdt_trainer`` (producer) —
makes that agreement a single source of truth rather than two copies that can
silently drift.

The gather-group partition itself lives on ``base.layerwise_groups``: it
defines what a group index means for ``WeightSource``, not just for this
transport.
"""

from collections.abc import Callable, Collection

import torch

# The op-chain contract, in one place because it has two enforcers.
#
# CONSUMER: ``LazyRDTTensor`` intercepts exactly these ``torch.Tensor`` methods
# during the bake and records the string name in the op chain. Anything else
# reaches ``__torch_dispatch__`` and raises, so a loader that needs real data
# (arithmetic, ``.to``, ``.float``, ``.item``, ``.data``, bool-mask indexing)
# fails loudly at init rather than silently transferring the wrong bytes.
#
# PRODUCER: replaying a chain is ``getattr(tensor, op)(*args, **kwargs)`` on a
# live trainer tensor, so it refuses any op outside ``ALLOWED_OPS`` -- a
# misbehaving or spoofed consumer must not be able to invoke arbitrary methods.
#
# Every entry must be a pure view / shape-only / byte-bounding operation. The two
# sets are DERIVED from this one table: when they were written out separately
# they drifted, leaving ``t`` consumer-emittable but producer-rejected (a loader
# calling ``.t()`` baked at init and then failed at first pull) while
# ``to``/``split``/``select`` were producer-allowed but unreachable -- and ``to``
# is exactly the dtype/device escape the bake exists to reject.
SUPPORTED_OPS: dict[Callable, str] = {
    torch.Tensor.narrow: "narrow",
    torch.Tensor.view: "view",
    torch.Tensor.reshape: "reshape",
    torch.Tensor.__getitem__: "__getitem__",
    torch.Tensor.unsqueeze: "unsqueeze",
    torch.Tensor.squeeze: "squeeze",
    torch.Tensor.transpose: "transpose",
    torch.Tensor.t: "t",
    torch.Tensor.permute: "permute",
    torch.Tensor.flatten: "flatten",
    torch.Tensor.contiguous: "contiguous",
    torch.Tensor.chunk: "chunk",
    # Multi-return, handled like chunk: the consumer emits one child per output
    # with a trailing __getitem__(i); the producer replays unbind()[i].
    torch.Tensor.unbind: "unbind",
}

ALLOWED_OPS = frozenset(SUPPORTED_OPS.values())


def assign_producer_indices(
    num_producers: int, num_consumers: int, consumer_idx: int
) -> list[int]:
    """Producers (global indices) that consumer ``consumer_idx`` binds."""
    p = max(1, num_producers)
    c = max(1, num_consumers)
    if p >= c:
        return list(range(consumer_idx * p // c, (consumer_idx + 1) * p // c))
    return [consumer_idx * p // c]


class RdtRouter:
    """Decides which producer serves (and is freed for) each gather group.

    Both engines build this from the same wire-carried data — ``num_producers``,
    ``num_consumers`` and ``group_owners`` — so producer and consumer always
    agree on who serves what. Disagreement is not a wrong answer but a hang:
    a producer waits forever for a name it never gathered, and a group nobody
    frees stalls ``end_sync``.

    ``group_owners[g]`` lists the producer ranks that gather and publish group
    ``g``; ``None`` means every producer owns every group (gather-to-all, the
    layout used when the trainer has no pipeline-parallel split). Each group is
    served by exactly ONE producer per consumer: splitting one pull across
    producers only multiplies produce calls, since the consumer's own NIC bounds
    the pull either way.
    """

    def __init__(
        self,
        num_producers: int,
        num_consumers: int,
        group_owners: list[list[int]] | None = None,
        num_groups: int = 0,
    ) -> None:
        self.num_producers = max(1, num_producers)
        self.num_consumers = max(1, num_consumers)
        self._owners = (
            [sorted(set(owners)) for owners in group_owners] if group_owners else None
        )
        self.num_groups = len(self._owners) if self._owners else max(0, num_groups)

    def owners(self, group_idx: int) -> list[int]:
        """Producer ranks that gather and publish ``group_idx``."""
        if self._owners is None:
            return list(range(self.num_producers))
        return self._owners[group_idx]

    def producer_for(self, consumer_id: int, group_idx: int) -> int:
        """The single producer ``consumer_id`` pulls ``group_idx`` from.

        Blocks consumers across the group's owner set with the same rule that
        binds producers to consumers globally, then rotates by group index. In
        the gather-to-all layout this reproduces the historical contiguous
        binding (consumer c and 16 producers -> {2c, 2c+1}, alternating by
        group), so every producer NIC still carries traffic. Routing all groups
        to one owner per consumer instead would leave the surplus owners
        publishing groups that nobody frees.
        """
        own = self.owners(group_idx)
        if not own:
            raise ValueError(f"group {group_idx} has no owner")
        block = assign_producer_indices(len(own), self.num_consumers, consumer_id)
        return own[block[group_idx % len(block)]]

    def bound_producers(self, consumer_id: int) -> list[int]:
        """Producers ``consumer_id`` pulls from, over all groups."""
        if self._owners is None and not self.num_groups:
            return assign_producer_indices(
                self.num_producers, self.num_consumers, consumer_id
            )
        return sorted(
            {self.producer_for(consumer_id, g) for g in range(self.num_groups)}
        )

    def free_target(
        self,
        producer_rank: int,
        group_idx: int,
        live_consumer_ids: Collection[int] | None = None,
    ) -> int:
        """How many consumers pull ``group_idx`` from ``producer_rank``.

        Zero is normal: an owner with more peers than consumers still has to run
        the group's collective, but must not publish it — nothing would free it.

        ``live_consumer_ids`` restricts the scan to the consumers still alive for
        THIS sync; ``None`` means the whole provisioned set (the behaviour of
        every deployment that does not tolerate consumer death). This is the
        entire producer-side mechanism for syncing to a fleet that has lost a
        consumer: ``producer_for`` is pure in the consumer id, so removing
        consumers cannot change any surviving consumer's binding — it can only
        lower some free targets, and a target that falls to zero turns that group
        into gather-and-drop, which the publish loop already handles. Liveness is
        a FILTER over the provisioned geometry, never a re-derivation of it:
        rebuilding the router from a shrunken consumer count would silently
        re-map every survivor.
        """
        live = None if live_consumer_ids is None else set(live_consumer_ids)
        return sum(
            1
            for c in range(self.num_consumers)
            if (live is None or c in live)
            and self.producer_for(c, group_idx) == producer_rank
        )

    def owned_groups(self, producer_rank: int) -> list[int]:
        """Groups ``producer_rank`` gathers and publishes."""
        return [g for g in range(self.num_groups) if producer_rank in self.owners(g)]

    def validate(self) -> None:
        """Check the ownership table can be served and fully freed.

        Raises:
            ValueError: a group has no owner, an owner is out of range, or the
                per-group free targets do not sum to the consumer count (which
                would leave a published group unfreed or double-freed).
        """
        if self._owners is not None and self.num_groups != len(self._owners):
            raise ValueError(
                f"group count {self.num_groups} != ownership rows {len(self._owners)}"
            )
        for g in range(self.num_groups):
            own = self.owners(g)
            if not own:
                raise ValueError(f"group {g} has no owner")
            bad = [p for p in own if not 0 <= p < self.num_producers]
            if bad:
                raise ValueError(f"group {g} owners out of range: {bad}")
            total = sum(self.free_target(p, g) for p in own)
            if total != self.num_consumers:
                raise ValueError(
                    f"group {g} free targets sum to {total}, "
                    f"expected {self.num_consumers}"
                )


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
    its run oversized (accepted). Used by the gather-group -> chunk split
    (``_chunk_group_scatters``)."""
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
