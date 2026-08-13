# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Helpers shared by the sharded-RDT consumer (worker) and producer (trainer)
engines.

Both sides must agree on the M:N producer/consumer binding and per-group
routing (``RdtRouter``), the arena byte sizing, and the op-chain allowlist.
Keeping them here — imported by both
``sharded_rdt_engine`` (consumer) and ``sharded_rdt_trainer`` (producer) —
makes that agreement a single source of truth rather than two copies that can
silently drift.

The gather-group partition itself lives on ``base.layerwise_groups``: it
defines what a group index means for ``WeightSource``, not just for this
transport.
"""

from collections.abc import Callable

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
    """Decides which producer serves each (gather group, ep_rank) pull unit.

    Both engines build this from the same wire-carried data — ``num_producers``,
    ``num_consumers``, ``group_owners`` and ``producer_ep_ranks`` — so producer
    and consumer always agree on who serves what. Disagreement is not a wrong
    answer but a hang or a loud misroute: a pull routed to a producer that never
    gathered the name trips its served-names guard.

    ``group_owners[g]`` lists the producer ranks that gather and publish group
    ``g``; ``None`` means every producer owns every group (gather-to-all, the
    layout used when the trainer has no pipeline-parallel split).

    ``producer_ep_ranks[r]`` is trainer rank r's expert-parallel coordinate —
    the second of the two stamp lists that must match. The first,
    ``name_ep_rank``, stamps each weight name with the coordinate holding it and
    rides the worker init info, not this router. A pull for a name stamped
    ``k >= 0`` must go to a group owner whose coordinate is ``k``; ``-1`` names
    match every group owner. ``None`` means no expert sharding (every stamp -1),
    which keeps the historical routing exactly.

    Each pull unit is served by exactly ONE producer per consumer: splitting one
    pull across producers only multiplies produce calls, since the consumer's
    own NIC bounds the pull either way.

    Freeing does NOT route through this class. Each consumer signals
    ``free_group(g)`` to EVERY owner of ``g`` (``owners(g)``), and each producer
    counts signals against the live consumer total handed to ``begin_sync`` —
    a per-group barrier, not routed ref-counting.
    """

    def __init__(
        self,
        num_producers: int,
        num_consumers: int,
        group_owners: list[list[int]] | None = None,
        num_groups: int = 0,
        producer_ep_ranks: list[int] | None = None,
    ) -> None:
        self.num_producers = max(1, num_producers)
        self.num_consumers = max(1, num_consumers)
        self._owners = (
            [sorted(set(owners)) for owners in group_owners] if group_owners else None
        )
        self.num_groups = len(self._owners) if self._owners else max(0, num_groups)
        self._ep_ranks = list(producer_ep_ranks) if producer_ep_ranks else None

    def owners(self, group_idx: int, ep_rank: int = -1) -> list[int]:
        """Producer ranks that publish ``group_idx`` — and, for ``ep_rank >= 0``,
        also hold that expert coordinate. The intersection is valid because
        Megatron expert-parallel groups never span pipeline stages: a group's
        owner set always contains every coordinate."""
        base = (
            list(range(self.num_producers))
            if self._owners is None
            else list(self._owners[group_idx])
        )
        if ep_rank < 0:
            return base
        if self._ep_ranks is None:
            raise ValueError(
                f"pull unit (group {group_idx}, ep_rank {ep_rank}) requested but no "
                "producer_ep_ranks were declared; the name stamps and the producer "
                "stamps must ship together"
            )
        return [r for r in base if self._ep_ranks[r] == ep_rank]

    def producer_for(self, consumer_id: int, group_idx: int, ep_rank: int = -1) -> int:
        """The single producer ``consumer_id`` pulls (``group_idx``,
        ``ep_rank``) from.

        Blocks consumers across the unit's owner set with the same rule that
        binds producers to consumers globally, then rotates by group index. In
        the gather-to-all layout this reproduces the historical contiguous
        binding (consumer c and 16 producers -> {2c, 2c+1}, alternating by
        group), so every producer NIC still carries traffic.
        """
        own = self.owners(group_idx, ep_rank)
        if not own:
            raise ValueError(
                f"pull unit (group {group_idx}, ep_rank {ep_rank}) has no owner"
            )
        block = assign_producer_indices(len(own), self.num_consumers, consumer_id)
        return own[block[group_idx % len(block)]]

    def owned_groups(self, producer_rank: int) -> list[int]:
        """Groups ``producer_rank`` gathers and publishes."""
        return [g for g in range(self.num_groups) if producer_rank in self.owners(g)]

    def validate(self) -> None:
        """Check the ownership tables can be served.

        Raises:
            ValueError: a group has no owner, an owner is out of range, or the
                producer coordinate list does not cover every producer. Empty
                (group, ep_rank) pull units are caught at plan build instead —
                ``producer_for`` raises — because only the consumer's baked
                copies know which units exist.
        """
        if self._owners is not None and self.num_groups != len(self._owners):
            raise ValueError(
                f"group count {self.num_groups} != ownership rows {len(self._owners)}"
            )
        if self._ep_ranks is not None and len(self._ep_ranks) != self.num_producers:
            raise ValueError(
                f"producer_ep_ranks has {len(self._ep_ranks)} entries for "
                f"{self.num_producers} producers"
            )
        for g in range(self.num_groups):
            own = self.owners(g)
            if not own:
                raise ValueError(f"group {g} has no owner")
            bad = [p for p in own if not 0 <= p < self.num_producers]
            if bad:
                raise ValueError(f"group {g} owners out of range: {bad}")


def arena_alloc_bytes(nbytes: int, presize: int = 0) -> int:
    """Size a NIXL arena / ring slot for ``nbytes``, rounded up so the buffer is
    allocated ONCE and never regrows: the max of the request, an optional
    ``presize`` floor, and a coarse 256MB round-up. Sizing once matters beyond
    perf — Ray's NIXL desc cache is keyed by ``data_ptr`` and its entries outlive
    their tensors, so repeated small regrowths can false-hit a recycled pointer
    and skip registering the new extent (see ``arena_presize_gb``). Shared by both
    sides (consumer receive arenas + producer serve rings)."""
    return max(nbytes, presize, -(-nbytes // (256 << 20)) * (256 << 20))
