# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Helpers shared by the sharded-RDT consumer (worker) and producer (trainer)
engines: M:N routing, arena sizing, and the op-chain allowlist. Both sides derive
them from here so the two cannot drift.

The gather-group partition itself is ``base.layerwise_groups``: it defines what a
group index means for any ``WeightSource``, not just this transport.
"""

from collections.abc import Callable

import torch

# The op-chain contract, with two enforcers. The consumer's ``LazyRDTTensor``
# intercepts exactly these methods during the bake and records the string name;
# anything else reaches ``__torch_dispatch__`` and raises, so a loader needing
# real data fails at init rather than transferring wrong bytes. The producer
# replays a chain as ``getattr(tensor, op)(...)`` and refuses anything outside
# ``ALLOWED_OPS``, so a spoofed consumer cannot invoke arbitrary methods.
#
# Every entry must be a pure view. Both sets derive from this one table: written
# out separately they drifted, leaving ``t`` consumer-emittable but
# producer-rejected, and ``to`` allowed but unreachable -- ``to`` being exactly
# the dtype/device escape the bake exists to reject.
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
    # Multi-return like chunk: one child per output, trailing __getitem__(i).
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

    Both engines build this from the same wire-carried data, so they always agree
    on who serves what. Disagreement is not a wrong answer but a hang or a loud
    misroute: a pull sent to a producer that never gathered the name trips its
    served-names guard.

    ``group_owners[g]`` lists the producer ranks that publish group ``g``;
    ``None`` means every producer owns every group (gather-to-all).

    ``producer_ep_ranks[r]`` is trainer rank r's expert-parallel coordinate. A
    pull for a name stamped ``k >= 0`` must go to a group owner whose coordinate
    is ``k``; ``-1`` matches every owner, and ``None`` means no expert sharding.
    The matching name stamps ride the worker init info, not this router.

    Exactly one producer serves each unit: splitting a pull only multiplies
    produce calls, since the consumer's own NIC bounds it either way.

    Freeing does not route through here. Each consumer signals ``free_group(g)``
    to every owner, counted against the live total handed to ``begin_sync``.
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
        """Producer ranks publishing ``group_idx`` and, for ``ep_rank >= 0``,
        holding that coordinate. The intersection is valid because Megatron
        expert-parallel groups never span pipeline stages, so a group's owner set
        always contains every coordinate."""
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
        """The single producer ``consumer_id`` pulls (``group_idx``, ``ep_rank``)
        from. Blocks consumers across the unit's owner set with the same rule that
        binds producers globally, then rotates by group index, so every producer
        NIC carries traffic."""
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
                coordinate list does not cover every producer. Empty
                (group, ep_rank) pull units are caught at plan build instead,
                since only the consumer's baked copies know which units exist.
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
    """Size a NIXL arena / ring slot for ``nbytes``: the max of the request, an
    optional ``presize`` floor, and a coarse 256MB round-up, so the buffer is
    allocated ONCE and never regrows. Regrowth is a correctness hazard, not just a
    perf one -- see ``arena_presize_gb``. Shared by the consumer's receive arenas
    and the producer's serve rings."""
    return max(nbytes, presize, -(-nbytes // (256 << 20)) * (256 << 20))
