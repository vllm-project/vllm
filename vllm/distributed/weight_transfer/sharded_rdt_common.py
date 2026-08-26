# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared pieces of the sharded-RDT backend: the op-chain allowlist, buffer
sizing and the minimum Ray version, plus ``RdtRouter`` — consumer-only, since
routing is a consumer decision.

The gather-group partition itself is ``base.layerwise_groups``: it defines what a
group index means for any ``WeightSource``, not just this transport.
"""

from collections.abc import Callable
from typing import Any

import torch

# The op-chain contract, with two enforcers. The consumer's ``FakeRDTTensor``
# intercepts exactly these methods during the bake and records the string name;
# anything else reaches ``__torch_dispatch__`` and raises, so a loader needing
# real data fails at init rather than transferring wrong bytes. The producer
# replays a chain as ``getattr(tensor, op)(...)`` and refuses anything outside
# ``ALLOWED_OPS``, so a spoofed consumer cannot invoke arbitrary methods.
#
# Every entry must be a pure view; ``to`` in particular must never be added, as
# it is the dtype/device escape the bake exists to reject. Both sets derive from
# this one table so they cannot disagree.
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

# Ray Direct Transport landed in halves: the actor opt-in
# ``enable_tensor_transport`` in 2.49, and ``ray.experimental``'s
# ``register_nixl_memory`` / ``set_target_for_ref`` — which this transport calls
# directly — in 2.55. The floor is the tested version rather than the earliest
# one that could import; 2.55 has never been run against.
RDT_MIN_RAY_VERSION = "2.56.0"


def check_ray_rdt_version() -> None:
    """Refuse an installed Ray older than the one this backend is tested on.

    vLLM does not depend on Ray, so there is no pin to carry this. Without the
    check the failure is an opaque option-validation error out of
    ``.options(enable_tensor_transport=True)`` (below 2.49) or an ImportError
    raised deep in the first pull, long after init reported success (below 2.55).

    Raises:
        ValueError: the installed Ray predates ``RDT_MIN_RAY_VERSION``.
    """
    import importlib.metadata

    from packaging import version

    required = version.parse(RDT_MIN_RAY_VERSION)
    current = version.parse(importlib.metadata.version("ray"))
    if current < required:
        raise ValueError(
            f"The 'sharded_rdt' weight transfer backend requires Ray "
            f">= {required}, the version it is tested against; Ray Direct "
            f"Transport (ray.experimental.register_nixl_memory / "
            f"set_target_for_ref) needs at least 2.55. Found {current}. "
            f"Run `pip install -U 'ray>={required}'`."
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


class RdtRouter:
    """Decides which producer serves each weight name, and — once bound — sends.

    Ownership is per NAME. ``owner_sets`` holds the few distinct producer sets
    that occur, and ``name_owner_class[i]`` indexes it for ``names[i]``; a name
    determines both its owner set and its gather group, so neither appears in the
    routing API. Empty tables mean every producer holds everything.

    Any placement is expressible this way: a pipeline stage is a set of names
    sharing one owner set, an expert is a name whose owner set is a single rank,
    and a group produced by two stages is just two classes inside one group.

    Both engines derive the same tables from the same wire data, so they agree on
    who serves what. Disagreement is not a wrong answer but a hang or a loud
    misroute: a pull sent to a producer that never gathered the name trips its
    served-names guard.

    Routing is consumer-only. The trainer publishes what it holds and answers
    whatever arrives; it never asks who serves what.
    """

    def __init__(
        self,
        num_producers: int,
        num_consumers: int,
        owner_sets: list[list[int]] | None = None,
        name_owner_class: list[int] | None = None,
        names: list[str] | None = None,
        group_lens: list[int] | None = None,
        workers_per_replica: int = 0,
    ) -> None:
        """Build the routing tables from the wire data both engines receive.

        The four table arguments travel together and default together: with all
        of them empty the router degrades to "every producer holds everything".

        Args:
            num_producers: Trainer ranks. Owner indices are positions in this
                range, and ``validate`` rejects any that fall outside it.
            num_consumers: Inference workers across the whole fleet. Fixes
                the id space; the block carve uses ``workers_per_replica``
                of it.
            owner_sets: The distinct producer sets that occur, one row per
                owner class; each row is deduplicated and sorted here. Empty
                means a single class owning every producer.
            name_owner_class: Parallel to ``names`` — ``name_owner_class[i]``
                indexes ``owner_sets`` for ``names[i]``. A name with no entry
                falls back to class 0.
            names: Every parameter name, in group-major order: concatenating
                the gather groups reproduces this list exactly. The other two
                tables are keyed by this order.
            group_lens: Length of each gather group, consecutive over
                ``names``, so they sum to ``len(names)``. Fixes name -> group
                and the per-group owner union the free barrier fans out to.
            workers_per_replica: Consumers per inference DEPLOYMENT. The block
                carve spreads this many consumers over an owner set and every
                deployment reuses that carve, so the same worker of each
                deployment resolves one producer (see ``producer_for``). 0
                means one deployment: carve over the whole fleet.

        Raises:
            ValueError: ``workers_per_replica`` does not divide
                ``num_consumers``, so the deployments are not uniform and two
                workers of one deployment would share a block index.
        """
        self.num_producers = max(1, num_producers)
        self.num_consumers = max(1, num_consumers)
        # Width of the block carve: one deployment's worth of consumers, so
        # deployments overlay rather than spread. See ``producer_for``.
        self._block_consumers = min(
            self.num_consumers, max(1, workers_per_replica or self.num_consumers)
        )
        if self.num_consumers % self._block_consumers:
            raise ValueError(
                f"workers_per_replica={workers_per_replica} does not divide "
                f"num_consumers={self.num_consumers}; the inference deployments "
                f"must be uniform for the per-deployment block carve to be "
                f"well defined."
            )
        self._owner_sets = (
            [sorted(set(row)) for row in owner_sets]
            if owner_sets
            else [list(range(self.num_producers))]
        )
        names = list(names or [])
        classes = list(name_owner_class or [])
        self._class_of = {
            n: (classes[i] if i < len(classes) else 0) for i, n in enumerate(names)
        }
        # name -> gather group, and the per-group owner union the free barrier
        # fans out to. Both are pure functions of the tables, so precompute once.
        self._group_of: dict[str, int] = {}
        self._group_owners: list[list[int]] = []
        pos = 0
        for gi, glen in enumerate(group_lens or []):
            union: set[int] = set()
            for n in names[pos : pos + glen]:
                self._group_of[n] = gi
                c = self._class_of.get(n, 0)
                # Bounds-checked: a bad class must surface from validate() as a
                # named ValueError, not as an IndexError from this precompute.
                if 0 <= c < len(self._owner_sets):
                    union.update(self._owner_sets[c])
            pos += glen
            self._group_owners.append(sorted(union))
        # Bound at init on the consumer; the rule methods never touch these.
        self._actors: list[Any] = []
        self._produce_methods: list[Any] = []
        self.consumer_id = 0

    # ---------------- rules (pure; safe to use unbound) ----------------

    def class_of(self, name: str) -> int:
        """The name's owner class — the planner's bucketing key, since all names
        of a chunk must share one producer."""
        return self._class_of.get(name, 0)

    def owners(self, name: str) -> list[int]:
        """Every producer holding ``name``."""
        return list(self._owner_sets[self.class_of(name)])

    def group_owners(self, group_idx: int) -> list[int]:
        """Every producer holding any name of ``group_idx``.

        The one group-keyed rule, because the free barrier is per group: a
        consumer signals ``free_group(gi)`` at each of these and the producer
        counts signals against its live-consumer total. Names route; groups free.
        """
        if not self._group_owners:
            return list(range(self.num_producers))
        return list(self._group_owners[group_idx])

    def producer_for(self, consumer_id: int, name: str) -> int:
        """The single producer ``consumer_id`` pulls ``name`` from.

        Blocks consumers across the name's owner set with the same rule that
        binds producers globally, then rotates by the name's GROUP index so a
        consumer spreads its groups over its block instead of hammering one NIC.
        Rotating per group (not per name) keeps every name of a chunk on one
        producer, which is what lets a chunk be a single pull.

        The carve is over ONE DEPLOYMENT (this consumer's index within its own
        deployment, over ``workers_per_replica`` of them), so the same worker of
        every deployment resolves the same producer for every name. Carving over
        the whole fleet instead spreads the multi-owner names of different
        deployments onto different producers while single-owner names still
        route by ownership, so a producer serves several distinct worker
        indices, each needing its own serve ring, and none of them can share a
        slot. With one deployment the width is the fleet, so this is the plain
        block rule.
        """
        own = self.owners(name)
        if not own:
            raise ValueError(f"{name!r} has no owner")
        block = assign_producer_indices(
            len(own), self._block_consumers, consumer_id % self._block_consumers
        )
        return own[block[self._group_of.get(name, 0) % len(block)]]

    def validate(self) -> None:
        """Check the ownership tables can be served.

        Raises:
            ValueError: an owner set is empty or out of range, or a class index
                does not resolve.
        """
        for c, row in enumerate(self._owner_sets):
            if not row:
                raise ValueError(f"owner set {c} is empty: no producer holds it")
            bad = [p for p in row if not 0 <= p < self.num_producers]
            if bad:
                raise ValueError(f"owner set {c} out of range: {bad}")
        for name, c in self._class_of.items():
            if not 0 <= c < len(self._owner_sets):
                raise ValueError(
                    f"{name!r} has owner class {c}, but only "
                    f"{len(self._owner_sets)} owner set(s) were shipped"
                )

    # ---------------- send surface (needs the binding) ----------------

    def bind(self, actors: list, produce_methods: list, consumer_id: int) -> None:
        """Attach this consumer's producer handles. Rebuilt wholesale per init,
        never appended to: every owner index is a position in these lists, so a
        rejoining engine that re-inits must not shift them."""
        if len(actors) != len(produce_methods):
            raise ValueError("actors and produce_methods must be parallel")
        self._actors = list(actors)
        self._produce_methods = list(produce_methods)
        self.consumer_id = consumer_id

    def _bound(self) -> None:
        if not self._produce_methods:
            raise RuntimeError("RdtRouter has no producers bound; call bind() first.")

    def pull(self, owner: int, keys: list, seq: int) -> Any:
        """Issue one packed pull to ``owner``.

        The consumer id keys the producer's serve ring; without it every worker
        is served out of ring 0 and concurrent pulls overwrite each other's blob.
        ``seq`` is this call's index in the stream to ``owner`` (see ``_Chunk``):
        it selects the serve slot in ISSUE order, so a slot is never repacked
        while the read of its previous contents is still in flight.
        """
        self._bound()
        return self._produce_methods[owner].remote(
            keys, consumer_id=self.consumer_id, seq=seq
        )

    def free_group(self, group_idx: int) -> list:
        """Signal the group done at every producer holding any of its names."""
        self._bound()
        return [
            self._actors[p].free_group.remote(group_idx)
            for p in self.group_owners(group_idx)
        ]

    def reserve_serve_buffers(
        self, bytes_by_producer: list[int], plan_digests: list[str] | None = None
    ) -> list:
        """Ask each producer to pre-register a serve ring sized to the most this
        consumer will pull from it.

        ``plan_digests[p]`` describes the chunks this consumer pulls from
        producer ``p``, in pull order. A producer that shares one serve ring
        across the consumers of several deployments compares it across them, so
        a fleet whose deployments are not identical fails at init instead of
        stalling mid-sync. A producer that shares nothing ignores it.
        """
        self._bound()
        digests: list = (
            list(plan_digests) if plan_digests else [None] * len(bytes_by_producer)
        )
        return [
            self._actors[p].reserve_serve_buffer.remote(
                self.consumer_id, nb, digests[p]
            )
            for p, nb in enumerate(bytes_by_producer)
            if nb > 0
        ]


def buffer_alloc_bytes(nbytes: int, presize: int = 0) -> int:
    """Size a NIXL buffer / ring slot for ``nbytes``: the max of the request, an
    optional ``presize`` floor, and a coarse 256MB round-up, so the buffer is
    allocated ONCE and never regrows. Regrowth is a correctness hazard, not just a
    perf one -- see ``buffer_presize_gb``. Shared by the consumer's receive buffers
    and the producer's serve rings."""
    return max(nbytes, presize, -(-nbytes // (256 << 20)) * (256 << 20))
