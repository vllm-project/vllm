# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Sharded RDT: the wire protocol, and what both sides compute from it.

Each inference worker (consumer) pulls from the trainer ranks (producers) the
boxes of each checkpoint tensor that its load plan names, over NIXL through
Ray Direct Transport (RDT). Nothing is pushed.

Init (once per trainer run, or when a consumer rejoins):
    The trainer's sender forwards `ShardedRDTWeightTransferInitInfo`: the
    producer actor names, the metadata (name, dtype, shape) in gather order,
    the owner table (`owner_sets`, `name_owner_class`), `buffer_bytes` and
    `num_buffers`. No model state is touched.

Gather units (every rank, every consumer, the same):
    `gather_units` cuts the metadata, in order, into consecutive runs of at
    most `buffer_bytes` full-tensor bytes (a larger tensor is a unit of its
    own). Every trainer rank gathers its owned names unit by unit, in that
    order, running at most `gather_lookahead` units ahead of the units its
    consumers have finished with. A rank owning no name of a unit skips it.
    Units are global, so ranks that gather collectively (FSDP, EP) reach the
    same collectives in the same order.

Plan (a consumer's first `start_weight_update`):
    The consumer runs `plan_loads` and routes each name it needs to one owner
    (`RdtRouter.producer_for`). Per producer and unit, it orders its pieces
    by destination address (so slices of one fused tensor, such as per-expert
    weights, are adjacent and land with one copy) and cuts them into buckets
    of at most `buffer_bytes`. A piece is one box of one name: copied into
    its `get_dest` tensors, or loaded in full through `load_weights` (a
    fallback). A piece over `buffer_bytes` is cut along dim 0 into segments
    of whole rows, each an ordinary piece; a fallback's segments are joined
    into the whole tensor before it is loaded. The consumer sends each producer, once,
    `register_consumer(consumer_id, buckets)`: per bucket, in pull order, a
    list of `(name, box)`.

Sync (`update_weights`, one per sync):
    The consumer pulls its buckets with `num_buffers` in flight, in unit
    order within each `RdtRouter.components` coupling component, interleaving
    components so independent pipeline stages pull in parallel.
    A pull is `bucket(consumer_id, seq, sync_idx)`: only the index travels.
    The producer waits until the bucket's unit is gathered in sync
    `sync_idx`, then packs its boxes, with one fused copy, into one byte
    tensor (pieces at `flat_layout`) carved from an arena it registered with
    NIXL once, so Ray only describes it. The consumer
    receives the bucket into a slot of its own receive arena, registered
    with NIXL once per plan (one slot of the largest bucket per pull in
    flight), issues its next pulls, then copies every piece into its
    destinations with one fused copy and loads the fallbacks. Ray holds a
    target only weakly and silently receives into a buffer of its own when
    the target is gone, so the consumer keeps each slot view alive until its
    pull is applied and fails the sync if a pull did not land in its slot.

Release:
    A consumer drops each ref once its bucket is applied, so Ray frees the
    object and drops its last reference to the producer's arena block, which
    a finalizer turns into the block's free. The producer releases a gathered
    unit once every live consumer has registered and every bucket reading
    from the unit has been freed; that is the trainer's credit to gather the
    next unit.

Arena: each producer registers an arena of `arena_bytes(...)` at startup and
admits a bucket only while a block for it can be carved from the arena next
to the served, unfreed ones, so a full arena slows pulls down instead of
failing them.
"""

import itertools
import math
from collections.abc import Sequence
from typing import Any

import torch

# `register_nixl_memory`, `set_target_for_ref` and concurrent `ray.get` on
# lists, all in Ray 2.56.
RDT_MIN_RAY_VERSION = "2.56.0"
DEFAULT_BUFFER_BYTES = 1 << 30
DEFAULT_NUM_BUFFERS = 2

Box = tuple[tuple[int, int], ...]
# One bucket on the wire: (name, box as [[start, stop], ...]) per piece.
WireBucket = list[tuple[str, list[list[int]]]]

_PIECE_ALIGNMENT = 16  # every piece of a flat bucket starts on 16 B


def validate_int(name: str, value: object, *, minimum: int) -> None:
    """Reject a non-int (bools included) or an int below `minimum`."""
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name}={value!r}: must be an integer >= {minimum}")


def validate_sharded_rdt_settings(
    *,
    buffer_bytes: object,
    num_buffers: object,
    num_consumers: object,
    allow_zero_consumers: bool,
) -> None:
    """Validate the settings shared by trainer and worker init."""
    validate_int("buffer_bytes", buffer_bytes, minimum=1)
    validate_int("num_buffers", num_buffers, minimum=1)
    validate_int(
        "num_consumers", num_consumers, minimum=0 if allow_zero_consumers else 1
    )


def check_ray_rdt_version() -> None:
    """Refuse an installed Ray older than `RDT_MIN_RAY_VERSION`.

    vLLM does not depend on Ray, so no pin carries this; without the check an
    old Ray fails deep in the first pull.

    Raises:
        ValueError: the installed Ray is too old.

    """
    import importlib.metadata

    from packaging import version

    current = version.parse(importlib.metadata.version("ray"))
    if current < version.parse(RDT_MIN_RAY_VERSION):
        raise ValueError(
            f"The 'sharded_rdt' weight transfer backend requires Ray >= "
            f"{RDT_MIN_RAY_VERSION}; found {current}. "
            f"Run `pip install -U 'ray>={RDT_MIN_RAY_VERSION}'`."
        )


def _ray_experimental() -> Any:
    from vllm.distributed.nixl_utils import alias_nixl_for_ray

    alias_nixl_for_ray()
    import ray.experimental

    return ray.experimental


def register_nixl_memory(tensor: torch.Tensor) -> None:
    """Register `tensor`'s whole storage with NIXL (reference counted)."""
    _ray_experimental().register_nixl_memory(tensor)


def deregister_nixl_memory(tensor: torch.Tensor) -> None:
    """Drop one registration `register_nixl_memory` took on `tensor`'s storage."""
    _ray_experimental().deregister_nixl_memory(tensor)


def box_shape(box: Box) -> tuple[int, ...]:
    return tuple(stop - start for start, stop in box)


def box_slices(box: Box) -> tuple[slice, ...]:
    return tuple(slice(start, stop) for start, stop in box)


def piece_nbytes(box: Box, dtype: torch.dtype) -> int:
    """A piece's bytes in a flat bucket, counting its 16 B alignment."""
    nbytes = math.prod(box_shape(box)) * dtype.itemsize
    return -(-nbytes // _PIECE_ALIGNMENT) * _PIECE_ALIGNMENT


FlatPiece = tuple[int, int, torch.dtype, tuple[int, ...]]
"""Where a piece sits in a flat bucket: byte offset, bytes, dtype, shape."""


def flat_layout(boxes: Sequence[Box], dtypes: Sequence[torch.dtype]) -> list[FlatPiece]:
    """The pieces of a flat bucket, in order, each starting on 16 B, so the
    bucket is `sum(piece_nbytes)` long. Producer and consumer derive it from
    the same `(name, box)` list."""
    layout, offset = [], 0
    for box, dtype in zip(boxes, dtypes, strict=True):
        shape = box_shape(box)
        layout.append((offset, math.prod(shape) * dtype.itemsize, dtype, shape))
        offset += piece_nbytes(box, dtype)
    return layout


def flat_view(flat: torch.Tensor, piece: FlatPiece) -> torch.Tensor:
    """One piece of a flat byte tensor, shaped like its box."""
    offset, nbytes, dtype, shape = piece
    return flat[offset : offset + nbytes].view(dtype).view(shape)


def gather_units(nbytes: Sequence[int], budget: int) -> list[int]:
    """The gather unit of each name: consecutive names, in order, of at most
    `budget` bytes together; a name larger than `budget` is a unit alone."""
    units: list[int] = []
    unit, used = -1, budget
    for n in nbytes:
        if used + n > budget or unit < 0:
            unit, used = unit + 1, 0
        units.append(unit)
        used += n
    return units


def wire_box(box: Box) -> list[list[int]]:
    return [[start, stop] for start, stop in box]


def from_wire_box(box: Sequence[Sequence[int]]) -> Box:
    return tuple((int(start), int(stop)) for start, stop in box)


def assign_producer_indices(
    num_producers: int, num_consumers: int, consumer_idx: int
) -> list[int]:
    """Producers (indices) consumer `consumer_idx` binds under M:N."""
    p = max(1, num_producers)
    c = max(1, num_consumers)
    if p >= c:
        return list(range(consumer_idx * p // c, (consumer_idx + 1) * p // c))
    return [consumer_idx * p // c]


class RdtRouter:
    """Which producer serves each name to each consumer. Pure: both sides
    derive the same answers from the same wire tables.

    Ownership is per name: `owner_sets` holds the distinct producer sets and
    `name_owner_class[i]` indexes it for `names[i]`; empty tables mean every
    producer holds everything. A pipeline stage is a set of names sharing an
    owner set, an expert is a name whose owner set is one rank.
    """

    def __init__(
        self,
        num_producers: int,
        num_consumers: int,
        owner_sets: list[list[int]] | None = None,
        name_owner_class: list[int] | None = None,
        names: list[str] | None = None,
        workers_per_replica: int = 0,
    ) -> None:
        """Build the routing tables.

        Args:
            num_producers: Trainer ranks.
            num_consumers: Inference workers across the whole fleet.
            owner_sets: The distinct owner sets, one per owner class; empty
                means one class owning every producer.
            name_owner_class: Per name of `names`, its index into
                `owner_sets`; a name without one is class 0.
            names: The metadata names.
            workers_per_replica: Consumers per inference deployment. The
                M:N carve spreads one deployment over an owner set and every
                deployment reuses it; 0 means one deployment.

        Raises:
            ValueError: `workers_per_replica` does not divide `num_consumers`.

        """
        self.num_producers = max(1, num_producers)
        self.num_consumers = max(1, num_consumers)
        self._block_consumers = min(
            self.num_consumers, max(1, workers_per_replica or self.num_consumers)
        )
        if self.num_consumers % self._block_consumers:
            raise ValueError(
                f"workers_per_replica={workers_per_replica} does not divide "
                f"num_consumers={self.num_consumers}: the inference deployments "
                "must be uniform."
            )
        self._owner_sets = (
            [sorted(set(row)) for row in owner_sets]
            if owner_sets
            else [list(range(self.num_producers))]
        )
        classes = list(name_owner_class or [])
        self._class_of = {
            n: (classes[i] if i < len(classes) else 0)
            for i, n in enumerate(names or [])
        }

    def owners(self, name: str) -> list[int]:
        """Every producer holding `name`."""
        return list(self._owner_sets[self._class_of.get(name, 0)])

    def producer_for(self, consumer_id: int, name: str, unit: int) -> int:
        """The producer `consumer_id` pulls `name` from.

        Spreads one deployment's consumers over the name's owners, then
        rotates by gather unit, so a consumer spreads its units over the
        owners it binds instead of loading one NIC.
        """
        own = self.owners(name)
        block = self._block(len(own), consumer_id)
        return own[block[unit % len(block)]]

    def _block(self, num_owners: int, consumer_id: int) -> list[int]:
        return assign_producer_indices(
            num_owners, self._block_consumers, consumer_id % self._block_consumers
        )

    def consumers_of(self, producer: int) -> list[int]:
        """Consumers that may pull from `producer`: the bound on its fan-in."""
        return [
            c
            for c in range(self.num_consumers)
            if any(
                producer in own and own.index(producer) in self._block(len(own), c)
                for own in self._owner_sets
            )
        ]

    def components(self) -> list[int]:
        """Per producer, its coupling component: producers that co-own a name
        may gather it collectively, so they progress through units together."""
        parent = list(range(self.num_producers))

        def root(p: int) -> int:
            while parent[p] != p:
                parent[p] = parent[parent[p]]
                p = parent[p]
            return p

        for own in self._owner_sets:
            for a, b in itertools.pairwise(own):
                parent[root(b)] = root(a)
        return [root(p) for p in range(self.num_producers)]

    def validate(self) -> None:
        """Raises:
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


def arena_bytes(*, buffer_bytes: int, num_buffers: int, num_consumers: int) -> int:
    """The NIXL arena one producer registers: `num_buffers` buckets of at most
    `buffer_bytes` for each consumer that may pull from it, plus one of slack
    for a free still on its way and for fragmentation."""
    bucket = buffer_bytes + _PIECE_ALIGNMENT
    return (num_buffers * max(1, num_consumers) + 1) * bucket
