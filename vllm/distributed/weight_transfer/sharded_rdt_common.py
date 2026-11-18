# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Sharded RDT: the wire protocol, and what both sides compute from it.

Each inference worker (consumer) pulls, over NIXL through Ray Direct Transport
(RDT), the boxes of each checkpoint tensor its load plan names from the
trainer ranks (producers) that own them. Both sides cut the metadata into the
same gather units, route each name the same way (`RdtRouter`), and lay a
bucket out the same way (`flat_layout`). The protocol is described in
docs/training/weight_transfer/sharded_rdt.md.
"""

import itertools
import math
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import torch

from vllm.utils.math_utils import round_up

if TYPE_CHECKING:
    from vllm.distributed.weight_transfer.base import ParamMeta

# `register_nixl_memory`, `set_target_for_ref` and concurrent `ray.get` on
# lists, all in Ray 2.56.
RDT_MIN_RAY_VERSION = "2.56.0"
DEFAULT_BUFFER_BYTES = 1 << 30
DEFAULT_NUM_BUFFERS = 2

Box = tuple[tuple[int, int], ...]
# One bucket on the wire: (name, box) per piece.
WireBucket = list[tuple[str, Box]]

PIECE_ALIGNMENT = 16  # every piece of a flat bucket starts on 16 B


def validate_int(name: str, value: object, *, minimum: int) -> None:
    """Reject a non-int (bools included) or an int below `minimum`."""
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name}={value!r}: must be an integer >= {minimum}")


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
    return round_up(math.prod(box_shape(box)) * dtype.itemsize, PIECE_ALIGNMENT)


def meta_nbytes(meta: "ParamMeta") -> int:
    """A full tensor's bytes."""
    return math.prod(meta.shape) * meta.dtype.itemsize


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
        """The owner indices `consumer_id` binds under M:N."""
        p, c = num_owners, self._block_consumers
        i = consumer_id % c
        if p >= c:
            return list(range(i * p // c, (i + 1) * p // c))
        return [i * p // c]

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
    bucket = buffer_bytes + PIECE_ALIGNMENT
    return (num_buffers * max(1, num_consumers) + 1) * bucket
