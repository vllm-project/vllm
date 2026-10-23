# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""KV hygiene between GLM-5.3's low-latency decode segment and vLLM's stock forwards."""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import ClassVar, Protocol

import torch

from vllm.models.glm53.nvidia.ops.launch import Glm53Geometry

PlaneKey = tuple[int, str]

_CACHE_NAME = re.compile(
    r"(?:^|\.)layers\.(?P<layer>\d+)\.self_attn\.(?P<kind>attn|indexer\.k_cache)$"
)


class TpGroup(Protocol):
    """The part of vLLM's ``GroupCoordinator`` the completion uses."""

    @property
    def rank_in_group(self) -> int: ...

    @property
    def world_size(self) -> int: ...

    def broadcast(self, input_: torch.Tensor, src: int = 0) -> torch.Tensor:
        """Broadcast ``input_`` in place from group rank ``src``."""
        ...


@dataclass(frozen=True)
class Plane:
    """One plane of one layer in vLLM's paged pool."""

    tensor: torch.Tensor
    block_size: int
    block_stride: int
    offset: int
    pitch: int
    width: int

    def blocks(self) -> torch.Tensor:
        """Return the plane as a ``[blocks, block_size, width]`` uint8 view."""
        return torch.as_strided(
            self.tensor,
            size=(int(self.tensor.shape[0]), self.block_size, self.width),
            stride=(self.block_stride, self.pitch, 1),
            storage_offset=self.tensor.storage_offset() + self.offset,
        )


def _page(cache: torch.Tensor, block_size: int) -> tuple[torch.Tensor, int, int]:
    if cache.dim() < 2 or cache.stride(-1) != 1:
        raise ValueError(
            "a cache view needs a block dimension and contiguous rows, got "
            f"{tuple(cache.shape)}"
        )
    element = cache.element_size()
    row_bytes = int(cache.shape[-1]) * element
    expected, rows = row_bytes, 1
    for size, stride in zip(
        reversed(cache.shape[1:-1]), reversed(cache.stride()[1:-1])
    ):
        if size == 1:
            continue
        if stride * element != expected:
            raise ValueError(
                f"cache view {tuple(cache.shape)} strides {cache.stride()} are not a "
                "dense page"
            )
        expected *= size
        rows *= size
    if rows != block_size:
        raise ValueError(
            f"cache view {tuple(cache.shape)} holds {rows} rows per block, not "
            f"{block_size}"
        )
    block_stride = cache.stride(0) * element
    if block_stride < block_size * row_bytes:
        raise ValueError(
            f"block stride {block_stride} is smaller than one page "
            f"({block_size * row_bytes})"
        )
    raw = cache if cache.dtype == torch.uint8 else cache.view(torch.uint8)
    return raw, block_stride, row_bytes


def _mla_row(geometry: Glm53Geometry) -> dict[str, tuple[int, int]]:
    """The ``(offset, bytes)`` of the planes of one fp8 MLA row: the latent, then the
    RoPE part.
    """
    fp8 = torch.float8_e4m3fn.itemsize
    latent, rope = geometry.kv_lora * fp8, geometry.rope_dim * fp8
    return {"latent": (0, latent), "rope": (latent, rope)}


def mla_planes(
    cache: torch.Tensor, geometry: Glm53Geometry, block_size: int
) -> dict[str, Plane]:
    """Return the planes of one layer's MLA cache: latent and RoPE."""
    raw, stride, row = _page(cache, block_size)
    planes = _mla_row(geometry)
    if row != sum(width for _, width in planes.values()):
        raise ValueError(
            f"MLA cache rows are {row} bytes, not fp8 latent and RoPE rows "
            "(--kv-cache-dtype fp8)"
        )
    return {
        name: Plane(raw, block_size, stride, offset, row, width)
        for name, (offset, width) in planes.items()
    }


def check_mla_pool(
    cache: torch.Tensor, geometry: Glm53Geometry, block_size: int
) -> None:
    """Refuse a layer's MLA cache that the decode kernels cannot address."""
    _, stride, row = _page(cache, block_size)
    expected = sum(width for _, width in _mla_row(geometry).values())
    if row != expected or stride != block_size * row:
        raise ValueError(
            f"MLA cache of {row}-byte rows and block stride {stride} is not a "
            "layer-outer pool "
            f"of {expected}-byte rows (--kv-cache-dtype fp8)"
        )


def index_planes(
    cache: torch.Tensor, geometry: Glm53Geometry, block_size: int
) -> dict[str, Plane]:
    """Return the key and scale planes of one layer's DSA index-key cache (a block's
    keys, then its scales).
    """
    raw, stride, row = _page(cache, block_size)
    key, scale = (
        geometry.index_dim * torch.float8_e4m3fn.itemsize,
        torch.float32.itemsize,
    )
    if row != key + scale:
        raise ValueError(
            f"index-key cache rows are {row} bytes, not {key} key bytes and a scale"
        )
    return {
        "index_key": Plane(raw, block_size, stride, 0, key, key),
        "index_scale": Plane(raw, block_size, stride, block_size * key, scale, scale),
    }


def pool_planes(
    caches: Mapping[str, torch.Tensor], geometry: Glm53Geometry, block_size: int
) -> dict[PlaneKey, Plane]:
    """Return every plane of vLLM's per-layer caches, keyed by ``(layer, plane)``."""
    planes: dict[PlaneKey, Plane] = {}
    for name, cache in caches.items():
        match = _CACHE_NAME.search(name)
        if match is None:
            continue
        layer = int(match.group("layer"))
        planes_of = mla_planes if match.group("kind") == "attn" else index_planes
        found = planes_of(cache, geometry, block_size)
        for plane, view in found.items():
            if (layer, plane) in planes:
                raise ValueError(f"two caches hold layer {layer}'s {plane} plane")
            planes[(layer, plane)] = view
    return planes


@dataclass(frozen=True)
class Rows:
    """The ``(physical block, row in block)`` pairs of some positions."""

    blocks: torch.Tensor
    rows: torch.Tensor

    @classmethod
    def of_positions(
        cls,
        block_ids: Sequence[int] | torch.Tensor,
        start: int,
        stop: int,
        *,
        block_size: int,
        device: torch.device | str,
    ) -> Rows:
        """Index logical positions ``[start, stop)`` through a request's block table."""
        if not 0 <= start <= stop or stop > len(block_ids) * block_size:
            raise ValueError(
                f"positions [{start}, {stop}) outside a table of {len(block_ids)} "
                "blocks"
            )
        if isinstance(block_ids, torch.Tensor):
            table = block_ids.to(device=device, dtype=torch.int64)
        else:
            table = torch.tensor(list(block_ids), dtype=torch.int64, device=device)
        positions = torch.arange(start, stop, dtype=torch.int64, device=device)
        return cls(table[positions // block_size], positions % block_size)

    def __len__(self) -> int:
        return int(self.rows.shape[0])

    def chunk(self, first: int, last: int) -> Rows:
        """Return positions ``[first, last)`` of this index."""
        return Rows(self.blocks[first:last], self.rows[first:last])

    def gather(self, plane: Plane) -> torch.Tensor:
        """Return ``plane``'s bytes at these rows as ``[n, width]`` uint8."""
        return plane.blocks()[self.blocks, self.rows]

    def scatter(self, plane: Plane, values: torch.Tensor) -> None:
        """Write ``[n, width]`` uint8 ``values`` into ``plane`` at these rows."""
        if values.dtype != torch.uint8 or tuple(values.shape) != (
            len(self),
            plane.width,
        ):
            raise ValueError(
                f"rows {tuple(values.shape)} {values.dtype} do not fit plane width "
                f"{plane.width}"
            )
        plane.blocks()[self.blocks, self.rows] = values


class PendingRows:
    """Per request, the position range written on the writer ranks only."""

    def __init__(self) -> None:
        self._pending: dict[str, tuple[int, int]] = {}

    def decoded(self, request_id: str, start: int, stop: int) -> None:
        """Record that a segment step may have written ``[start, stop)``."""
        start = max(start, 0)
        if stop <= start:
            return
        low, high = self._pending.get(request_id, (start, stop))
        self._pending[request_id] = (min(low, start), max(high, stop))

    def has(self, request_id: str) -> bool:
        """Whether ``request_id`` has a pending range."""
        return request_id in self._pending

    def take(self, request_id: str | None = None) -> list[tuple[str, int, int]]:
        """Remove and return pending ranges (one request's, or all)."""
        if request_id is not None:
            span = self._pending.pop(request_id, None)
            return [] if span is None else [(request_id, *span)]
        pending = sorted((key, *span) for key, span in self._pending.items())
        self._pending.clear()
        return pending

    def __bool__(self) -> bool:
        return bool(self._pending)


class Glm53KvHygiene:
    """One rank's KV hygiene for the decode segment."""

    #: Positions copied per broadcast.
    CHUNK: ClassVar[int] = 4096

    def __init__(
        self,
        planes: Mapping[PlaneKey, Plane],
        geometry: Glm53Geometry,
        *,
        rank: int,
        block_size: int,
        partial_writers: Mapping[PlaneKey, frozenset[int]] | None = None,
    ) -> None:
        self.planes: dict[PlaneKey, Plane] = dict(planes)
        self.rank: int = rank
        self.world: int = geometry.world
        self.block_size: int = block_size
        if partial_writers is None:
            selectors = frozenset(range(geometry.selector_ranks))
            partial_writers = {
                key: selectors
                for key in self.planes
                if key[1] in ("index_key", "index_scale") and key[0] < geometry.layers
            }
        sources: dict[int, list[PlaneKey]] = {}
        self.receivers: dict[PlaneKey, frozenset[int]] = {}
        for key in sorted(partial_writers):
            writers = frozenset(partial_writers[key])
            if key not in self.planes:
                raise ValueError(f"writer-only plane {key} is not in the pool")
            if not writers or not writers <= frozenset(range(self.world)):
                raise ValueError(
                    f"plane {key} writers {sorted(writers)} are not ranks of "
                    f"{self.world}"
                )
            if len(writers) < self.world:
                sources.setdefault(min(writers), []).append(key)
                self.receivers[key] = frozenset(range(self.world)) - writers
        self.transfers: tuple[tuple[int, tuple[PlaneKey, ...]], ...] = tuple(
            (source, tuple(keys)) for source, keys in sorted(sources.items())
        )
        self.pending: PendingRows = PendingRows()
        self._device: torch.device = next(iter(self.planes.values())).tensor.device

    @classmethod
    def from_caches(
        cls,
        caches: Mapping[str, torch.Tensor],
        geometry: Glm53Geometry,
        *,
        rank: int,
        block_size: int,
        layers: Iterable[int] | None = None,
    ) -> Glm53KvHygiene:
        """Build over vLLM's per-layer caches by layer name (:func:`pool_planes`)."""
        planes = pool_planes(caches, geometry, block_size)
        if layers is not None:
            wanted = set(layers)
            planes = {key: plane for key, plane in planes.items() if key[0] in wanted}
        return cls(planes, geometry, rank=rank, block_size=block_size)

    def complete(
        self,
        block_ids: Sequence[int] | torch.Tensor,
        start: int,
        stop: int,
        group: TpGroup,
    ) -> None:
        """Copy the writer-only planes' rows ``[start, stop)`` to the other ranks."""
        if group.world_size != self.world:
            raise ValueError(
                f"hygiene for {self.world} ranks used on a group of {group.world_size}"
            )
        stop = min(stop, len(block_ids) * self.block_size)
        if stop <= start or not self.transfers:
            return
        rows = Rows.of_positions(
            block_ids, start, stop, block_size=self.block_size, device=self._device
        )
        for source, keys in self.transfers:
            widths = [self.planes[key].width for key in keys]
            for first in range(0, len(rows), self.CHUNK):
                part = rows.chunk(first, min(first + self.CHUNK, len(rows)))
                count = len(part)
                if self.rank == source:
                    buffer = torch.cat(
                        [part.gather(self.planes[key]).reshape(-1) for key in keys]
                    )
                else:
                    buffer = torch.empty(
                        count * sum(widths), dtype=torch.uint8, device=self._device
                    )
                group.broadcast(buffer, src=source)
                offset = 0
                for key, width in zip(keys, widths):
                    size = count * width
                    if self.rank in self.receivers[key]:
                        part.scatter(
                            self.planes[key],
                            buffer[offset : offset + size].view(count, width),
                        )
                    offset += size

    def complete_pending(
        self,
        block_ids_of: Callable[[str], Sequence[int] | torch.Tensor],
        group: TpGroup,
    ) -> None:
        """Complete every pending range (before a stock forward reads them)."""
        for request_id, start, stop in self.pending.take():
            self.complete(block_ids_of(request_id), start, stop, group)

    def release(
        self, request_id: str, block_ids: Sequence[int] | torch.Tensor, group: TpGroup
    ) -> None:
        """Complete a finished or preempted request's pending range."""
        for _, start, stop in self.pending.take(request_id):
            self.complete(block_ids, start, stop, group)
