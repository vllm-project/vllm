# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Deprecated trainer-side weight sources for the NCCL M2N backend.

The base `ParamMeta` now carries each parameter's owners, mesh and placements,
and `WeightSource.iter_local()` yields local shards, so any `WeightSource` that
implements `iter_local()` (such as `ModuleSource`) feeds m2n directly. The
classes here remain for one deprecation cycle.
"""

from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from typing import Any

import torch

from vllm.distributed.weight_transfer.base import (
    ModuleSource,
    WeightBatch,
    WeightSource,
)
from vllm.distributed.weight_transfer.m2n_common import (
    REPLICATE,
    REPLICATED,
    M2NLayout,
    M2NMesh,
    M2NParamMeta,
    Placements,
)

__all__ = [
    "M2NWeightSource",
    "M2NManifestEntry",
    "ManifestM2NWeightSource",
    "DTensorModuleSource",
    "mesh_from_tensor",
    "placements_from_tensor",
]


class M2NWeightSource(WeightSource):
    """Deprecated: implement `WeightSource.iter_local()` and declare layouts
    on the base `ParamMeta` instead.

    Metadata is `M2NParamMeta`, and `__iter__` yields each rank's local shard,
    one parameter at a time; `iter_local()` adapts that.
    """

    def iter_local(self) -> Iterator[WeightBatch]:
        """Adapt `__iter__` to one-parameter batches."""
        for name, buffer in self:
            yield [(name, buffer)]

    def metadata(self) -> list[M2NParamMeta]:  # type: ignore[override]
        """Name, dtype, full shape, and source layout for each parameter."""
        raise NotImplementedError

    def __iter__(self) -> Iterator[tuple[str, Any]]:  # type: ignore[override]
        """Yield `(name, local buffer)` pairs in metadata order.

        Providers must enqueue materialization on the caller's current CUDA
        stream, or return a buffer whose writes are already visible to it.
        Buffers may be torch tensors, CUDA Array Interface objects, or raw
        device pointers accepted by nccl4py. Providers own correct
        communicator-device placement. Raw-pointer providers additionally own
        allocation lifetime, dtype, and shape correctness.
        """
        raise NotImplementedError


@dataclass(frozen=True)
class M2NManifestEntry:
    """One immutable source descriptor with a lazy rank-local tensor view."""

    key: str
    dtype: torch.dtype
    global_shape: tuple[int, ...]
    source_layout: M2NLayout
    local_tensor: Callable[[], Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "global_shape", tuple(self.global_shape))
        if not self.key:
            raise ValueError("M2N manifest key must not be empty")
        if not isinstance(self.dtype, torch.dtype):
            raise TypeError(
                f"manifest entry '{self.key}' has invalid dtype {self.dtype!r}"
            )
        if not isinstance(self.source_layout, M2NLayout):
            raise TypeError(
                f"manifest entry '{self.key}' source_layout must be an M2NLayout"
            )
        if not callable(self.local_tensor):
            raise TypeError(
                f"manifest entry '{self.key}' local_tensor must be callable"
            )

    @property
    def metadata(self) -> M2NParamMeta:
        """Return the serializable portion consumed during initialization."""
        return M2NParamMeta(
            self.key,
            self.dtype,
            self.global_shape,
            self.source_layout,
        )


class ManifestM2NWeightSource(M2NWeightSource):
    """An ordered M2N source backed by lazy, rank-local manifest entries."""

    def __init__(self, entries: Sequence[M2NManifestEntry]) -> None:
        self._entries = tuple(entries)
        self._metadata = tuple(entry.metadata for entry in self._entries)

    def metadata(self) -> list[M2NParamMeta]:  # type: ignore[override]
        """Return a fresh list while preserving the immutable manifest order."""
        return list(self._metadata)

    def __iter__(self) -> Iterator[tuple[str, Any]]:  # type: ignore[override]
        """Materialize one rank-local buffer at a time."""
        for entry in self._entries:
            yield entry.key, entry.local_tensor()


def _placement_code(placement: Any) -> int:
    """Map a `torch.distributed` placement onto an m2n placement code."""
    name = type(placement).__name__
    if name == "Replicate":
        return REPLICATE
    if name == "Shard":
        return int(placement.dim)
    raise ValueError(
        f"nccl_m2n cannot express the {name} placement; only Replicate and "
        "Shard are supported"
    )


def mesh_from_tensor(tensor: torch.Tensor, num_trainer_ranks: int) -> M2NMesh:
    """The mesh a parameter lives on, as an `M2NMesh`.

    Tensors without DeviceMesh metadata are treated as replicated. Implicitly
    sharded tensors require a custom source with an explicit layout.
    """
    device_mesh = getattr(tensor, "device_mesh", None)
    if device_mesh is None:
        return M2NMesh((num_trainer_ranks, 1), 0)

    grid = device_mesh.mesh
    ranks = grid.flatten().tolist()
    if ranks != list(range(len(ranks))):
        raise ValueError(
            "nccl_m2n requires the trainer's device mesh to cover the "
            f"contiguous rank interval [0, {len(ranks)}); got {ranks}"
        )
    if grid.ndim > 2:
        raise ValueError(
            f"nccl_m2n supports 1-D and 2-D device meshes, got {grid.ndim}-D"
        )
    dims = tuple(grid.shape)
    return M2NMesh(dims if len(dims) == 2 else (dims[0], 1), 0)


def placements_from_tensor(tensor: torch.Tensor) -> Placements | None:
    """How a parameter is placed over its mesh, or `REPLICATED`.

    `REPLICATED` is returned rather than a `(REPLICATE, REPLICATE)` pair: m2n
    cannot express that, and the size-1-axis encoding it needs instead is
    applied later by `resolve_layout`, which owns that workaround.
    """
    placements = getattr(tensor, "placements", None)
    if placements is None:
        return REPLICATED
    codes = [_placement_code(p) for p in placements]
    if all(code == REPLICATE for code in codes):
        return REPLICATED
    if len(codes) == 1:
        return (codes[0], REPLICATE)
    return (codes[0], codes[1])


class DTensorModuleSource(ModuleSource):
    """Deprecated alias of `ModuleSource`, which now declares DTensor layouts
    itself; `num_trainer_ranks` is ignored."""

    def __init__(self, module: torch.nn.Module, num_trainer_ranks: int) -> None:
        super().__init__(module)
