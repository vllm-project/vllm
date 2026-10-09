# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Trainer-side weight sources for the NCCL M2N backend.

M2N plans a transfer from both sides' layouts, so the trainer has to say how it
holds each parameter — which the base `WeightSource` / `ParamMeta` pair does not
express. Each parameter owns its source mesh and placements because dense and
expert weights can use different factorizations over the same trainer ranks.
"""

from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from typing import Any

import torch

from vllm.distributed.weight_transfer.base import WeightSource
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
    """A `WeightSource` that also describes how the trainer holds its weights.

    Unlike `ModuleSource`, iteration yields each rank's **local shard**, not a
    materialized full tensor: gathering is exactly the cost m2n removes.
    """

    def metadata(self) -> list[M2NParamMeta]:  # type: ignore[override]
        """Name, dtype, full shape, and source layout for each parameter."""
        raise NotImplementedError

    def __iter__(self) -> Iterator[tuple[str, torch.Tensor]]:
        """Yield `(name, local shard)` pairs in metadata order.

        Providers must enqueue materialization on the caller's current CUDA
        stream, or return a tensor whose writes are already visible to it.
        """
        raise NotImplementedError


@dataclass(frozen=True)
class M2NManifestEntry:
    """One immutable source descriptor with a lazy rank-local tensor view."""

    key: str
    dtype: torch.dtype
    global_shape: tuple[int, ...]
    source_layout: M2NLayout
    local_tensor: Callable[[], torch.Tensor]
    allow_full_fallback: bool = True

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
        if not isinstance(self.allow_full_fallback, bool):
            raise TypeError(
                f"manifest entry '{self.key}' allow_full_fallback must be a bool"
            )

    @property
    def metadata(self) -> M2NParamMeta:
        """Return the serializable portion consumed during initialization."""
        return M2NParamMeta(
            self.key,
            self.dtype,
            self.global_shape,
            self.source_layout,
            self.allow_full_fallback,
        )


class ManifestM2NWeightSource(M2NWeightSource):
    """An ordered M2N source backed by lazy, rank-local manifest entries.

    The provider callable is invoked on every iteration, so it can return a
    current parameter view or perform rank-local conversion and LoRA merging.
    Callable identities and tensors remain local and never enter the wire plan.
    Callbacks run synchronously and must honor `M2NWeightSource.__iter__`'s
    current-stream readiness contract.
    """

    def __init__(self, entries: Sequence[M2NManifestEntry]) -> None:
        self._entries = tuple(entries)
        self._metadata = tuple(entry.metadata for entry in self._entries)

    def metadata(self) -> list[M2NParamMeta]:  # type: ignore[override]
        """Return a fresh list while preserving the immutable manifest order."""
        return list(self._metadata)

    def __iter__(self) -> Iterator[tuple[str, torch.Tensor]]:
        """Materialize one rank-local tensor at a time."""
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

    This adapter treats tensors without DeviceMesh metadata as replicated.
    Implicitly sharded tensors, such as Megatron parameters, require a custom
    `M2NWeightSource` with an explicit source layout.

    Megatron integrations must use `ManifestM2NWeightSource` or another custom
    `M2NWeightSource` that explicitly supplies TP/EP layouts. Megatron parameters
    must not be passed to `DTensorModuleSource`.
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


class DTensorModuleSource(M2NWeightSource):
    """`M2NWeightSource` over `module.named_parameters()`.

    Covers both the FSDP/DTensor trainer (placement read off each parameter,
    local shard yielded via `to_local()`) and the plain replicated trainer, with
    no special casing. Trainers with a custom producer (a Megatron export, MoE
    re-fusing) subclass `M2NWeightSource` instead.
    """

    def __init__(self, module: torch.nn.Module, num_trainer_ranks: int) -> None:
        """Wrap `module`; `num_trainer_ranks` sizes the mesh for plain tensors."""
        self._module = module
        self._num_trainer_ranks = num_trainer_ranks

    def metadata(self) -> list[M2NParamMeta]:  # type: ignore[override]
        """Read each global shape and source layout without gathering shards."""
        return [
            M2NParamMeta(
                name,
                p.dtype,
                tuple(p.shape),
                M2NLayout(
                    mesh_from_tensor(p, self._num_trainer_ranks),
                    placements_from_tensor(p),
                ),
            )
            for name, p in self._module.named_parameters()
        ]

    def __iter__(self) -> Iterator[tuple[str, torch.Tensor]]:
        """Yield each parameter's local shard (`to_local()`), or the tensor itself."""
        for name, param in self._module.named_parameters():
            to_local = getattr(param, "to_local", None)
            yield name, (to_local() if callable(to_local) else param)
