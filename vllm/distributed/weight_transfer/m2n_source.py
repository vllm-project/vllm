# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Trainer-side weight sources for the NCCL M2N backend.

m2n plans a transfer from both sides' layouts, so the trainer has to say how it
holds each parameter — which the base `WeightSource` / `ParamMeta` pair does not
express. This module supplies the m2n flavor of both, and the DTensor-backed
implementation that covers the common trainer.

A source declares its mesh once (`mesh()`) and a placement per parameter, since
one topology describes every tensor on the side.
"""

from collections.abc import Iterator
from typing import Any

import torch

from vllm.distributed.weight_transfer.base import WeightSource
from vllm.distributed.weight_transfer.m2n_common import (
    REPLICATE,
    REPLICATED,
    M2NMesh,
    M2NParamMeta,
    Placements,
)

__all__ = [
    "M2NWeightSource",
    "DTensorModuleSource",
    "mesh_from_tensor",
    "placements_from_tensor",
]


class M2NWeightSource(WeightSource):
    """A `WeightSource` that also describes how the trainer holds its weights.

    Unlike `ModuleSource`, iteration yields each rank's **local shard**, not a
    materialized full tensor: gathering is exactly the cost m2n removes.
    """

    def mesh(self) -> M2NMesh:
        """The trainer's rank topology, shared by every parameter."""
        raise NotImplementedError

    def metadata(self) -> list[M2NParamMeta]:
        """Name, dtype, full shape, and trainer placement for each parameter."""
        raise NotImplementedError

    def __iter__(self) -> Iterator[tuple[str, torch.Tensor]]:
        """Yield `(name, local shard)` pairs in the same order as `metadata()`."""
        raise NotImplementedError


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

    A plain tensor is identical on every trainer rank, so it spans all of them.
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

    def mesh(self) -> M2NMesh:
        """The mesh every sharded parameter agrees on.

        Replicated parameters span the trainer without a device mesh of their
        own, so they never decide this; a model whose *sharded* parameters
        disagree cannot be described by one side mesh and is rejected.
        """
        meshes = {
            mesh_from_tensor(p, self._num_trainer_ranks)
            for _, p in self._module.named_parameters()
            if placements_from_tensor(p) is not REPLICATED
        }
        if len(meshes) > 1:
            raise ValueError(
                "nccl_m2n needs one mesh per side, but this module's sharded "
                f"parameters span several: {sorted(m.dims for m in meshes)}"
            )
        return meshes.pop() if meshes else M2NMesh((self._num_trainer_ranks, 1), 0)

    def metadata(self) -> list[M2NParamMeta]:
        """Read global shape/dtype and placements without gathering shards."""
        return [
            M2NParamMeta(name, p.dtype, tuple(p.shape), placements_from_tensor(p))
            for name, p in self._module.named_parameters()
        ]

    def __iter__(self) -> Iterator[tuple[str, torch.Tensor]]:
        """Yield each parameter's local shard (`to_local()`), or the tensor itself."""
        for name, param in self._module.named_parameters():
            to_local = getattr(param, "to_local", None)
            yield name, (to_local() if callable(to_local) else param)
