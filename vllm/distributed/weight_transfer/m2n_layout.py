# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Destination-layout resolution for the NCCL M2N weight transfer backend.

For every incoming checkpoint parameter the worker needs a destination buffer
plus its placement over the inference mesh. Two outcomes are possible:

* **direct** — when the model is not quantized, the parameter maps 1:1 onto a
  live vLLM parameter whose shape is either identical to the checkpoint shape
  (replicated) or differs on exactly one dim by the shard-axis factor. The
  reshard writes straight into the live parameter, so each rank receives only
  its own shard and nothing is copied afterwards.
* **fallback** — anything else. The reshard delivers the whole tensor to every
  rank and `load_weights` does the sharding, exactly as the broadcast NCCL
  backend does. Fused parameters (`qkv_proj`, `gate_up_proj`, MoE `w13`/`w2`)
  take this path: the checkpoint name does not name a vLLM parameter, so there
  is nothing to resolve against.

Correctness never depends on a parameter resolving — the fallback is always
available and is the same path the existing backend uses.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import cast

import torch

from vllm.distributed.weight_transfer.m2n_common import (
    DESTINATION_REPLICA_AXIS,
    DESTINATION_SHARD_AXIS,
    MESH_NDIMS,
    REPLICATE,
    REPLICATED,
    Placements,
)
from vllm.logger import init_logger

logger = init_logger(__name__)


def _destination_placements(shard_dim: int) -> Placements:
    """Place one tensor shard on the configured destination mesh axis."""
    placements = [REPLICATE] * MESH_NDIMS
    placements[DESTINATION_REPLICA_AXIS] = REPLICATE
    placements[DESTINATION_SHARD_AXIS] = shard_dim
    return cast(Placements, tuple(placements))


@dataclass
class M2NDestination:
    """Where one checkpoint parameter lands on this worker."""

    name: str
    placements: Placements | None
    """Placement over the inference mesh, or `REPLICATED` for the fallback."""
    tensor: torch.Tensor | None
    """The live parameter view to reshard into, or None for the fallback path
    (the engine allocates a full replica per round and calls `load_weights`)."""

    @property
    def direct(self) -> bool:
        return self.tensor is not None


def _shard_dim(
    global_shape: Sequence[int],
    local_shape: Sequence[int],
    shard_axis_size: int,
) -> int | None:
    """Tensor dim the local shape shards, `REPLICATE`, or None if unresolvable.

    Derived from the shapes alone rather than the parameter's `output_dim` /
    `input_dim`, so it stays honest for any layer type: a mismatch anywhere it
    cannot explain simply falls back.
    """
    if len(global_shape) != len(local_shape):
        return None
    differing = [
        i
        for i, (whole, local) in enumerate(zip(global_shape, local_shape))
        if whole != local
    ]
    if not differing:
        return REPLICATE
    if len(differing) > 1:
        return None
    dim = differing[0]
    if local_shape[dim] * shard_axis_size != global_shape[dim]:
        return None
    return dim


def resolve_parameter_destinations(
    model: torch.nn.Module,
    names: Sequence[str],
    dtypes: Sequence[torch.dtype],
    shapes: Sequence[Sequence[int]],
    *,
    num_workers: int,
    shard_axis_size: int,
    allow_direct: bool,
) -> list[M2NDestination]:
    """Build this worker's destination plan, one entry per checkpoint parameter.

    Placements are relative to the inference mesh: axis 0 replicates and axis 1
    shards. `allow_direct=False` forces every parameter onto the fallback path;
    the engine sets it for pipeline-parallel or quantized deployments, where a
    parameter's local shape is not simply the checkpoint shape split across the
    shard axis.
    """
    params = dict(model.named_parameters()) if allow_direct else {}

    destinations: list[M2NDestination] = []
    for name, dtype, shape in zip(names, dtypes, shapes):
        param = params.get(name)
        dim = None
        if (
            param is not None
            and param.dtype == dtype
            and param.data.is_contiguous()
            and num_workers % shard_axis_size == 0
        ):
            dim = _shard_dim(shape, param.shape, shard_axis_size)

        if dim is None or dim == REPLICATE:
            # A replicated parameter is identical on every rank, so it needs no
            # placement of its own — REPLICATED lets resolve_layout spread it
            # over all the workers regardless of how the mesh is factored.
            tensor = param.data if dim == REPLICATE else None
            destinations.append(M2NDestination(name, REPLICATED, tensor))
        else:
            destinations.append(
                M2NDestination(name, _destination_placements(dim), param.data)
            )

    num_direct = sum(d.direct for d in destinations)
    logger.info(
        "nccl_m2n destination plan: %d/%d parameters resharded directly into "
        "the model, %d via full-tensor fallback",
        num_direct,
        len(destinations),
        len(destinations) - num_direct,
    )
    return destinations
