# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Destination-layout resolution for the NCCL M2N weight transfer backend.

For every incoming checkpoint parameter the worker needs a destination buffer
plus its placement over the inference mesh. Two outcomes are possible:

* **direct** — when the model is not quantized, the parameter maps 1:1 onto a
  live vLLM parameter and its loader is a small, known-safe copy or tensor-
  parallel loader. The loader's declared input/output dimension determines the
  placement; shapes are validation, not an inference mechanism. The reshard
  writes straight into the live parameter, so each rank receives only its own
  shard and nothing is copied afterwards.
* **fallback** — anything else. The reshard delivers the whole tensor to every
  rank and `load_weights` does the sharding, exactly as the broadcast NCCL
  backend does. Fused parameters (`qkv_proj`, `gate_up_proj`, MoE `w13`/`w2`)
  take this path: the checkpoint name does not name a vLLM parameter, so there
  is nothing to resolve against.

Correctness never depends on a parameter resolving — the fallback is always
available and is the same path the existing backend uses.
"""

import inspect
import math
from collections.abc import Sequence
from dataclasses import dataclass
from functools import partial
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

_DIRECT_WEIGHT_LOADING_BLOCKLIST = {
    "vllm.model_executor.models.gpt2.GPT2Model",
}


def _module_type_name(module: torch.nn.Module) -> str:
    cls = type(module)
    return f"{cls.__module__}.{cls.__name__}"


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


def _base_loader_shard_dim(
    param: torch.nn.Parameter,
    shard_axis_size: int,
) -> int | None:
    """Return a known-safe loader's declared shard dim, or reject it.

    Loader identity is deliberately fail-closed. A missing loader is not
    treated as ``default_weight_loader`` because a model-level ``load_weights``
    implementation may transform the tensor before it reaches that default.
    Wrappers and partials are also rejected: recognizing their wrapped callable
    would ignore behavior added by the wrapper.
    """
    loader = getattr(param, "weight_loader", None)
    if loader is None or isinstance(loader, partial):
        return None

    owner = loader.__self__ if inspect.ismethod(loader) else None
    loader_fn = loader.__func__ if inspect.ismethod(loader) else loader

    # Lazy imports avoid making the weight-transfer registry import all model
    # executor layers during normal vLLM startup.
    from vllm.model_executor.layers.linear import (
        ColumnParallelLinear,
        ReplicatedLinear,
        RowParallelLinear,
    )
    from vllm.model_executor.model_loader.weight_utils import default_weight_loader

    replicated_loaders = {
        default_weight_loader,
        ReplicatedLinear.weight_loader,
    }
    column_loaders = {
        ColumnParallelLinear.weight_loader,
        ColumnParallelLinear.weight_loader_v2,
    }
    row_loaders = {
        RowParallelLinear.weight_loader,
        RowParallelLinear.weight_loader_v2,
    }

    # These attributes signal packing, a transform, or pre-sharded checkpoint
    # storage. Such semantics cannot be reproduced by a plain M2N placement.
    # Check them before accepting replicated loaders too: model-level loading
    # may transform a tensor before handing it to an otherwise plain copier.
    metadata_attrs = ("packed_dim", "packed_factor", "marlin_tile_size")
    flag_attrs = ("needs_scalar_to_array", "is_sharded_weight", "is_transposed")
    if any(getattr(param, attr, None) is not None for attr in metadata_attrs):
        return None
    if any(bool(getattr(param, attr, False)) for attr in flag_attrs):
        return None

    if loader_fn in replicated_loaders:
        return REPLICATE
    if loader_fn not in column_loaders | row_loaders:
        return None

    if owner is None or getattr(owner, "tp_size", None) != shard_axis_size:
        return None

    attr = "output_dim" if loader_fn in column_loaders else "input_dim"
    dim = getattr(param, attr, None)
    return dim if isinstance(dim, int) and dim >= 0 else None


def _validated_shard_dim(
    global_shape: Sequence[int],
    local_shape: Sequence[int],
    declared_dim: int,
    shard_axis_size: int,
) -> int | None:
    """Validate a loader-declared shard dimension against both shapes."""
    if len(global_shape) != len(local_shape):
        return None

    if declared_dim == REPLICATE:
        return REPLICATE if tuple(global_shape) == tuple(local_shape) else None
    if declared_dim >= len(global_shape):
        return None
    for dim, (whole, local) in enumerate(zip(global_shape, local_shape)):
        expected = local * shard_axis_size if dim == declared_dim else local
        if whole != expected:
            return None
    return declared_dim


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
    shard axis. Parameters below a model type with known model-level checkpoint
    preprocessing also fall back, since bypassing `load_weights` would skip the
    transformation.
    """
    params = dict(model.named_parameters()) if allow_direct else {}
    blocked_param_ids = {
        id(param)
        for module in model.modules()
        if _module_type_name(module) in _DIRECT_WEIGHT_LOADING_BLOCKLIST
        for param in module.parameters()
    }

    destinations: list[M2NDestination] = []
    for name, dtype, shape in zip(names, dtypes, shapes):
        param = params.get(name)
        dim = None
        if (
            param is not None
            and id(param) not in blocked_param_ids
            and param.dtype == dtype
            and param.data.is_contiguous()
            and num_workers % shard_axis_size == 0
        ):
            declared_dim = _base_loader_shard_dim(param, shard_axis_size)
            if declared_dim is not None:
                dim = _validated_shard_dim(
                    shape, param.shape, declared_dim, shard_axis_size
                )

        if dim is None:
            # A replicated parameter is identical on every rank, so it needs no
            # placement of its own — REPLICATED lets resolve_layout spread it
            # over all the workers regardless of how the mesh is factored.
            destinations.append(M2NDestination(name, REPLICATED, None))
        elif dim == REPLICATE:
            assert param is not None
            destinations.append(M2NDestination(name, REPLICATED, param.data))
        else:
            assert param is not None
            destinations.append(
                M2NDestination(name, _destination_placements(dim), param.data)
            )

    # Layerwise reload finalizes a module as a unit. Mixing direct and fallback
    # parameters within one module can copy stale storage over the direct load.
    fallback_modules = {
        name.rpartition(".")[0]
        for name, destination in zip(names, destinations)
        if not destination.direct and name in params
    }
    destinations = [
        M2NDestination(destination.name, REPLICATED, None)
        if destination.direct
        and destination.name.rpartition(".")[0] in fallback_modules
        else destination
        for destination in destinations
    ]

    num_direct = sum(d.direct for d in destinations)
    parameter_bytes = [
        math.prod(shape) * dtype.itemsize for dtype, shape in zip(dtypes, shapes)
    ]
    total_bytes = sum(parameter_bytes)
    direct_bytes = sum(
        nbytes
        for destination, nbytes in zip(destinations, parameter_bytes)
        if destination.direct
    )
    direct_byte_percentage = 100 * direct_bytes / total_bytes if total_bytes else 0
    logger.info(
        "nccl_m2n destination plan: %d/%d parameters resharded directly into "
        "the model, %d via full-tensor fallback; direct byte coverage: "
        "%d/%d bytes (%.1f%%)",
        num_direct,
        len(destinations),
        len(destinations) - num_direct,
        direct_bytes,
        total_bytes,
        direct_byte_percentage,
    )
    return destinations
