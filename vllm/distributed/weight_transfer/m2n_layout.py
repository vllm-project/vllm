# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Destination-layout resolution for the NCCL M2N weight transfer backend.

For every incoming checkpoint parameter the worker needs a destination buffer
plus its placement over the inference mesh. Three outcomes are possible:

* **in-place** — when the model is not quantized, the parameter maps 1:1 onto a
  live vLLM parameter and its loader is a small, known-safe copy or tensor-
  parallel loader. The loader's declared input/output dimension determines the
  placement; shapes are validation, not an inference mechanism. The reshard
  writes straight into the live parameter, so each rank receives only its own
  shard and nothing is copied afterwards.
* **sharded staging** — a model module exposes the logical shard owned by this
  rank and an update-scoped callback that consumes it through the module's
  native loading path. M2N transfers only that shard into bounded staging.
* **full fallback** — when explicitly allowed, the reshard delivers the whole
  tensor to every rank and `load_weights` performs the model-specific loading,
  matching the broadcast NCCL backend.

Parameters that require owner-local transfer fail closed when neither an
in-place nor a semantic sharded destination can be resolved.
"""

import inspect
import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from functools import partial
from typing import cast

import torch

from vllm.distributed.weight_transfer.m2n_common import (
    DESTINATION_REPLICA_AXIS,
    DESTINATION_SHARD_AXIS,
    MESH_NDIMS,
    REPLICATE,
    REPLICATED,
    M2NDestinationMode,
    Placements,
)
from vllm.logger import init_logger
from vllm.model_executor.model_loader.sharded_weight import (
    ShardedWeightRequest,
    ShardedWeightSpec,
    ShardedWeightTarget,
    resolve_sharded_weight_target,
)

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


@dataclass(frozen=True)
class M2NDestination:
    """Immutable worker-local plan for one logical checkpoint tensor."""

    name: str
    mode: M2NDestinationMode
    placements: Placements | None
    local_shape: tuple[int, ...]
    tensor: torch.Tensor | None = field(compare=False, repr=False)
    """Stable live destination for in-place updates; omitted from wire plans."""
    sharded_spec: ShardedWeightSpec | None = None
    staging_group: int | None = None
    staging_slot: int | None = None
    staging_group_size: int | None = None

    @property
    def direct(self) -> bool:
        return self.mode is M2NDestinationMode.IN_PLACE


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


def _validate_semantic_target(
    request: ShardedWeightRequest,
    target: ShardedWeightTarget,
    *,
    shard_axis_size: int,
    destination_shard_index: int,
) -> None:
    spec = target.spec
    if spec.num_shards != shard_axis_size:
        raise ValueError(
            f"semantic target for '{request.name}' uses {spec.num_shards} "
            f"shards, but the destination mesh shard axis has {shard_axis_size}"
        )
    if spec.shard_index != destination_shard_index:
        raise ValueError(
            f"semantic target for '{request.name}' reports shard index "
            f"{spec.shard_index}, but this M2N rank maps to "
            f"{destination_shard_index}"
        )


def resolve_parameter_destinations(
    model: torch.nn.Module,
    names: Sequence[str],
    dtypes: Sequence[torch.dtype],
    shapes: Sequence[Sequence[int]],
    *,
    num_workers: int,
    shard_axis_size: int,
    allow_direct: bool,
    destination_shard_index: int,
    allow_full_fallback: Sequence[bool],
) -> list[M2NDestination]:
    """Resolve semantic targets, then conservative exact-name destinations."""
    if not (len(names) == len(dtypes) == len(shapes)):
        raise ValueError("destination inputs must have the same length")
    fallback_allowed = tuple(allow_full_fallback)
    if len(fallback_allowed) != len(names) or any(
        not isinstance(value, bool) for value in fallback_allowed
    ):
        raise ValueError("allow_full_fallback must contain one bool per destination")
    if shard_axis_size <= 0 or num_workers % shard_axis_size:
        raise ValueError(
            f"destination shard axis {shard_axis_size} must divide "
            f"{num_workers} workers"
        )
    if not 0 <= destination_shard_index < shard_axis_size:
        raise ValueError(
            f"destination shard index {destination_shard_index} must be in "
            f"[0, {shard_axis_size})"
        )

    params = dict(model.named_parameters()) if allow_direct else {}
    blocked_param_ids = {
        id(param)
        for module in model.modules()
        if _module_type_name(module) in _DIRECT_WEIGHT_LOADING_BLOCKLIST
        for param in module.parameters()
    }

    destinations: list[M2NDestination] = []
    closed_retention_keys: set[object] = set()
    active_retention_key: object | None = None
    active_staging_slot = -1
    active_staging_group_size = 0

    def close_staging_group() -> None:
        nonlocal active_retention_key
        if active_retention_key is None:
            return
        actual_size = active_staging_slot + 1
        if actual_size != active_staging_group_size:
            raise ValueError(
                "semantic destination group starting at "
                f"'{destinations[-actual_size].name}' contains "
                f"{actual_size} tensors, but its provider requires "
                f"{active_staging_group_size}"
            )
        closed_retention_keys.add(active_retention_key)
        active_retention_key = None

    for name, dtype, shape_value, fallback_ok in zip(
        names, dtypes, shapes, fallback_allowed
    ):
        shape = tuple(shape_value)
        request = ShardedWeightRequest(name, dtype, shape)
        target = resolve_sharded_weight_target(model, request)
        # Resolve first so malformed recognized tensors still fail closed.
        if not allow_direct:
            target = None
        if target is not None:
            _validate_semantic_target(
                request,
                target,
                shard_axis_size=shard_axis_size,
                destination_shard_index=destination_shard_index,
            )
            retention_key = target.retention_key
            if retention_key != active_retention_key:
                close_staging_group()
                if retention_key in closed_retention_keys:
                    raise ValueError(
                        f"semantic destination group for '{name}' is not "
                        "contiguous in transfer order"
                    )
                active_retention_key = retention_key
                active_staging_slot = 0
                active_staging_group_size = target.retention_group_size
            else:
                if target.retention_group_size != active_staging_group_size:
                    raise ValueError(
                        f"semantic destination group for '{name}' disagrees on "
                        "its required tensor count"
                    )
                active_staging_slot += 1
            destinations.append(
                M2NDestination(
                    name=name,
                    mode=M2NDestinationMode.SHARDED_STAGING,
                    placements=_destination_placements(target.spec.shard_dim),
                    local_shape=target.spec.local_shape,
                    tensor=None,
                    sharded_spec=target.spec,
                    staging_group=len(closed_retention_keys),
                    staging_slot=active_staging_slot,
                    staging_group_size=active_staging_group_size,
                )
            )
            continue

        close_staging_group()
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
            if not fallback_ok:
                raise ValueError(
                    f"parameter '{name}' requires a sharded destination, but "
                    "the model did not provide one"
                )
            destinations.append(
                M2NDestination(
                    name=name,
                    mode=M2NDestinationMode.FULL_FALLBACK,
                    placements=REPLICATED,
                    local_shape=shape,
                    tensor=None,
                )
            )
            continue

        assert param is not None
        destinations.append(
            M2NDestination(
                name=name,
                mode=M2NDestinationMode.IN_PLACE,
                placements=(
                    REPLICATED if dim == REPLICATE else _destination_placements(dim)
                ),
                local_shape=tuple(param.shape),
                tensor=param.data,
            )
        )

    close_staging_group()

    # Layerwise reload restores model-format tensors as a unit. Mixing direct
    # and model-loader parameters within one module can copy stale storage
    # over a direct load.
    reload_modules = {
        name.rpartition(".")[0]
        for name, destination in zip(names, destinations)
        if not destination.direct and name in params
    }
    for index, destination in enumerate(destinations):
        if destination.direct and destination.name.rpartition(".")[0] in reload_modules:
            if not fallback_allowed[index]:
                raise ValueError(
                    f"parameter '{destination.name}' cannot share a module "
                    "with a model-loader destination without a full fallback"
                )
            destinations[index] = M2NDestination(
                name=destination.name,
                mode=M2NDestinationMode.FULL_FALLBACK,
                placements=REPLICATED,
                local_shape=tuple(shapes[index]),
                tensor=None,
            )

    num_direct = sum(d.direct for d in destinations)
    num_staged = sum(
        destination.mode is M2NDestinationMode.SHARDED_STAGING
        for destination in destinations
    )
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
        "the model, %d via sharded staging, %d via full-tensor fallback; "
        "direct byte coverage: "
        "%d/%d bytes (%.1f%%)",
        num_direct,
        len(destinations),
        num_staged,
        len(destinations) - num_direct - num_staged,
        direct_bytes,
        total_bytes,
        direct_byte_percentage,
    )
    return destinations
