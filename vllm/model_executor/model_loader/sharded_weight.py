# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Transport-neutral contracts for loading an already-sharded weight.

The resolver in this module deliberately knows nothing about a transfer
backend.  A backend supplies a logical checkpoint name, dtype, and global
shape; a model module may describe the rank-local logical shard it can safely
consume through its native loading path.

Targets are update-scoped.  Resolve them after the model has entered its
weight-update lifecycle, and do not cache target callbacks or tensor pointers
across updates.
"""

from collections.abc import Callable
from dataclasses import dataclass, field

import torch

__all__ = [
    "ShardedWeightRequest",
    "ShardedWeightSpec",
    "ShardedWeightTarget",
    "resolve_sharded_weight_target",
]


@dataclass(frozen=True)
class ShardedWeightRequest:
    """Logical checkpoint tensor offered to model-side resolvers."""

    name: str
    dtype: torch.dtype
    global_shape: tuple[int, ...]

    def __post_init__(self) -> None:
        if (
            not self.name
            or self.name.startswith(".")
            or self.name.endswith(".")
            or ".." in self.name
        ):
            raise ValueError("name must be a non-empty qualified tensor name")
        if not isinstance(self.dtype, torch.dtype):
            raise TypeError("dtype must be a torch.dtype")
        if any(
            not isinstance(dim, int) or isinstance(dim, bool) or dim <= 0
            for dim in self.global_shape
        ):
            raise ValueError("global_shape must contain positive integer dimensions")


@dataclass(frozen=True)
class ShardedWeightSpec:
    """Rank-local shard geometry, independent of any transfer backend."""

    semantic_id: str
    dtype: torch.dtype
    shard_dim: int
    local_shape: tuple[int, ...]
    shard_index: int
    num_shards: int

    # TODO: Add explicit ownership indices for EPLB, expert replication, or
    # custom placement, where shard_index and local_shape are insufficient.

    def __post_init__(self) -> None:
        if not self.semantic_id:
            raise ValueError("semantic_id must be non-empty")
        if not isinstance(self.dtype, torch.dtype):
            raise TypeError("dtype must be a torch.dtype")
        if not self.local_shape or any(
            not isinstance(dim, int) or isinstance(dim, bool) or dim <= 0
            for dim in self.local_shape
        ):
            raise ValueError("local_shape must contain positive integer dimensions")
        if self.shard_dim < 0 or self.shard_dim >= len(self.local_shape):
            raise ValueError(
                f"shard_dim {self.shard_dim} is invalid for local shape "
                f"{self.local_shape}"
            )
        if self.num_shards <= 0:
            raise ValueError("num_shards must be positive")
        if not 0 <= self.shard_index < self.num_shards:
            raise ValueError(
                f"shard_index {self.shard_index} must be in [0, {self.num_shards})"
            )


@dataclass(frozen=True, eq=False)
class ShardedWeightTarget:
    """A rank-local logical destination and its model-native consumer.

    ``retention_key`` groups input buffers whose views may remain referenced
    until the owning layer finishes loading.  A transfer engine must retain
    all buffers in that group until ``load`` returns ``True`` or the update is
    finalized.
    """

    spec: ShardedWeightSpec
    retention_key: object = field(repr=False)
    consume: Callable[[torch.Tensor], bool] = field(repr=False)
    retention_group_size: int = 1

    def __post_init__(self) -> None:
        if (
            not isinstance(self.retention_group_size, int)
            or isinstance(self.retention_group_size, bool)
            or self.retention_group_size <= 0
        ):
            raise ValueError("retention_group_size must be a positive integer")

    def load(self, weight: torch.Tensor) -> bool:
        """Validate and consume one rank-local logical shard."""
        if weight.dtype != self.spec.dtype:
            raise ValueError(
                f"local shard dtype {weight.dtype} does not match "
                f"target dtype {self.spec.dtype}"
            )
        if tuple(weight.shape) != self.spec.local_shape:
            raise ValueError(
                f"local shard shape {tuple(weight.shape)} does not match "
                f"target shape {self.spec.local_shape}"
            )
        result = self.consume(weight)
        if not isinstance(result, bool):
            raise TypeError("sharded weight consumer returned an invalid result")
        return result


def _validate_target(
    request: ShardedWeightRequest,
    target: ShardedWeightTarget,
) -> None:
    if target.spec.dtype != request.dtype:
        raise ValueError(
            f"target for {request.name!r} has dtype {target.spec.dtype}, "
            f"expected {request.dtype}"
        )
    local_shape = target.spec.local_shape
    if len(local_shape) != len(request.global_shape):
        raise ValueError(
            f"target for {request.name!r} has rank {len(local_shape)}, "
            f"expected {len(request.global_shape)}"
        )
    shard_dim = target.spec.shard_dim
    for dim, (local, global_) in enumerate(zip(local_shape, request.global_shape)):
        expected = local * target.spec.num_shards if dim == shard_dim else local
        if expected != global_:
            raise ValueError(
                f"target for {request.name!r} has incompatible local shape "
                f"{local_shape} for global shape {request.global_shape}"
            )
    if target.retention_key is None:
        raise ValueError(f"target for {request.name!r} has no retention key")
    try:
        hash(target.retention_key)
    except TypeError as error:
        raise ValueError(
            f"target for {request.name!r} has an unhashable retention key"
        ) from error


def resolve_sharded_weight_target(
    model: torch.nn.Module,
    request: ShardedWeightRequest,
) -> ShardedWeightTarget | None:
    """Resolve a logical checkpoint tensor through the nearest model module.

    A module may implement ``resolve_sharded_weight_target(relative_name,
    request)`` to return a target. Matching ancestors are visited
    deepest-first. A deeper provider may decline a name by returning ``None``,
    after which its parents are tried. Provider errors propagate so a
    recognized but unsupported semantic tensor cannot fall back silently.
    """
    parts = request.name.split(".")
    for depth in range(len(parts), -1, -1):
        prefix = ".".join(parts[:depth])
        relative_name = ".".join(parts[depth:])
        try:
            module = model.get_submodule(prefix)
        except AttributeError:
            continue
        resolver = getattr(module, "resolve_sharded_weight_target", None)
        if not callable(resolver):
            continue
        target = resolver(relative_name, request)
        if target is None:
            continue
        if not isinstance(target, ShardedWeightTarget):
            raise TypeError(
                f"provider {prefix or '<root>'!r} returned an invalid "
                f"target for {request.name!r}"
            )
        _validate_target(request, target)
        return target
    return None
