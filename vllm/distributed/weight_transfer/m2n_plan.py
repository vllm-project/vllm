# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Canonical destination-plan agreement for NCCL M2N."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from vllm.distributed.weight_transfer.m2n_common import (
    DESTINATION_SHARD_AXIS,
    M2N_WIRE_SCHEMA_VERSION,
    REPLICATE,
    M2NDestinationMode,
    M2NMesh,
    M2NWireParam,
    Placements,
    check_placements,
    source_plan_digest,
)

if TYPE_CHECKING:
    from vllm.distributed.utils import StatelessProcessGroup
    from vllm.distributed.weight_transfer.m2n_layout import M2NDestination


@dataclass(frozen=True)
class M2NWireDestination:
    """JSON-safe, callback-free destination descriptor for one worker."""

    name: str
    mode: str
    dtype_name: str
    placements: Placements | None
    local_shape: tuple[int, ...]
    semantic_id: str | None
    shard_dim: int | None
    shard_index: int
    num_shards: int
    staging_group: int | None = None
    staging_slot: int | None = None
    staging_group_size: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not isinstance(self.dtype_name, str):
            raise TypeError("destination name and dtype must be strings")
        if not self.name or not self.dtype_name:
            raise ValueError("destination name and dtype must be non-empty")
        if not isinstance(self.mode, str):
            raise TypeError("destination mode must be a string")
        M2NDestinationMode(self.mode)
        if self.placements is not None:
            object.__setattr__(self, "placements", tuple(self.placements))
            check_placements(
                self.placements, f"parameter '{self.name}' destination placements"
            )
        object.__setattr__(self, "local_shape", tuple(self.local_shape))
        if not self.local_shape or any(
            not isinstance(dim, int) or isinstance(dim, bool) or dim <= 0
            for dim in self.local_shape
        ):
            raise ValueError("destination local_shape must contain positive integers")
        for field, value in (
            ("shard_index", self.shard_index),
            ("num_shards", self.num_shards),
        ):
            if not isinstance(value, int) or isinstance(value, bool):
                raise TypeError(f"destination {field} must be an integer")
        if self.num_shards <= 0 or not 0 <= self.shard_index < self.num_shards:
            raise ValueError("destination shard coordinate is invalid")
        if self.shard_dim is not None and (
            not isinstance(self.shard_dim, int) or isinstance(self.shard_dim, bool)
        ):
            raise TypeError("destination shard_dim must be an integer or null")
        if self.shard_dim is not None and not (
            0 <= self.shard_dim < len(self.local_shape)
        ):
            raise ValueError("destination shard_dim is outside local_shape")
        staged = self.mode == M2NDestinationMode.SHARDED_STAGING.value
        staging_coordinates = (
            self.staging_group,
            self.staging_slot,
            self.staging_group_size,
        )
        if staged:
            if self.placements is None or self.shard_dim is None:
                raise ValueError("a staged destination must be sharded")
            if not isinstance(self.semantic_id, str) or not self.semantic_id:
                raise ValueError("a staged destination needs a semantic_id")
            if any(coordinate is None for coordinate in staging_coordinates):
                raise ValueError("a staged destination needs staging coordinates")
            for field, coordinate in (
                ("staging_group", self.staging_group),
                ("staging_slot", self.staging_slot),
                ("staging_group_size", self.staging_group_size),
            ):
                if (
                    not isinstance(coordinate, int)
                    or isinstance(coordinate, bool)
                    or coordinate < 0
                ):
                    raise ValueError(f"{field} must be a non-negative integer")
            assert self.staging_slot is not None
            assert self.staging_group_size is not None
            if self.staging_group_size == 0:
                raise ValueError("staging_group_size must be positive")
            if self.staging_slot >= self.staging_group_size:
                raise ValueError("staging_slot is outside its group")
        elif self.semantic_id is not None or any(
            coordinate is not None for coordinate in staging_coordinates
        ):
            raise ValueError(
                "only staged destinations carry a semantic_id or coordinates"
            )
        if self.placements is None:
            if (
                self.shard_dim is not None
                or self.shard_index != 0
                or self.num_shards != 1
            ):
                raise ValueError("a replicated destination has shard metadata")
        elif self.shard_dim is None:
            raise ValueError("a sharded destination needs shard_dim")
        if (
            self.mode == M2NDestinationMode.FULL_FALLBACK.value
            and self.placements is not None
        ):
            raise ValueError("a full fallback must be replicated")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> M2NWireDestination:
        fields = {
            "name",
            "mode",
            "dtype_name",
            "placements",
            "local_shape",
            "semantic_id",
            "shard_dim",
            "shard_index",
            "num_shards",
            "staging_group",
            "staging_slot",
            "staging_group_size",
        }
        missing = fields - value.keys()
        extra = value.keys() - fields
        if missing or extra:
            details = []
            if missing:
                details.append(f"missing {sorted(missing)}")
            if extra:
                details.append(f"unexpected {sorted(extra)}")
            raise ValueError(f"invalid M2N wire destination: {', '.join(details)}")
        return cls(**value)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "mode": self.mode,
            "dtype_name": self.dtype_name,
            "placements": (None if self.placements is None else list(self.placements)),
            "local_shape": list(self.local_shape),
            "semantic_id": self.semantic_id,
            "shard_dim": self.shard_dim,
            "shard_index": self.shard_index,
            "num_shards": self.num_shards,
            "staging_group": self.staging_group,
            "staging_slot": self.staging_slot,
            "staging_group_size": self.staging_group_size,
        }

    @classmethod
    def from_destination(
        cls,
        destination: M2NDestination,
        *,
        dtype_name: str,
        destination_shard_index: int,
        shard_axis_size: int,
    ) -> M2NWireDestination:
        """Strip live state from one worker-local destination plan."""
        placements = destination.placements
        shard_dim = None
        shard_index = 0
        num_shards = 1
        semantic_id = None
        if placements is not None:
            shard_dim = placements[DESTINATION_SHARD_AXIS]
            shard_index = destination_shard_index
            num_shards = shard_axis_size
        if destination.mode is M2NDestinationMode.SHARDED_STAGING:
            assert destination.sharded_spec is not None
            spec = destination.sharded_spec
            semantic_id = spec.semantic_id
            shard_dim = spec.shard_dim
            shard_index = spec.shard_index
            num_shards = spec.num_shards
        return cls(
            name=destination.name,
            mode=destination.mode.value,
            dtype_name=dtype_name,
            placements=placements,
            local_shape=destination.local_shape,
            semantic_id=semantic_id,
            shard_dim=shard_dim,
            shard_index=shard_index,
            num_shards=num_shards,
            staging_group=destination.staging_group,
            staging_slot=destination.staging_slot,
            staging_group_size=destination.staging_group_size,
        )


def _parse_worker_plan(
    value: Any,
    *,
    rank: int,
    params: Sequence[M2NWireParam],
) -> list[M2NWireDestination]:
    if not isinstance(value, list):
        raise TypeError(f"worker rank {rank} destination plan must be a list")
    if len(value) != len(params):
        raise ValueError(
            f"worker rank {rank} planned {len(value)} parameters, expected "
            f"{len(params)}"
        )
    plan: list[M2NWireDestination] = []
    for index, (record, param) in enumerate(zip(value, params)):
        if not isinstance(record, Mapping):
            raise TypeError(f"worker rank {rank} destination {index} must be a mapping")
        destination = M2NWireDestination.from_dict(record)
        if destination.name != param.name:
            raise ValueError(
                f"worker rank {rank} planned {destination.name!r} at index "
                f"{index}, expected {param.name!r}"
            )
        if destination.dtype_name != param.dtype_name:
            raise ValueError(
                f"worker rank {rank} destination dtype for {param.name!r} is "
                f"{destination.dtype_name}, expected {param.dtype_name}"
            )
        if (
            destination.mode == M2NDestinationMode.FULL_FALLBACK.value
            and not param.allow_full_fallback
        ):
            raise ValueError(
                f"worker rank {rank} selected a forbidden full fallback for "
                f"{param.name!r}"
            )
        plan.append(destination)
    return plan


def _validate_geometry(
    destination: M2NWireDestination,
    param: M2NWireParam,
    dst_mesh: M2NMesh,
    worker_rank: int,
) -> None:
    if not dst_mesh.start_rank <= worker_rank < dst_mesh.start_rank + dst_mesh.size:
        raise ValueError(f"rank {worker_rank} is outside the destination mesh")
    local_worker_rank = worker_rank - dst_mesh.start_rank
    expected_shard_index = local_worker_rank % dst_mesh.dims[DESTINATION_SHARD_AXIS]
    if destination.placements is None:
        if (
            destination.local_shape != param.shape
            or destination.shard_dim is not None
            or destination.num_shards != 1
            or destination.shard_index != 0
        ):
            raise ValueError(
                f"worker rank {worker_rank} has invalid replicated geometry "
                f"for {param.name!r}"
            )
        return

    placements = destination.placements
    if placements[0] != REPLICATE:
        raise ValueError(
            f"worker rank {worker_rank} has unsupported destination placements "
            f"for {param.name!r}: {placements}"
        )
    shard_dim = placements[DESTINATION_SHARD_AXIS]
    num_shards = dst_mesh.dims[DESTINATION_SHARD_AXIS]
    if (
        destination.shard_dim != shard_dim
        or destination.num_shards != num_shards
        or destination.shard_index != expected_shard_index
    ):
        raise ValueError(
            f"worker rank {worker_rank} has invalid shard coordinates for "
            f"{param.name!r}"
        )
    expected_shape = list(param.shape)
    if param.shape[shard_dim] % num_shards:
        raise ValueError(f"destination shards do not evenly divide {param.name!r}")
    expected_shape[shard_dim] //= num_shards
    if destination.local_shape != tuple(expected_shape):
        raise ValueError(
            f"worker rank {worker_rank} has local shape "
            f"{destination.local_shape} for {param.name!r}, expected "
            f"{tuple(expected_shape)}"
        )


def _validate_staging_groups(
    plan: Sequence[M2NWireDestination], worker_rank: int
) -> None:
    """Require complete, contiguous semantic groups in canonical order."""
    closed: set[int] = set()
    active_group: int | None = None
    active_slots: list[int] = []
    active_size = 0

    def close() -> None:
        nonlocal active_group, active_size
        if active_group is None:
            return
        if active_slots != list(range(active_size)):
            raise ValueError(
                f"worker rank {worker_rank} has incomplete staging group "
                f"{active_group}: slots {active_slots}, expected "
                f"{list(range(active_size))}"
            )
        closed.add(active_group)
        active_group = None
        active_slots.clear()
        active_size = 0

    for destination in plan:
        group = destination.staging_group
        if group is None:
            close()
            continue
        assert destination.staging_slot is not None
        assert destination.staging_group_size is not None
        if group != active_group:
            close()
            if group in closed:
                raise ValueError(
                    f"worker rank {worker_rank} reopens staging group {group}"
                )
            active_group = group
            active_size = destination.staging_group_size
        elif destination.staging_group_size != active_size:
            raise ValueError(
                f"worker rank {worker_rank} staging group {group} "
                "disagrees on its required size"
            )
        active_slots.append(destination.staging_slot)
    close()
    if closed and closed != set(range(len(closed))):
        raise ValueError(
            f"worker rank {worker_rank} staging groups are not densely numbered"
        )


def _validate_plan_set(
    plans: Sequence[tuple[int, list[M2NWireDestination]]],
    params: Sequence[M2NWireParam],
    dst_mesh: M2NMesh,
) -> None:
    expected_ranks = list(
        range(dst_mesh.start_rank, dst_mesh.start_rank + dst_mesh.size)
    )
    actual_ranks = [rank for rank, _ in plans]
    if actual_ranks != expected_ranks:
        raise ValueError(
            f"destination plan ranks {actual_ranks} do not match mesh ranks "
            f"{expected_ranks}"
        )
    reference = plans[0][1]
    invariant_fields = (
        "mode",
        "placements",
        "semantic_id",
        "staging_group",
        "staging_slot",
        "staging_group_size",
    )
    for worker_rank, plan in plans:
        _validate_staging_groups(plan, worker_rank)
        for index, (destination, param) in enumerate(zip(plan, params)):
            _validate_geometry(destination, param, dst_mesh, worker_rank)
            expected = reference[index]
            differing = [
                field
                for field in invariant_fields
                if getattr(destination, field) != getattr(expected, field)
            ]
            if differing:
                raise ValueError(
                    f"worker rank {worker_rank} disagrees on {param.name!r} "
                    f"destination fields {differing}"
                )


def agree_destination_plan(
    group: StatelessProcessGroup,
    *,
    source_digest: str,
    params: Sequence[M2NWireParam],
    dst_mesh: M2NMesh,
    local_plan: Sequence[M2NWireDestination] | None,
    local_error: str | None,
) -> tuple[list[M2NWireDestination], str]:
    """All-gather worker plans and issue a unanimous pre-NCCL commit token."""
    prepared_error = local_error
    serialized_plan: list[dict[str, Any]] | None = None
    if prepared_error is None:
        try:
            if not isinstance(source_digest, str) or not source_digest:
                raise ValueError("source_digest must be a non-empty string")
            if not params:
                raise ValueError("source parameter manifest must not be empty")
            if source_plan_digest(params) != source_digest:
                raise ValueError("source_digest does not match local parameters")
            if group.world_size != dst_mesh.start_rank + dst_mesh.size:
                raise ValueError(
                    "destination mesh does not cover the tail of the group"
                )
            if local_plan is not None:
                serialized_plan = [item.to_dict() for item in local_plan]
        except Exception as exc:
            prepared_error = f"{type(exc).__name__}: {exc}"
    envelope = {
        "ok": prepared_error is None,
        "error": prepared_error,
        "plan": serialized_plan,
        "source_digest": source_digest,
        "dst_mesh_dims": list(dst_mesh.dims),
        "dst_mesh_start_rank": dst_mesh.start_rank,
    }
    gathered = group.all_gather_obj(envelope)
    errors: list[str] = []
    declarations: list[tuple[int, str, M2NMesh]] = []
    expected_fields = {
        "ok",
        "error",
        "plan",
        "source_digest",
        "dst_mesh_dims",
        "dst_mesh_start_rank",
    }
    if len(gathered) != group.world_size:
        errors.append(
            "control-plane all-gather returned "
            f"{len(gathered)} records for world size {group.world_size}"
        )
    for rank, item in enumerate(gathered):
        if not isinstance(item, Mapping):
            errors.append(f"rank {rank}: invalid preflight envelope")
            continue
        if set(item) != expected_fields:
            errors.append(f"rank {rank}: invalid preflight envelope fields")
            continue
        if not isinstance(item["ok"], bool):
            errors.append(f"rank {rank}: invalid preflight status")
            continue
        error = item["error"]
        if item["ok"] != (error is None) or (
            error is not None and (not isinstance(error, str) or not error)
        ):
            errors.append(f"rank {rank}: invalid preflight error")
            continue
        if item["ok"] is not True:
            errors.append(f"rank {rank}: {error}")
            continue
        try:
            declared_digest = item["source_digest"]
            if not isinstance(declared_digest, str) or not declared_digest:
                raise ValueError("source_digest must be a non-empty string")
            declared_mesh = M2NMesh(
                tuple(item["dst_mesh_dims"]), item["dst_mesh_start_rank"]
            )
            declarations.append((rank, declared_digest, declared_mesh))
        except Exception as exc:
            errors.append(f"rank {rank}: {type(exc).__name__}: {exc}")
    if errors:
        raise RuntimeError(
            "nccl_m2n destination preflight rejected before NCCL init: "
            + "; ".join(errors)
        )

    _, canonical_digest, canonical_mesh = declarations[0]
    for rank, declared_digest, declared_mesh in declarations[1:]:
        if declared_digest != canonical_digest:
            errors.append(f"rank {rank}: source_digest disagrees with rank 0")
        if declared_mesh != canonical_mesh:
            errors.append(f"rank {rank}: destination mesh disagrees with rank 0")
    if group.world_size != canonical_mesh.start_rank + canonical_mesh.size:
        errors.append("agreed destination mesh does not cover the group tail")
    if not 0 < canonical_mesh.start_rank < group.world_size:
        errors.append("agreed destination mesh must leave trainers and workers")
    if errors:
        raise RuntimeError(
            "nccl_m2n destination preflight rejected before NCCL init: "
            + "; ".join(errors)
        )

    worker_plans: list[tuple[int, list[M2NWireDestination]]] = []
    for rank, item in enumerate(gathered):
        raw_plan = item["plan"]
        if rank < canonical_mesh.start_rank:
            if raw_plan is not None:
                errors.append(f"trainer rank {rank} published a destination plan")
            continue
        try:
            plan = _parse_worker_plan(raw_plan, rank=rank, params=params)
            worker_plans.append((rank, plan))
        except Exception as exc:
            errors.append(f"rank {rank}: {type(exc).__name__}: {exc}")
    if errors:
        raise RuntimeError(
            "nccl_m2n destination preflight rejected before NCCL init: "
            + "; ".join(errors)
        )
    _validate_plan_set(worker_plans, params, canonical_mesh)

    payload = {
        "schema_version": M2N_WIRE_SCHEMA_VERSION,
        "source_digest": canonical_digest,
        "dst_mesh_dims": list(canonical_mesh.dims),
        "dst_mesh_start_rank": canonical_mesh.start_rank,
        "workers": [
            {
                "rank": rank,
                "plan": [destination.to_dict() for destination in plan],
            }
            for rank, plan in worker_plans
        ],
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return worker_plans[0][1], f"m2n-plan:{digest}"
