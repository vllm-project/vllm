# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared helpers for the NCCL M2N (`nccl_m2n`) weight transfer backend.

M2N reshards a tensor between two disjoint meshes of ranks that live in one
communicator. Each transfer route places one ordered trainer source group at
`[0, T)` and an ordered destination group at `[T, T + N)`. This module holds
everything both sides need — the optional runtime import, topology/layout
descriptors, and conversion to `nccl.m2n` types — so engine modules stay about
transfer execution.
"""

import hashlib
import json
import math
import os
from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any, cast

import torch

from vllm.distributed.weight_transfer.base import ParamMeta
from vllm.utils.nccl import unpinned_nccl_env

if TYPE_CHECKING:
    from vllm.distributed.utils import StatelessProcessGroup

# Placement code for a mesh axis that replicates. Any other (non-negative) code
# is the tensor dim that axis shards. Ints rather than `nccl.m2n` objects so
# layouts survive the JSON init handshake without a custom encoder.
REPLICATE = -1

# `NCCL_RESHARD_MAX_TENSOR_DIMS`; 4-D and higher are rejected by the library.
MAX_TENSOR_DIMS = 3

MESH_NDIMS = 2

# Destination mesh convention. Keeping these roles named avoids scattering
# positional axis assumptions through the worker-side planner.
DESTINATION_REPLICA_AXIS = 0
DESTINATION_SHARD_AXIS = 1

# Conservative capacities accepted by the supported nccl_m2n algorithms. They
# are not exposed by its Python API yet; remove these mirrors once nccl_m2n
# provides communication-free plan validation.
MAX_SOURCE_SHARDS = 16
MAX_DEST_SHARDS = 64

# Increment when the JSON init payload changes incompatibly. M2N peers must
# agree on this before any rank constructs a NCCL communicator.
M2N_WIRE_SCHEMA_VERSION = 4

_MULTI_COMM_LAUNCH_ENV = "NCCL_LAUNCH_ORDER_IMPLICIT"


class M2NDestinationMode(str, Enum):
    """How one logical checkpoint tensor is installed on a worker."""

    IN_PLACE = "in_place"
    SHARDED_STAGING = "sharded_staging"
    FULL_FALLBACK = "full_fallback"


def m2n_load_group(
    mode: M2NDestinationMode | str, staging_group: int | None
) -> int | None:
    """Return the ordered model-load group, or ``None`` for in-place loads."""
    resolved = M2NDestinationMode(mode)
    if resolved is M2NDestinationMode.IN_PLACE:
        return None
    if resolved is M2NDestinationMode.FULL_FALLBACK:
        # The model's fallback loader has no out-of-order input contract.
        return -1
    if not isinstance(staging_group, int) or isinstance(staging_group, bool):
        raise ValueError("a staged destination needs an integer staging group")
    return staging_group


SUPPORTED_DTYPES: frozenset[torch.dtype] = frozenset(
    dtype
    for dtype in (
        getattr(torch, name, None)
        for name in (
            "int8",
            "uint8",
            "float8_e4m3fn",
            "float8_e5m2",
            "float16",
            "bfloat16",
            "int32",
            "uint32",
            "float32",
            "int64",
            "uint64",
            "float64",
        )
    )
    if dtype is not None
)

_IMPORT_HINT = (
    "The nccl_m2n weight transfer backend requires the `nccl-extensions` "
    "package (and its `nccl4py` dependency), which is not installed by vLLM. "
    "See https://github.com/NVIDIA/nccl-extensions for build "
    "and install instructions. It needs NCCL 2.30.5 or newer."
)


def prepare_m2n_local_runtime(m2n: Any, max_cta: int | None) -> tuple[Any, int, Any]:
    """Prepare rank-local resources before any rank enters NCCL init."""
    import nccl.core as nccl

    device = torch.accelerator.current_device_index()
    if not isinstance(device, int) or isinstance(device, bool) or device < 0:
        raise RuntimeError(f"invalid current accelerator device index: {device!r}")
    handle = m2n.Handle.create(m2n.Config(max_cta=max_cta))
    return nccl, device, handle


def check_multi_communicator_launch_order(num_routes: int) -> None:
    """Require NCCL's ordered multi-communicator launch mode when needed.

    Pipeline transfers use one NCCL communicator and CUDA stream per route.
    NCCL requires implicit launch ordering for overlapping
    operations on independent communicators. This is process-global NCCL
    configuration, so callers must set it before starting the process rather
    than mutating the environment after NCCL may already have initialized.
    """
    if (
        not isinstance(num_routes, int)
        or isinstance(num_routes, bool)
        or num_routes <= 0
    ):
        raise ValueError("nccl_m2n needs at least one transfer route")
    if num_routes > 1 and os.environ.get(_MULTI_COMM_LAUNCH_ENV) != "1":
        value = os.environ.get(_MULTI_COMM_LAUNCH_ENV)
        raise RuntimeError(
            f"{num_routes} nccl_m2n transfer routes use independent NCCL "
            "communicators and streams; set NCCL_LAUNCH_ORDER_IMPLICIT=1 "
            f"before starting every process (current value: {value!r})"
        )


def create_m2n_communicator(
    nccl: Any,
    group: "StatelessProcessGroup",
    unique_id_bytes: bytes | None,
    *,
    members: Sequence[int] | None = None,
) -> Any | None:
    """Create one M2N data-plane communicator through NCCL4Py.

    ``members`` uses metadata-group ranks in communicator-local order. Every
    metadata rank must call this helper for each route, including nonmembers, so
    automatic UID bootstrap advances the shared control plane identically.
    Nonmembers return ``None`` after that bootstrap.
    """
    ordered_members = (
        tuple(range(group.world_size)) if members is None else tuple(members)
    )
    if not ordered_members:
        raise ValueError("M2N communicator members must not be empty")
    if any(
        not isinstance(rank, int)
        or isinstance(rank, bool)
        or not 0 <= rank < group.world_size
        for rank in ordered_members
    ):
        raise ValueError(
            "M2N communicator members must be metadata ranks in "
            f"[0, {group.world_size})"
        )
    if len(set(ordered_members)) != len(ordered_members):
        raise ValueError("M2N communicator members must be unique")

    if unique_id_bytes is None:
        envelope: dict[str, bytes | str | None] | None = None
        uid_root = ordered_members[0]
        if group.rank == uid_root:
            try:
                envelope = {
                    "unique_id": bytes(nccl.get_unique_id().as_bytes),
                    "error": None,
                }
            except Exception as exc:
                envelope = {
                    "unique_id": None,
                    "error": f"{type(exc).__name__}: {exc}",
                }
        envelope = group.broadcast_obj(envelope, src=uid_root)
        if not isinstance(envelope, Mapping) or set(envelope) != {"unique_id", "error"}:
            raise RuntimeError("invalid NCCL4Py unique-id envelope")
        error = envelope["error"]
        received_unique_id = envelope["unique_id"]
        if error is not None:
            if not isinstance(error, str) or not error:
                raise RuntimeError("invalid NCCL4Py unique-id error")
            raise RuntimeError(f"NCCL4Py unique-id generation failed: {error}")
        if not isinstance(received_unique_id, bytes):
            raise RuntimeError("NCCL4Py unique-id broadcast did not return bytes")
        unique_id_bytes = received_unique_id
    elif not isinstance(unique_id_bytes, bytes):
        raise TypeError("NCCL unique id must be bytes or None")

    if group.rank not in ordered_members:
        return None

    unique_id = nccl.UniqueId.from_bytes(unique_id_bytes)
    with unpinned_nccl_env():
        return nccl.Communicator.init(
            nranks=len(ordered_members),
            rank=ordered_members.index(group.rank),
            unique_id=unique_id,
        )


def import_m2n() -> Any:
    """Import `nccl.m2n` lazily, with an actionable error when it is missing.

    Deferred so that importing vLLM — or any other weight transfer backend —
    never requires the m2n runtime to be present.
    """
    try:
        import nccl.m2n as m2n
    except ImportError as e:
        raise ImportError(f"{_IMPORT_HINT} (import failed: {e})") from e
    return m2n


@dataclass(frozen=True)
class M2NMesh:
    """One side's rank topology — pure topology, no tensor placement.

    Mirrors `ncclMesh_t`: a 2-axis mesh owning the contiguous rank interval
    `[start_rank, start_rank + dims[0] * dims[1])`. There is no 1-D mesh; a
    single-axis topology is spelled with a second axis of size 1.

    A model may use a different factorization for each tensor. For example,
    dense weights can use a DP x TP mesh while expert weights use EDP x EP.
    """

    dims: tuple[int, int]
    start_rank: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "dims", tuple(self.dims))
        if len(self.dims) != MESH_NDIMS or any(
            not isinstance(dim, int) or isinstance(dim, bool) or dim <= 0
            for dim in self.dims
        ):
            raise ValueError(f"mesh dims must be {MESH_NDIMS} positive ints")
        if (
            not isinstance(self.start_rank, int)
            or isinstance(self.start_rank, bool)
            or self.start_rank < 0
        ):
            raise ValueError(
                f"mesh start_rank must be non-negative, got {self.start_rank}"
            )

    @property
    def size(self) -> int:
        return self.dims[0] * self.dims[1]


# A tensor's placement over its side's mesh: one code per mesh axis
# (`REPLICATE`, or the tensor dim that axis shards), or `REPLICATED` for a
# tensor every rank holds in full.
Placements = tuple[int, int]
REPLICATED: Placements | None = None


@dataclass(frozen=True)
class M2NLayout:
    """A tensor's mesh and placement on one side of a transfer."""

    mesh: M2NMesh
    placements: Placements | None

    def __post_init__(self) -> None:
        if not isinstance(self.mesh, M2NMesh):
            raise TypeError("M2N layout mesh must be an M2NMesh")
        if self.placements is not None:
            object.__setattr__(self, "placements", tuple(self.placements))


def check_placements(placements: Placements, context: str = "placements") -> None:
    """Reject placement pairs nccl_m2n cannot express."""
    if len(placements) != MESH_NDIMS:
        raise ValueError(f"{context} must have {MESH_NDIMS} entries")
    if any(not isinstance(code, int) or isinstance(code, bool) for code in placements):
        raise TypeError(f"{context} must contain integer placement codes")
    invalid = [code for code in placements if code < REPLICATE]
    if invalid:
        raise ValueError(
            f"{context} contains invalid placement code {invalid[0]}; valid "
            f"codes are {REPLICATE} (Replicate) and non-negative tensor dimensions"
        )
    num_sharded = sum(code != REPLICATE for code in placements)
    if num_sharded == 0:
        raise ValueError(
            f"{context}: a fully replicated tensor is carried as REPLICATED, "
            "not as two REPLICATE axes"
        )
    if num_sharded == MESH_NDIMS:
        raise ValueError(
            f"{context}: nccl_m2n needs one REPLICATE mesh axis, but "
            f"{placements} shards both"
        )


def resolve_layout(
    mesh: M2NMesh,
    placements: Placements | None,
    context: str = "placements",
) -> tuple[M2NMesh, Placements]:
    """Pair one tensor's placement with the mesh m2n should see for it.

    A replicated tensor needs a size-1 mesh axis to carry a no-op shard, since
    m2n has no `{REPLICATE, REPLICATE}`. It is therefore described over the
    *same rank interval* re-factored as `(size, 1)`. That re-factoring is sound
    precisely because replication is order-independent — every rank holds the
    whole tensor, so it does not matter that `(a, b)` and `(size, 1)` walk the
    interval in a different order. A sharded tensor keeps its side's own
    factorization, where rank order decides who owns which shard.
    """
    if placements is None:
        return M2NMesh((mesh.size, 1), mesh.start_rank), (REPLICATE, 0)
    check_placements(placements, context)
    return mesh, placements


def validate_layout(
    mesh: M2NMesh,
    placements: Placements,
    shape: Sequence[int],
    context: str,
) -> tuple[int, ...]:
    """Validate one resolved layout and return its required local shape."""
    local_shape = list(shape)
    for axis, tensor_dim in enumerate(placements):
        if tensor_dim == REPLICATE:
            continue
        if tensor_dim >= len(shape):
            raise ValueError(
                f"{context} shards tensor dim {tensor_dim}, but parameter has "
                f"rank {len(shape)}"
            )
        factor = mesh.dims[axis]
        if shape[tensor_dim] % factor:
            raise ValueError(
                f"{context} shards dim {tensor_dim} (size {shape[tensor_dim]}) "
                f"over {factor} ranks, which does not divide evenly"
            )
        local_shape[tensor_dim] //= factor
    return tuple(local_shape)


def expected_local_shape(meta: "M2NParamMeta") -> tuple[int, ...]:
    """Validate a parameter's source layout and return its local shape."""
    mesh, placements = resolve_layout(
        meta.source_layout.mesh,
        meta.source_layout.placements,
        f"parameter '{meta.name}' source placements",
    )
    return validate_layout(
        mesh, placements, meta.shape, f"parameter '{meta.name}' source layout"
    )


def check_transferable(name: str, dtype: torch.dtype, shape: Sequence[int]) -> None:
    """Reject parameters nccl_m2n cannot transfer."""
    if dtype not in SUPPORTED_DTYPES:
        raise ValueError(
            f"parameter '{name}' has dtype {dtype}, which nccl_m2n does not "
            f"support. Supported: {sorted(str(d) for d in SUPPORTED_DTYPES)}"
        )
    if not 1 <= len(shape) <= MAX_TENSOR_DIMS:
        raise ValueError(
            f"parameter '{name}' has rank {len(shape)}; nccl_m2n supports "
            f"rank 1..{MAX_TENSOR_DIMS}"
        )


def _shard_spec(mesh: M2NMesh, placements: Placements) -> tuple[int, int]:
    return next(
        (code, mesh.dims[axis])
        for axis, code in enumerate(placements)
        if code != REPLICATE
    )


def _max_partition_overlaps(
    extent: int,
    partitions: int,
    other_partitions: int,
) -> int:
    """Return the maximum other partitions intersected by any one partition."""
    if extent == 0:
        return 0
    width = extent // partitions
    other_width = extent // other_partitions
    max_overlaps = 0
    for index in range(partitions):
        start = index * width
        end = start + width
        first_other = start // other_width
        last_other = (end - 1) // other_width
        max_overlaps = max(max_overlaps, last_other - first_other + 1)
    return max_overlaps


def check_plan_limits(
    src: tuple[M2NMesh, Placements],
    dst: tuple[M2NMesh, Placements],
    name: str,
    shape: Sequence[int],
) -> None:
    """Reject plans whose per-shard fan-in or fan-out exceeds M2N capacity."""
    src_dim, src_shards = _shard_spec(*src)
    dst_dim, dst_shards = _shard_spec(*dst)
    if any(extent == 0 for extent in shape):
        max_sources = max_targets = 0
    elif src_dim != dst_dim:
        max_sources = src_shards
        max_targets = dst_shards
    else:
        extent = shape[src_dim]
        max_sources = _max_partition_overlaps(extent, dst_shards, src_shards)
        max_targets = _max_partition_overlaps(extent, src_shards, dst_shards)

    if max_sources > MAX_SOURCE_SHARDS:
        raise ValueError(
            f"parameter '{name}' would feed up to {max_sources} source shards "
            f"into one destination shard, over nccl_m2n MAX_SOURCES="
            f"{MAX_SOURCE_SHARDS}"
        )
    if max_targets > MAX_DEST_SHARDS:
        raise ValueError(
            f"parameter '{name}' would fan one source shard out to up to "
            f"{max_targets} destination shards, over nccl_m2n "
            f"MAX_TARGETS={MAX_DEST_SHARDS}"
        )


def validate_transfer_plan(
    metas: Sequence["M2NParamMeta"],
    dst_mesh: M2NMesh,
    dst_placements: Sequence[Placements | None],
) -> None:
    """Validate the complete ordered source-to-destination transfer plan."""
    if len(dst_placements) != len(metas):
        raise ValueError(
            f"destination plan has {len(dst_placements)} parameters, expected "
            f"{len(metas)}"
        )
    for meta, placement in zip(metas, dst_placements):
        check_transferable(meta.name, meta.dtype, meta.shape)
        src = resolve_layout(
            meta.source_layout.mesh,
            meta.source_layout.placements,
            f"parameter '{meta.name}' source placements",
        )
        validate_layout(*src, meta.shape, f"parameter '{meta.name}' source layout")
        dst = resolve_layout(
            dst_mesh,
            placement,
            f"parameter '{meta.name}' destination placements",
        )
        validate_layout(*dst, meta.shape, f"parameter '{meta.name}' destination layout")
        check_plan_limits(src, dst, meta.name, meta.shape)


@dataclass(frozen=True)
class M2NParamMeta(ParamMeta):
    """`ParamMeta` extended with how the trainer places this tensor.

    The base class carries only name / dtype / full shape, which is not enough
    to plan a reshard. `source_layout` carries both the per-parameter mesh and
    its placement on that mesh.
    """

    source_layout: M2NLayout
    allow_full_fallback: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.source_layout, M2NLayout):
            raise TypeError("source_layout must be an M2NLayout")
        if not isinstance(self.allow_full_fallback, bool):
            raise TypeError("allow_full_fallback must be a bool")


@dataclass(frozen=True)
class M2NWireParam:
    """JSON-safe source metadata for one parameter in the init handshake."""

    name: str
    dtype_name: str
    shape: tuple[int, ...]
    src_mesh_dims: tuple[int, int]
    src_placements: Placements | None
    allow_full_fallback: bool = True

    def __post_init__(self) -> None:
        shape = tuple(self.shape)
        dims = tuple(self.src_mesh_dims)
        placements = None if self.src_placements is None else tuple(self.src_placements)
        object.__setattr__(self, "shape", shape)
        object.__setattr__(self, "src_mesh_dims", dims)
        object.__setattr__(self, "src_placements", placements)
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("M2N wire parameter name must not be empty")
        if not isinstance(self.dtype_name, str) or not self.dtype_name:
            raise ValueError(f"parameter '{self.name}' has an empty wire dtype name")
        if not shape or any(
            not isinstance(dim, int) or isinstance(dim, bool) or dim <= 0
            for dim in shape
        ):
            raise ValueError(
                f"parameter '{self.name}' shape must contain positive integers"
            )
        M2NMesh(cast(tuple[int, int], dims), 0)
        if placements is not None:
            check_placements(
                cast(Placements, placements),
                f"parameter '{self.name}' source placements",
            )
        if not isinstance(self.allow_full_fallback, bool):
            raise TypeError(
                f"parameter '{self.name}' allow_full_fallback must be a bool"
            )

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "M2NWireParam":
        """Parse one strict wire record from its JSON-decoded mapping."""
        fields = {
            "name",
            "dtype_name",
            "shape",
            "src_mesh_dims",
            "src_placements",
            "allow_full_fallback",
        }
        missing = fields - value.keys()
        extra = value.keys() - fields
        if missing or extra:
            details = []
            if missing:
                details.append(f"missing {sorted(missing)}")
            if extra:
                details.append(f"unexpected {sorted(extra)}")
            raise ValueError(f"invalid M2N wire parameter: {', '.join(details)}")
        return cls(**value)

    def to_dict(self) -> dict[str, Any]:
        """Serialize this record using only JSON-compatible values."""
        return {
            "name": self.name,
            "dtype_name": self.dtype_name,
            "shape": list(self.shape),
            "src_mesh_dims": list(self.src_mesh_dims),
            "src_placements": (
                None if self.src_placements is None else list(self.src_placements)
            ),
            "allow_full_fallback": self.allow_full_fallback,
        }


@dataclass(frozen=True)
class M2NSourceTopology:
    """Ordered source rank groups and each parameter's owning group.

    Rank ids live in the global metadata-group namespace. Rank order within a
    group defines route-local source-mesh coordinates, so callers must not sort
    it. ``param_groups[i]`` selects the group that exports ``params[i]``.
    Transfer-route identity is derived separately from this source ownership.

    An empty topology represents an empty manifest and is also used as a
    lifecycle sentinel. A usable non-empty plan must pass :meth:`validate`.
    Source groups are non-empty and pairwise disjoint, so a trainer rank belongs
    to at most one group. Each parameter selects exactly one canonical group.
    """

    groups: tuple[tuple[int, ...], ...]
    param_groups: tuple[int, ...]

    def __post_init__(self) -> None:
        groups = tuple(tuple(ranks) for ranks in self.groups)
        param_groups = tuple(self.param_groups)
        object.__setattr__(self, "groups", groups)
        object.__setattr__(self, "param_groups", param_groups)

        seen_ranks: set[int] = set()
        for group_index, ranks in enumerate(groups):
            if not ranks:
                raise ValueError(f"source group {group_index} must not be empty")
            if any(
                not isinstance(rank, int) or isinstance(rank, bool) or rank < 0
                for rank in ranks
            ):
                raise ValueError(
                    f"source group {group_index} ranks must be non-negative integers"
                )
            if len(set(ranks)) != len(ranks):
                raise ValueError(
                    f"source group {group_index} contains duplicate trainer ranks"
                )
            overlap = seen_ranks.intersection(ranks)
            if overlap:
                raise ValueError(
                    f"source group {group_index} overlaps an earlier group at "
                    f"trainer rank {min(overlap)}"
                )
            seen_ranks.update(ranks)

        for param_index, group_index in enumerate(param_groups):
            if not isinstance(group_index, int) or isinstance(group_index, bool):
                raise TypeError(
                    f"parameter {param_index} source group must be an integer"
                )
            if not 0 <= group_index < len(groups):
                raise ValueError(
                    f"parameter {param_index} references source group "
                    f"{group_index}, but there are {len(groups)} groups"
                )

    @classmethod
    def single_group(
        cls, num_trainer_ranks: int, num_params: int
    ) -> "M2NSourceTopology":
        """Build the legacy topology: every trainer exports every parameter."""
        if (
            not isinstance(num_trainer_ranks, int)
            or isinstance(num_trainer_ranks, bool)
            or num_trainer_ranks <= 0
        ):
            raise ValueError("num_trainer_ranks must be a positive integer")
        if (
            not isinstance(num_params, int)
            or isinstance(num_params, bool)
            or num_params < 0
        ):
            raise ValueError("num_params must be a non-negative integer")
        if num_params == 0:
            return cls((), ())
        return cls(
            (tuple(range(num_trainer_ranks)),),
            (0,) * num_params,
        )

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "M2NSourceTopology":
        """Parse one strict JSON-decoded topology record."""
        fields = {"groups", "param_groups"}
        missing = fields - value.keys()
        extra = value.keys() - fields
        if missing or extra:
            details = []
            if missing:
                details.append(f"missing {sorted(missing)}")
            if extra:
                details.append(f"unexpected {sorted(extra)}")
            raise ValueError(f"invalid M2N source topology: {', '.join(details)}")
        raw_groups = value["groups"]
        raw_param_groups = value["param_groups"]
        if not isinstance(raw_groups, list):
            raise TypeError("source topology groups must be a list")
        if not isinstance(raw_param_groups, list):
            raise TypeError("source topology param_groups must be a list")
        if any(not isinstance(ranks, list) for ranks in raw_groups):
            raise TypeError("each source topology group must be a list")
        return cls(
            tuple(tuple(ranks) for ranks in raw_groups),
            tuple(raw_param_groups),
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize this topology using JSON-compatible values."""
        return {
            "groups": [list(ranks) for ranks in self.groups],
            "param_groups": list(self.param_groups),
        }

    def validate(self, *, num_trainer_ranks: int, num_params: int) -> None:
        """Validate this topology against one concrete source manifest."""
        if (
            not isinstance(num_trainer_ranks, int)
            or isinstance(num_trainer_ranks, bool)
            or num_trainer_ranks <= 0
        ):
            raise ValueError("num_trainer_ranks must be a positive integer")
        if (
            not isinstance(num_params, int)
            or isinstance(num_params, bool)
            or num_params < 0
        ):
            raise ValueError("num_params must be a non-negative integer")
        if len(self.param_groups) != num_params:
            raise ValueError(
                f"source topology maps {len(self.param_groups)} parameters, "
                f"expected {num_params}"
            )
        if num_params == 0:
            if self.groups:
                raise ValueError("an empty source manifest must not declare groups")
            return
        if not self.groups:
            raise ValueError("a non-empty source manifest needs a source group")
        for group_index, ranks in enumerate(self.groups):
            invalid = [rank for rank in ranks if rank >= num_trainer_ranks]
            if invalid:
                raise ValueError(
                    f"source group {group_index} rank {invalid[0]} is outside "
                    f"trainer ranks [0, {num_trainer_ranks})"
                )
        used = set(self.param_groups)
        unused = sorted(set(range(len(self.groups))) - used)
        if unused:
            raise ValueError(f"source topology contains unused groups {unused}")


@dataclass(frozen=True)
class M2NTransferRoute:
    """One source and destination group in route-local communicator order."""

    source_group_index: int
    members: tuple[int, ...]
    param_indices: tuple[int, ...]
    dst_mesh: M2NMesh

    def __post_init__(self) -> None:
        members = tuple(self.members)
        param_indices = tuple(self.param_indices)
        object.__setattr__(self, "members", members)
        object.__setattr__(self, "param_indices", param_indices)
        if (
            not isinstance(self.source_group_index, int)
            or isinstance(self.source_group_index, bool)
            or self.source_group_index < 0
        ):
            raise ValueError("transfer route source_group_index must be non-negative")
        if not members or any(
            not isinstance(rank, int) or isinstance(rank, bool) or rank < 0
            for rank in members
        ):
            raise ValueError(
                "transfer route members must be non-empty non-negative integers"
            )
        if len(set(members)) != len(members):
            raise ValueError("transfer route contains duplicate members")
        if not param_indices or any(
            not isinstance(index, int) or isinstance(index, bool) or index < 0
            for index in param_indices
        ):
            raise ValueError(
                "transfer route param_indices must be non-empty non-negative integers"
            )
        if tuple(sorted(set(param_indices))) != param_indices:
            raise ValueError(
                "transfer route param_indices must be unique and increasing"
            )
        if not isinstance(self.dst_mesh, M2NMesh):
            raise TypeError("transfer route dst_mesh must be an M2NMesh")
        if not 0 < self.dst_mesh.start_rank < len(members):
            raise ValueError(
                "transfer route destination must follow at least one source rank"
            )
        if self.dst_mesh.start_rank + self.dst_mesh.size != len(members):
            raise ValueError(
                "transfer route destination mesh must cover the communicator tail"
            )

    @property
    def source_ranks(self) -> tuple[int, ...]:
        """Global metadata ranks forming the ordered source mesh."""
        return self.members[: self.dst_mesh.start_rank]

    @property
    def worker_ranks(self) -> tuple[int, ...]:
        """Global metadata ranks forming the ordered destination mesh."""
        return self.members[self.dst_mesh.start_rank :]


@dataclass(frozen=True)
class M2NTransferWave:
    """One deterministic bounded issue wave shared by senders and receivers."""

    issue_indices: tuple[int, ...]
    consume_index: int | None


def build_m2n_transfer_waves(
    routes: Sequence[M2NTransferRoute],
    load_groups: Sequence[int | None],
    *,
    active_indices: Sequence[int] | None = None,
) -> tuple[M2NTransferWave, ...]:
    """Plan bounded launches while preserving model-loader dependencies.

    ``None`` identifies an in-place destination. Equal integer values identify
    callbacks that must run in manifest order. A wave launches at most one new
    parameter on every unblocked route before consuming one ready result. Ready
    results are drained before the next launch wave, so independent PP groups
    advance together without reordering fused or fallback loader inputs.
    """
    groups = tuple(load_groups)
    if any(
        group is not None and (not isinstance(group, int) or isinstance(group, bool))
        for group in groups
    ):
        raise TypeError("load_groups must contain integers or None")
    if active_indices is None:
        active = tuple(range(len(groups)))
    else:
        active = tuple(active_indices)
    if any(
        not isinstance(index, int)
        or isinstance(index, bool)
        or not 0 <= index < len(groups)
        for index in active
    ):
        raise ValueError("active parameter index is outside load_groups")
    if len(set(active)) != len(active):
        raise ValueError("active parameter indices must be unique")
    if tuple(sorted(active)) != active:
        raise ValueError("active parameter indices must be increasing")

    param_routes = build_m2n_param_routes(routes, len(groups))

    active_set = set(active)
    queues = {
        route_index: deque(
            index for index in route.param_indices if index in active_set
        )
        for route_index, route in enumerate(routes)
    }
    predecessors: dict[int, int | None] = {}
    last_by_group: dict[int, int] = {}
    for index in active:
        group = groups[index]
        if group is None:
            continue
        predecessors[index] = last_by_group.get(group)
        last_by_group[group] = index

    pending: dict[int, int] = {}
    consumed: set[int] = set()
    waves: list[M2NTransferWave] = []

    while any(queues.values()) or pending:
        issued: list[int] = []
        ready = sorted(
            index
            for index in pending.values()
            if predecessors[index] is None or predecessors[index] in consumed
        )
        if not ready:
            for route_index in range(len(routes)):
                queue = queues[route_index]
                if not queue or route_index in pending:
                    continue
                index = queue.popleft()
                issued.append(index)
                if groups[index] is not None:
                    pending[route_index] = index
            ready = sorted(
                index
                for index in pending.values()
                if predecessors[index] is None or predecessors[index] in consumed
            )

        consume_index = ready[0] if ready else None
        if consume_index is not None:
            del pending[param_routes[consume_index]]
            consumed.add(consume_index)
        if not issued and consume_index is None:
            blocked = sorted(pending.values())
            raise ValueError(
                "cannot build bounded M2N transfer schedule; blocked callbacks "
                f"are {blocked!r}"
            )
        waves.append(M2NTransferWave(tuple(issued), consume_index))

    if consumed != set(predecessors):
        raise ValueError("M2N transfer schedule left callbacks unconsumed")
    return tuple(waves)


def build_m2n_param_routes(
    routes: Sequence[M2NTransferRoute], num_params: int
) -> tuple[int, ...]:
    """Map each manifest parameter to its canonical transfer-route index."""
    if (
        not isinstance(num_params, int)
        or isinstance(num_params, bool)
        or num_params < 0
    ):
        raise ValueError("num_params must be a non-negative integer")
    param_routes = [-1] * num_params
    for route_index, route in enumerate(routes):
        for param_index in route.param_indices:
            if param_index >= num_params:
                raise ValueError(
                    f"transfer route {route_index} parameter {param_index} is outside "
                    f"the {num_params}-parameter manifest"
                )
            if param_routes[param_index] != -1:
                raise ValueError(
                    f"parameter {param_index} belongs to multiple transfer routes"
                )
            param_routes[param_index] = route_index
    try:
        missing = param_routes.index(-1)
    except ValueError:
        return tuple(param_routes)
    raise ValueError(f"parameter {missing} has no transfer route")


def build_m2n_transfer_routes(
    topology: M2NSourceTopology,
    *,
    num_trainer_ranks: int,
    world_size: int,
    dst_mesh_dims: tuple[int, int],
) -> tuple[M2NTransferRoute, ...]:
    """Build canonical source-group-to-all-workers transfer routes."""
    if not isinstance(topology, M2NSourceTopology):
        raise TypeError("topology must be an M2NSourceTopology")
    if (
        not isinstance(num_trainer_ranks, int)
        or isinstance(num_trainer_ranks, bool)
        or num_trainer_ranks <= 0
    ):
        raise ValueError("num_trainer_ranks must be a positive integer")
    if (
        not isinstance(world_size, int)
        or isinstance(world_size, bool)
        or world_size <= num_trainer_ranks
    ):
        raise ValueError(
            f"world_size {world_size!r} must exceed num_trainer_ranks "
            f"{num_trainer_ranks!r}"
        )
    topology.validate(
        num_trainer_ranks=num_trainer_ranks,
        num_params=len(topology.param_groups),
    )
    if not topology.groups:
        raise ValueError("source topology has no transfer routes")
    dims = tuple(dst_mesh_dims)
    worker_mesh = M2NMesh(cast(tuple[int, int], dims), 0)
    num_workers = world_size - num_trainer_ranks
    if worker_mesh.size != num_workers:
        raise ValueError(
            f"destination mesh dims {dims} cover {worker_mesh.size} ranks, "
            f"expected {num_workers} workers"
        )
    workers = tuple(range(num_trainer_ranks, world_size))
    routes: list[M2NTransferRoute] = []
    for group_index, source_ranks in enumerate(topology.groups):
        # Keep manifest order: route members must issue matching NCCL
        # collective sequences.
        param_indices = tuple(
            index
            for index, owner in enumerate(topology.param_groups)
            if owner == group_index
        )
        routes.append(
            M2NTransferRoute(
                source_group_index=group_index,
                members=source_ranks + workers,
                param_indices=param_indices,
                dst_mesh=M2NMesh(
                    cast(tuple[int, int], dims),
                    len(source_ranks),
                ),
            )
        )
    return tuple(routes)


def _legacy_source_topology(
    params: Sequence[M2NWireParam],
) -> M2NSourceTopology:
    """Infer the pre-PP single source group for compatibility callers."""
    if not params:
        return M2NSourceTopology((), ())
    num_trainer_ranks = math.prod(params[0].src_mesh_dims)
    for param in params[1:]:
        if math.prod(param.src_mesh_dims) != num_trainer_ranks:
            raise ValueError(
                "mixed source mesh sizes require an explicit M2NSourceTopology"
            )
    return M2NSourceTopology.single_group(num_trainer_ranks, len(params))


def source_plan_digest(
    params: Sequence[M2NWireParam],
    topology: M2NSourceTopology | None = None,
) -> str:
    """Return a rank-independent digest of the ordered source wire plan."""
    resolved_topology = topology or _legacy_source_topology(params)
    if params:
        inferred_num_trainers = (
            max(
                (rank for ranks in resolved_topology.groups for rank in ranks),
                default=0,
            )
            + 1
        )
        resolved_topology.validate(
            num_trainer_ranks=inferred_num_trainers,
            num_params=len(params),
        )
    elif resolved_topology.groups or resolved_topology.param_groups:
        raise ValueError("an empty source plan must use an empty topology")
    payload = json.dumps(
        {
            "schema_version": M2N_WIRE_SCHEMA_VERSION,
            "params": [param.to_dict() for param in params],
            "source_topology": resolved_topology.to_dict(),
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def check_source_plan_agreement(
    group: "StatelessProcessGroup",
    params: Sequence[M2NWireParam],
    expected_digest: str | None = None,
    local_error: str | None = None,
    *,
    topology: M2NSourceTopology | None = None,
) -> str:
    """Verify every participant entered with the same source plan."""
    prepared_error = local_error
    digest: str | None = None
    declared: str | None = expected_digest
    if prepared_error is None:
        try:
            digest = source_plan_digest(params, topology)
            declared = digest if expected_digest is None else expected_digest
            if not isinstance(declared, str) or not declared:
                raise ValueError("declared source digest must be a non-empty string")
        except Exception as exc:
            prepared_error = f"{type(exc).__name__}: {exc}"
    elif not isinstance(prepared_error, str) or not prepared_error:
        prepared_error = "TypeError: local source error must be a non-empty string"

    gathered = group.all_gather_obj(
        {
            "phase": "source",
            "digest": digest,
            "declared_digest": declared,
            "error": prepared_error,
        }
    )
    errors: list[str] = []
    expected_fields = {"phase", "digest", "declared_digest", "error"}
    if len(gathered) != group.world_size:
        errors.append(
            "control-plane all-gather returned "
            f"{len(gathered)} records for world size {group.world_size}"
        )
    agreed: list[tuple[int, str, str]] = []
    for rank, item in enumerate(gathered):
        if not isinstance(item, Mapping) or set(item) != expected_fields:
            errors.append(f"rank {rank}: invalid source preflight envelope")
            continue
        if item["phase"] != "source":
            errors.append(f"rank {rank}: invalid source preflight phase")
            continue
        error = item["error"]
        if error is not None:
            if not isinstance(error, str) or not error:
                errors.append(f"rank {rank}: invalid source preflight error")
            else:
                errors.append(f"rank {rank}: {error}")
            continue
        computed = item["digest"]
        claimed = item["declared_digest"]
        if (
            not isinstance(computed, str)
            or not computed
            or not isinstance(claimed, str)
            or not claimed
        ):
            errors.append(f"rank {rank}: invalid source digest")
            continue
        agreed.append((rank, computed, claimed))
    if errors:
        raise RuntimeError(
            "nccl_m2n source preflight rejected before NCCL init: " + "; ".join(errors)
        )

    canonical = agreed[0][1]
    for rank, computed, claimed in agreed:
        if computed != claimed:
            errors.append(
                f"rank {rank}: computed digest {computed} does not match "
                f"declared digest {claimed}"
            )
        if computed != canonical:
            errors.append(f"rank {rank}: source digest disagrees with rank 0")
    if errors:
        raise RuntimeError(
            "nccl_m2n source preflight rejected before NCCL init: " + "; ".join(errors)
        )
    return canonical


def check_runtime_ready_agreement(
    group: "StatelessProcessGroup", local_error: str | None
) -> None:
    """Require every rank to prepare local resources before NCCL init."""
    prepared_error = local_error
    if prepared_error is not None and (
        not isinstance(prepared_error, str) or not prepared_error
    ):
        prepared_error = "TypeError: local runtime error must be a non-empty string"
    gathered = group.all_gather_obj({"phase": "runtime_ready", "error": prepared_error})
    errors: list[str] = []
    if len(gathered) != group.world_size:
        errors.append(
            "control-plane all-gather returned "
            f"{len(gathered)} records for world size {group.world_size}"
        )
    for rank, item in enumerate(gathered):
        if (
            not isinstance(item, Mapping)
            or set(item) != {"phase", "error"}
            or item.get("phase") != "runtime_ready"
        ):
            errors.append(f"rank {rank}: invalid runtime readiness envelope")
            continue
        error = item.get("error")
        if error is not None:
            if not isinstance(error, str) or not error:
                errors.append(f"rank {rank}: invalid runtime readiness error")
            else:
                errors.append(f"rank {rank}: {error}")
    if errors:
        raise RuntimeError(
            "nccl_m2n local runtime preflight failed before NCCL init: "
            + "; ".join(errors)
        )


def check_data_plane_agreement(
    group: "StatelessProcessGroup",
    unique_id_bytes: bytes | Sequence[bytes | None] | None,
    local_error: str | None = None,
) -> None:
    """Agree on ordered NCCL route rendezvous identities before NCCL init.

    A scalar preserves the single-communicator API. Multi-route callers pass one
    bytes-or-None entry per canonical transfer route.
    """
    prepared_error = local_error
    identities: list[dict[str, str | None]] | None = None
    if prepared_error is None:
        try:
            unique_ids: tuple[bytes | None, ...]
            if unique_id_bytes is None or isinstance(unique_id_bytes, bytes):
                unique_ids = (unique_id_bytes,)
            elif isinstance(unique_id_bytes, Sequence) and not isinstance(
                unique_id_bytes, (str, bytearray)
            ):
                unique_ids = cast(tuple[bytes | None, ...], tuple(unique_id_bytes))
            else:
                raise TypeError(
                    "NCCL unique ids must be bytes, None, or a sequence of them"
                )
            if not unique_ids:
                raise ValueError("NCCL data plane needs at least one transfer route")
            explicit_unique_ids = tuple(
                unique_id for unique_id in unique_ids if unique_id is not None
            )
            if len(set(explicit_unique_ids)) != len(explicit_unique_ids):
                raise ValueError(
                    "each nccl_m2n transfer communicator needs a distinct "
                    "pre-shared NCCL unique id"
                )
            identities = []
            for route_index, unique_id in enumerate(unique_ids):
                if unique_id is not None and not isinstance(unique_id, bytes):
                    raise TypeError(
                        f"NCCL unique id for route {route_index} must decode to bytes"
                    )
                identities.append(
                    {
                        "mode": "uid" if unique_id is not None else "tcp",
                        "uid_digest": (
                            None
                            if unique_id is None
                            else hashlib.sha256(unique_id).hexdigest()
                        ),
                    }
                )
        except Exception as exc:
            prepared_error = f"{type(exc).__name__}: {exc}"
    elif not isinstance(prepared_error, str) or not prepared_error:
        prepared_error = "TypeError: local data-plane error must be non-empty"

    gathered = group.all_gather_obj(
        {
            "phase": "data_plane",
            "routes": identities,
            "error": prepared_error,
        }
    )
    expected_fields = {
        "phase",
        "routes",
        "error",
    }
    errors: list[str] = []
    valid: list[tuple[int, tuple[tuple[str, str | None], ...]]] = []
    if len(gathered) != group.world_size:
        errors.append(
            "control-plane all-gather returned "
            f"{len(gathered)} records for world size {group.world_size}"
        )
    for rank, item in enumerate(gathered):
        if (
            not isinstance(item, Mapping)
            or set(item) != expected_fields
            or item.get("phase") != "data_plane"
        ):
            errors.append(f"rank {rank}: invalid data-plane preflight envelope")
            continue
        error = item["error"]
        if error is not None:
            if not isinstance(error, str) or not error:
                errors.append(f"rank {rank}: invalid data-plane preflight error")
            else:
                errors.append(f"rank {rank}: {error}")
            continue
        route_identities = item["routes"]
        if not isinstance(route_identities, list) or not route_identities:
            errors.append(f"rank {rank}: invalid data-plane route identities")
            continue
        parsed_identities: list[tuple[str, str | None]] = []
        malformed = False
        for route_index, identity in enumerate(route_identities):
            if not isinstance(identity, Mapping) or set(identity) != {
                "mode",
                "uid_digest",
            }:
                errors.append(
                    f"rank {rank}: invalid data-plane identity for route {route_index}"
                )
                malformed = True
                break
            mode = identity["mode"]
            uid = identity["uid_digest"]
            if mode not in {"tcp", "uid"} or (mode == "uid") != isinstance(uid, str):
                errors.append(
                    f"rank {rank}: invalid data-plane identity for route {route_index}"
                )
                malformed = True
                break
            parsed_identities.append((mode, uid))
        if not malformed:
            valid.append((rank, tuple(parsed_identities)))
    if not errors:
        _, canonical_identities = valid[0]
        for rank, other_identities in valid[1:]:
            if other_identities != canonical_identities:
                errors.append(f"rank {rank}: NCCL data-plane identity disagrees")
    if errors:
        raise RuntimeError(
            "nccl_m2n data-plane preflight rejected before NCCL init: "
            + "; ".join(errors)
        )


def check_communicator_init_agreement(
    group: "StatelessProcessGroup",
    local_errors: Sequence[str | None],
) -> None:
    """Agree once on the complete ordered communicator-init result."""
    prepared_errors: tuple[str | None, ...] | None = None
    prepared_error: str | None = None
    try:
        prepared_errors = tuple(local_errors)
        if not prepared_errors:
            raise ValueError("communicator initialization needs at least one group")
        if any(
            error is not None and (not isinstance(error, str) or not error)
            for error in prepared_errors
        ):
            raise TypeError(
                "communicator initialization errors must be null or non-empty strings"
            )
    except Exception as exc:
        prepared_error = f"{type(exc).__name__}: {exc}"

    gathered = group.all_gather_obj(
        {
            "phase": "communicator_init",
            "errors": (None if prepared_errors is None else list(prepared_errors)),
            "error": prepared_error,
        }
    )
    errors: list[str] = []
    valid: list[tuple[int, tuple[str | None, ...]]] = []
    expected_fields = {"phase", "errors", "error"}
    if len(gathered) != group.world_size:
        errors.append(
            "control-plane all-gather returned "
            f"{len(gathered)} records for world size {group.world_size}"
        )
    for rank, item in enumerate(gathered):
        if (
            not isinstance(item, Mapping)
            or set(item) != expected_fields
            or item.get("phase") != "communicator_init"
        ):
            errors.append(f"rank {rank}: invalid communicator-init envelope")
            continue
        error = item["error"]
        if error is not None:
            if not isinstance(error, str) or not error:
                errors.append(f"rank {rank}: invalid communicator-init error")
            else:
                errors.append(f"rank {rank}: {error}")
            continue
        rank_errors = item["errors"]
        if not isinstance(rank_errors, list) or not rank_errors:
            errors.append(f"rank {rank}: invalid communicator-init results")
            continue
        if any(
            value is not None and (not isinstance(value, str) or not value)
            for value in rank_errors
        ):
            errors.append(f"rank {rank}: invalid communicator-init result")
            continue
        valid.append((rank, tuple(rank_errors)))
    if not errors:
        expected_count = len(valid[0][1])
        for rank, rank_errors in valid:
            if len(rank_errors) != expected_count:
                errors.append(
                    f"rank {rank}: reported {len(rank_errors)} communicator "
                    f"results, expected {expected_count}"
                )
                continue
            for communicator_index, error in enumerate(rank_errors):
                if error is not None:
                    errors.append(
                        f"rank {rank}, communicator {communicator_index}: {error}"
                    )
    if errors:
        raise RuntimeError(
            "nccl_m2n communicator initialization failed: " + "; ".join(errors)
        )


def validate_local_buffer(m2n: Any, meta: M2NParamMeta, buffer: Any) -> None:
    """Check source dtype/shape while delegating buffer validity to M2N."""
    # A raw pointer carries no inspectable metadata. The source owns its
    # correctness; _send supplies the declared dtype and local shape to M2N.
    if isinstance(buffer, int):
        return

    mesh, placements = resolve_layout(
        meta.source_layout.mesh, meta.source_layout.placements
    )
    actual = m2n.DistTensor(
        buffer,
        mesh=to_mesh(m2n, mesh),
        placements=to_placements(m2n, placements),
    )
    declared = m2n.DistTensor(
        None,
        local_shape=actual.local_shape,
        dtype=meta.dtype,
        mesh=to_mesh(m2n, mesh),
        placements=to_placements(m2n, placements),
    )
    if actual.dtype != declared.dtype:
        raise ValueError(
            f"parameter '{meta.name}' source returned dtype {actual.dtype}, "
            f"but its metadata declared {declared.dtype}"
        )

    expected = expected_local_shape(meta)
    if actual.local_shape != expected:
        raise ValueError(
            f"parameter '{meta.name}' source returned local shape "
            f"{actual.local_shape}, but layout {meta.source_layout} implies "
            f"{expected} from global shape {meta.shape}"
        )


def to_mesh(m2n: Any, mesh: M2NMesh) -> Any:
    return m2n.Mesh(mesh.dims, start_rank=mesh.start_rank)


def to_placements(m2n: Any, placements: Placements) -> list[Any]:
    return [
        m2n.Replicate() if code == REPLICATE else m2n.Shard(code) for code in placements
    ]
