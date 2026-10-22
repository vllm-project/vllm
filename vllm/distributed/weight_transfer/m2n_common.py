# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared helpers for the NCCL M2N (`nccl_m2n`) weight transfer backend.

M2N reshards a tensor between two disjoint meshes of ranks that live in one
communicator: the trainer occupies ranks `[0, T)` and the inference workers
`[T, T + N)`. This module holds everything both sides need — the optional
runtime import, layout descriptors, and the conversion to `nccl.m2n` types —
so the engine module stays about the transfer itself.
"""

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
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

# Increment when the JSON init payload changes incompatibly.
M2N_WIRE_SCHEMA_VERSION = 2

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


def create_m2n_communicator(
    nccl: Any,
    group: "StatelessProcessGroup",
    unique_id_bytes: bytes | None,
) -> Any:
    """Create the M2N data-plane communicator through NCCL4Py."""
    if unique_id_bytes is None:
        envelope: dict[str, bytes | str | None] | None = None
        if group.rank == 0:
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
        envelope = group.broadcast_obj(envelope, src=0)
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

    unique_id = nccl.UniqueId.from_bytes(unique_id_bytes)
    with unpinned_nccl_env():
        return nccl.Communicator.init(
            nranks=group.world_size,
            rank=group.rank,
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

    def __post_init__(self) -> None:
        if not isinstance(self.source_layout, M2NLayout):
            raise TypeError("source_layout must be an M2NLayout")


@dataclass(frozen=True)
class M2NWireParam:
    """JSON-safe source metadata for one parameter in the init handshake."""

    name: str
    dtype_name: str
    shape: tuple[int, ...]
    src_mesh_dims: tuple[int, int]
    src_placements: Placements | None

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

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "M2NWireParam":
        """Parse one strict wire record from its JSON-decoded mapping."""
        fields = {
            "name",
            "dtype_name",
            "shape",
            "src_mesh_dims",
            "src_placements",
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
        }


def source_plan_digest(params: Sequence[M2NWireParam]) -> str:
    """Return a rank-independent digest of the ordered source wire plan."""
    payload = json.dumps(
        {
            "schema_version": M2N_WIRE_SCHEMA_VERSION,
            "params": [param.to_dict() for param in params],
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def complete_plan_digest(
    source_digest: str,
    dst_mesh: M2NMesh,
    dst_placements: Sequence[Placements | None],
) -> str:
    """Return a digest of the agreed source and destination plan."""
    if not isinstance(source_digest, str) or not source_digest:
        raise ValueError("source digest must be a non-empty string")
    payload = json.dumps(
        {
            "schema_version": M2N_WIRE_SCHEMA_VERSION,
            "source_digest": source_digest,
            "destination_mesh": {
                "dims": list(dst_mesh.dims),
                "start_rank": dst_mesh.start_rank,
            },
            "destination_placements": [
                None if placement is None else list(placement)
                for placement in dst_placements
            ],
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
) -> str:
    """Verify every participant entered with the same source plan."""
    prepared_error = local_error
    digest: str | None = None
    declared: str | None = expected_digest
    if prepared_error is None:
        try:
            digest = source_plan_digest(params)
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


def check_complete_plan_agreement(
    group: "StatelessProcessGroup",
    source_digest: str,
    dst_mesh: M2NMesh,
    dst_placements: Sequence[Placements | None],
    local_error: str | None = None,
) -> str:
    """Require every rank to accept the same complete transfer plan."""
    prepared_error = local_error
    digest: str | None = None
    if prepared_error is None:
        try:
            digest = complete_plan_digest(source_digest, dst_mesh, dst_placements)
        except Exception as exc:
            prepared_error = f"{type(exc).__name__}: {exc}"
    elif not isinstance(prepared_error, str) or not prepared_error:
        prepared_error = "TypeError: local plan error must be a non-empty string"

    gathered = group.all_gather_obj(
        {"phase": "complete_plan", "digest": digest, "error": prepared_error}
    )
    errors: list[str] = []
    valid: list[tuple[int, str]] = []
    expected_fields = {"phase", "digest", "error"}
    if len(gathered) != group.world_size:
        errors.append(
            "control-plane all-gather returned "
            f"{len(gathered)} records for world size {group.world_size}"
        )
    for rank, item in enumerate(gathered):
        if (
            not isinstance(item, Mapping)
            or set(item) != expected_fields
            or item.get("phase") != "complete_plan"
        ):
            errors.append(f"rank {rank}: invalid complete-plan envelope")
            continue
        error = item["error"]
        if error is not None:
            if not isinstance(error, str) or not error:
                errors.append(f"rank {rank}: invalid complete-plan error")
            else:
                errors.append(f"rank {rank}: {error}")
            continue
        rank_digest = item["digest"]
        if not isinstance(rank_digest, str) or not rank_digest:
            errors.append(f"rank {rank}: invalid complete-plan digest")
            continue
        valid.append((rank, rank_digest))
    if not errors:
        canonical = valid[0][1]
        for rank, rank_digest in valid[1:]:
            if rank_digest != canonical:
                errors.append(
                    f"rank {rank}: complete-plan digest disagrees with rank 0"
                )
    if errors:
        raise RuntimeError(
            "nccl_m2n complete-plan preflight rejected before first reshard: "
            + "; ".join(errors)
        )
    return valid[0][1]


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
    unique_id_bytes: bytes | None,
    local_error: str | None = None,
) -> None:
    """Agree on the NCCL rendezvous identity before NCCL init."""
    prepared_error = local_error
    uid_digest: str | None = None
    if prepared_error is None:
        try:
            if unique_id_bytes is not None:
                if not isinstance(unique_id_bytes, bytes):
                    raise TypeError("NCCL unique id must decode to bytes")
                uid_digest = hashlib.sha256(unique_id_bytes).hexdigest()
        except Exception as exc:
            prepared_error = f"{type(exc).__name__}: {exc}"
    elif not isinstance(prepared_error, str) or not prepared_error:
        prepared_error = "TypeError: local data-plane error must be non-empty"

    gathered = group.all_gather_obj(
        {
            "phase": "data_plane",
            "mode": "uid" if unique_id_bytes is not None else "tcp",
            "uid_digest": uid_digest,
            "error": prepared_error,
        }
    )
    expected_fields = {
        "phase",
        "mode",
        "uid_digest",
        "error",
    }
    errors: list[str] = []
    valid: list[tuple[int, str, str | None]] = []
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
        mode = item["mode"]
        uid = item["uid_digest"]
        if mode not in {"tcp", "uid"} or (mode == "uid") != isinstance(uid, str):
            errors.append(f"rank {rank}: invalid data-plane identity")
            continue
        valid.append((rank, mode, uid))
    if not errors:
        _, mode, uid = valid[0]
        for rank, other_mode, other_uid in valid[1:]:
            if (other_mode, other_uid) != (mode, uid):
                errors.append(f"rank {rank}: NCCL data-plane identity disagrees")
    if errors:
        raise RuntimeError(
            "nccl_m2n data-plane preflight rejected before NCCL init: "
            + "; ".join(errors)
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


def publish_destination_placements(
    comm: Any,
    device: torch.device,
    stream: torch.cuda.Stream,
    first_worker_rank: int,
    placements: "Sequence[Placements | None] | None",
    num_parameters: int,
) -> list[Placements | None]:
    """Share the worker-side destination plan with every rank in the group.

    The trainer must issue each reshard with the same destination the workers
    use, but once destinations are per-parameter they depend on the inference
    model, which only the workers can see. The first worker publishes them here,
    over the shared NCCL communicator; trainer ranks pass `None` and receive
    them, and the other workers pass their own so a disagreement is caught
    rather than deadlocking later.

    Only placements travel: the destination *mesh* is still derived from the
    rank split, identically on both sides. The plan is published on the data
    plane before any M2N reshard collective.
    """
    replicated_sentinel = REPLICATE - 1
    encoded = torch.full(
        (num_parameters, MESH_NDIMS),
        replicated_sentinel,
        dtype=torch.int8,
        device=device,
    )
    if comm.rank == first_worker_rank:
        if placements is None or len(placements) != num_parameters:
            raise ValueError(
                f"publishing rank needs {num_parameters} destination placements"
            )
        rows = [
            [replicated_sentinel] * MESH_NDIMS if placement is REPLICATED else placement
            for placement in placements
        ]
        if rows:
            encoded.copy_(torch.tensor(rows, dtype=torch.int8, device=device))

    comm.broadcast(encoded, encoded, root=first_worker_rank, stream=stream)
    return [
        REPLICATED
        if all(code == replicated_sentinel for code in row)
        else cast(Placements, tuple(row))
        for row in encoded.tolist()
    ]
