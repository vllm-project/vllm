# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared helpers for the NCCL M2N (`nccl_m2n`) weight transfer backend.

M2N reshards a tensor between two disjoint meshes of ranks that live in one
communicator: the trainer occupies ranks `[0, T)` and the inference workers
`[T, T + N)`. This module holds everything both sides need — the optional
runtime import, layout descriptors, and the conversion to `nccl.m2n` types —
so the engine module stays about the transfer itself.
"""

import ctypes
import functools
import hashlib
import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

import torch

from vllm import envs
from vllm.distributed.weight_transfer.base import ParamMeta
from vllm.logger import init_logger

logger = init_logger(__name__)

if TYPE_CHECKING:
    from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
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

# m2n bounds how many source shards may feed one destination shard, and how many
# destination shards one source shard may feed, with compile-time arrays in
# `reshard_limits.h`. The bindings do not expose them, so they are mirrored
# here; a build with larger arrays makes these conservative, never wrong.
MAX_SOURCE_SHARDS = 16
MAX_DEST_SHARDS = 64

# Increment when the JSON init payload changes incompatibly.
M2N_WIRE_SCHEMA_VERSION = 2

_NCCL_RESHARD_ENV_PREFIX = "NCCL_RESHARD_"
_NCCL_RESHARD_DIAGNOSTIC_ENV_VARS = frozenset(
    {
        "NCCL_RESHARD_LOG_LEVEL",
        "NCCL_RESHARD_SPLIT_KERNEL_TRACE",
    }
)

# Wire dtypes `ncclReshard` accepts. Notably excludes fp4 and any packed
# sub-byte type, so quantized checkpoints are out of scope for now.
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
    "and install instructions. It needs NCCL 2.30.5 or newer. If runtime "
    "validation reports mismatched NCCL libraries, preload the NCCL linked "
    "by libnccl_m2n.so before starting Python."
)


@dataclass(frozen=True)
class M2NNcclRuntime:
    """The process-wide NCCL DSO selected for both PyNccl and M2N."""

    library_path: str
    library_handle: int
    library: Any


@functools.cache
def prepare_m2n_nccl_runtime() -> M2NNcclRuntime:
    """Resolve and globally promote the NCCL DSO used by ``nccl.m2n``."""
    try:
        from cuda.pathfinder import load_nvidia_dynamic_lib
    except ImportError as exc:
        raise ImportError(
            f"{_IMPORT_HINT} (cuda.pathfinder import failed: {exc})"
        ) from exc

    loaded = load_nvidia_dynamic_lib("nccl")
    library_path = loaded.abs_path
    if not library_path:
        raise RuntimeError(
            "cuda.pathfinder found NCCL but could not resolve its absolute path; "
            "preload the NCCL linked by libnccl_m2n.so before starting Python"
        )
    library = ctypes.CDLL(library_path, mode=os.RTLD_NOW | os.RTLD_GLOBAL)
    loaded_handle = int(loaded._handle_uint)
    promoted_handle = int(library._handle)
    if promoted_handle != loaded_handle:
        raise RuntimeError(
            "globally promoted NCCL differs from cuda.pathfinder's loaded NCCL: "
            f"path={library_path}, pathfinder_handle={loaded_handle:#x}, "
            f"promoted_handle={promoted_handle:#x}"
        )

    configured_path = os.environ.get("VLLM_NCCL_SO_PATH")
    if configured_path:
        try:
            same_file = os.path.samefile(configured_path, library_path)
        except OSError:
            same_file = os.path.realpath(configured_path) == os.path.realpath(
                library_path
            )
        if not same_file:
            logger.warning_once(
                "NCCL M2N is reusing the already-loaded NCCL at %s instead of "
                "VLLM_NCCL_SO_PATH=%s so its communicator and extension share "
                "one process runtime. Preload the configured library before "
                "Python starts to force that file.",
                library_path,
                configured_path,
            )

    return M2NNcclRuntime(library_path, promoted_handle, library)


def validate_m2n_nccl_library(library: Any, runtime: M2NNcclRuntime) -> None:
    """Reject a PyNccl wrapper bound to a different NCCL DSO."""
    loaded = getattr(library, "lib", None)
    handle = getattr(loaded, "_handle", None)
    if handle is None:
        raise RuntimeError("PyNccl does not expose its NCCL library handle")
    if int(handle) != runtime.library_handle:
        comm_path = getattr(loaded, "_name", "<unknown>")
        raise RuntimeError(
            "PyNccl and NCCL M2N loaded different NCCL runtimes: "
            f"pynccl={comm_path} handle={int(handle):#x}, "
            f"m2n={runtime.library_path} handle={runtime.library_handle:#x}. "
            "Preload one NCCL library before Python starts."
        )


def validate_m2n_nccl_communicator(
    comm: "PyNcclCommunicator", runtime: M2NNcclRuntime
) -> None:
    """Reject a PyNccl communicator created by a different NCCL DSO."""
    library = getattr(comm, "nccl", None)
    if library is None:
        raise RuntimeError("PyNcclCommunicator does not expose its NCCL library")
    validate_m2n_nccl_library(library, runtime)


def prepare_m2n_local_runtime(
    m2n: Any, max_cta: int | None
) -> tuple[M2NNcclRuntime, int, Any]:
    """Prepare rank-local resources before any rank enters NCCL init."""
    from vllm.distributed.device_communicators.pynccl_wrapper import NCCLLibrary
    from vllm.utils.nccl import unpinned_nccl_env

    runtime = prepare_m2n_nccl_runtime()
    with unpinned_nccl_env():
        library = NCCLLibrary(runtime.library_path)
        library.ncclGetRawVersion()
    validate_m2n_nccl_library(library, runtime)
    device = torch.accelerator.current_device_index()
    if not isinstance(device, int) or isinstance(device, bool) or device < 0:
        raise RuntimeError(f"invalid current accelerator device index: {device!r}")
    handle = m2n.Handle.create(m2n.Config(max_cta=max_cta))
    return runtime, device, handle


@functools.cache
def _load_m2n_library(library_path: str) -> Any:
    """Keep the M2N DSO and its dependency scope alive process-wide."""
    return ctypes.CDLL(library_path, mode=os.RTLD_NOW | os.RTLD_GLOBAL)


def _symbol_address(library: Any, symbol: str) -> int:
    try:
        function = getattr(library, symbol)
    except AttributeError as exc:
        raise RuntimeError(
            f"NCCL library {getattr(library, '_name', '<unknown>')} does not "
            f"export required symbol {symbol}"
        ) from exc
    address = ctypes.cast(function, ctypes.c_void_p).value
    if address is None:
        raise RuntimeError(f"NCCL symbol {symbol} resolved to a null function pointer")
    return int(address)


def validate_m2n_nccl_library_binding(
    runtime: M2NNcclRuntime, m2n_library_path: str
) -> None:
    """Attest that M2N's NCCL calls resolve through the PyNccl runtime."""
    m2n_library = _load_m2n_library(os.path.realpath(m2n_library_path))
    symbol = "ncclCommWindowRegister"
    runtime_address = _symbol_address(runtime.library, symbol)
    m2n_address = _symbol_address(m2n_library, symbol)
    if m2n_address != runtime_address:
        raise RuntimeError(
            "libnccl_m2n and PyNccl resolve NCCL through different runtimes: "
            f"symbol={symbol}, m2n_library={m2n_library_path}, "
            f"m2n_address={m2n_address:#x}, "
            f"pynccl_library={runtime.library_path}, "
            f"pynccl_address={runtime_address:#x}. Preload the NCCL linked by "
            "libnccl_m2n before Python starts."
        )


def import_m2n() -> Any:
    """Import `nccl.m2n` lazily, with an actionable error when it is missing.

    Deferred so that importing vLLM — or any other weight transfer backend —
    never requires the m2n runtime to be present.
    """
    if envs.VLLM_DISABLE_PYNCCL:
        raise ValueError(
            "nccl_m2n requires PyNccl; unset VLLM_DISABLE_PYNCCL on every rank"
        )
    runtime = prepare_m2n_nccl_runtime()
    try:
        import nccl.m2n as m2n
    except ImportError as e:
        raise ImportError(f"{_IMPORT_HINT} (import failed: {e})") from e
    m2n_library_path = os.environ.get("NCCL_M2N_LIBRARY")
    if m2n_library_path:
        validate_m2n_nccl_library_binding(runtime, m2n_library_path)
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
            check_placements(self.placements)


def check_placements(placements: Placements, context: str = "placements") -> None:
    """Reject placement pairs m2n cannot express.

    The header requires exactly one SHARD axis and one REPLICATE axis.
    `{REPLICATE, REPLICATE}` hits a degenerate prepare branch, which is why
    full replication is carried as `REPLICATED` and resolved separately;
    sharding both axes is not expressible at all.
    """
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
            "not as two "
            "REPLICATE axes"
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


def shard_count(mesh: M2NMesh, placements: Placements) -> int:
    """How many pieces this layout splits the tensor into."""
    return next(
        (mesh.dims[axis] for axis, code in enumerate(placements) if code != REPLICATE),
        1,
    )


def check_plan_limits(
    src: tuple[M2NMesh, Placements],
    dst: tuple[M2NMesh, Placements],
    name: str,
) -> None:
    """Reject plans that exceed m2n's static per-shard fan-in / fan-out arrays.

    Only the unambiguous cases are checked: when one side is a single shard it
    is fed by (or feeds) every shard on the other side, so the count is exactly
    the other side's shard count. In the general sharded-to-sharded case the
    overlap depends on the library's chunking, and reproducing that arithmetic
    here would duplicate internals that can change under us.

    m2n re-checks authoritatively and, because the plan is derived from the
    shared descriptors, fails identically on every rank -- so this is about
    reporting at init with a message that names the parameter, not about
    avoiding a hang.
    """
    src_shards = shard_count(*src)
    dst_shards = shard_count(*dst)
    if dst_shards == 1 and src_shards > MAX_SOURCE_SHARDS:
        raise ValueError(
            f"parameter '{name}' would feed {src_shards} source shards into one "
            f"replicated destination, over m2n's MAX_SOURCES={MAX_SOURCE_SHARDS}. "
            "Shard the destination, replicate the source, or rebuild m2n with "
            "larger arrays."
        )
    if src_shards == 1 and dst_shards > MAX_DEST_SHARDS:
        raise ValueError(
            f"parameter '{name}' would fan one source shard out to {dst_shards} "
            f"destination shards, over m2n's MAX_TARGETS={MAX_DEST_SHARDS}. "
            "Rebuild m2n with larger arrays to raise the bound."
        )


def validate_layout(
    mesh: M2NMesh, placements: Placements, shape: Sequence[int], side: str
) -> None:
    """Check a resolved layout can describe `shape` before any collective runs.

    Called on both sides at init so a bad plan surfaces as an error from the
    init RPC rather than as a hang inside the first reshard.
    """
    for axis, code in enumerate(placements):
        if code == REPLICATE:
            continue
        if code >= len(shape):
            raise ValueError(
                f"{side} layout shards tensor dim {code}, but the tensor "
                f"has rank {len(shape)}"
            )
        factor = mesh.dims[axis]
        if shape[code] % factor:
            raise ValueError(
                f"{side} layout shards dim {code} (size {shape[code]}) over "
                f"{factor} ranks, which does not divide evenly"
            )


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


def _nccl_reshard_environment() -> dict[str, str]:
    """Snapshot rank-local M2N settings that can affect execution."""
    return {
        name: value
        for name, value in sorted(os.environ.items())
        if name.startswith(_NCCL_RESHARD_ENV_PREFIX)
        and name not in _NCCL_RESHARD_DIAGNOSTIC_ENV_VARS
    }


def check_data_plane_agreement(
    group: "StatelessProcessGroup",
    unique_id_bytes: bytes | None,
    max_cta: int | None,
    local_error: str | None = None,
) -> None:
    """Agree on NCCL rendezvous identity and M2N config before NCCL init."""
    prepared_error = local_error
    uid_digest: str | None = None
    if prepared_error is None:
        try:
            if unique_id_bytes is not None:
                if not isinstance(unique_id_bytes, bytes):
                    raise TypeError("NCCL unique id must decode to bytes")
                uid_digest = hashlib.sha256(unique_id_bytes).hexdigest()
            if max_cta is not None and (
                not isinstance(max_cta, int)
                or isinstance(max_cta, bool)
                or max_cta <= 0
            ):
                raise ValueError("max_cta must be a positive integer or null")
        except Exception as exc:
            prepared_error = f"{type(exc).__name__}: {exc}"
    elif not isinstance(prepared_error, str) or not prepared_error:
        prepared_error = "TypeError: local data-plane error must be non-empty"

    gathered = group.all_gather_obj(
        {
            "phase": "data_plane",
            "mode": "uid" if unique_id_bytes is not None else "tcp",
            "uid_digest": uid_digest,
            "max_cta": max_cta,
            "reshard_env": _nccl_reshard_environment(),
            "error": prepared_error,
        }
    )
    expected_fields = {
        "phase",
        "mode",
        "uid_digest",
        "max_cta",
        "reshard_env",
        "error",
    }
    errors: list[str] = []
    valid: list[tuple[int, str, str | None, int | None, dict[str, str]]] = []
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
        cta = item["max_cta"]
        reshard_env = item["reshard_env"]
        if mode not in {"tcp", "uid"} or (mode == "uid") != isinstance(uid, str):
            errors.append(f"rank {rank}: invalid data-plane identity")
            continue
        if cta is not None and (
            not isinstance(cta, int) or isinstance(cta, bool) or cta <= 0
        ):
            errors.append(f"rank {rank}: invalid max_cta")
            continue
        if not isinstance(reshard_env, Mapping) or any(
            not isinstance(name, str)
            or not name.startswith(_NCCL_RESHARD_ENV_PREFIX)
            or name in _NCCL_RESHARD_DIAGNOSTIC_ENV_VARS
            or not isinstance(value, str)
            for name, value in reshard_env.items()
        ):
            errors.append(f"rank {rank}: invalid NCCL_RESHARD environment")
            continue
        valid.append((rank, mode, uid, cta, dict(reshard_env)))
    if not errors:
        _, mode, uid, cta, reshard_env = valid[0]
        for rank, other_mode, other_uid, other_cta, other_env in valid[1:]:
            if (other_mode, other_uid) != (mode, uid):
                errors.append(f"rank {rank}: NCCL data-plane identity disagrees")
            if other_cta != cta:
                errors.append(f"rank {rank}: max_cta disagrees")
            if other_env != reshard_env:
                different = sorted(
                    name
                    for name in set(reshard_env) | set(other_env)
                    if reshard_env.get(name) != other_env.get(name)
                )
                errors.append(
                    f"rank {rank}: NCCL_RESHARD environment disagrees with "
                    f"rank 0 for {', '.join(different)}"
                )
    if errors:
        raise RuntimeError(
            "nccl_m2n data-plane preflight rejected before NCCL init: "
            + "; ".join(errors)
        )


def validate_local_tensor(
    meta: M2NParamMeta,
    tensor: Any,
    expected_device: torch.device | None = None,
) -> None:
    """Check one source value against its initialization-time metadata."""
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(
            f"parameter '{meta.name}' source returned {type(tensor).__name__}; "
            "expected a torch.Tensor"
        )
    if tensor.dtype != meta.dtype:
        raise ValueError(
            f"parameter '{meta.name}' source returned dtype {tensor.dtype}, "
            f"but its metadata declared {meta.dtype}"
        )
    mesh, placements = resolve_layout(
        meta.source_layout.mesh, meta.source_layout.placements
    )
    validate_layout(mesh, placements, meta.shape, "source")
    expected_shape = list(meta.shape)
    for axis, tensor_dim in enumerate(placements):
        if tensor_dim != REPLICATE:
            expected_shape[tensor_dim] //= mesh.dims[axis]
    expected = tuple(expected_shape)
    if tuple(tensor.shape) != expected:
        raise ValueError(
            f"parameter '{meta.name}' source returned local shape "
            f"{tuple(tensor.shape)}, but layout {meta.source_layout} implies "
            f"{expected} from global shape {meta.shape}"
        )
    if expected_device is not None and tensor.device != expected_device:
        raise ValueError(
            f"parameter '{meta.name}' source returned tensor on "
            f"{tensor.device}, expected {expected_device}"
        )
    if not tensor.is_contiguous():
        raise ValueError(
            f"parameter '{meta.name}' source returned a non-contiguous tensor"
        )


def check_transferable(name: str, dtype: torch.dtype, shape: Sequence[int]) -> None:
    """Reject tensors m2n cannot move at all, with the parameter named."""
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


def to_mesh(m2n: Any, mesh: M2NMesh) -> Any:
    return m2n.Mesh(mesh.dims, start_rank=mesh.start_rank)


def to_placements(m2n: Any, placements: Placements) -> list[Any]:
    return [
        m2n.Replicate() if code == REPLICATE else m2n.Shard(code) for code in placements
    ]


def publish_destination_placements(
    comm: "PyNcclCommunicator",
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
    rank split, identically on both sides. Use the NCCL communicator directly
    because unique-id rendezvous deliberately has no bootstrap process group.
    """
    replicated_sentinel = REPLICATE - 1
    encoded = torch.full(
        (num_parameters, MESH_NDIMS),
        replicated_sentinel,
        dtype=torch.int8,
        device=comm.device,
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
            encoded.copy_(torch.tensor(rows, dtype=torch.int8, device=comm.device))

    comm.broadcast(encoded, src=first_worker_rank)
    return [
        REPLICATED
        if all(code == replicated_sentinel for code in row)
        else cast(Placements, tuple(row))
        for row in encoded.tolist()
    ]


def comm_ptr(comm: "PyNcclCommunicator") -> int:
    """Raw `ncclComm_t` behind vLLM's `PyNcclCommunicator`.

    ``prepare_m2n_nccl_runtime`` selects the shared NCCL runtime and
    ``validate_m2n_nccl_communicator`` validates this handle before use.
    """
    handle = comm.comm
    ptr = getattr(handle, "value", handle)
    if not ptr:
        raise RuntimeError("PyNcclCommunicator has no live NCCL communicator")
    return int(ptr)
