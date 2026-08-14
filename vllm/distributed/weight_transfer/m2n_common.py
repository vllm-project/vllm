# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared helpers for the NCCL M2N (`nccl_m2n`) weight transfer backend.

M2N reshards a tensor between two disjoint meshes of ranks that live in one
communicator: the trainer occupies ranks `[0, T)` and the inference workers
`[T, T + N)`. This module holds everything both sides need — the optional
runtime import, layout descriptors, and the conversion to `nccl.m2n` types —
so the engine module stays about the transfer itself.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch

from vllm.distributed.weight_transfer.base import ParamMeta

if TYPE_CHECKING:
    from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator

# Placement code for a mesh axis that replicates. Any other (non-negative) code
# is the tensor dim that axis shards. Ints rather than `nccl.m2n` objects so
# layouts survive the JSON init handshake without a custom encoder.
REPLICATE = -1

# `NCCL_RESHARD_MAX_TENSOR_DIMS`; 4-D and higher are rejected by the library.
MAX_TENSOR_DIMS = 3

MESH_NDIMS = 2

# m2n bounds how many source shards may feed one destination shard, and how many
# destination shards one source shard may feed, with compile-time arrays in
# `reshard_limits.h`. The bindings do not expose them, so they are mirrored
# here; a build with larger arrays makes these conservative, never wrong.
MAX_SOURCE_SHARDS = 16
MAX_DEST_SHARDS = 64

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
    "and install instructions. It needs NCCL 2.30.5 or newer, and "
    "VLLM_NCCL_SO_PATH must point at the same libnccl.so that libnccl_m2n.so "
    "was linked against."
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

    One mesh describes every tensor on its side, so it is exchanged once at the
    init handshake rather than per parameter.
    """

    dims: tuple[int, int]
    start_rank: int

    def __post_init__(self) -> None:
        if len(self.dims) != MESH_NDIMS or any(d <= 0 for d in self.dims):
            raise ValueError(f"mesh dims must be {MESH_NDIMS} positive ints")
        if self.start_rank < 0:
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


def check_placements(placements: Placements, context: str = "placements") -> None:
    """Reject placement pairs m2n cannot express.

    The header requires exactly one SHARD axis and one REPLICATE axis.
    `{REPLICATE, REPLICATE}` hits a degenerate prepare branch, which is why
    full replication is carried as `REPLICATED` and resolved separately;
    sharding both axes is not expressible at all.
    """
    if len(placements) != MESH_NDIMS:
        raise ValueError(f"{context} must have {MESH_NDIMS} entries")
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
    to plan a reshard. `placements` is relative to its side's `M2NMesh`, or
    `REPLICATED` when every rank holds the whole tensor.
    """

    placements: Placements | None


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


def comm_ptr(comm: "PyNcclCommunicator") -> int:
    """Raw `ncclComm_t` behind vLLM's `PyNcclCommunicator`.

    m2n links its own NCCL, so the handle only means anything if vLLM loaded
    the same `libnccl.so` — set `VLLM_NCCL_SO_PATH` accordingly.
    """
    handle = comm.comm
    ptr = getattr(handle, "value", handle)
    if not ptr:
        raise RuntimeError("PyNcclCommunicator has no live NCCL communicator")
    return int(ptr)
