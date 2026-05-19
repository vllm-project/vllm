# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Etha-style M-to-N weight transfer engine.

The planner/executor here are independent re-implementations of the
ideas in https://github.com/cmriat/Etha 's `comm/` module: declarative
`(DeviceMesh, Placement)` on both sides, compute one M-to-N chunk plan
per pair at init time, then move bytes straight from each source
rank's local shard into each target rank's local shard.

Two simplifications vs Etha proper:

1. We piggyback on vLLM's `StatelessProcessGroup` + `PyNcclCommunicator`
   rendezvous (same shape `NCCLWeightTransferEngine` uses) rather than
   spawning a separate "agent" process group. One process group covers
   trainer + vLLM workers; both sides run the same planning code and
   arrive at identical M2M maps.

2. The M2M planner is pure shape math — `DeviceMesh.distribute_tensor`
   would need a real `torch.distributed` PG, which we don't have. The
   ownership and routing it would compute is deterministic from
   `(mesh, placements)`, so we compute it directly.

3. Broadcasts (one source slice fanning out to multiple targets) are
   degraded to fan-out P2P sends. NCCL broadcast over an arbitrary
   subgroup would require a second `PyNcclCommunicator` per subset,
   which is more rendezvous than it's worth for MVP. Bytes-heavy
   pairs (`experts_*`) are pure P2P anyway.
"""

from __future__ import annotations

import itertools
import math
import time
from collections import OrderedDict, defaultdict
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch
from torch.distributed.tensor.placement_types import Placement, Replicate, Shard

from vllm.config.parallel import ParallelConfig
from vllm.config.weight_transfer import WeightTransferConfig
from vllm.distributed.utils import StatelessProcessGroup
from vllm.distributed.weight_transfer.base import (
    WeightTransferEngine,
    WeightTransferInitInfo,
    WeightTransferUpdateInfo,
)
from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator

logger = init_logger(__name__)


# ============================================================================
# Handler taxonomy (Qwen3-MoE specific; lift to model config later)
# ============================================================================

# Order matters: most-specific keyword first.
HANDLER_KEYWORDS: tuple[tuple[str, str], ...] = (
    ("q_norm", "layernorm"),
    ("k_norm", "layernorm"),
    ("input_layernorm", "layernorm"),
    ("post_attention_layernorm", "layernorm"),
    ("model.norm", "layernorm"),
    ("embed_tokens", "embed_tokens"),
    ("q_proj", "qkv_proj"),
    ("k_proj", "qkv_proj"),
    ("v_proj", "qkv_proj"),
    ("o_proj", "o_proj"),
    ("mlp.gate.weight", "router"),
    ("experts.gate_up_proj", "experts_gate_up"),
    ("experts.down_proj", "experts_down"),
    ("lm_head", "lm_head"),
)

MOE_HANDLERS = frozenset({"experts_gate_up", "experts_down"})


def get_handler_name(param_name: str) -> str | None:
    for kw, handler in HANDLER_KEYWORDS:
        if kw in param_name:
            return handler
    return None


# Trainer-side per-pair placements. Hierarchy follows the Etha
# vllm_weight_sync example:
#   att_mesh = (dp_replicate, dp_shard) 2D — dense weights
#   moe_mesh = (dp_replicate, dp_shard, ep) 3D — grouped MoE
TRAINER_HANDLER_PLACEMENTS: dict[str, tuple[Placement, ...]] = {
    "embed_tokens": (Replicate(), Shard(0)),
    "qkv_proj": (Replicate(), Shard(0)),
    "o_proj": (Replicate(), Shard(1)),
    "router": (Replicate(), Replicate()),
    "experts_gate_up": (Replicate(), Shard(0), Shard(0)),
    "experts_down": (Replicate(), Shard(0), Shard(0)),
    "lm_head": (Replicate(), Shard(0)),
    "layernorm": (Replicate(), Replicate()),
}


# ============================================================================
# Init / Update info dataclasses
# ============================================================================


@dataclass
class EthaWeightTransferInitInfo(WeightTransferInitInfo):
    """Rendezvous info for the Etha backend.

    Mirrors `NCCLWeightTransferInitInfo`: trainer is the lower-rank
    half of the joint process group, vLLM workers are the upper half.
    """

    master_address: str
    master_port: int
    rank_offset: int  # added to (dp_rank * world_size_per_dp + local_rank)
    world_size: int  # M trainer + N vLLM
    # Trainer mesh dims for the dense ("att") and MoE meshes. Must
    # match what the trainer-side helper announces; we hardcode here
    # for parity with the Etha example. Lift to per-pair handshake later.
    trainer_attn_dp_replicate: int = 2
    trainer_attn_dp_shard: int = 2
    trainer_moe_dp_replicate: int = 1
    trainer_moe_dp_shard: int = 2
    trainer_ep_size: int = 2


@dataclass
class EthaWeightTransferUpdateInfo(WeightTransferUpdateInfo):
    """Per-round update info. Intentionally empty — everything (pair
    placements, chunk plan) is baked in at init_transfer_engine."""

    version: int = 0


# ============================================================================
# Pure-math M2M planner
# ============================================================================


def _tensor_ndim(placements: tuple[Placement, ...]) -> int:
    """Tensor ndim implied by the max Shard dim across placements."""
    return (
        max(
            (p.dim for p in placements if isinstance(p, Shard)),
            default=0,
        )
        + 1
    )


def _shard_shape(
    mesh_shape: tuple[int, ...],
    placements: tuple[Placement, ...],
    tensor_ndim: int,
) -> list[int]:
    """Sharding factor per tensor dim: product of mesh dims that Shard it."""
    out = [1] * tensor_ndim
    for mesh_dim, p in enumerate(placements):
        if isinstance(p, Shard):
            out[p.dim] *= mesh_shape[mesh_dim]
    return out


def _slicer_tuples(
    tensor_shape: tuple[int, ...], num_slicers: list[int]
) -> list[tuple[slice, ...]]:
    """Slice tuples for chunking a tensor into num_slicers pieces per dim."""
    per_dim = []
    for d, n in enumerate(num_slicers):
        size = tensor_shape[d] // n
        per_dim.append([slice(i * size, (i + 1) * size) for i in range(n)])
    return list(itertools.product(*per_dim))


def _idx_to_linear(idx: tuple[int, ...], shape: list[int] | tuple[int, ...]) -> int:
    """Row-major flatten of a multi-dim index."""
    flat = 0
    for c, s in zip(idx, shape):
        flat = flat * s + c
    return flat


def _cell_owners(
    middle_idx: tuple[int, ...],
    mesh_shape: tuple[int, ...],
    mesh_ranks: list[int],
    placements: tuple[Placement, ...],
    middle_shape: tuple[int, ...],
) -> list[tuple[int, tuple[int, ...]]]:
    """All `(rank, local_idx)` pairs that hold a replica of this cell.

    Shard dims along a given tensor dim fix the mesh position
    deterministically — when multiple mesh dims shard the same tensor
    dim, we apply them in declaration order (coarser to finer).
    Replicate dims leave the mesh position free, so the cell has one
    replica per position along that mesh dim. The returned list
    enumerates the cross-product of free positions.
    """
    fixed_coord: list[int | None] = [None] * len(mesh_shape)
    local_idx = list(middle_idx)
    for d in range(len(middle_idx)):
        remaining = middle_idx[d]
        remaining_size = middle_shape[d]
        for mesh_dim, p in enumerate(placements):
            if isinstance(p, Shard) and p.dim == d:
                slice_size = remaining_size // mesh_shape[mesh_dim]
                fixed_coord[mesh_dim] = remaining // slice_size
                remaining = remaining % slice_size
                remaining_size = slice_size
        local_idx[d] = remaining

    options = [
        [c] if c is not None else list(range(mesh_shape[mesh_dim]))
        for mesh_dim, c in enumerate(fixed_coord)
    ]
    out: list[tuple[int, tuple[int, ...]]] = []
    local_tuple = tuple(local_idx)
    for coord in itertools.product(*options):
        out.append((mesh_ranks[_idx_to_linear(coord, mesh_shape)], local_tuple))
    return out


def compute_m2m_map(
    source_mesh: torch.Tensor,
    source_placements: tuple[Placement, ...],
    target_mesh: torch.Tensor,
    target_placements: tuple[Placement, ...],
) -> tuple[
    dict[int, dict[tuple, list[tuple[int, tuple]]]],
    list[int],
    list[int],
]:
    """Compute the M-to-N redistribution map.

    Both sides call this with identical inputs and arrive at the same
    map — it's pure shape math. The map is keyed by source rank and
    source chunk index in middle-shape units; values are lists of
    (target_rank, target_chunk_idx) recipients.
    """
    tensor_ndim = max(_tensor_ndim(source_placements), _tensor_ndim(target_placements))

    src_mesh_shape = tuple(source_mesh.shape)
    tgt_mesh_shape = tuple(target_mesh.shape)
    src_mesh_ranks = source_mesh.flatten().tolist()
    tgt_mesh_ranks = target_mesh.flatten().tolist()

    src_shard = _shard_shape(src_mesh_shape, source_placements, tensor_ndim)
    tgt_shard = _shard_shape(tgt_mesh_shape, target_placements, tensor_ndim)
    middle_shape = tuple(math.lcm(s, t) for s, t in zip(src_shard, tgt_shard))
    src_num_slicers = [m // s for m, s in zip(middle_shape, src_shard)]
    tgt_num_slicers = [m // t for m, t in zip(middle_shape, tgt_shard)]

    m2m: defaultdict[int, defaultdict[tuple, list[tuple[int, tuple]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    seen: set[tuple] = set()
    for mid in itertools.product(*[range(d) for d in middle_shape]):
        src_owners = _cell_owners(
            mid, src_mesh_shape, src_mesh_ranks, source_placements, middle_shape
        )
        tgt_owners = _cell_owners(
            mid, tgt_mesh_shape, tgt_mesh_ranks, target_placements, middle_shape
        )
        # Each target replica gets exactly one send. Pair via round-robin
        # over the source replicas — equivalent to Etha's "source idx i
        # routes to target idxs {i, i+|src|, ...}" pattern.
        for i, (dst_rank, dst_idx) in enumerate(tgt_owners):
            src_rank, src_idx = src_owners[i % len(src_owners)]
            key = (src_rank, src_idx, dst_rank, dst_idx)
            if key in seen:
                continue
            seen.add(key)
            m2m[src_rank][src_idx].append((dst_rank, dst_idx))

    return (
        {k: dict(v) for k, v in m2m.items()},
        src_num_slicers,
        tgt_num_slicers,
    )


# ============================================================================
# Chunk specialization
# ============================================================================


@dataclass
class Chunk:
    """A planned send / recv / self-copy operation on the local tensor.

    `is_source` is True for sends and self-copies (the rank is producing
    bytes); False for recvs.
    """

    is_source: bool
    src_rank: int
    dst_rank: int
    slice_tuples: tuple[slice, ...]
    tensor: torch.Tensor
    transfer_dtype: torch.dtype
    # For self-copy only: the slice of `tensor` to read from.
    src_slice_tuples: tuple[slice, ...] | None = None

    @property
    def is_self_copy(self) -> bool:
        return self.src_slice_tuples is not None


def map_to_chunk_ops(
    m2m_map: dict[int, dict[tuple, list[tuple[int, tuple]]]],
    rank: int,
    src_num_slicers: list[int],
    tgt_num_slicers: list[int],
    src_tensor: torch.Tensor | None,
    tgt_tensor: torch.Tensor | None,
    transfer_dtype: torch.dtype,
) -> list[Chunk]:
    """Specialize a shape-independent M2M map into per-rank send/recv chunks.

    Fan-out broadcasts are emitted as N separate P2P sends. This is the
    MVP shortcut documented in the engine module docstring.
    """
    def _build_slicers(tensor, num_slicers):
        if tensor is None:
            return None
        pad = [1] * (len(tensor.shape) - len(num_slicers))
        return _slicer_tuples(tuple(tensor.shape), num_slicers + pad)

    src_slicers = _build_slicers(src_tensor, src_num_slicers)
    tgt_slicers = _build_slicers(tgt_tensor, tgt_num_slicers)

    chunks: list[Chunk] = []
    for src_rank, sub in m2m_map.items():
        for src_idx, dst_list in sub.items():
            for dst_rank, dst_idx in dst_list:
                if src_rank == rank and dst_rank == rank:
                    # Self-copy: no NCCL op needed.
                    assert tgt_tensor is not None and src_tensor is not None
                    assert tgt_slicers is not None and src_slicers is not None
                    chunks.append(
                        Chunk(
                            is_source=True,
                            src_rank=src_rank,
                            dst_rank=dst_rank,
                            slice_tuples=tgt_slicers[
                                _idx_to_linear(dst_idx, tgt_num_slicers)
                            ],
                            src_slice_tuples=src_slicers[
                                _idx_to_linear(src_idx, src_num_slicers)
                            ],
                            tensor=tgt_tensor,
                            transfer_dtype=transfer_dtype,
                        )
                    )
                elif src_rank == rank:
                    assert src_tensor is not None and src_slicers is not None
                    chunks.append(
                        Chunk(
                            is_source=True,
                            src_rank=src_rank,
                            dst_rank=dst_rank,
                            slice_tuples=src_slicers[
                                _idx_to_linear(src_idx, src_num_slicers)
                            ],
                            tensor=src_tensor,
                            transfer_dtype=transfer_dtype,
                        )
                    )
                elif dst_rank == rank:
                    assert tgt_tensor is not None and tgt_slicers is not None
                    chunks.append(
                        Chunk(
                            is_source=False,
                            src_rank=src_rank,
                            dst_rank=dst_rank,
                            slice_tuples=tgt_slicers[
                                _idx_to_linear(dst_idx, tgt_num_slicers)
                            ],
                            tensor=tgt_tensor,
                            transfer_dtype=transfer_dtype,
                        )
                    )
    return chunks


# ============================================================================
# Executor
# ============================================================================


def chunk_comm(
    chunks: list[Chunk],
    pynccl: PyNcclCommunicator,
    label: str = "chunk_comm",
) -> None:
    """Execute a chunk list via NCCL. Sends + recvs are batched in one
    NCCL group; self-copies happen outside the group. After the group
    closes we synchronize the current stream and copy any
    dtype-converted recv buffers back into the destination slice.

    Logs prep / NCCL / post-process timings via `time.monotonic` and
    reports the total bytes moved over the wire.
    """
    intermediates: list[tuple[Chunk, torch.Tensor]] = []
    ops: list[tuple[Chunk, torch.Tensor]] = []
    wire_bytes = 0

    t0 = time.monotonic()
    for chunk in chunks:
        if chunk.is_self_copy:
            assert chunk.src_slice_tuples is not None
            src = chunk.tensor[chunk.src_slice_tuples]
            dst = chunk.tensor[chunk.slice_tuples]
            if src.dtype != dst.dtype:
                dst.copy_(src.to(dst.dtype))
            else:
                dst.copy_(src)
            continue

        slot = chunk.tensor[chunk.slice_tuples]
        if chunk.is_source:
            buf = slot if slot.is_contiguous() else slot.contiguous()
            if buf.dtype != chunk.transfer_dtype:
                buf = buf.to(chunk.transfer_dtype)
        else:
            # Recv buffer: must be contiguous and in transfer_dtype. If
            # the slot already matches, recv straight into it; otherwise
            # we recv into an intermediate and copy back below.
            if slot.is_contiguous() and slot.dtype == chunk.transfer_dtype:
                buf = slot
            else:
                buf = torch.empty(
                    slot.shape, dtype=chunk.transfer_dtype, device=slot.device
                )
                intermediates.append((chunk, buf))
        wire_bytes += buf.numel() * buf.element_size()
        ops.append((chunk, buf))
    t_prep = time.monotonic() - t0

    t0 = time.monotonic()
    pynccl.group_start()
    try:
        for chunk, buf in ops:
            if chunk.is_source:
                pynccl.send(buf, dst=chunk.dst_rank)
            else:
                pynccl.recv(buf, src=chunk.src_rank)
    finally:
        pynccl.group_end()

    # Stream-sync before touching recv buffers from the host side.
    torch.cuda.current_stream().synchronize()
    t_nccl = time.monotonic() - t0

    t0 = time.monotonic()
    for chunk, buf in intermediates:
        dst = chunk.tensor[chunk.slice_tuples]
        dst.copy_(buf.to(dst.dtype) if buf.dtype != dst.dtype else buf)
    t_post = time.monotonic() - t0

    total = t_prep + t_nccl + t_post
    gib = wire_bytes / (1024**3)
    bw = (gib / t_nccl) if t_nccl > 0 else float("inf")
    logger.info(
        "[%s] %d ops, %.3f GiB on the wire | "
        "prep=%.3fs nccl+sync=%.3fs post=%.3fs | "
        "total=%.3fs (%.2f GiB/s on-wire)",
        label,
        len(ops),
        gib,
        t_prep,
        t_nccl,
        t_post,
        total,
        bw,
    )


# ============================================================================
# vLLM placement inference (walks model, classifies modules by type)
# ============================================================================


def _vllm_dptp_mesh(parallel_config: ParallelConfig, base_rank: int) -> torch.Tensor:
    dp = parallel_config.data_parallel_size
    tp = parallel_config.tensor_parallel_size
    return torch.arange(base_rank, base_rank + dp * tp).view(dp, tp)


def _vllm_ep_mesh(ep_size: int, base_rank: int) -> torch.Tensor:
    return torch.arange(base_rank, base_rank + ep_size).view(ep_size)


def _vllm_param_placements(
    model: torch.nn.Module,
    dptp_mesh: torch.Tensor,
    ep_mesh: torch.Tensor | None,
) -> dict[str, tuple[torch.Tensor, tuple[Placement, ...]]]:
    """For each named parameter, infer (mesh, placements) by module type.

    Mirrors the `_get_placements` walk in
    `Etha/examples/vllm_weight_sync/vllm_server.py`. Skips params we
    don't transfer (no handler match).
    """
    from vllm.model_executor.layers.fused_moe import FusedMoE
    from vllm.model_executor.layers.linear import (
        ColumnParallelLinear,
        RowParallelLinear,
    )
    from vllm.model_executor.layers.vocab_parallel_embedding import (
        VocabParallelEmbedding,
    )

    rep_rep = (Replicate(), Replicate())
    rep_sh0 = (Replicate(), Shard(0))
    rep_sh1 = (Replicate(), Shard(1))

    out: dict[str, tuple[torch.Tensor, tuple[Placement, ...]]] = {}
    for module_name, module in model.named_modules():
        if isinstance(module, ColumnParallelLinear):
            mesh, weight_pl, bias_pl = dptp_mesh, rep_sh0, rep_sh0
        elif isinstance(module, RowParallelLinear):
            mesh, weight_pl, bias_pl = dptp_mesh, rep_sh1, rep_rep
        elif isinstance(module, VocabParallelEmbedding):
            mesh, weight_pl, bias_pl = dptp_mesh, rep_sh0, None
        elif isinstance(module, FusedMoE):
            if ep_mesh is None:
                continue
            mesh, weight_pl, bias_pl = ep_mesh, (Shard(0),), (Shard(0),)
        else:
            mesh, weight_pl, bias_pl = dptp_mesh, rep_rep, rep_rep

        for pname, _ in module.named_parameters(recurse=False):
            full = f"{module_name}.{pname}"
            if "weight" in pname and weight_pl is not None:
                out[full] = (mesh, weight_pl)
            elif "bias" in pname and bias_pl is not None:
                out[full] = (mesh, bias_pl)
    return out


# ============================================================================
# vLLM ↔ HF name conversion (Qwen3-MoE)
# ============================================================================


def _qkv_split(
    qkv: torch.Tensor, hf_config
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    head_dim = (
        hf_config.head_dim
        if hasattr(hf_config, "head_dim")
        else hf_config.hidden_size // hf_config.num_attention_heads
    )
    nq, nkv = hf_config.num_attention_heads, hf_config.num_key_value_heads
    total = nq + 2 * nkv
    qkv_3d = qkv.view(-1, head_dim, qkv.shape[-1])
    scale = total / qkv_3d.shape[0]
    q_n, kv_n = int(nq / scale), int(nkv / scale)
    q = qkv_3d[:q_n].reshape(-1, qkv.shape[-1])
    k = qkv_3d[q_n : q_n + kv_n].reshape(-1, qkv.shape[-1])
    v = qkv_3d[q_n + kv_n : q_n + 2 * kv_n].reshape(-1, qkv.shape[-1])
    return q, k, v


def _convert_vllm_state_dict(
    state_dict: dict[str, torch.Tensor], hf_config
) -> OrderedDict[str, tuple[str, torch.Tensor]]:
    """vLLM state-dict → {trainer-side name: (vllm orig name, tensor)}.

    Splits fused qkv into q/k/v views; renames FusedMoE w13/w2 to the
    transformers Qwen3MoeExperts grouped layout.
    """
    out: OrderedDict[str, tuple[str, torch.Tensor]] = OrderedDict()
    for vllm_name, tensor in state_dict.items():
        if not isinstance(tensor, torch.Tensor):
            continue
        if vllm_name.endswith("self_attn.qkv_proj.weight"):
            base = vllm_name[: -len("self_attn.qkv_proj.weight")]
            q, k, v = _qkv_split(tensor, hf_config)
            out[base + "self_attn.q_proj.weight"] = (vllm_name, q)
            out[base + "self_attn.k_proj.weight"] = (vllm_name, k)
            out[base + "self_attn.v_proj.weight"] = (vllm_name, v)
            continue
        if vllm_name.endswith("mlp.experts.w13_weight"):
            out[vllm_name[: -len("w13_weight")] + "gate_up_proj"] = (vllm_name, tensor)
            continue
        if vllm_name.endswith("mlp.experts.w2_weight"):
            out[vllm_name[: -len("w2_weight")] + "down_proj"] = (vllm_name, tensor)
            continue
        out[vllm_name] = (vllm_name, tensor)
    return out


def _detect_cutlass_swap(model: torch.nn.Module) -> bool:
    """FlashInfer CUTLASS/TRTLLM MoE backends swap w13 from [gate; up] to
    [up; gate] in `process_weights_after_loading`. Detect that so we
    register the gate/up views into the right halves."""
    from vllm.model_executor.layers.fused_moe import FusedMoE

    try:
        from vllm.model_executor.layers.fused_moe.oracle.unquantized import (
            UnquantizedMoeBackend,
        )
    except ImportError:
        return False

    for _, module in model.named_modules():
        if isinstance(module, FusedMoE):
            backend = getattr(module.quant_method, "unquantized_backend", None)
            return backend in (
                getattr(UnquantizedMoeBackend, "FLASHINFER_CUTLASS", None),
                getattr(UnquantizedMoeBackend, "FLASHINFER_TRTLLM", None),
            )
    return False


# ============================================================================
# Trainer meshes (mirror of the Etha example's HANDLER_PLACEMENTS table)
# ============================================================================


def trainer_meshes(
    init_info: EthaWeightTransferInitInfo,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build the trainer-side att_mesh and moe_mesh from init info.

    Trainer ranks are [0, M) where M = attn_r * attn_s = moe_r * moe_s * ep.
    """
    a_r, a_s = init_info.trainer_attn_dp_replicate, init_info.trainer_attn_dp_shard
    m_r, m_s, ep = (
        init_info.trainer_moe_dp_replicate,
        init_info.trainer_moe_dp_shard,
        init_info.trainer_ep_size,
    )
    att = torch.arange(a_r * a_s).view(a_r, a_s)
    moe = torch.arange(m_r * m_s * ep).view(m_r, m_s, ep)
    return att, moe


# ============================================================================
# Engine
# ============================================================================


@dataclass
class _RegisteredTensor:
    pair_name: str
    handler: str
    name: str
    tensor: torch.Tensor


class EthaWeightTransferEngine(
    WeightTransferEngine[EthaWeightTransferInitInfo, EthaWeightTransferUpdateInfo]
):
    """Weight transfer engine that uses M-to-N resharding + NCCL P2P.

    Designed to be used with `start_weight_update(is_checkpoint_format=False)`:
    chunks write directly into kernel-format parameter storage, so the
    `load_weights` callable passed to `receive_weights` is ignored. This
    matches what the Etha example does — call `process_weights_after_loading`
    once at vLLM startup (with `--load-format dummy`), then keep writing
    into the kernel-format tensors forever.
    """

    init_info_cls = EthaWeightTransferInitInfo
    update_info_cls = EthaWeightTransferUpdateInfo

    def __init__(
        self, config: WeightTransferConfig, parallel_config: ParallelConfig
    ) -> None:
        super().__init__(config, parallel_config)
        self.stateless_pg: StatelessProcessGroup | None = None
        self.pynccl: PyNcclCommunicator | None = None
        self._recv_chunks: list[Chunk] = []
        self._registered: list[_RegisteredTensor] = []
        self._model: torch.nn.Module | None = None

    # ------------------------------------------------------------------
    # init_weight_transfer_engine path
    # ------------------------------------------------------------------

    def init_transfer_engine(
        self,
        init_info: EthaWeightTransferInitInfo,
        model: torch.nn.Module | None = None,
    ) -> None:
        # We need the model to walk the module graph for per-parameter
        # placement inference. The gpu_worker passes this in for us.
        if model is None:
            raise RuntimeError(
                "EthaWeightTransferEngine.init_transfer_engine requires the "
                "loaded model — gpu_worker passes this for you. If you're "
                "calling the engine directly, pass model=<your inference model>."
            )
        self._model = model

        # 1. Rendezvous (same shape as NCCLWeightTransferEngine).
        worker_rank = (
            self.parallel_config.data_parallel_index
            * self.parallel_config.world_size
            + self.parallel_config.rank
        )
        rank = worker_rank + init_info.rank_offset
        device = torch.accelerator.current_device_index()
        self.stateless_pg = StatelessProcessGroup.create(
            host=init_info.master_address,
            port=init_info.master_port,
            rank=rank,
            world_size=init_info.world_size,
        )
        from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator

        self.pynccl = PyNcclCommunicator(self.stateless_pg, device=device)
        self.rank = rank
        logger.info(
            "Etha engine rank=%d world_size=%d initialized",
            rank,
            init_info.world_size,
        )

        # 2. Plan + register tensors.
        self._build_plan(init_info)

    def _build_plan(self, init_info: EthaWeightTransferInitInfo) -> None:
        assert self._model is not None
        assert self.pynccl is not None

        # vLLM-side meshes (ranks shifted by trainer_world_size).
        trainer_world_size = (
            init_info.trainer_attn_dp_replicate * init_info.trainer_attn_dp_shard
        )
        dptp_mesh = _vllm_dptp_mesh(self.parallel_config, base_rank=trainer_world_size)
        from vllm.model_executor.layers.fused_moe import FusedMoE

        ep_size = 0
        for _, m in self._model.named_modules():
            if isinstance(m, FusedMoE):
                ep_size = m.ep_size
                break
        ep_mesh = (
            _vllm_ep_mesh(ep_size, base_rank=trainer_world_size) if ep_size else None
        )

        vllm_placements = _vllm_param_placements(self._model, dptp_mesh, ep_mesh)

        # Trainer-side meshes (ranks 0..M-1).
        trainer_att, trainer_moe = trainer_meshes(init_info)

        # Convert state dict to trainer-side names.
        from vllm.config import ModelConfig  # noqa: F401 (typing hint only)

        hf_config = getattr(
            getattr(self._model, "config", None),
            "text_config",
            getattr(self._model, "config", None),
        )
        if hf_config is None:
            # Fall back to the engine's parallel config; if we don't have
            # an HF config the QKV split will fail later.
            raise RuntimeError("Could not locate hf_config on the model")

        converted = _convert_vllm_state_dict(self._model.state_dict(), hf_config)
        cutlass_swap = _detect_cutlass_swap(self._model)

        # Group registered tensors by handler/pair name.
        by_pair: dict[str, list[tuple[str, torch.Tensor]]] = defaultdict(list)
        for trainer_name, (vllm_orig_name, tensor) in converted.items():
            handler = get_handler_name(trainer_name)
            if handler is None:
                continue
            if vllm_orig_name not in vllm_placements:
                continue
            if trainer_name.endswith("mlp.experts.gate_up_proj"):
                # Boundary split: vLLM stores w13 as [gate; up] under
                # TRITON and as [up; gate] under CUTLASS post-load.
                half = tensor.shape[1] // 2
                base = trainer_name[: -len("gate_up_proj")]
                if cutlass_swap:
                    gate_view, up_view = tensor[:, half:, :], tensor[:, :half, :]
                else:
                    gate_view, up_view = tensor[:, :half, :], tensor[:, half:, :]
                by_pair[handler].append((base + "gate_proj", gate_view))
                by_pair[handler].append((base + "up_proj", up_view))
                continue
            by_pair[handler].append((trainer_name, tensor))

        # Plan each pair: compute M2M map, then specialize chunks per tensor.
        for handler in sorted(by_pair.keys()):
            mesh_v, placements_v = self._pair_vllm_mesh_placements(
                handler, dptp_mesh, ep_mesh, by_pair[handler], vllm_placements
            )
            mesh_t, placements_t = self._pair_trainer_mesh_placements(
                handler, trainer_att, trainer_moe
            )
            m2m_map, src_num, tgt_num = compute_m2m_map(
                mesh_t, placements_t, mesh_v, placements_v
            )
            logger.info(
                "Etha pair %s: src_mesh=%s tgt_mesh=%s "
                "src_pl=%s tgt_pl=%s src_num=%s tgt_num=%s",
                handler,
                tuple(mesh_t.shape),
                tuple(mesh_v.shape),
                placements_t,
                placements_v,
                src_num,
                tgt_num,
            )
            for name, view in sorted(by_pair[handler], key=lambda x: x[0]):
                self._registered.append(
                    _RegisteredTensor(
                        pair_name=handler,
                        handler=handler,
                        name=name,
                        tensor=view,
                    )
                )
                chunks = map_to_chunk_ops(
                    m2m_map,
                    rank=self.rank,
                    src_num_slicers=src_num,
                    tgt_num_slicers=tgt_num,
                    src_tensor=None,  # we never produce sends on the vLLM side
                    tgt_tensor=view,
                    transfer_dtype=view.dtype,
                )
                self._recv_chunks.extend(chunks)

        from collections import Counter as _Counter

        per_pair = _Counter(
            (c.src_rank, c.dst_rank) for c in self._recv_chunks
        )
        logger.info(
            "Etha engine planned %d recv chunks across %d tensors / %d pairs; "
            "per (src→dst): %s",
            len(self._recv_chunks),
            len(self._registered),
            len(by_pair),
            dict(sorted(per_pair.items())),
        )

    def _pair_vllm_mesh_placements(
        self,
        handler: str,
        dptp_mesh: torch.Tensor,
        ep_mesh: torch.Tensor | None,
        tensors: list[tuple[str, torch.Tensor]],
        vllm_placements: dict[str, tuple[torch.Tensor, tuple[Placement, ...]]],
    ) -> tuple[torch.Tensor, tuple[Placement, ...]]:
        # Pick the first tensor's mesh/placements; they're identical across
        # tensors that share a handler (same module type).
        for name, _ in tensors:
            # Slice-derived views (gate_proj/up_proj/q_proj/...) won't be
            # in vllm_placements directly; fall back to their fused parent.
            if name in vllm_placements:
                return vllm_placements[name]
            for parent in (
                name.replace("gate_proj", "gate_up_proj.weight").replace(
                    "up_proj", "gate_up_proj.weight"
                ),
                name.replace("q_proj", "qkv_proj").replace(
                    "k_proj", "qkv_proj"
                ).replace("v_proj", "qkv_proj"),
            ):
                if parent in vllm_placements:
                    return vllm_placements[parent]
        # Last resort.
        if handler in MOE_HANDLERS and ep_mesh is not None:
            return ep_mesh, (Shard(0),)
        return dptp_mesh, (Replicate(), Replicate())

    def _pair_trainer_mesh_placements(
        self, handler: str, att_mesh: torch.Tensor, moe_mesh: torch.Tensor
    ) -> tuple[torch.Tensor, tuple[Placement, ...]]:
        placements = TRAINER_HANDLER_PLACEMENTS[handler]
        mesh = moe_mesh if handler in MOE_HANDLERS else att_mesh
        return mesh, placements

    # ------------------------------------------------------------------
    # update_weights path
    # ------------------------------------------------------------------

    def receive_weights(
        self,
        update_info: EthaWeightTransferUpdateInfo,
        load_weights: Callable[[list[tuple[str, torch.Tensor]]], None],
    ) -> None:
        if self.pynccl is None:
            raise RuntimeError("Etha engine not initialized")
        t0 = time.monotonic()
        torch.cuda.synchronize()
        chunk_comm(
            self._recv_chunks,
            self.pynccl,
            label=f"recv rank={self.rank} v={update_info.version}",
        )
        torch.cuda.synchronize()
        logger.info(
            "Etha receive_weights rank=%d v=%d wall=%.3fs",
            self.rank,
            update_info.version,
            time.monotonic() - t0,
        )
        # load_weights is deliberately unused — Etha wrote in place above.
        # IMPORTANT: callers must use start_weight_update(is_checkpoint_format=False);
        # layerwise reload would clobber the writes we just did.
        del load_weights

    def shutdown(self) -> None:
        self.pynccl = None
        self.stateless_pg = None
        self._recv_chunks = []
        self._registered = []

    @staticmethod
    def trainer_send_weights(
        iterator: Iterator[tuple[str, torch.Tensor]],
        trainer_args: dict[str, Any] | Any,
    ) -> None:
        raise NotImplementedError(
            "Etha is symmetric — use the helpers in this module from the "
            "trainer process (trainer_init_etha + send_weights_etha). "
            "See examples/rl/rlhf_etha.py."
        )


# ============================================================================
# Trainer-side helpers (used by the example launcher; live here so the
# constants and planner are shared with the engine)
# ============================================================================


@dataclass
class EthaTrainerHandle:
    rank: int
    world_size: int
    pynccl: PyNcclCommunicator
    stateless_pg: StatelessProcessGroup
    send_chunks: list[Chunk]


def trainer_init_etha(
    master_address: str,
    master_port: int,
    rank: int,
    world_size: int,
    device_index: int,
    trainer_state_dict: dict[str, torch.Tensor],
    trainer_att_mesh: torch.Tensor,
    trainer_moe_mesh: torch.Tensor,
    vllm_dptp_mesh_shape: tuple[int, int],
    vllm_ep_mesh_shape: tuple[int],
    trainer_world_size: int,
    wire_dtype: torch.dtype = torch.bfloat16,
) -> EthaTrainerHandle:
    """Trainer-side init: rendezvous + plan send chunks.

    `trainer_state_dict` maps trainer-side parameter names → local
    shard tensor (a `DTensor.to_local()`). The planner runs identically
    to the vLLM side; both arrive at the same M2M map.

    `wire_dtype` must match the dtype the vLLM-side parameters hold
    on receive (typically `torch.bfloat16`). If the trainer's tensors
    are in a wider dtype (e.g. fp32 master weights), `chunk_comm`
    downcasts to `wire_dtype` before issuing the send. NCCL pairs ops
    by byte count, so both sides must agree on the wire dtype or the
    NCCL group will deadlock waiting for matching bytes.
    """
    pg = StatelessProcessGroup.create(
        host=master_address, port=master_port, rank=rank, world_size=world_size
    )
    from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator

    pynccl = PyNcclCommunicator(pg, device=device_index)

    # Build the same vLLM-side meshes the engine builds, with the same
    # rank offset. Trainer ranks are 0..M-1, vLLM ranks are M..M+N-1.
    dp, tp = vllm_dptp_mesh_shape
    vllm_dptp = torch.arange(trainer_world_size, trainer_world_size + dp * tp).view(
        dp, tp
    )
    (ep,) = vllm_ep_mesh_shape
    vllm_ep = torch.arange(trainer_world_size, trainer_world_size + ep).view(ep)

    # Group tensors by handler. Mirror the boundary split the engine
    # does on the vLLM side: a grouped MoE `experts.gate_up_proj` is
    # registered as two views (`gate_proj`, `up_proj`) under the same
    # handler. Handler lookup is done on the PRE-split name so both
    # sides bake the same chunk list.
    by_pair: dict[str, list[tuple[str, torch.Tensor]]] = defaultdict(list)
    for name, t in trainer_state_dict.items():
        h = get_handler_name(name)
        if h is None:
            continue
        if name.endswith("mlp.experts.gate_up_proj"):
            half = t.shape[1] // 2
            base = name[: -len("gate_up_proj")]
            by_pair[h].append((base + "gate_proj", t[:, :half, :]))
            by_pair[h].append((base + "up_proj", t[:, half:, :]))
            continue
        by_pair[h].append((name, t))

    send_chunks: list[Chunk] = []
    for handler in sorted(by_pair.keys()):
        placements_t = TRAINER_HANDLER_PLACEMENTS[handler]
        mesh_t = trainer_moe_mesh if handler in MOE_HANDLERS else trainer_att_mesh
        if handler in MOE_HANDLERS:
            mesh_v, placements_v = vllm_ep, (Shard(0),)
        elif handler == "o_proj":
            mesh_v, placements_v = vllm_dptp, (Replicate(), Shard(1))
        elif handler == "router" or handler == "layernorm":
            mesh_v, placements_v = vllm_dptp, (Replicate(), Replicate())
        else:
            mesh_v, placements_v = vllm_dptp, (Replicate(), Shard(0))
        m2m_map, src_num, tgt_num = compute_m2m_map(
            mesh_t, placements_t, mesh_v, placements_v
        )
        for name, t in sorted(by_pair[handler], key=lambda x: x[0]):
            chunks = map_to_chunk_ops(
                m2m_map,
                rank=rank,
                src_num_slicers=src_num,
                tgt_num_slicers=tgt_num,
                src_tensor=t,
                tgt_tensor=None,
                transfer_dtype=wire_dtype,
            )
            send_chunks.extend(chunks)

    from collections import Counter as _Counter

    per_pair = _Counter((c.src_rank, c.dst_rank) for c in send_chunks)
    logger.info(
        "Etha trainer rank=%d planned %d send chunks; per (src→dst): %s",
        rank,
        len(send_chunks),
        dict(sorted(per_pair.items())),
    )

    return EthaTrainerHandle(
        rank=rank,
        world_size=world_size,
        pynccl=pynccl,
        stateless_pg=pg,
        send_chunks=send_chunks,
    )


def send_weights_etha(handle: EthaTrainerHandle) -> None:
    """Trainer-side per-round send."""
    t0 = time.monotonic()
    torch.cuda.synchronize()
    chunk_comm(
        handle.send_chunks, handle.pynccl, label=f"send rank={handle.rank}"
    )
    torch.cuda.synchronize()
    logger.info(
        "Etha send_weights_etha rank=%d wall=%.3fs",
        handle.rank,
        time.monotonic() - t0,
    )
