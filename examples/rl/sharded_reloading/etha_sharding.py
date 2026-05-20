# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Etha sharding strategy — transport-agnostic M-to-N planning.

A `EthaShardingStrategy` discovers per-handler (mesh, placements,
tensors) for one side of the transfer (trainer or inference worker),
runs the trace-based mark-and-recapture planner against the peer
side's declared placements, and emits `list[Chunk]` for this rank.

The strategy never touches NCCL. It uses gloo (via a temporary
torch.distributed default ProcessGroup) for the planning critical
section because `DeviceMesh` + `distribute_tensor` + `all_gather_object`
require a real PG, and the planner only moves tiny CPU tensors. A
caller wanting a different transport (NIXL, RDMA, ...) reuses the
strategy unchanged and supplies its own `chunk_comm` equivalent.

Two concrete subclasses:

- `VllmEthaShardingStrategy` — discovers placements by walking the
  loaded vLLM model's module tree (empirical, ground truth). Role
  is "tgt" (receiver).
- `TrainerEthaShardingStrategy` — discovers placements by looking up
  handler names in the static `TRAINER_HANDLER_PLACEMENTS` table.
  Role is "src" (sender).

Both share `plan_and_specialize`, the orchestration loop that builds
the cross-cluster PG, runs `get_m2m_map` per pair, and turns the
abstract maps into Chunks via `map_to_chunk_ops`. The only thing
subclasses contribute is `_build_pair_table`.

VLLM placements appear in TWO places today on the trainer side:
hardcoded in `TrainerEthaShardingStrategy._build_pair_table` and
empirically in `_vllm_param_placements`. The single source of truth
is `VLLM_HANDLER_PLACEMENTS` in this file; the vLLM-side strategy
cross-checks the empirical walker against the table at init and
raises if they disagree — turning a silent placement-drift bug into
a loud failure at startup.
"""

from __future__ import annotations

import contextlib
import datetime
import itertools
import math
import time
from abc import ABC, abstractmethod
from collections import OrderedDict, defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

import torch
import torch.distributed as dist
from torch.distributed import DeviceMesh
from torch.distributed.distributed_c10d import (
    _get_default_group,
    _new_process_group_helper,
    _update_default_pg,
)
from torch.distributed.tensor import distribute_tensor
from torch.distributed.tensor.placement_types import Placement, Replicate, Shard

from vllm.logger import init_logger

from etha_chunk import Chunk, map_to_chunk_ops

if TYPE_CHECKING:
    from vllm.config.parallel import ParallelConfig

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

# Single source of truth for vLLM-side placements per handler.
# Mesh kind: "dptp" = the (dp, tp) 2D mesh, "ep" = the 1D expert mesh.
# Worker-side strategy cross-checks the table against an empirical walk
# of the loaded model and raises on disagreement; trainer-side strategy
# uses the table directly as peer placement info.
VLLM_HANDLER_PLACEMENTS: dict[str, tuple[str, tuple[Placement, ...]]] = {
    "embed_tokens": ("dptp", (Replicate(), Shard(0))),
    "qkv_proj": ("dptp", (Replicate(), Shard(0))),
    "o_proj": ("dptp", (Replicate(), Shard(1))),
    "router": ("dptp", (Replicate(), Replicate())),
    "experts_gate_up": ("ep", (Shard(0),)),
    "experts_down": ("ep", (Shard(0),)),
    "lm_head": ("dptp", (Replicate(), Shard(0))),
    "layernorm": ("dptp", (Replicate(), Replicate())),
}


# ============================================================================
# Trace-based M2M planner (mark-and-recapture, vendored from Etha)
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


@contextlib.contextmanager
def _temporary_default_pg(pg: dist.ProcessGroup):
    """Swap torch's default ProcessGroup to `pg` for the with-block.

    Restored on exit (including on exception). Init-time only — caller
    is responsible for ensuring no concurrent torch.distributed calls
    in other threads of this process during the critical section.
    """
    saved = _get_default_group()
    _update_default_pg(pg)
    try:
        yield
    finally:
        _update_default_pg(saved)


def _build_cross_pg(
    store: torch._C._distributed_c10d.Store,
    rank: int,
    world_size: int,
    group_name: str = "etha_planner",
    timeout_s: float = 300.0,
) -> dist.ProcessGroup:
    """Build a real torch.distributed.ProcessGroup spanning all
    `world_size` ranks, backed by `store`.

    Caller must already have a default PG in this process (vLLM workers
    and trainer Ray actors do). Uses gloo over CPU — tiny metadata
    tensors only. Returned PG is registered in torch's world map (so
    `DeviceMesh` constructor and `new_group` against it work), but is
    NOT set as the default — callers swap it in via `_temporary_default_pg`
    for the duration of planning.

    `group_name` must be IDENTICAL on every participating rank — it is
    used as a store prefix to scope rendezvous keys. If ranks pick
    different names (e.g. each generates its own uuid) they never meet
    on the store and ProcessGroupGloo's constructor hangs forever.
    """
    cross_pg, _ = _new_process_group_helper(
        group_size=world_size,
        group_rank=rank,
        global_ranks_in_group=list(range(world_size)),
        backend="gloo",
        store=store,
        group_name=group_name,
        timeout=datetime.timedelta(seconds=timeout_s),
    )
    return cross_pg


def _destroy_mesh_pgs(meshes: list[DeviceMesh], my_rank: int) -> None:
    """Destroy the dim-subgroups DeviceMesh created. Only ranks that
    participate in a given mesh own its subgroups."""
    for mesh in meshes:
        if my_rank in mesh.mesh.flatten().tolist():
            for pg in mesh.get_all_groups():
                with contextlib.suppress(Exception):
                    dist.destroy_process_group(pg)


def get_m2m_map(
    source_mesh: DeviceMesh,
    source_placements: tuple[Placement, ...],
    target_mesh: DeviceMesh,
    target_placements: tuple[Placement, ...],
    group: dist.ProcessGroup,
    device: str = "cpu",
) -> tuple[dict[int, dict[tuple, list[tuple[int, tuple]]]], list[int], list[int]]:
    """Mark-and-recapture M2M planner.

    Each source rank fills its local shard of a small LCM-sized "middle
    tensor" with an encoded `(rank, local_idx)` fingerprint. DTensor
    `full_tensor()` gathers across the source mesh; the assembled tensor
    is shipped to target ranks (round-robin), where another
    `distribute_tensor` reshards by the target placements. Reading each
    cell of the target's local shard decodes back into
    `(src_rank, src_idx) → (dst_rank, dst_idx)`.

    Collective. Every rank in `group` must call in the same order.
    Caller must have set `group` as the current default PG.
    """
    rank = dist.get_rank()
    target_mesh_ranks = target_mesh.mesh.flatten().tolist()
    source_mesh_ranks = source_mesh.mesh.flatten().tolist()

    source_tensor_ndim = _tensor_ndim(source_placements)
    target_tensor_ndim = _tensor_ndim(target_placements)
    tensor_ndim = max(source_tensor_ndim, target_tensor_ndim)

    source_shard_shape = _shard_shape(
        source_mesh.mesh.shape, source_placements, tensor_ndim
    )
    target_shard_shape = _shard_shape(
        target_mesh.mesh.shape, target_placements, tensor_ndim
    )
    middle_tensor_shape = tuple(
        math.lcm(source_shard_shape[i], target_shard_shape[i])
        for i in range(len(source_shard_shape))
    )

    source_num_slicers: list[int] = []
    target_num_slicers: list[int] = []
    for o, m, t in zip(
        source_shard_shape, middle_tensor_shape, target_shard_shape, strict=False
    ):
        source_num_slicers.append(m // o)
        target_num_slicers.append(m // t)

    reqs: list[Any] = []
    full_tensor_restored: torch.Tensor | None = None
    if rank in source_mesh_ranks:
        middle_tensor = torch.zeros(middle_tensor_shape, device=device)
        dtensor_source = distribute_tensor(
            middle_tensor, source_mesh, source_placements
        )
        local_shard = dtensor_source.to_local()
        encoded_tensor = torch.zeros_like(local_shard)
        base = max(middle_tensor_shape) + 1
        for idx in itertools.product(*[range(d) for d in local_shard.shape]):
            encoded_value = rank
            for coord in idx:
                encoded_value = encoded_value * base + coord
            encoded_tensor[idx] = encoded_value
        local_shard.copy_(encoded_tensor)
        full_tensor_restored = dtensor_source.full_tensor()

        source_idx = source_mesh_ranks.index(rank)
        for target_idx in range(
            source_idx, len(target_mesh_ranks), len(source_mesh_ranks)
        ):
            target_rank = target_mesh_ranks[target_idx]
            reqs.append(dist.isend(full_tensor_restored, dst=target_rank))
    elif rank in target_mesh_ranks:
        full_tensor_restored = torch.empty(middle_tensor_shape, device=device)
        target_idx = target_mesh_ranks.index(rank)
        source_rank = source_mesh_ranks[target_idx % len(source_mesh_ranks)]
        reqs.append(dist.irecv(full_tensor_restored, src=source_rank))

    for req in reqs:
        if req is not None:
            req.wait()

    m2m_map: defaultdict[int, defaultdict[tuple, list[tuple[int, tuple]]]] = (
        defaultdict(lambda: defaultdict(list))
    )

    if rank in target_mesh_ranks:
        assert full_tensor_restored is not None
        dtensor_target = distribute_tensor(
            full_tensor_restored, target_mesh, target_placements, src_data_rank=None
        )
        local_target_shard = dtensor_target.to_local()
        base = max(middle_tensor_shape) + 1
        for target_idx_tuple in itertools.product(
            *[range(d) for d in local_target_shard.shape]
        ):
            encoded_value = int(local_target_shard[target_idx_tuple].item())
            source_indices: list[int] = []
            temp = encoded_value
            for _ in range(len(target_idx_tuple)):
                coord = temp % base
                source_indices.append(coord)
                temp = temp // base
            source_rank = temp
            source_indices.reverse()
            source_idx_tuple = tuple(source_indices)
            m2m_map[source_rank][source_idx_tuple].append((rank, target_idx_tuple))

    m2m_map_regular = {k: dict(v) for k, v in m2m_map.items()}

    group_world_size = dist.get_world_size(group)
    all_m2m_maps: list[Any] = [None] * group_world_size
    dist.all_gather_object(all_m2m_maps, m2m_map_regular, group=group)

    merged: defaultdict[int, defaultdict[tuple, list[tuple[int, tuple]]]] = (
        defaultdict(lambda: defaultdict(list))
    )
    for rank_map in all_m2m_maps:
        if rank_map is not None:
            for src_rank, src_idx_map in rank_map.items():
                for src_idx, dst_list in src_idx_map.items():
                    merged[src_rank][src_idx].extend(dst_list)

    final: dict[int, dict[tuple, list[tuple[int, tuple]]]] = {}
    for src_rank, src_idx_map in merged.items():
        final[src_rank] = dict(src_idx_map)

    return final, source_num_slicers, target_num_slicers


# ============================================================================
# vLLM-side state-dict conversion + CUTLASS detection
# ============================================================================


def _qkv_split(
    qkv: torch.Tensor, hf_config: Any
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
    state_dict: dict[str, torch.Tensor], hf_config: Any
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


def _vllm_param_placements(
    model: torch.nn.Module,
    dptp_mesh: torch.Tensor,
    ep_mesh: torch.Tensor | None,
) -> dict[str, tuple[torch.Tensor, tuple[Placement, ...]]]:
    """For each named parameter, infer (mesh, placements) by module type.

    Mirrors the `_get_placements` walk in Etha's example. Skips params
    we don't transfer (no handler match). Result is the ground-truth
    side of the cross-check against VLLM_HANDLER_PLACEMENTS.
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
# Pair specification + strategy ABC
# ============================================================================


@dataclass
class PairSpec:
    """One pair handler's planning input.

    `own_*` describes this side's mesh + placements; `peer_*` describes
    the other side's. `tensors` are this side's local tensors registered
    to the pair (post-split for gate_up / qkv).
    """

    handler: str
    own_mesh: torch.Tensor
    own_placements: tuple[Placement, ...]
    peer_mesh: torch.Tensor
    peer_placements: tuple[Placement, ...]
    tensors: list[tuple[str, torch.Tensor]] = field(default_factory=list)


class EthaShardingStrategy(ABC):
    """Plans M-to-N tensor reshardings between trainer and inference meshes.

    Transport-agnostic: produces a list of `Chunk`s but does not move
    bytes. Callers (the worker engine or trainer engine) execute the
    chunks over whatever transport they prefer.
    """

    role: Literal["src", "tgt"]

    @abstractmethod
    def _build_pair_table(self) -> dict[str, PairSpec]:
        """Side-specific: discover handler → PairSpec for this side."""

    def _transfer_dtype(self, tensor: torch.Tensor) -> torch.dtype:
        """Wire dtype for a registered tensor. Default: keep storage dtype.
        Trainer subclass overrides to coerce to a configured wire_dtype."""
        return tensor.dtype

    def plan_and_specialize(
        self,
        store: torch._C._distributed_c10d.Store,
        rank: int,
        world_size: int,
    ) -> list[Chunk]:
        """Entry point. Builds cross-PG, runs trace-based planner per
        pair, returns chunks for this rank to execute.
        """
        pair_table = self._build_pair_table()
        if not pair_table:
            logger.warning("EthaShardingStrategy: empty pair table")
            return []

        cross_pg = _build_cross_pg(store, rank, world_size)
        M2MMap = dict[int, dict[tuple, list[tuple[int, tuple]]]]
        pair_plans: dict[str, tuple[M2MMap, list[int], list[int]]] = {}
        meshes_to_destroy: list[DeviceMesh] = []
        t_plan = time.monotonic()
        try:
            with _temporary_default_pg(cross_pg):
                # Build each unique mesh tensor exactly once.
                #
                # WARNING: do not move this inside the per-pair loop. Recreating
                # the same mesh per pair triggers a deadlock inside gloo's
                # broadcast path during DTensor's mesh_broadcast for all-Replicate
                # placements (e.g. the `layernorm` and `router` pairs). py-spy at
                # hang time showed: trainer-side ranks at different stages of
                # get_m2m_map for the same pair — half at `broadcast` inside
                # distribute_tensor, half at `all_gather_object` at the end —
                # with matching group_names on both sides of the [2,3] subgroup
                # PGG, so it is NOT a counter-drift / store-prefix mismatch.
                # The deadlock appears to be inside gloo when multiple live PGGs
                # share the same rank composition (in the broken pattern, by
                # pair 4 the composition `[2,3]` exists in 3 distinct PGGs:
                # one from pair 1's trainer_att, one from pair 2/3's trainer_moe
                # dim-2 subgroups, one from pair 4's trainer_att). Caching
                # avoids creating duplicate PGGs for the same rank composition.
                #
                # Residual risk: this fix works for any topology where the
                # unique meshes used in the pair table have disjoint
                # dim-subgroup rank compositions. If a future topology
                # introduces two distinct meshes whose dim-subgroups happen
                # to share a rank set (e.g. a 3D trainer mesh whose dim-1
                # subgroups overlap with a 2D mesh's dim-0 subgroups), the
                # same gloo deadlock could re-emerge. If that happens, the
                # gloo broadcast hang inside `distribute_tensor` is where to
                # look — minimal repro: build a gloo cross-PG, create 3+
                # DeviceMesh instances under it whose dim subgroups overlap
                # on the same rank pairs, then call `distribute_tensor` with
                # all-Replicate placements on a later one.
                mesh_cache: dict[tuple, DeviceMesh] = {}

                def _get_mesh(mesh_tensor: torch.Tensor) -> DeviceMesh:
                    key = (tuple(mesh_tensor.shape), tuple(mesh_tensor.flatten().tolist()))
                    if key not in mesh_cache:
                        mesh_cache[key] = DeviceMesh("cpu", mesh_tensor)
                        meshes_to_destroy.append(mesh_cache[key])
                    return mesh_cache[key]

                for handler in sorted(pair_table):
                    spec = pair_table[handler]
                    if self.role == "src":
                        src_mesh = _get_mesh(spec.own_mesh)
                        tgt_mesh = _get_mesh(spec.peer_mesh)
                        src_pl = spec.own_placements
                        tgt_pl = spec.peer_placements
                    else:
                        src_mesh = _get_mesh(spec.peer_mesh)
                        tgt_mesh = _get_mesh(spec.own_mesh)
                        src_pl = spec.peer_placements
                        tgt_pl = spec.own_placements
                    m2m_map, src_num, tgt_num = get_m2m_map(
                        source_mesh=src_mesh,
                        source_placements=src_pl,
                        target_mesh=tgt_mesh,
                        target_placements=tgt_pl,
                        group=cross_pg,
                    )
                    pair_plans[handler] = (m2m_map, src_num, tgt_num)
                    logger.info(
                        "Etha pair %s: own_mesh=%s peer_mesh=%s "
                        "own_pl=%s peer_pl=%s src_num=%s tgt_num=%s",
                        handler,
                        tuple(spec.own_mesh.shape),
                        tuple(spec.peer_mesh.shape),
                        spec.own_placements,
                        spec.peer_placements,
                        src_num,
                        tgt_num,
                    )
                _destroy_mesh_pgs(meshes_to_destroy, my_rank=rank)
        finally:
            with contextlib.suppress(Exception):
                dist.destroy_process_group(cross_pg)
        logger.info(
            "Etha planner (role=%s) trace took %.3fs for %d pairs",
            self.role,
            time.monotonic() - t_plan,
            len(pair_plans),
        )

        chunks: list[Chunk] = []
        for handler in sorted(pair_table):
            spec = pair_table[handler]
            m2m_map, src_num, tgt_num = pair_plans[handler]
            for name, view in sorted(spec.tensors, key=lambda x: x[0]):
                pair_chunks = map_to_chunk_ops(
                    m2m_map,
                    rank=rank,
                    src_num_slicers=src_num,
                    tgt_num_slicers=tgt_num,
                    src_tensor=view if self.role == "src" else None,
                    tgt_tensor=view if self.role == "tgt" else None,
                    transfer_dtype=self._transfer_dtype(view),
                    role=self.role,
                )
                chunks.extend(pair_chunks)

        from collections import Counter as _Counter

        per_pair = _Counter((c.src_rank, c.dst_rank) for c in chunks)
        logger.info(
            "Etha strategy (role=%s, rank=%d) emitted %d chunks; per (src→dst): %s",
            self.role,
            rank,
            len(chunks),
            dict(sorted(per_pair.items())),
        )
        return chunks


# ============================================================================
# Mesh helpers used by both subclasses
# ============================================================================


def _vllm_dptp_mesh(
    dp_size: int, tp_size: int, base_rank: int
) -> torch.Tensor:
    return torch.arange(base_rank, base_rank + dp_size * tp_size).view(dp_size, tp_size)


def _vllm_ep_mesh(ep_size: int, base_rank: int) -> torch.Tensor:
    return torch.arange(base_rank, base_rank + ep_size).view(ep_size)


def _trainer_att_mesh(dp_replicate: int, dp_shard: int) -> torch.Tensor:
    return torch.arange(dp_replicate * dp_shard).view(dp_replicate, dp_shard)


def _trainer_moe_mesh(
    dp_replicate: int, dp_shard: int, ep: int
) -> torch.Tensor:
    return torch.arange(dp_replicate * dp_shard * ep).view(dp_replicate, dp_shard, ep)


# ============================================================================
# vLLM-side strategy (inference worker)
# ============================================================================


class VllmEthaShardingStrategy(EthaShardingStrategy):
    """Strategy for the inference worker side.

    Discovers own placements by walking the loaded vLLM model's module
    tree (`_vllm_param_placements`). Cross-checks against the static
    `VLLM_HANDLER_PLACEMENTS` table at init and raises on disagreement
    so placement drift surfaces loudly.
    """

    role: Literal["src", "tgt"] = "tgt"

    def __init__(
        self,
        model: torch.nn.Module,
        parallel_config: ParallelConfig,
        trainer_world_size: int,
        trainer_attn_dp_replicate: int,
        trainer_attn_dp_shard: int,
        trainer_moe_dp_replicate: int,
        trainer_moe_dp_shard: int,
        trainer_ep_size: int,
    ) -> None:
        self.model = model
        self.parallel_config = parallel_config
        self.trainer_world_size = trainer_world_size
        self.trainer_attn_dp_replicate = trainer_attn_dp_replicate
        self.trainer_attn_dp_shard = trainer_attn_dp_shard
        self.trainer_moe_dp_replicate = trainer_moe_dp_replicate
        self.trainer_moe_dp_shard = trainer_moe_dp_shard
        self.trainer_ep_size = trainer_ep_size

    def _build_pair_table(self) -> dict[str, PairSpec]:
        from vllm.model_executor.layers.fused_moe import FusedMoE

        # vLLM-side meshes (own). Ranks shifted by trainer_world_size.
        dp_size = self.parallel_config.data_parallel_size
        tp_size = self.parallel_config.tensor_parallel_size
        dptp_mesh = _vllm_dptp_mesh(dp_size, tp_size, base_rank=self.trainer_world_size)
        ep_size = 0
        for _, m in self.model.named_modules():
            if isinstance(m, FusedMoE):
                ep_size = m.ep_size
                break
        ep_mesh = (
            _vllm_ep_mesh(ep_size, base_rank=self.trainer_world_size)
            if ep_size
            else None
        )

        # Trainer-side meshes (peer). Ranks 0..M-1.
        trainer_att = _trainer_att_mesh(
            self.trainer_attn_dp_replicate, self.trainer_attn_dp_shard
        )
        trainer_moe = _trainer_moe_mesh(
            self.trainer_moe_dp_replicate,
            self.trainer_moe_dp_shard,
            self.trainer_ep_size,
        )

        # Empirical placement walk + cross-check against the table.
        empirical = _vllm_param_placements(self.model, dptp_mesh, ep_mesh)

        hf_config = getattr(
            getattr(self.model, "config", None),
            "text_config",
            getattr(self.model, "config", None),
        )
        if hf_config is None:
            raise RuntimeError("Could not locate hf_config on the model")

        converted = _convert_vllm_state_dict(self.model.state_dict(), hf_config)
        cutlass_swap = _detect_cutlass_swap(self.model)

        # Group tensors by handler; emit boundary splits for gate_up.
        by_pair: dict[str, list[tuple[str, torch.Tensor]]] = defaultdict(list)
        for trainer_name, (vllm_orig_name, tensor) in converted.items():
            handler = get_handler_name(trainer_name)
            if handler is None:
                continue
            if vllm_orig_name not in empirical:
                continue
            if trainer_name.endswith("mlp.experts.gate_up_proj"):
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

        # Cross-check: VLLM_HANDLER_PLACEMENTS must match the walker.
        # We check at the handler granularity using one tensor per pair.
        pair_table: dict[str, PairSpec] = {}
        for handler, tensors in by_pair.items():
            mesh_kind, table_placements = VLLM_HANDLER_PLACEMENTS[handler]
            expected_mesh = (
                ep_mesh if mesh_kind == "ep" else dptp_mesh
            )
            if expected_mesh is None:
                raise RuntimeError(
                    f"Handler {handler!r} needs ep_mesh but no FusedMoE found "
                    f"in the model — refusing to plan."
                )

            # Empirical check: every tensor in this pair must agree with
            # the table. Slice-derived views (gate_proj/up_proj/q_proj/...)
            # are not in `empirical` directly, so we look up their
            # canonical parent.
            for name, _ in tensors:
                emp_key = name
                if name not in empirical:
                    for parent in (
                        name.replace("gate_proj", "gate_up_proj.weight").replace(
                            "up_proj", "gate_up_proj.weight"
                        ),
                        name.replace("q_proj", "qkv_proj")
                        .replace("k_proj", "qkv_proj")
                        .replace("v_proj", "qkv_proj"),
                    ):
                        if parent in empirical:
                            emp_key = parent
                            break
                if emp_key not in empirical:
                    continue
                emp_mesh, emp_placements = empirical[emp_key]
                if (
                    tuple(emp_mesh.shape) != tuple(expected_mesh.shape)
                    or emp_placements != table_placements
                ):
                    raise RuntimeError(
                        f"VLLM_HANDLER_PLACEMENTS drift for handler={handler!r} "
                        f"tensor={name!r}: empirical={emp_placements} "
                        f"on mesh shape {tuple(emp_mesh.shape)}, table="
                        f"{table_placements} on mesh shape "
                        f"{tuple(expected_mesh.shape)}. Update "
                        f"VLLM_HANDLER_PLACEMENTS in etha_sharding.py."
                    )

            # Peer (trainer) side comes from the static trainer table.
            trainer_placements = TRAINER_HANDLER_PLACEMENTS[handler]
            trainer_mesh = (
                trainer_moe if handler in MOE_HANDLERS else trainer_att
            )

            pair_table[handler] = PairSpec(
                handler=handler,
                own_mesh=expected_mesh,
                own_placements=table_placements,
                peer_mesh=trainer_mesh,
                peer_placements=trainer_placements,
                tensors=tensors,
            )
        return pair_table


# ============================================================================
# Trainer-side strategy (sender)
# ============================================================================


class TrainerEthaShardingStrategy(EthaShardingStrategy):
    """Strategy for the trainer side.

    Takes a state_dict mapping trainer-side names → local shard tensors
    (typically `DTensor.to_local()` results). Uses `TRAINER_HANDLER_PLACEMENTS`
    for own placements and `VLLM_HANDLER_PLACEMENTS` for peer placements
    — both static tables, kept consistent with the worker side via the
    cross-check in `VllmEthaShardingStrategy`.

    `wire_dtype` overrides the storage dtype on the wire. Trainers
    holding fp32 master weights typically downcast to bf16 to match
    the inference side's wire dtype; NCCL pairs ops by byte count, so
    both sides must agree on this.
    """

    role: Literal["src", "tgt"] = "src"

    def __init__(
        self,
        state_dict: dict[str, torch.Tensor],
        trainer_attn_dp_replicate: int,
        trainer_attn_dp_shard: int,
        trainer_moe_dp_replicate: int,
        trainer_moe_dp_shard: int,
        trainer_ep_size: int,
        trainer_world_size: int,
        vllm_dp_size: int,
        vllm_tp_size: int,
        vllm_ep_size: int,
        wire_dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        self.state_dict = state_dict
        self.trainer_attn_dp_replicate = trainer_attn_dp_replicate
        self.trainer_attn_dp_shard = trainer_attn_dp_shard
        self.trainer_moe_dp_replicate = trainer_moe_dp_replicate
        self.trainer_moe_dp_shard = trainer_moe_dp_shard
        self.trainer_ep_size = trainer_ep_size
        self.trainer_world_size = trainer_world_size
        self.vllm_dp_size = vllm_dp_size
        self.vllm_tp_size = vllm_tp_size
        self.vllm_ep_size = vllm_ep_size
        self.wire_dtype = wire_dtype

    def _transfer_dtype(self, tensor: torch.Tensor) -> torch.dtype:
        return self.wire_dtype

    def _build_pair_table(self) -> dict[str, PairSpec]:
        trainer_att = _trainer_att_mesh(
            self.trainer_attn_dp_replicate, self.trainer_attn_dp_shard
        )
        trainer_moe = _trainer_moe_mesh(
            self.trainer_moe_dp_replicate,
            self.trainer_moe_dp_shard,
            self.trainer_ep_size,
        )
        vllm_dptp = _vllm_dptp_mesh(
            self.vllm_dp_size, self.vllm_tp_size, base_rank=self.trainer_world_size
        )
        vllm_ep = _vllm_ep_mesh(self.vllm_ep_size, base_rank=self.trainer_world_size)

        # Group state dict by handler; split grouped gate_up into
        # gate / up views matching the canonical [gate; up] layout.
        by_pair: dict[str, list[tuple[str, torch.Tensor]]] = defaultdict(list)
        for name, t in self.state_dict.items():
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

        pair_table: dict[str, PairSpec] = {}
        for handler, tensors in by_pair.items():
            own_placements = TRAINER_HANDLER_PLACEMENTS[handler]
            own_mesh = trainer_moe if handler in MOE_HANDLERS else trainer_att

            mesh_kind, peer_placements = VLLM_HANDLER_PLACEMENTS[handler]
            peer_mesh = vllm_ep if mesh_kind == "ep" else vllm_dptp

            pair_table[handler] = PairSpec(
                handler=handler,
                own_mesh=own_mesh,
                own_placements=own_placements,
                peer_mesh=peer_mesh,
                peer_placements=peer_placements,
                tensors=tensors,
            )
        return pair_table
