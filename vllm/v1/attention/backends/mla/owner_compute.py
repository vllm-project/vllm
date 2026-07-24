# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Correctness helpers for owner-local sparse MLA."""

from __future__ import annotations

from typing import Literal, cast

import torch

from vllm import envs
from vllm.v1.attention.ops.dcp_alltoall import _lse_weighted_combine

_OWNER_COMPUTE_MIN_PREFILL_SEQ_LEN = 1536
OwnerPrefillMode = Literal["auto", "direct", "materialize"]


def get_owner_prefill_mode() -> OwnerPrefillMode:
    mode = envs.VLLM_PCP_OWNER_PREFILL_MODE
    if mode not in ("auto", "direct", "materialize"):
        raise ValueError(
            "VLLM_PCP_OWNER_PREFILL_MODE must be auto, direct, or materialize, "
            f"got {mode!r}."
        )
    return cast(OwnerPrefillMode, mode)


def validate_owner_prefill_materialization_support(
    *,
    owner_history_enabled: bool,
    supports_materialization: bool,
) -> None:
    """Reject materialization when the backend cannot provide it."""
    if (
        owner_history_enabled
        and get_owner_prefill_mode() == "materialize"
        and not supports_materialization
    ):
        raise RuntimeError(
            "VLLM_PCP_OWNER_PREFILL_MODE=materialize requires an attention "
            "backend that supports bounded prefill history materialization."
        )


def should_use_owner_compute(
    *,
    owner_history_enabled: bool,
    num_decodes: int,
    max_prefill_seq_len: int,
) -> bool:
    """Select owner compute uniformly for long pure-prefill PCP batches.

    Decode rows are replicated on every PCP rank, whereas a short prefill may
    give one rank zero local prefill rows. Keying only on ``num_decodes`` keeps
    all ranks on the same collective schedule. Direct peer reads avoid the
    owner-compute collectives for short prefills, while owner compute avoids
    scaling peer traffic with long history. Mixed and decode-only batches
    retain direct peer reads.
    """
    if num_decodes < 0:
        raise ValueError("Owner-compute decode count cannot be negative.")
    if max_prefill_seq_len < 0:
        raise ValueError("Owner-compute prefill length cannot be negative.")
    return (
        owner_history_enabled
        and get_owner_prefill_mode() == "auto"
        and num_decodes == 0
        and max_prefill_seq_len > _OWNER_COMPUTE_MIN_PREFILL_SEQ_LEN
    )


def validate_owner_compute_scope(
    *,
    pcp_world_size: int,
    dcp_world_size: int,
    pcp_rank: int,
    dcp_rank: int,
    cp_kv_cache_interleave_size: int,
    block_size: int,
) -> None:
    """Fail closed outside the initial PCP4=DCP4 owner-compute contract."""
    if pcp_world_size != 4 or dcp_world_size != 4:
        raise RuntimeError(
            "Owner-local sparse MLA requires PCP4=DCP4, got "
            f"PCP={pcp_world_size} and DCP={dcp_world_size}."
        )
    if not 0 <= pcp_rank < pcp_world_size or not 0 <= dcp_rank < dcp_world_size:
        raise RuntimeError(
            "Owner-local sparse MLA received an invalid PCP/DCP rank: "
            f"PCP rank={pcp_rank}, DCP rank={dcp_rank}."
        )
    if pcp_rank != dcp_rank:
        raise RuntimeError(
            "Owner-local sparse MLA requires identical PCP/DCP rank ordering, "
            f"got PCP rank={pcp_rank} and DCP rank={dcp_rank}."
        )
    if cp_kv_cache_interleave_size != block_size:
        raise RuntimeError(
            "Owner-local sparse MLA requires page-granular KV ownership: "
            "cp_kv_cache_interleave_size must equal block_size, got "
            f"{cp_kv_cache_interleave_size} and {block_size}."
        )


def filter_peer_slots_to_owner_local_reference(
    peer_slots: torch.Tensor,
    *,
    owner_rank: int,
    dcp_world_size: int,
    blocks_per_peer: int,
    block_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Stable reference filter for rank-major peer physical slots.

    The origin rank has already converted per-request global selected IDs into
    ``[owner, physical_block, block_offset]`` peer slots. Each owner retains
    only its rank interval, subtracts the interval base to recover slots in its
    local cache, and compacts valid slots to a contiguous prefix.
    """
    if peer_slots.dtype != torch.int32 or peer_slots.ndim != 2:
        raise ValueError("Owner-compute peer slots must be a 2D int32 tensor.")
    if dcp_world_size <= 1 or not 0 <= owner_rank < dcp_world_size:
        raise ValueError("Owner-compute filtering requires a valid DCP owner.")
    if blocks_per_peer <= 0 or block_size <= 0:
        raise ValueError("Owner-compute peer stride and block size must be positive.")
    if peer_slots.shape[1] == 0:
        raise ValueError("Owner-compute selected-slot rows cannot be empty.")

    slots_per_peer = blocks_per_peer * block_size
    owner_start = owner_rank * slots_per_peer
    owner_stop = owner_start + slots_per_peer
    valid = (peer_slots >= owner_start) & (peer_slots < owner_stop)
    local_slots = torch.where(valid, peer_slots - owner_start, -1)

    rows, topk = peer_slots.shape
    columns = torch.arange(topk, device=peer_slots.device).expand(rows, topk)
    sort_keys = torch.where(valid, columns, columns + topk)
    order = torch.argsort(sort_keys, dim=1, stable=True)
    compacted = local_slots.gather(1, order)
    valid_counts = valid.sum(dim=1, dtype=torch.int32)
    return compacted, valid_counts


def merge_owner_compute_partials_reference(
    owner_outputs: torch.Tensor,
    owner_lses: torch.Tensor,
    *,
    return_lse: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Reference owner-rank-ordered base-2 LSE reduction."""
    if owner_outputs.ndim != 4:
        raise ValueError("Owner outputs must have shape [owners, rows, heads, dim].")
    if owner_lses.ndim != 3 or owner_lses.shape != owner_outputs.shape[:3]:
        raise ValueError("Owner LSEs must have shape [owners, rows, heads].")
    if owner_outputs.device != owner_lses.device:
        raise ValueError("Owner outputs and LSEs must share a device.")
    return _lse_weighted_combine(
        owner_outputs,
        owner_lses,
        return_lse=return_lse,
        is_lse_base_on_e=False,
    )
