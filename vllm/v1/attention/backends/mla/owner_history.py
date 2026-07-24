# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch


def select_owner_slot_mapping(
    mla_slot: torch.Tensor | None,
    *,
    owner_history_expected: bool,
    pcp_rank: int,
    pcp_size: int,
    num_tokens: int,
    device: torch.device,
) -> torch.Tensor | None:
    """Select this producer's owner mapping, rejecting malformed owner mode."""
    if not owner_history_expected:
        return None
    if mla_slot is None:
        raise RuntimeError("Owner-sharded PCP history requires an owner-slot mapping.")
    expected_shape = (pcp_size * num_tokens, 2)
    if mla_slot.ndim != 2 or tuple(mla_slot.shape) != expected_shape:
        raise RuntimeError(
            "Owner-sharded PCP history requires owner-slot mapping shape "
            f"{expected_shape}, got {tuple(mla_slot.shape)}."
        )
    if mla_slot.dtype not in (torch.int32, torch.int64):
        raise RuntimeError(
            "Owner-sharded PCP history requires an integer owner-slot mapping."
        )
    if mla_slot.device != device:
        raise RuntimeError(
            "Owner-sharded PCP history requires the owner-slot mapping on "
            f"{device}, got {mla_slot.device}."
        )
    if not 0 <= pcp_rank < pcp_size:
        raise RuntimeError(
            "Owner-sharded PCP history received invalid PCP rank "
            f"{pcp_rank} for size {pcp_size}."
        )
    return mla_slot.view(pcp_size, num_tokens, 2)[pcp_rank]


def validate_owner_fused_cache_contract(
    *,
    mla_slot: torch.Tensor | None,
    indexer_slot: torch.Tensor | None,
    mla_peer_cache: torch.Tensor,
    indexer_peer_cache: torch.Tensor | None,
) -> None:
    """Validate the shared slot identity used by fused MLA/indexer writes."""
    if indexer_slot is not mla_slot:
        raise RuntimeError(
            "Owner-history fused publication requires the MLA and indexer caches "
            "to share one KV-cache group and the exact same slot tensor."
        )
    if indexer_peer_cache is None:
        raise RuntimeError(
            "Owner-history fused publication requires an indexer peer cache."
        )
    if (
        mla_peer_cache.ndim < 3
        or indexer_peer_cache.ndim < 3
        or mla_peer_cache.shape[2] != indexer_peer_cache.shape[2]
    ):
        raise RuntimeError(
            "Owner-history fused publication requires identical MLA and indexer "
            "cache block sizes."
        )
