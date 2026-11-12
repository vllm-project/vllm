# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import TYPE_CHECKING

import torch

from vllm.distributed import get_pcp_group

if TYPE_CHECKING:
    from vllm.v1.attention.backends.mla.sparse_swa import DeepseekSparseSWAMetadata


def insert_swa_kv(
    kv: torch.Tensor,
    cache: torch.Tensor,
    metadata: "DeepseekSparseSWAMetadata",
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    *,
    use_pcp: bool = False,
    fp8_scale: torch.Tensor | None = None,
    kv_mxfp8: bool = False,
) -> torch.Tensor:
    """Insert SWA KV, gathering PCP rows, and return the cache-write positions."""
    slots = metadata.slot_mapping
    if use_pcp:
        assert metadata.cache_slot_mapping is not None
        assert metadata.cache_positions is not None
        kv = get_pcp_group().all_gather(kv.contiguous(), dim=0)
        slots = metadata.cache_slot_mapping
        positions = metadata.cache_positions
    torch.ops._C.fused_deepseek_v4_kv_rope_insert(
        kv,
        cache,
        slots,
        positions,
        cos_sin_cache,
        metadata.block_size,
        fp8_scale,
        kv_mxfp8,
    )
    return positions
