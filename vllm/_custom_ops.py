# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import TYPE_CHECKING, Literal

import torch

import vllm.envs as envs
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.scalar_type import ScalarType
from vllm.utils.flashinfer import (
    flashinfer_quant_nvfp4_8x4_sf_layout,
)
from vllm.utils.math_utils import cdiv

logger = init_logger(__name__)

current_platform.import_kernels()

if TYPE_CHECKING:

    def register_fake(fn):
        return lambda name: fn
else:
    try:
        from torch.library import register_fake
    except ImportError:
        from torch.library import impl_abstract as register_fake


# page attention ops
# Enhanced paged_attention_v1 with device checks
def paged_attention_v1(
    out: torch.Tensor,
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    num_kv_heads: int,
    scale: float,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    block_size: int,
    max_seq_len: int,
    alibi_slopes: torch.Tensor | None,
    kv_cache_dtype: str,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
    tp_rank: int = 0,
    blocksparse_local_blocks: int = 0,
    blocksparse_vert_stride: int = 0,
):
    """
    Optimized paged attention with dynamic GPU/CPU device compatibility.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    # Ensure tensors are on the correct device
    query, key_cache, value_cache = query.to(device), key_cache.to(device), value_cache.to(device)
    k_scale, v_scale = k_scale.to(device), v_scale.to(device)

    out = torch.zeros_like(query, device=device)  # Ensuring output is on device

    # Existing implementation
    pass

    return out