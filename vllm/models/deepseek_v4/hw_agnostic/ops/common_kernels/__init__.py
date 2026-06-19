# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Triton kernels for the DeepSeek-V4 hw-agnostic path.

Vendored copies of the kernels that previously lived under
``vllm/models/deepseek_v4/common/ops/``. The originals are imported by
the vendor branches (``nvidia/``, ``amd/``, ``xpu/``) which can layer
hardware-specific fast paths on top; the copies here run only the
portable Triton path. CUTeDSL fast-path branches and CUDA-only helpers
have been stripped per the §54 isolation policy.
"""

from .cache_utils import (
    combine_topk_swa_indices,
    compute_global_topk_indices_and_lens,
    dequantize_and_gather_k_cache,
    quantize_and_insert_k_cache,
)
from .fused_compress_quant_cache import compress_norm_rope_store_triton
from .fused_indexer_q import MXFP4_BLOCK_SIZE, fused_indexer_q_rope_quant
from .fused_qk_rmsnorm import fused_q_kv_rmsnorm
from .save_partial_states import save_partial_states

__all__ = [
    "MXFP4_BLOCK_SIZE",
    "combine_topk_swa_indices",
    "compress_norm_rope_store_triton",
    "compute_global_topk_indices_and_lens",
    "dequantize_and_gather_k_cache",
    "fused_indexer_q_rope_quant",
    "fused_q_kv_rmsnorm",
    "quantize_and_insert_k_cache",
    "save_partial_states",
]
