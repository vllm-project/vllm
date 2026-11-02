# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek V4.1 fused ops.

Single bridging point for the kernels V4.1 shares verbatim with V4: the
re-exports below are the only place this package reaches into
``vllm.models.deepseek_v4``, so submodules import them from ``.`` instead of
naming the V4 path themselves. Keep these imports above the ``.``-relative
ones — ``indexer_k_store`` reads them off this partially-initialized module.
"""

from vllm.models.deepseek_v4.common.ops.fused_indexer_q import (
    MXFP4_BLOCK_SIZE,
    fused_indexer_q_rope_quant,
)
from vllm.models.deepseek_v4.common.ops.fused_indexer_q import (
    _fp32x2_to_fp4x2 as _fp32x2_to_fp4x2,  # re-export for .indexer_k_store
)
from vllm.models.deepseek_v4.common.ops.fused_inv_rope_fp8_quant import (
    fused_inv_rope_fp8_quant,
)

from .cache_utils import (
    build_flashinfer_mixed_sparse_indices,
    combine_topk_swa_indices,
    compute_global_topk_indices_and_lens,
    dequantize_and_gather_k_cache,
    quantize_and_insert_k_cache,
)
from .indexer_k_store import indexer_k_norm_rope_store

__all__ = [
    "MXFP4_BLOCK_SIZE",
    "build_flashinfer_mixed_sparse_indices",
    "combine_topk_swa_indices",
    "compute_global_topk_indices_and_lens",
    "dequantize_and_gather_k_cache",
    "fused_indexer_q_rope_quant",
    "fused_inv_rope_fp8_quant",
    "indexer_k_norm_rope_store",
    "quantize_and_insert_k_cache",
]
