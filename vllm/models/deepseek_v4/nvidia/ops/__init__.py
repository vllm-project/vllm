# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""NVIDIA-only (cutedsl/cutlass) kernels for DeepSeek V4.

These modules import ``cutlass``/``cutedsl`` at module top level, so they must
not be imported on non-CUDA platforms. Callers should gate on
``vllm.utils.import_utils.has_cutedsl()`` before importing from here.
"""

from .dequant_gather_k_cutedsl import dequantize_and_gather_k_cache_cutedsl
from .fused_indexer_q_cutedsl import (
    fused_indexer_q_rope_quant_fp8_cutedsl,
    fused_indexer_q_rope_quant_mxfp4_cutedsl,
)
from .prepare_megamoe import prepare_megamoe_inputs
from .sparse_attn_compress_cutedsl import compress_norm_rope_store_cutedsl

__all__ = [
    "compress_norm_rope_store_cutedsl",
    "dequantize_and_gather_k_cache_cutedsl",
    "fused_indexer_q_rope_quant_fp8_cutedsl",
    "fused_indexer_q_rope_quant_mxfp4_cutedsl",
    "prepare_megamoe_inputs",
]
