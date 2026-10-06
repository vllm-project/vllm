# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
RDNA3 HIP INT8 per-token-head paged prefill attention kernel.

This module wraps the compiled HIP kernel for use in the triton_attn backend.
The kernel is 8x faster than the Triton per-token-head prefill path on
gfx1100 due to cooperative K/V loads (4 waves) + WMMA + fused scale dequant.

The kernel handles:
- Phase 1 (cached prefix): reads INT8 paged KV cache with per-token-head
  scale dequantization. k_scale fused post-QK-WMMA, v_scale fused into P.
- Phase 2 (current chunk): reads FP16 K/V directly (not yet quantized).
"""

import os

import torch

from vllm.logger import init_logger
from vllm.platforms import current_platform

logger = init_logger(__name__)

_module = None
_available = None


def is_available() -> bool:
    """Check if the RDNA3 INT8 prefill kernel is available."""
    global _available
    if _available is not None:
        return _available

    if not current_platform.is_rocm():
        _available = False
        return False

    # Check if we have the pre-compiled extension
    try:
        _load_module()
        _available = True
    except Exception:
        _available = False

    return _available


def _load_module():
    """Load the compiled extension module."""
    global _module
    if _module is not None:
        return _module

    # Try loading from torch cache (already compiled)
    import importlib.util

    # Check common paths
    cache_paths = [
        "/root/.cache/torch_extensions/py312_cpu/rdna3_int8_attn/rdna3_int8_attn.so",
        os.path.expanduser(
            "~/.cache/torch_extensions/py312_cpu/rdna3_int8_attn/rdna3_int8_attn.so"
        ),
    ]

    for path in cache_paths:
        if os.path.exists(path):
            spec = importlib.util.spec_from_file_location("rdna3_int8_attn", path)
            _module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(_module)  # type: ignore
            logger.info("Loaded RDNA3 INT8 prefill kernel from %s", path)
            return _module

    # Try via torch.ops._C if registered in the build
    if hasattr(torch.ops, "_C") and hasattr(
        torch.ops._C, "paged_prefill_attn_rdna3_int8"
    ):
        _module = torch.ops._C
        return _module

    raise RuntimeError("RDNA3 INT8 prefill kernel not found")


def rdna3_int8_paged_prefill(
    out: torch.Tensor,
    q: torch.Tensor,
    k_chunk: torch.Tensor,
    v_chunk: torch.Tensor,
    k_cache: torch.Tensor,  # int8, 5D paged
    v_cache: torch.Tensor,  # int8, 4D paged
    k_scale_cache: torch.Tensor,  # float32, [blocks, slots, heads]
    v_scale_cache: torch.Tensor,  # float32, [blocks, slots, heads]
    block_table: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    seq_lens: torch.Tensor,
    max_query_len: int,
    sm_scale: float,
    causal: bool = True,
) -> None:
    """Run the RDNA3 INT8 per-token-head paged prefill attention."""
    mod = _load_module()
    mod.paged_prefill_attn_rdna3_int8(
        out,
        q,
        k_chunk,
        v_chunk,
        k_cache,
        v_cache,
        k_scale_cache,
        v_scale_cache,
        block_table,
        cu_seqlens_q,
        seq_lens,
        max_query_len,
        sm_scale,
        causal,
    )
