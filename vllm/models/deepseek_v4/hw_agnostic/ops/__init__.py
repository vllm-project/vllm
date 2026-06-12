# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Triton kernels backing the DeepSeek V4 hw-agnostic attention path.

Each kernel lives in its own file. This package never imports from
``nvidia/``, ``amd/``, or ``xpu/`` — the hw-agnostic stream is
self-contained at the kernel level (see study doc §52).
"""
from .triton_inv_rope_einsum import triton_inv_rope_einsum
from .triton_mla_sparse import triton_bf16_mla_sparse_interface
from .triton_qnorm_rope_kv_fp8_insert import triton_qnorm_rope_kv_fp8_insert
from .triton_sparse_decode_fp8 import triton_sparse_decode_fp8

__all__ = [
    "triton_bf16_mla_sparse_interface",
    "triton_inv_rope_einsum",
    "triton_qnorm_rope_kv_fp8_insert",
    "triton_sparse_decode_fp8",
]
