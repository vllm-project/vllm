# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Attention backend registry"""

import enum


class _Backend(enum.Enum):
    FLASH_ATTN = enum.auto()
    TRITON_ATTN = enum.auto()
    XFORMERS = enum.auto()
    ROCM_FLASH = enum.auto()
    ROCM_AITER_MLA = enum.auto()
    ROCM_AITER_FA = enum.auto()  # used for ViT attn backend
    TORCH_SDPA = enum.auto()
    FLASHINFER = enum.auto()
    FLASHINFER_MLA = enum.auto()
    TRITON_MLA = enum.auto()
    CUTLASS_MLA = enum.auto()
    FLASHMLA = enum.auto()
    FLASH_ATTN_MLA = enum.auto()
    PALLAS = enum.auto()
    IPEX = enum.auto()
    NO_ATTENTION = enum.auto()
    FLEX_ATTENTION = enum.auto()
    TREE_ATTN = enum.auto()
    ROCM_ATTN = enum.auto()
