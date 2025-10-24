# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Attention backend registry"""

import enum
from typing import TYPE_CHECKING

from vllm.utils.import_utils import resolve_obj_by_qualname

if TYPE_CHECKING:
    from vllm.attention.backends.abstract import AttentionBackend


class _Backend(enum.Enum):
    FLASH_ATTN = enum.auto()
    TRITON_ATTN = enum.auto()
    XFORMERS = enum.auto()
    ROCM_ATTN = enum.auto()
    ROCM_AITER_MLA = enum.auto()
    ROCM_AITER_FA = enum.auto()  # used for ViT attn backend
    TORCH_SDPA = enum.auto()
    FLASHINFER = enum.auto()
    FLASHINFER_MLA = enum.auto()
    TRITON_MLA = enum.auto()
    CUTLASS_MLA = enum.auto()
    FLASHMLA = enum.auto()
    FLASHMLA_SPARSE = enum.auto()
    FLASH_ATTN_MLA = enum.auto()
    PALLAS = enum.auto()
    IPEX = enum.auto()
    NO_ATTENTION = enum.auto()
    FLEX_ATTENTION = enum.auto()
    TREE_ATTN = enum.auto()
    ROCM_AITER_UNIFIED_ATTN = enum.auto()


BACKEND_MAP = {
    _Backend.FLASH_ATTN: "vllm.v1.attention.backends.flash_attn.FlashAttentionBackend",  # noqa: E501
    _Backend.TRITON_ATTN: "vllm.v1.attention.backends.triton_attn.TritonAttentionBackend",  # noqa: E501
    _Backend.XFORMERS: "vllm.v1.attention.backends.xformers.XFormersAttentionBackend",  # noqa: E501
    _Backend.ROCM_ATTN: "vllm.v1.attention.backends.rocm_attn.RocmAttentionBackend",  # noqa: E501
    _Backend.ROCM_AITER_MLA: "vllm.v1.attention.backends.mla.rocm_aiter_mla.AiterMLABackend",  # noqa: E501
    _Backend.ROCM_AITER_FA: "vllm.v1.attention.backends.rocm_aiter_fa.AiterFlashAttentionBackend",  # noqa: E501
    _Backend.TORCH_SDPA: "vllm.v1.attention.backends.cpu_attn.TorchSDPABackend",  # noqa: E501
    _Backend.FLASHINFER: "vllm.v1.attention.backends.flashinfer.FlashInferBackend",  # noqa: E501
    _Backend.FLASHINFER_MLA: "vllm.v1.attention.backends.mla.flashinfer_mla.FlashInferMLABackend",  # noqa: E501
    _Backend.TRITON_MLA: "vllm.v1.attention.backends.mla.triton_mla.TritonMLABackend",  # noqa: E501
    _Backend.CUTLASS_MLA: "vllm.v1.attention.backends.mla.cutlass_mla.CutlassMLABackend",  # noqa: E501
    _Backend.FLASHMLA: "vllm.v1.attention.backends.mla.flashmla.FlashMLABackend",  # noqa: E501
    _Backend.FLASHMLA_SPARSE: "vllm.v1.attention.backends.mla.flashmla_sparse.FlashMLASparseBackend",  # noqa: E501
    _Backend.FLASH_ATTN_MLA: "vllm.v1.attention.backends.mla.flashattn_mla.FlashAttnMLABackend",  # noqa: E501
    _Backend.PALLAS: "vllm.v1.attention.backends.pallas.PallasAttentionBackend",  # noqa: E501
    _Backend.FLEX_ATTENTION: "vllm.v1.attention.backends.flex_attention.FlexAttentionBackend",  # noqa: E501
    _Backend.TREE_ATTN: "vllm.v1.attention.backends.tree_attn.TreeAttentionBackend",  # noqa: E501
    _Backend.ROCM_AITER_UNIFIED_ATTN: "vllm.v1.attention.backends.rocm_aiter_unified_attn.RocmAiterUnifiedAttentionBackend",  # noqa: E501
}


def register_attn_backend(backend: _Backend, class_path: str | None = None):
    """
    Decorator: register a custom attention backend into BACKEND_MAPPING.
    - If class_path is provided, use it.
    - Otherwise, auto-generate from the class object.
    Validation: only checks if 'backend' is a valid _Backend enum member.
    Overwriting existing mappings is allowed. This enables other hardware
    platforms to plug in custom out-of-tree backends.
    """
    if not isinstance(backend, _Backend):
        raise ValueError(f"{backend} is not a valid _Backend enum value.")

    def decorator(cls):
        path = class_path or f"{cls.__module__}.{cls.__qualname__}"
        BACKEND_MAP[backend] = path
        return cls

    return decorator


def backend_to_class_str(backend: _Backend) -> str:
    """Get the backend class string

    Args:
        backend: The backend enum value

    Returns:
        The backend class string
    """
    return BACKEND_MAP[backend]


def backend_to_class(backend: _Backend) -> "type[AttentionBackend]":
    """Get the backend class.

    Args:
        backend: The backend enum value

    Returns:
        The backend class
    """
    backend_class_name = backend_to_class_str(backend)
    return resolve_obj_by_qualname(backend_class_name)
