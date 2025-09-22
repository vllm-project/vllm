# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Attention backend registry"""

import enum
from typing import Optional, Type, Union

from vllm.utils import resolve_obj_by_qualname


class _Backend(enum.Enum):
    FLASH_ATTN_VLLM_V1 = enum.auto()
    TRITON_ATTN_VLLM_V1 = enum.auto()
    ROCM_AITER_MLA_VLLM_V1 = enum.auto()
    ROCM_AITER_FA = enum.auto()  # used for ViT attn backend
    TORCH_SDPA = enum.auto()
    TORCH_SDPA_VLLM_V1 = enum.auto()
    FLASHINFER = enum.auto()
    FLASHINFER_VLLM_V1 = enum.auto()
    FLASHINFER_MLA = enum.auto()
    TRITON_MLA = enum.auto()  # Supported by V1
    TRITON_MLA_VLLM_V1 = enum.auto()
    CUTLASS_MLA = enum.auto()
    FLASHMLA_VLLM_V1 = enum.auto()
    FLASH_ATTN_MLA = enum.auto()  # Supported by V1
    PALLAS = enum.auto()
    PALLAS_VLLM_V1 = enum.auto()
    IPEX = enum.auto()
    NO_ATTENTION = enum.auto()
    FLEX_ATTENTION = enum.auto()
    TREE_ATTN = enum.auto()
    XFORMERS_VLLM_V1 = enum.auto()


BACKEND_MAPPING = {
    _Backend.FLASH_ATTN_VLLM_V1:
    "vllm.v1.attention.backends.flash_attn.FlashAttentionBackend",  # noqa: E501
    _Backend.TRITON_ATTN_VLLM_V1:
    "vllm.v1.attention.backends.triton_attn.TritonAttentionBackend",  # noqa: E501
    _Backend.ROCM_AITER_MLA_VLLM_V1:
    "vllm.v1.attention.backends.mla.rocm_aiter_mla.AiterMLABackend",  # noqa: E501
    _Backend.ROCM_AITER_FA:
    "vllm.v1.attention.backends.rocm_aiter_fa.AiterFlashAttentionBackend",  # noqa: E501
    _Backend.TORCH_SDPA:
    "vllm.v1.attention.backends.cpu_attn.TorchSDPABackend",  # noqa: E501
    _Backend.TORCH_SDPA_VLLM_V1:
    "vllm.v1.attention.backends.cpu_attn.TorchSDPABackend",  # noqa: E501
    _Backend.FLASHINFER:
    "vllm.v1.attention.backends.flashinfer.FlashInferBackend",  # noqa: E501
    _Backend.FLASHINFER_VLLM_V1:
    "vllm.v1.attention.backends.flashinfer.FlashInferBackend",  # noqa: E501
    _Backend.FLASHINFER_MLA:
    "vllm.v1.attention.backends.mla.flashinfer_mla.FlashInferMLABackend",  # noqa: E501
    _Backend.TRITON_MLA:
    "vllm.v1.attention.backends.mla.triton_mla.TritonMLABackend",  # noqa: E501
    _Backend.TRITON_MLA_VLLM_V1:
    "vllm.v1.attention.backends.mla.triton_mla.TritonMLABackend",  # noqa: E501
    _Backend.CUTLASS_MLA:
    "vllm.v1.attention.backends.mla.cutlass_mla.CutlassMLABackend",  # noqa: E501
    _Backend.FLASHMLA_VLLM_V1:
    "vllm.v1.attention.backends.mla.flashmla.FlashMLABackend",  # noqa: E501
    _Backend.FLASH_ATTN_MLA:
    "vllm.v1.attention.backends.mla.flashattn_mla.FlashAttnMLABackend",  # noqa: E501
    _Backend.PALLAS:
    "vllm.v1.attention.backends.pallas.PallasAttentionBackend",  # noqa: E501
    _Backend.PALLAS_VLLM_V1:
    "vllm.v1.attention.backends.pallas.PallasAttentionBackend",  # noqa: E501
    _Backend.NO_ATTENTION:
    "vllm.attention.backends.placeholder_attn.PlaceholderAttentionBackend",  # noqa: E501
    _Backend.FLEX_ATTENTION:
    "vllm.v1.attention.backends.flex_attention.FlexAttentionBackend",  # noqa: E501
    _Backend.TREE_ATTN:
    "vllm.v1.attention.backends.tree_attn.TreeAttentionBackend",  # noqa: E501
    _Backend.XFORMERS_VLLM_V1:
    "vllm.v1.attention.backends.xformers.XFormersAttentionBackend",  # noqa: E501
}


def backend_to_class_str(backend: _Backend,
                         use_v1: Union[None, bool] = None) -> str:
    """Get the backend class string, optionally resolving to V1 version.
    This function DOES NOT validate if the backend supports V1 or not.
    
    Args:
        backend: The backend enum value
        use_v1: If True, return the V1 version of the backend if available
        
    Returns:
        The backend class string
    """
    if use_v1 is None:
        return BACKEND_MAPPING[backend]

    backend_name = backend.name
    if use_v1:
        v1_backend_name = f"{backend_name}_VLLM_V1"
        try:
            v1_backend = _Backend[v1_backend_name]
            return BACKEND_MAPPING[v1_backend]
        except KeyError:
            return BACKEND_MAPPING[backend]
    else:
        if backend_name.endswith('_VLLM_V1'):
            raise ValueError(f"{backend_name} is a V1-only backend.")
        return BACKEND_MAPPING[backend]


def backend_to_class(backend: _Backend,
                     use_v1: Union[None, bool] = None) -> Type:
    """Get the backend class, optionally resolving to V1 version.
    This function DOES NOT validate if the backend supports V1 or not.

    Args:
        backend: The backend enum value
        use_v1: If True, return the V1 version of the backend if available

    Returns:
        The backend class
    """
    backend_class_name = backend_to_class_str(backend, use_v1)
    return resolve_obj_by_qualname(backend_class_name)


def backend_name_to_enum(backend_name: str) -> Optional[_Backend]:
    """
    Convert a string backend name to a _Backend enum value.

    Returns:
    * _Backend: enum value if backend_name is a valid in-tree type
    * None: otherwise it's an invalid in-tree type or an out-of-tree platform is
            loaded.
    """
    assert backend_name is not None
    return _Backend[backend_name] if backend_name in _Backend.__members__ else \
          None
