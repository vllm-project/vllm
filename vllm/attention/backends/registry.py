# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Attention backend registry"""

import enum
from typing import Optional, Type

from vllm.utils import resolve_obj_by_qualname


class _Backend(enum.Enum):
    FLASH_ATTN = enum.auto()
    TRITON_ATTN = enum.auto()
    XFORMERS = enum.auto()
    ROCM_FLASH = enum.auto()
    ROCM_AITER_MLA = enum.auto()  # Supported by V1
    ROCM_AITER_FA = enum.auto()  # used for ViT attn backend
    TORCH_SDPA = enum.auto()
    FLASHINFER = enum.auto()
    FLASHINFER_MLA = enum.auto()
    TRITON_MLA = enum.auto()  # Supported by V1
    CUTLASS_MLA = enum.auto()
    FLASHMLA = enum.auto()  # Supported by V1
    FLASH_ATTN_MLA = enum.auto()  # Supported by V1
    PALLAS = enum.auto()
    IPEX = enum.auto()
    DUAL_CHUNK_FLASH_ATTN = enum.auto()
    DIFFERENTIAL_FLASH_ATTN = enum.auto()
    NO_ATTENTION = enum.auto()
    FLEX_ATTENTION = enum.auto()
    TREE_ATTN = enum.auto()
    ROCM_ATTN = enum.auto()


BACKEND_MAPPING = {}


def register_attn_backend(backend: _Backend, class_path: str):
    """
    Decorator: register a custom attention backend into BACKEND_MAPPING.
    Validation: only checks if 'backend' is a valid _Backend enum member.
    Overwriting existing mappings is allowed.
    """
    if not isinstance(backend, _Backend):
        raise ValueError(f"{backend} is not a valid _Backend enum value.")

    def decorator(cls):
        BACKEND_MAPPING[backend] = class_path
        return cls

    return decorator


def backend_to_class_str(backend: _Backend) -> str:
    """Get the backend class string
    
    Args:
        backend: The backend enum value
        
    Returns:
        The backend class string
    """
    return BACKEND_MAPPING[backend]


def backend_to_class(backend: _Backend) -> Type:
    """Get the backend class.

    Args:
        backend: The backend enum value

    Returns:
        The backend class
    """
    backend_class_name = backend_to_class_str(backend)
    return resolve_obj_by_qualname(backend_class_name)


def backend_name_to_enum(backend_name: str) -> Optional[_Backend]:
    """
    Convert a string backend name to a _Backend enum value.

    Returns:
        _Backend: enum value if backend_name is a valid in-tree type
        None: otherwise it's an invalid in-tree type or an out-of-tree platform
              is loaded.
    """
    assert backend_name is not None
    backend_name = backend_name.removesuffix("_VLLM_V1")
    return _Backend[backend_name] if backend_name in _Backend.__members__ else \
          None
