# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Attention backend registry"""

import enum
from typing import Optional

from vllm.utils import resolve_obj_by_qualname


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
    FLASH_ATTN_MLA = enum.auto()
    PALLAS = enum.auto()
    IPEX = enum.auto()
    NO_ATTENTION = enum.auto()
    FLEX_ATTENTION = enum.auto()
    TREE_ATTN = enum.auto()


BACKEND_MAP = {}


def register_attn_backend(backend: _Backend, class_path: Optional[str] = None):
    """
    Decorator: register a custom attention backend into BACKEND_MAPPING.
    - If class_path is provided, use it.
    - Otherwise, auto-generate from the class object.
    Validation: only checks if 'backend' is a valid _Backend enum member.
    Overwriting existing mappings is allowed.
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


def backend_to_class(backend: _Backend) -> type:
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
    return _Backend[backend_name] if backend_name in _Backend.__members__ else None
