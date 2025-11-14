# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Attention backend registry"""

import enum
from collections.abc import Callable
from typing import TYPE_CHECKING, cast

from vllm.logger import init_logger
from vllm.utils.import_utils import resolve_obj_by_qualname

if TYPE_CHECKING:
    from vllm.attention.backends.abstract import AttentionBackend

logger = init_logger(__name__)


class _AttentionBackendEnumMeta(enum.EnumMeta):
    """Metaclass for AttentionBackendEnum to provide better error messages."""

    def __getitem__(cls, name: str):
        """Get backend by name with helpful error messages."""
        try:
            return super().__getitem__(name)
        except KeyError:
            members = cast("dict[str, AttentionBackendEnum]", cls.__members__).values()
            valid_backends = ", ".join(m.name for m in members)
            raise ValueError(
                f"Unknown attention backend: '{name}'. "
                f"Valid options are: {valid_backends}"
            ) from None


class AttentionBackendEnum(enum.Enum, metaclass=_AttentionBackendEnumMeta):
    """Enumeration of all supported attention backends.

    The enum value is the default class path, but this can be overridden
    at runtime using register_backend().

    To get the actual backend class (respecting overrides), use:
        backend.get_class()
    """

    FLASH_ATTN = "vllm.v1.attention.backends.flash_attn.FlashAttentionBackend"
    TRITON_ATTN = "vllm.v1.attention.backends.triton_attn.TritonAttentionBackend"
    XFORMERS = "vllm.v1.attention.backends.xformers.XFormersAttentionBackend"
    ROCM_ATTN = "vllm.v1.attention.backends.rocm_attn.RocmAttentionBackend"
    ROCM_AITER_MLA = "vllm.v1.attention.backends.mla.rocm_aiter_mla.AiterMLABackend"
    ROCM_AITER_FA = (
        "vllm.v1.attention.backends.rocm_aiter_fa.AiterFlashAttentionBackend"
    )
    TORCH_SDPA = ""  # this tag is only used for ViT
    FLASHINFER = "vllm.v1.attention.backends.flashinfer.FlashInferBackend"
    FLASHINFER_MLA = (
        "vllm.v1.attention.backends.mla.flashinfer_mla.FlashInferMLABackend"
    )
    TRITON_MLA = "vllm.v1.attention.backends.mla.triton_mla.TritonMLABackend"
    CUTLASS_MLA = "vllm.v1.attention.backends.mla.cutlass_mla.CutlassMLABackend"
    FLASHMLA = "vllm.v1.attention.backends.mla.flashmla.FlashMLABackend"
    FLASHMLA_SPARSE = (
        "vllm.v1.attention.backends.mla.flashmla_sparse.FlashMLASparseBackend"
    )
    FLASH_ATTN_MLA = "vllm.v1.attention.backends.mla.flashattn_mla.FlashAttnMLABackend"
    PALLAS = "vllm.v1.attention.backends.pallas.PallasAttentionBackend"
    IPEX = "vllm.v1.attention.backends.ipex.IpexAttentionBackend"
    NO_ATTENTION = "vllm.v1.attention.backends.no_attention.NoAttentionBackend"
    FLEX_ATTENTION = "vllm.v1.attention.backends.flex_attention.FlexAttentionBackend"
    TREE_ATTN = "vllm.v1.attention.backends.tree_attn.TreeAttentionBackend"
    ROCM_AITER_UNIFIED_ATTN = (
        "vllm.v1.attention.backends.rocm_aiter_unified_attn."
        "RocmAiterUnifiedAttentionBackend"
    )
    CPU_ATTN = "vllm.v1.attention.backends.cpu_attn.CPUAttentionBackend"
    # Placeholder for third-party/custom backends - must be registered before use
    CUSTOM = ""

    def get_path(self, include_classname: bool = True) -> str:
        """Get the class path for this backend (respects overrides).

        Returns:
            The fully qualified class path string

        Raises:
            ValueError: If Backend.CUSTOM is used without being registered
        """
        path = _OVERRIDES.get(self, self.value)
        if not path:
            raise ValueError(
                f"Backend {self.name} must be registered before use. "
                f"Use register_backend(Backend.{self.name}, 'your.module.YourClass')"
            )
        if not include_classname:
            path = path.rsplit(".", 1)[0]
        return path

    def get_class(self) -> "type[AttentionBackend]":
        """Get the backend class (respects overrides).

        Returns:
            The backend class

        Raises:
            ImportError: If the backend class cannot be imported
            ValueError: If Backend.CUSTOM is used without being registered
        """
        return resolve_obj_by_qualname(self.get_path())

    def is_overridden(self) -> bool:
        """Check if this backend has been overridden.

        Returns:
            True if the backend has a registered override
        """
        return self in _OVERRIDES

    def clear_override(self) -> None:
        """Clear any override for this backend, reverting to the default."""
        _OVERRIDES.pop(self, None)


_OVERRIDES: dict[AttentionBackendEnum, str] = {}


def register_backend(
    backend: AttentionBackendEnum, class_path: str | None = None
) -> Callable[[type], type]:
    """Register or override a backend implementation.

    Args:
        backend: The AttentionBackendEnum member to register
        class_path: Optional class path. If not provided and used as
            decorator, will be auto-generated from the class.

    Returns:
        Decorator function if class_path is None, otherwise a no-op

    Examples:
        # Override an existing backend
        @register_backend(AttentionBackendEnum.FLASH_ATTN)
        class MyCustomFlashAttn:
            ...

        # Register a custom third-party backend
        @register_backend(AttentionBackendEnum.CUSTOM)
        class MyCustomBackend:
            ...

        # Direct registration
        register_backend(
            AttentionBackendEnum.CUSTOM,
            "my.module.MyCustomBackend"
        )
    """

    def decorator(cls: type) -> type:
        _OVERRIDES[backend] = f"{cls.__module__}.{cls.__qualname__}"
        return cls

    if class_path is not None:
        _OVERRIDES[backend] = class_path
        return lambda x: x

    return decorator


# Backwards compatibility alias for plugins
class _BackendMeta(type):
    """Metaclass to provide deprecation warnings when accessing _Backend."""

    def __getattribute__(cls, name: str):
        if name not in ("__class__", "__mro__", "__name__"):
            logger.warning(
                "_Backend has been renamed to AttentionBackendEnum. "
                "Please update your code to use AttentionBackendEnum instead. "
                "_Backend will be removed in a future release."
            )
        return getattr(AttentionBackendEnum, name)

    def __getitem__(cls, name: str):
        logger.warning(
            "_Backend has been renamed to AttentionBackendEnum. "
            "Please update your code to use AttentionBackendEnum instead. "
            "_Backend will be removed in a future release."
        )
        return AttentionBackendEnum[name]


class _Backend(metaclass=_BackendMeta):
    """Deprecated: Use AttentionBackendEnum instead.

    This class is provided for backwards compatibility with plugins
    and will be removed in a future release.
    """

    pass
