# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Registry for MLA prefill backends.

This module provides an enumeration of all available MLA prefill backends
and utilities for loading and registering them.
"""

from collections.abc import Callable
from enum import Enum, EnumMeta
from typing import TYPE_CHECKING

from vllm.utils.import_utils import resolve_obj_by_qualname

if TYPE_CHECKING:
    from vllm.v1.attention.backends.mla.prefill.base import MLAPrefillBackend


class _MLAPrefillBackendEnumMeta(EnumMeta):
    """Metaclass for MLAPrefillBackendEnum to provide better error messages."""

    def __getitem__(cls, name: str):
        try:
            return super().__getitem__(name)
        except KeyError:
            members = cls.__members__.keys()
            valid_backends = ", ".join(members)
            raise ValueError(
                f"Unknown MLA prefill backend: '{name}'. "
                f"Valid options are: {valid_backends}"
            ) from None


class MLAPrefillBackendEnum(Enum, metaclass=_MLAPrefillBackendEnumMeta):
    """Enumeration of all supported MLA prefill backends."""

    FLASH_ATTN = (
        "vllm.v1.attention.backends.mla.prefill.flash_attn.FlashAttnPrefillBackend"
    )
    FLASHINFER = (
        "vllm.v1.attention.backends.mla.prefill.flashinfer.FlashInferPrefillBackend"
    )
    TRTLLM_RAGGED = (
        "vllm.v1.attention.backends.mla.prefill.trtllm_ragged."
        "TrtllmRaggedPrefillBackend"
    )
    TOKENSPEED_MLA = (
        "vllm.v1.attention.backends.mla.prefill.tokenspeed_mla."
        "TokenspeedMLAPrefillBackend"
    )
    AITER_ASM = (
        "vllm.v1.attention.backends.mla.prefill.aiter_asm.AiterAsmPrefillBackend"
    )
    ROCM_AITER_FA = (
        "vllm.v1.attention.backends.mla.prefill.aiter_flash_attn."
        "AiterFlashAttnPrefillBackend"
    )
    # Placeholder for third-party/custom backends - must be registered before use
    # set to None to avoid alias with other backend, whose value is an empty string
    CUSTOM = None

    def get_path(self) -> str:
        """Get the class path for this backend (respects overrides).

        Returns:
            The fully qualified class path string

        Raises:
            ValueError: If Backend.CUSTOM is used without being registered
        """
        path = _MLA_PREFILL_OVERRIDES.get(self, self.value)
        if not path:
            raise ValueError(
                f"MLA prefill backend {self.name} must be registered before "
                f"use. Use register_mla_prefill_backend("
                f"MLAPrefillBackendEnum.{self.name}, "
                f"'your.module.YourClass')"
            )
        return path

    def get_class(self) -> "type[MLAPrefillBackend]":
        """Get the backend class (respects overrides).

        Returns:
            The backend class

        Raises:
            ImportError: If the backend class cannot be imported
            ValueError: If CUSTOM is used without being registered
        """
        return resolve_obj_by_qualname(self.get_path())

    def is_overridden(self) -> bool:
        """Check if this backend has been overridden."""
        return self in _MLA_PREFILL_OVERRIDES

    def clear_override(self) -> None:
        """Clear any override for this backend, reverting to the default."""
        _MLA_PREFILL_OVERRIDES.pop(self, None)


_MLA_PREFILL_OVERRIDES: dict[MLAPrefillBackendEnum, str] = {}


def register_mla_prefill_backend(
    backend: MLAPrefillBackendEnum,
    class_path: str | None = None,
) -> Callable[[type], type]:
    """Register or override an MLA prefill backend implementation.

    Args:
        backend: The MLAPrefillBackendEnum member to register.
        class_path: Optional class path. If not provided and used as
            decorator, will be auto-generated from the class.

    Returns:
        Decorator function if class_path is None, otherwise a no-op.

    Examples:
        # Override an existing MLA prefill backend
        @register_mla_prefill_backend(MLAPrefillBackendEnum.FLASH_ATTN)
        class MyCustomFlashAttn(MLAPrefillBackend):
            ...

        # Register a custom third-party MLA prefill backend
        @register_mla_prefill_backend(MLAPrefillBackendEnum.CUSTOM)
        class MyCustomPrefillBackend(MLAPrefillBackend):
            ...

        # Direct registration
        register_mla_prefill_backend(
            MLAPrefillBackendEnum.CUSTOM,
            "my.module.MyCustomPrefillBackend"
        )
    """

    def decorator(cls: type) -> type:
        _MLA_PREFILL_OVERRIDES[backend] = f"{cls.__module__}.{cls.__qualname__}"
        return cls

    if class_path is not None:
        _MLA_PREFILL_OVERRIDES[backend] = class_path
        return lambda x: x

    return decorator
