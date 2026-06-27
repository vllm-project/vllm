# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Registry for MLA prefill backends.

This module provides an enumeration of all available MLA prefill backends
and utilities for loading and registering them.
"""

from collections.abc import Callable
from enum import Enum, EnumMeta
from typing import TYPE_CHECKING

from vllm.logger import init_logger
from vllm.utils.import_utils import resolve_obj_by_qualname

if TYPE_CHECKING:
    from vllm.v1.attention.backends.mla.prefill.base import MLAPrefillBackend

logger = init_logger(__name__)


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
    # Placeholder for third-party/custom backends - must be registered before use
    # set to None to avoid alias with other backend, whose value is an empty string
    CUSTOM = None

    @classmethod
    def register(cls, name: str, value: str) -> "MLAPrefillBackendEnum":
        """Dynamically register a new MLA prefill backend enum member.

        Args:
            name: The name for the new enum member
            value: The fully qualified class path string

        Returns:
            The newly created enum member
        """
        if name in cls._member_map_:
            raise ValueError(
                f"MLA prefill backend {name} already exists in {cls.__name__}. "
                f"Use register_mla_prefill_backend("
                f"{cls.__name__}.{name}, '{value}') to override."
            )
        if not name.isidentifier() or hasattr(cls, name):
            raise ValueError(f"Invalid or reserved backend name: {name}")

        member = object.__new__(cls)
        member._name_ = name
        member._value_ = value
        setattr(cls, name, member)
        cls._member_map_[name] = member
        cls._value2member_map_[value] = member
        cls._member_names_.append(name)
        logger.info("Registered new MLA prefill backend: %s -> %s", name, value)
        return member

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
    backend: MLAPrefillBackendEnum | str,
    class_path: str | None = None,
) -> Callable[[type], type]:
    """Register or override an MLA prefill backend implementation.

    Args:
        backend: The MLAPrefillBackendEnum member to register,
                 or a string name for a new custom backend
                 (e.g., "CUSTOM_MLA_PREFILL").
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

        # Register a new custom MLA prefill backend with a dynamic enum name
        @register_mla_prefill_backend("CUSTOM_MLA_PREFILL")
        class CustomMLAPrefillBackend(MLAPrefillBackend):
            ...

        # Direct registration
        register_mla_prefill_backend(
            MLAPrefillBackendEnum.CUSTOM,
            "my.module.MyCustomPrefillBackend"
        )

        # Direct registration with string name
        register_mla_prefill_backend(
            "CUSTOM_MLA_PREFILL",
            "custom.attention.mla_prefill.CustomMLAPrefillBackend"
        )
    """
    # Handle dynamic enum creation for string backend names
    if isinstance(backend, str):
        backend = MLAPrefillBackendEnum.register(backend, class_path or "")

    def decorator(cls: type) -> type:
        _MLA_PREFILL_OVERRIDES[backend] = f"{cls.__module__}.{cls.__qualname__}"
        return cls

    if class_path is not None:
        _MLA_PREFILL_OVERRIDES[backend] = class_path
        return lambda x: x

    return decorator
