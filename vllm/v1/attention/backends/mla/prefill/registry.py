# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Registry for MLA prefill backends.

This module provides an enumeration of all available MLA prefill backends
and utilities for loading them.
"""

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
    CUDNN = "vllm.v1.attention.backends.mla.prefill.cudnn.CudnnPrefillBackend"
    TRTLLM_RAGGED = (
        "vllm.v1.attention.backends.mla.prefill.trtllm_ragged."
        "TrtllmRaggedPrefillBackend"
    )

    def get_path(self) -> str:
        """Get the fully qualified class path for this backend."""
        return self.value

    def get_class(self) -> "type[MLAPrefillBackend]":
        """Lazy load and return the backend class."""
        return resolve_obj_by_qualname(self.get_path())
