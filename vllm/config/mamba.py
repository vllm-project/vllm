# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from enum import Enum, EnumMeta
from typing import Any

from pydantic import field_validator

from vllm.config.utils import config


class _MambaBackendEnumMeta(EnumMeta):
    """Metaclass for MambaBackendEnum to provide better error messages."""

    def __getitem__(cls, name: str):
        try:
            return super().__getitem__(name)
        except KeyError:
            valid = ", ".join(cls.__members__.keys())
            raise ValueError(
                f"Unknown Mamba SSU backend: '{name}'. Valid options are: {valid}"
            ) from None


class MambaBackendEnum(Enum, metaclass=_MambaBackendEnumMeta):
    """Enumeration of supported Mamba SSU (selective state update) backends."""

    TRITON = "triton"
    FLASHINFER = "flashinfer"


@config
class MambaConfig:
    """Configuration for Mamba SSM backends."""

    backend: MambaBackendEnum | None = None
    """Mamba SSU backend to use. If None, defaults to triton."""

    @field_validator("backend", mode="before")
    @classmethod
    def validate_backend_before(cls, value: Any) -> Any:
        """Enable parsing of the `backend` enum type from string."""
        if isinstance(value, str):
            return MambaBackendEnum[value.upper()]
        return value
