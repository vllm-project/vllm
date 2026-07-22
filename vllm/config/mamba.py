# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from enum import Enum, EnumMeta
from typing import Any

from pydantic import AliasChoices, Field, field_validator

from vllm.config.utils import config


class _MambaDecodeBackendEnumMeta(EnumMeta):
    """Metaclass providing actionable Mamba decode backend errors."""

    def __getitem__(cls, name: str):
        try:
            return super().__getitem__(name)
        except KeyError:
            valid = ", ".join(cls.__members__.keys())
            raise ValueError(
                f"Unknown Mamba decode backend: '{name}'. Valid options are: {valid}"
            ) from None


class MambaDecodeBackendEnum(Enum, metaclass=_MambaDecodeBackendEnumMeta):
    """Enumeration of Mamba selective-state-update backends used for decode."""

    TRITON = "triton"
    FLASHINFER = "flashinfer"
    CPU = "cpu"


MambaBackendEnum = MambaDecodeBackendEnum


class _MambaPrefillBackendEnumMeta(EnumMeta):
    """Metaclass providing actionable Mamba prefill backend errors."""

    def __getitem__(cls, name: str):
        try:
            return super().__getitem__(name)
        except KeyError:
            valid = ", ".join(cls.__members__.keys())
            raise ValueError(
                f"Unknown Mamba prefill backend: '{name}'. Valid options are: {valid}"
            ) from None


class MambaPrefillBackendEnum(Enum, metaclass=_MambaPrefillBackendEnumMeta):
    """Enumeration of supported Mamba2 SSD prefill backends."""

    TRITON = "triton"
    FLASHINFER = "flashinfer"


@config
class MambaConfig:
    """Configuration for Mamba SSM backends."""

    decode_backend: MambaDecodeBackendEnum = Field(
        default=MambaDecodeBackendEnum.TRITON,
        validation_alias=AliasChoices("decode_backend", "backend"),
    )
    """Mamba selective-state-update backend used for decode."""

    prefill_backend: MambaPrefillBackendEnum = MambaPrefillBackendEnum.TRITON
    """Mamba2 SSD prefill backend to use. This is independent of the SSU
    backend used for decode."""

    enable_stochastic_rounding: bool = False
    """Enable stochastic rounding when writing SSM state to fp16 cache.
    Uses random bits to unbias the rounding error, which can improve
    numerical stability for long sequences."""
    stochastic_rounding_philox_rounds: int = 0
    """Number of Philox PRNG rounds for stochastic rounding random number
    generation. 0 uses the Triton default. Higher values improve randomness
    quality at the cost of compute."""

    @field_validator("decode_backend", mode="before")
    @classmethod
    def validate_decode_backend_before(cls, value: Any) -> Any:
        """Enable parsing of the decode backend enum from string."""
        if isinstance(value, str):
            return MambaDecodeBackendEnum[value.upper()]
        return value

    @field_validator("prefill_backend", mode="before")
    @classmethod
    def validate_prefill_backend_before(cls, value: Any) -> Any:
        """Enable parsing of the prefill backend enum from string."""
        if isinstance(value, str):
            return MambaPrefillBackendEnum[value.upper()]
        return value

    @property
    def backend(self) -> MambaDecodeBackendEnum:
        return self.decode_backend

    @backend.setter
    def backend(self, value: MambaDecodeBackendEnum) -> None:
        self.decode_backend = value

    def __post_init__(self):
        if self.enable_stochastic_rounding:
            from vllm.platforms import current_platform

            if not current_platform.is_cuda():
                raise ValueError(
                    "Stochastic rounding for Mamba cache is only supported "
                    "on NVIDIA CUDA platforms. Please do not specify  "
                    "`--enable-mamba-cache-stochastic-rounding`."
                )
            if (
                self.decode_backend == MambaDecodeBackendEnum.TRITON
                and not current_platform.is_device_capability_family(100)
            ):
                raise ValueError(
                    "Stochastic rounding for Mamba cache with triton backend requires "
                    "compute capability 10.0 (data center Blackwell). The `cvt.rs` "
                    "PTX instruction is not supported on your GPU. Please do not "
                    "specify `--enable-mamba-cache-stochastic-rounding`, "
                    "or set `--mamba-decode-backend flashinfer`."
                )
