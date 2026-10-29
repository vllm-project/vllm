# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from enum import Enum, EnumMeta
from typing import Any, Literal, get_args

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
                f"Unknown Mamba backend: '{name}'. Valid options are: {valid}"
            ) from None


class MambaBackendEnum(Enum, metaclass=_MambaBackendEnumMeta):
    """Enumeration of supported Mamba backends."""

    TRITON = "triton"
    FLASHINFER = "flashinfer"
    CPU = "cpu"


MambaSSUAlgorithm = Literal["auto", "simple", "vertical", "horizontal"]


@config
class MambaConfig:
    """Configuration for Mamba SSM backends."""

    backend: MambaBackendEnum = MambaBackendEnum.TRITON
    """Default backend for Mamba operations."""

    decode_backend: MambaBackendEnum | None = None
    """Decode backend for SSU or ReplaySSM. None inherits `backend`."""

    enable_stochastic_rounding: bool = False
    """Enable stochastic rounding when writing SSM state to fp16 cache.
    Uses random bits to unbias the rounding error, which can improve
    numerical stability for long sequences. The Triton backend requires
    compute capability 10.0 on CUDA or gfx950 on ROCm."""
    stochastic_rounding_philox_rounds: int = 0
    """Number of Philox PRNG rounds for stochastic rounding random number
    generation. 0 uses the Triton default. Higher values improve randomness
    quality at the cost of compute."""

    ssu_algorithm: MambaSSUAlgorithm | None = None
    """Selective state update algorithm to use with the FlashInfer backend.
    None defaults to FlashInfer's "auto" algorithm. Forced algorithms must
    be supported by FlashInfer for the active GPU, state dtype, and decoding
    mode."""

    prefill_backend: MambaBackendEnum | None = None
    """Kernel backend for the SSD scan of Mamba2 prefills. "flashinfer" runs
    FlashInfer's fused SSD scan (SM100/SM103 only). None inherits `backend`,
    falling back to Triton if an inherited FlashInfer SSD is unavailable."""

    @field_validator("backend", "decode_backend", "prefill_backend", mode="before")
    @classmethod
    def validate_backend_before(cls, value: Any) -> Any:
        """Enable parsing of the backend enum fields from strings."""
        if isinstance(value, str):
            return MambaBackendEnum[value.upper()]
        return value

    @property
    def effective_decode_backend(self) -> MambaBackendEnum:
        return self.decode_backend if self.decode_backend is not None else self.backend

    @property
    def effective_prefill_backend(self) -> MambaBackendEnum:
        return (
            self.prefill_backend if self.prefill_backend is not None else self.backend
        )

    def validate_ssu_algorithm(self) -> None:
        if self.ssu_algorithm is None:
            return
        valid_algorithms = get_args(MambaSSUAlgorithm)
        if self.ssu_algorithm not in valid_algorithms:
            valid = ", ".join(valid_algorithms)
            raise ValueError(
                f"Unknown Mamba SSU algorithm: '{self.ssu_algorithm}'. "
                f"Valid options are: {valid}"
            )
        if self.effective_decode_backend != MambaBackendEnum.FLASHINFER:
            raise ValueError(
                "Mamba SSU algorithm selection is only supported with the "
                "FlashInfer backend. Please set `--mamba-decode-backend flashinfer`, "
                "or omit `--mamba-ssu-algorithm`."
            )

    def __post_init__(self):
        self.validate_ssu_algorithm()
        if self.enable_stochastic_rounding:
            from vllm.platforms import current_platform

            if current_platform.is_rocm():
                from vllm.platforms.rocm import on_gfx950

                if not on_gfx950():
                    raise ValueError(
                        "Stochastic rounding for Mamba cache on ROCm requires "
                        "gfx950 (MI350/MI355). Please do not specify "
                        "`--enable-mamba-cache-stochastic-rounding`."
                    )
            elif not current_platform.is_cuda():
                raise ValueError(
                    "Stochastic rounding for Mamba cache is only supported "
                    "on NVIDIA CUDA and AMD gfx950 platforms. Please do not specify "
                    "`--enable-mamba-cache-stochastic-rounding`."
                )
            elif (
                self.effective_decode_backend == MambaBackendEnum.TRITON
                and not current_platform.is_device_capability_family(100)
            ):
                raise ValueError(
                    "Stochastic rounding for Mamba cache with triton backend requires "
                    "compute capability 10.0 (data center Blackwell). The `cvt.rs` "
                    "PTX instruction is not supported on your GPU. Please do not "
                    "specify `--enable-mamba-cache-stochastic-rounding`, "
                    "or set `--mamba-decode-backend flashinfer`."
                )
