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
                f"Unknown Mamba SSU backend: '{name}'. Valid options are: {valid}"
            ) from None


class MambaBackendEnum(Enum, metaclass=_MambaBackendEnumMeta):
    """Enumeration of supported Mamba SSU (selective state update) backends."""

    TRITON = "triton"
    FLASHINFER = "flashinfer"
    CPU = "cpu"


MambaSSUAlgorithm = Literal["auto", "simple", "vertical", "horizontal"]

_FLOAT_SSM_DTYPES = ("auto", "float32", "float16", "bfloat16")
_SCALED_SSM_DTYPES = ("int8", "fp8")


def decode_backend_ssm_dtypes(
    backend: MambaBackendEnum, use_replayssm: bool
) -> tuple[str, ...]:
    """`--mamba-ssm-cache-dtype` values a Mamba decode backend can read and write.

    Scaled 8-bit state needs a kernel that applies the per-row scale on every
    read and write; only FlashInfer ReplaySSM has one.
    """
    if backend == MambaBackendEnum.FLASHINFER and use_replayssm:
        return _FLOAT_SSM_DTYPES + _SCALED_SSM_DTYPES
    return _FLOAT_SSM_DTYPES


def prefill_backend_ssm_dtypes(backend: MambaBackendEnum) -> tuple[str, ...]:
    """`--mamba-ssm-cache-dtype` values a Mamba2 prefill backend supports.

    The Triton SSD scan works in FP32 and vLLM converts scaled 8-bit state
    around it. FlashInfer's fused scan reads and writes the state itself.
    """
    if backend == MambaBackendEnum.TRITON:
        return _FLOAT_SSM_DTYPES + _SCALED_SSM_DTYPES
    return _FLOAT_SSM_DTYPES


@config
class MambaConfig:
    """Configuration for Mamba SSM backends."""

    backend: MambaBackendEnum = MambaBackendEnum.TRITON
    """Mamba SSU backend to use."""

    enable_stochastic_rounding: bool = False
    """Enable stochastic rounding when writing SSM state to an FP16, INT8 or
    FP8 cache.
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

    prefill_backend: MambaBackendEnum = MambaBackendEnum.TRITON
    """Kernel backend for the SSD scan of Mamba2 prefills. "flashinfer" runs
    FlashInfer's fused SSD scan (SM100/SM103 only); prefills that store Mamba
    checkpoints always use "triton"."""

    @field_validator("backend", "prefill_backend", mode="before")
    @classmethod
    def validate_backend_before(cls, value: Any) -> Any:
        """Enable parsing of the backend enum fields from strings."""
        if isinstance(value, str):
            return MambaBackendEnum[value.upper()]
        return value

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
        if self.backend != MambaBackendEnum.FLASHINFER:
            raise ValueError(
                "Mamba SSU algorithm selection is only supported with the "
                "FlashInfer backend. Please set `--mamba-backend flashinfer`, "
                "or omit `--mamba-ssu-algorithm`."
            )

    def validate_prefill_backend(self) -> None:
        if self.prefill_backend == MambaBackendEnum.CPU:
            raise ValueError(
                "Unknown Mamba prefill backend: 'cpu'. "
                "Valid options are: triton, flashinfer"
            )
        if self.prefill_backend == MambaBackendEnum.FLASHINFER:
            from vllm.platforms import current_platform

            if not current_platform.is_device_capability_family(100):
                raise ValueError(
                    "--mamba-prefill-backend flashinfer requires SM100/SM103."
                )

    def __post_init__(self):
        self.validate_ssu_algorithm()
        self.validate_prefill_backend()
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
                self.backend == MambaBackendEnum.TRITON
                and not current_platform.is_device_capability_family(100)
            ):
                raise ValueError(
                    "Stochastic rounding for Mamba cache with triton backend requires "
                    "compute capability 10.0 (data center Blackwell). The `cvt.rs` "
                    "PTX instruction is not supported on your GPU. Please do not "
                    "specify `--enable-mamba-cache-stochastic-rounding`, "
                    "or set `--mamba-backend flashinfer`."
                )
