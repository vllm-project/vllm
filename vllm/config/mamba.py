# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from enum import EnumMeta, StrEnum
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


class MambaBackendEnum(StrEnum, metaclass=_MambaBackendEnumMeta):
    """Enumeration of supported Mamba backends."""

    TRITON = "triton"
    FLASHINFER = "flashinfer"
    CPU = "cpu"


MambaSSUAlgorithm = Literal["auto", "simple", "vertical", "horizontal"]


@config
class MambaConfig:
    """Configuration for Mamba SSM backends."""

    backend: MambaBackendEnum | None = None
    """Kernel backend for Mamba decode (SSU, and ReplaySSM's provider). A
    backend that can't run fails at startup. None selects automatically:
    Triton on GPUs and "cpu" on CPU, or FlashInfer where only it can run what
    is asked."""

    enable_stochastic_rounding: bool = False
    """Enable stochastic rounding when writing SSM state to fp16 cache.
    Uses random bits to unbias the rounding error, which can improve
    numerical stability for long sequences. Needs a float16 SSM cache
    (`--mamba-ssm-cache-dtype float16`) and a decode kernel that supports it:
    Triton on compute capability 10.x (CUDA) or gfx950 (ROCm), or FlashInfer."""
    stochastic_rounding_philox_rounds: int = 0
    """Number of Philox PRNG rounds for stochastic rounding random number
    generation. 0 uses the Triton default. Higher values improve randomness
    quality at the cost of compute."""

    ssu_algorithm: MambaSSUAlgorithm | None = None
    """Selective state update algorithm for FlashInfer's decode kernel, the
    only one that takes one. None defaults to FlashInfer's "auto" algorithm.
    Forced algorithms must be supported by FlashInfer for the active GPU,
    state dtype, and decoding mode."""

    prefill_backend: MambaBackendEnum | None = None
    """Kernel backend for the SSD scan of Mamba2 prefills. "flashinfer" runs
    FlashInfer's fused SSD scan (SM10x). A backend that can't run fails at
    startup. None selects automatically: Triton on GPUs, "cpu" on CPU."""

    conv1d_prefill_backend: MambaBackendEnum | None = None
    """Kernel backend for the causal conv1d of prefills, in the layers that
    select one (Mamba2, ShortConv). "flashinfer" runs FlashInfer's
    width-4 SiLU conv1d (SM100/SM103, BF16). A backend that can't run fails at
    startup. None selects automatically: Triton on GPUs, "cpu" on CPU."""

    conv1d_decode_backend: MambaBackendEnum | None = None
    """Kernel backend for the causal conv1d of decodes, as
    `conv1d_prefill_backend` is for prefills."""

    @field_validator(
        "backend",
        "prefill_backend",
        "conv1d_prefill_backend",
        "conv1d_decode_backend",
        mode="before",
    )
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

    def __post_init__(self):
        self.validate_ssu_algorithm()
