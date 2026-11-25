# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from enum import Enum, EnumMeta
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, get_args

import regex as re
import torch
from pydantic import Field, field_validator

from vllm.config.utils import config

if TYPE_CHECKING:
    from vllm.config.vllm import VllmConfig


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


@config(config={"frozen": True}, kw_only=True)
class SpareConfig:
    """Fixed, uniform per-head projections for Granite Mamba2."""

    rank: int = Field(gt=0, strict=True)
    """Uniform temporal state dimension after projection."""
    basis_path: str = Field(min_length=1)
    """Absolute path to a trusted SafeTensors file."""
    basis_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    """Expected lowercase SHA-256 digest of the file."""
    method: Literal["state_pca", "identity"] = "state_pca"
    """Offline basis method. Identity is a full-rank control."""

    def __post_init__(self) -> None:
        if not Path(self.basis_path).is_absolute():
            raise ValueError("SPARE basis_path must be absolute")


@config
class MambaConfig:
    """Configuration for Mamba SSM backends."""

    backend: MambaBackendEnum = MambaBackendEnum.TRITON
    """Mamba SSU backend to use."""

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

    prefill_backend: MambaBackendEnum = MambaBackendEnum.TRITON
    """Kernel backend for the SSD scan of Mamba2 prefills. "flashinfer" runs
    FlashInfer's fused SSD scan (SM100/SM103 only); prefills that store Mamba
    checkpoints always use "triton"."""

    spare: SpareConfig | None = None
    """Experimental uniform-rank Mamba2 state compression with a fixed basis.
    None uses the standard full-dimensional Mamba2 recurrence."""

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


def validate_spare(config: "VllmConfig") -> None:
    mamba = config.mamba_config
    spare = mamba.spare
    if spare is None:
        return

    from vllm.config.compilation import CompilationMode, CUDAGraphMode
    from vllm.platforms import current_platform

    model = config.model_config
    cache = config.cache_config
    parallel = config.parallel_config
    compilation = config.compilation_config
    if (
        model is None
        or model.architecture != "GraniteMoeHybridForCausalLM"
        or model.model_impl not in ("auto", "vllm")
    ):
        raise ValueError("SPARE requires the native GraniteMoeHybridForCausalLM model")
    if not re.fullmatch(r"[0-9a-f]{40}", model.revision or ""):
        raise ValueError("SPARE requires a pinned checkpoint commit revision")
    size = model.hf_config.mamba_d_state
    if spare.rank not in (16, 32, 64, size) or spare.rank > size:
        raise ValueError("SPARE rank must be 16, 32, 64, or the native state size")
    if spare.method == "identity" and spare.rank != size:
        raise ValueError("SPARE identity requires full rank")
    if (
        parallel.world_size_across_dp != 1
        or parallel.prefill_context_parallel_size != 1
        or parallel.decode_context_parallel_size != 1
    ):
        raise ValueError("SPARE requires single-device serving")
    if model.dtype != torch.bfloat16 or cache.mamba_ssm_cache_dtype != "float32":
        raise ValueError("SPARE requires BF16 model dtype and FP32 temporal state")
    if cache.enable_prefix_caching or cache.enable_mamba_shared_prefix_checkpoint:
        raise ValueError("SPARE requires prefix caching disabled")
    if config.scheduler_config.disable_hybrid_kv_cache_manager:
        raise ValueError("SPARE requires the hybrid KV cache manager")
    if config.speculative_config is not None or cache.use_replayssm:
        raise ValueError("SPARE does not support speculative decoding or ReplaySSM")
    if config.kv_transfer_config is not None or cache.kv_offloading_size is not None:
        raise ValueError("SPARE does not support KV transfer or offloading")
    if config.lora_config is not None or model.quantization is not None:
        raise ValueError("SPARE requires the unquantized model without LoRA")
    if compilation.mode != CompilationMode.NONE or compilation.cudagraph_mode not in (
        CUDAGraphMode.NONE,
        CUDAGraphMode.FULL_DECODE_ONLY,
    ):
        raise ValueError(
            "SPARE requires eager or decode-only CUDA Graph without compilation"
        )
    if (
        not current_platform.is_cuda()
        or mamba.backend != MambaBackendEnum.TRITON
        or mamba.prefill_backend != MambaBackendEnum.TRITON
        or mamba.enable_stochastic_rounding
    ):
        raise ValueError(
            "SPARE requires NVIDIA CUDA with Triton SSU and SSD "
            "without stochastic rounding"
        )
