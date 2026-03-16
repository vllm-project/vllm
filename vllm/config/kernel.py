# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Callable
from typing import Any, Literal

from pydantic import Field, field_validator

from vllm.config.utils import config
from vllm.utils.hashing import safe_hash

MoEBackend = Literal[
    "auto",
    "triton",
    "deep_gemm",
    "cutlass",
    "flashinfer_trtllm",
    "flashinfer_cutlass",
    "flashinfer_cutedsl",
    "marlin",
    "aiter",
]


@config
class KernelConfig:
    """Configuration for kernel selection and warmup behavior."""

    enable_flashinfer_autotune: bool = Field(default=None)
    """If True, run FlashInfer autotuning during kernel warmup."""

    moe_backend: MoEBackend = "auto"
    """Backend for MoE expert computation kernels. Available options:

    - "auto": Automatically select the best backend based on model and hardware\n
    - "triton": Use Triton-based fused MoE kernels\n
    - "deep_gemm": Use DeepGEMM kernels (FP8 block-quantized only)\n
    - "cutlass": Use vLLM CUTLASS kernels\n
    - "flashinfer_trtllm": Use FlashInfer with TRTLLM-GEN kernels\n
    - "flashinfer_cutlass": Use FlashInfer with CUTLASS kernels\n
    - "flashinfer_cutedsl": Use FlashInfer with CuteDSL kernels (FP4 only)\n
    - "marlin": Use Marlin kernels (weight-only quantization)\n
    - "aiter": Use AMD AITer kernels (ROCm only)"""

    @field_validator("moe_backend", mode="before")
    @classmethod
    def _normalize_moe_backend(cls, value: Any) -> Any:
        if isinstance(value, str):
            return value.lower().replace("-", "_")
        return value

    def compute_hash(self) -> str:
        """
        WARNING: Whenever a new field is added to this config,
        ensure that it is included in the factors list if
        it affects the computation graph.

        Provide a hash that uniquely identifies all the configs
        that affect the structure of the computation
        graph from input ids/embeddings to the final hidden states,
        excluding anything before input ids/embeddings and after
        the final hidden states.
        """
        # no factors to consider.
        # this config will not affect the computation graph.
        factors: list[Any] = []
        hash_str = safe_hash(str(factors).encode(), usedforsecurity=False).hexdigest()
        return hash_str

    @field_validator("enable_flashinfer_autotune", mode="wrap")
    @classmethod
    def _skip_none_validation(cls, value: Any, handler: Callable) -> Any:
        """Skip validation if the value is `None` when initialization is delayed."""
        if value is None:
            return value
        return handler(value)
