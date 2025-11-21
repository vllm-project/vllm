# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Any

from pydantic import field_validator
from pydantic.dataclasses import dataclass

from vllm.attention.backends.registry import AttentionBackendEnum
from vllm.config.utils import config
from vllm.logger import init_logger

logger = init_logger(__name__)


@config
@dataclass
class AttentionConfig:
    """Configuration for attention mechanisms in vLLM."""

    backend: AttentionBackendEnum | None = None
    """Attention backend to use. If None, will be selected automatically.
    Example options: FLASH_ATTN, XFORMERS, FLASHINFER, etc."""

    flash_attn_version: int | None = None
    """Force vllm to use a specific flash-attention version (2 or 3).
    Only valid when using the flash-attention backend."""

    v1_use_prefill_decode_attention: bool = False
    """Use separate prefill and decode kernels for V1 attention instead of
    the unified triton kernel."""

    flash_attn_max_num_splits_for_cuda_graph: int = 32
    """Flash Attention max number splits for cuda graph decode."""

    use_cudnn_prefill: bool = False
    """Whether to use cudnn prefill."""

    use_trtllm_ragged_deepseek_prefill: bool = False
    """Whether to use TRTLLM ragged deepseek prefill."""

    use_trtllm_attention: bool | None = None
    """If set to True/False, use or don't use the TRTLLM attention backend
    in flashinfer. If None, auto-detect the attention backend in flashinfer."""

    disable_flashinfer_prefill: bool = False
    """Whether to disable flashinfer prefill."""

    flashinfer_disable_q_quantization: bool = False
    """If set, when using fp8 kv, do not quantize Q to fp8."""

    def compute_hash(self) -> str:
        """
        Provide a hash that uniquely identifies all the configs
        that affect the structure of the computation
        graph from input ids/embeddings to the final hidden states,
        excluding anything before input ids/embeddings and after
        the final hidden states.
        """
        from vllm.config.utils import get_hash_factors, hash_factors

        ignored_factors: list[str] = []
        factors = get_hash_factors(self, ignored_factors)
        return hash_factors(factors)

    @field_validator("backend", mode="before")
    @classmethod
    def validate_backend_before(cls, value: Any) -> Any:
        """Enable parsing of the `backend` enum type from string."""
        if isinstance(value, str):
            return AttentionBackendEnum[value.upper()]
        return value

    def __post_init__(self) -> None:
        from vllm import envs

        if self.backend is None:
            self.backend = envs.VLLM_ATTENTION_BACKEND
            if envs.is_set("VLLM_ATTENTION_BACKEND"):
                logger.warning(
                    "Using VLLM_ATTENTION_BACKEND environment variable is deprecated "
                    "and will be removed in a future release. "
                    "Please use --attention-backend CLI argument instead."
                )
        elif envs.is_set("VLLM_ATTENTION_BACKEND"):
            logger.warning(
                "Both --attention-backend CLI argument and "
                "VLLM_ATTENTION_BACKEND environment variable are set. "
                "--attention-backend will take precedence. VLLM_ATTENTION_BACKEND "
                "is deprecated and will be removed in a future release."
            )

        if envs.is_set("VLLM_FLASH_ATTN_VERSION"):
            self.flash_attn_version = envs.VLLM_FLASH_ATTN_VERSION

        if envs.is_set("VLLM_V1_USE_PREFILL_DECODE_ATTENTION"):
            self.v1_use_prefill_decode_attention = (
                envs.VLLM_V1_USE_PREFILL_DECODE_ATTENTION
            )

        if envs.is_set("VLLM_FLASH_ATTN_MAX_NUM_SPLITS_FOR_CUDA_GRAPH"):
            self.flash_attn_max_num_splits_for_cuda_graph = (
                envs.VLLM_FLASH_ATTN_MAX_NUM_SPLITS_FOR_CUDA_GRAPH
            )

        if envs.is_set("VLLM_USE_CUDNN_PREFILL"):
            self.use_cudnn_prefill = envs.VLLM_USE_CUDNN_PREFILL

        if envs.is_set("VLLM_USE_TRTLLM_RAGGED_DEEPSEEK_PREFILL"):
            self.use_trtllm_ragged_deepseek_prefill = (
                envs.VLLM_USE_TRTLLM_RAGGED_DEEPSEEK_PREFILL
            )

        if envs.is_set("VLLM_USE_TRTLLM_ATTENTION"):
            self.use_trtllm_attention = envs.VLLM_USE_TRTLLM_ATTENTION

        if envs.is_set("VLLM_DISABLE_FLASHINFER_PREFILL"):
            self.disable_flashinfer_prefill = envs.VLLM_DISABLE_FLASHINFER_PREFILL

        if envs.is_set("VLLM_FLASHINFER_DISABLE_Q_QUANTIZATION"):
            self.flashinfer_disable_q_quantization = (
                envs.VLLM_FLASHINFER_DISABLE_Q_QUANTIZATION
            )
