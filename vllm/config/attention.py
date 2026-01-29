# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Any, Literal

from pydantic import field_validator
from pydantic.dataclasses import dataclass

from vllm.config.utils import config
from vllm.logger import init_logger
from vllm.v1.attention.backends.registry import AttentionBackendEnum

logger = init_logger(__name__)


@config
@dataclass
class AttentionConfig:
    """Configuration for attention mechanisms in vLLM."""

    backend: AttentionBackendEnum | None = None
    """Attention backend to use. If None, will be selected automatically."""

    flash_attn_version: Literal[2, 3] | None = None
    """Force vllm to use a specific flash-attention version (2 or 3).
    Only valid when using the flash-attention backend."""

    use_prefill_decode_attention: bool = False
    """Use separate prefill and decode kernels for attention instead of
    the unified triton kernel."""

    flash_attn_max_num_splits_for_cuda_graph: int = 32
    """Flash Attention max number splits for cuda graph decode."""

    use_cudnn_prefill: bool = False
    """Whether to use cudnn prefill."""

    use_trtllm_ragged_deepseek_prefill: bool = True
    """Whether to use TRTLLM ragged deepseek prefill."""

    use_trtllm_attention: bool | None = None
    """If set to True/False, use or don't use the TRTLLM attention backend
    in flashinfer. If None, auto-detect the attention backend in flashinfer."""

    disable_flashinfer_prefill: bool = False
    """Whether to disable flashinfer prefill."""

    disable_flashinfer_q_quantization: bool = False
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

    def _set_from_env_if_set(self, field_name: str, env_var_name: str) -> None:
        """Set field from env var if set, with deprecation warning."""
        from vllm import envs

        if envs.is_set(env_var_name):
            value = getattr(envs, env_var_name)
            if field_name == "backend":
                value = self.validate_backend_before(value)
            setattr(self, field_name, value)
            logger.warning_once(
                "Using %s environment variable is deprecated and will be removed in "
                "v0.14.0 or v1.0.0, whichever is soonest. Please use "
                "--attention-config.%s command line argument or "
                "AttentionConfig(%s=...) config field instead.",
                env_var_name,
                field_name,
                field_name,
            )

    def __post_init__(self) -> None:
        self._set_from_env_if_set("backend", "VLLM_ATTENTION_BACKEND")
        self._set_from_env_if_set("flash_attn_version", "VLLM_FLASH_ATTN_VERSION")
        self._set_from_env_if_set(
            "use_prefill_decode_attention", "VLLM_V1_USE_PREFILL_DECODE_ATTENTION"
        )
        self._set_from_env_if_set(
            "flash_attn_max_num_splits_for_cuda_graph",
            "VLLM_FLASH_ATTN_MAX_NUM_SPLITS_FOR_CUDA_GRAPH",
        )
        self._set_from_env_if_set("use_cudnn_prefill", "VLLM_USE_CUDNN_PREFILL")
        self._set_from_env_if_set(
            "use_trtllm_ragged_deepseek_prefill",
            "VLLM_USE_TRTLLM_RAGGED_DEEPSEEK_PREFILL",
        )
        self._set_from_env_if_set("use_trtllm_attention", "VLLM_USE_TRTLLM_ATTENTION")
        self._set_from_env_if_set(
            "disable_flashinfer_prefill", "VLLM_DISABLE_FLASHINFER_PREFILL"
        )
        self._set_from_env_if_set(
            "disable_flashinfer_q_quantization",
            "VLLM_FLASHINFER_DISABLE_Q_QUANTIZATION",
        )
