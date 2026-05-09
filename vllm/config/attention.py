# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Any, Literal

from pydantic import field_validator

from vllm.config.utils import config
from vllm.logger import init_logger
from vllm.v1.attention.backends.mla.prefill.registry import MLAPrefillBackendEnum
from vllm.v1.attention.backends.registry import AttentionBackendEnum

logger = init_logger(__name__)


@config
class AttentionConfig:
    """Configuration for attention mechanisms in vLLM."""

    backend: AttentionBackendEnum | None = None
    """Attention backend to use. Use "auto" or None for automatic selection."""

    flash_attn_version: Literal[2, 3, 4] | None = None
    """Force vllm to use a specific flash-attention version (2, 3, or 4).
    Only valid when using the flash-attention backend."""

    use_prefill_decode_attention: bool = False
    """Use separate prefill and decode kernels for attention instead of
    the unified triton kernel."""

    flash_attn_max_num_splits_for_cuda_graph: int = 32
    """Flash Attention max number splits for cuda graph decode."""

    tq_max_kv_splits_for_cuda_graph: int = 32
    """TurboQuant max NUM_KV_SPLITS for cuda graph decode.
    Fixes the split count so grid dimensions are constant across captures,
    and buffers can be pre-allocated to avoid inflating the memory estimate."""

    use_cudnn_prefill: bool = False
    """Deprecated: cuDNN prefill backend has been removed."""

    use_trtllm_ragged_deepseek_prefill: bool = False
    """Whether to use TRTLLM ragged deepseek prefill."""

    use_trtllm_attention: bool | None = None
    """If set to True/False, use or don't use the TRTLLM attention backend
    in flashinfer. If None, auto-detect the attention backend in flashinfer."""

    disable_flashinfer_prefill: bool | None = None
    """Whether to disable flashinfer prefill."""

    disable_flashinfer_q_quantization: bool = False
    """If set, when using fp8 kv, do not quantize Q to fp8."""

    mla_prefill_backend: MLAPrefillBackendEnum | None = None
    """MLA prefill backend to use. If None, will be selected automatically.
    Valid options: FLASH_ATTN (FA3/FA4), FLASHINFER, TRTLLM_RAGGED.
    This option supersedes use_trtllm_ragged_deepseek_prefill
    and disable_flashinfer_prefill which are deprecated."""

    use_prefill_query_quantization: bool = False
    """If set, quantize query for attention in prefill."""

    use_fp4_indexer_cache: bool = False
    """If set, use fp4 indexer cache for dsv32 family model (not support yet)"""

    use_non_causal: bool = False
    """Whether to use non-causal (bidirectional) attention."""

    def compute_hash(self) -> str:
        """
        Provide a hash that uniquely identifies all the configs
        that affect the structure of the computation
        graph from input ids/embeddings to the final hidden states,
        excluding anything before input ids/embeddings and after
        the final hidden states.
        """
        from vllm.config.utils import get_hash_factors, hash_factors

        ignored_factors: set[str] = set()
        factors = get_hash_factors(self, ignored_factors)
        return hash_factors(factors)

    @field_validator("backend", mode="before")
    @classmethod
    def validate_backend_before(cls, value: Any) -> Any:
        """Enable parsing of the `backend` enum type from string.

        The special value "auto" is treated as None, which triggers
        automatic backend selection.
        """
        if isinstance(value, str):
            if value.lower() == "auto":
                return None
            return AttentionBackendEnum[value.upper()]
        return value

    @field_validator("mla_prefill_backend", mode="before")
    @classmethod
    def validate_mla_prefill_backend_before(cls, value: Any) -> Any:
        """Enable parsing of the `mla_prefill_backend` enum type from string."""
        if isinstance(value, str):
            return MLAPrefillBackendEnum[value.upper()]
        return value

    def __post_init__(self) -> None:
        self._migrate_deprecated_mla_prefill_flags()

    def _migrate_deprecated_mla_prefill_flags(self) -> None:
        """Migrate deprecated MLA prefill flags to mla_prefill_backend."""
        # If the new option is already set, it takes precedence
        if self.mla_prefill_backend is not None:
            return

        # Check for deprecated flags and migrate them.
        # Only the first flag encountered sets the backend.
        if self.use_cudnn_prefill:
            raise ValueError(
                "The cuDNN MLA prefill backend has been removed. "
                "Use --attention-config.mla_prefill_backend=FLASH_ATTN or "
                "FLASHINFER or TRTLLM_RAGGED instead."
            )

        if self.use_trtllm_ragged_deepseek_prefill:
            if self.mla_prefill_backend is None:
                self.mla_prefill_backend = MLAPrefillBackendEnum.TRTLLM_RAGGED
            logger.warning_once(
                "use_trtllm_ragged_deepseek_prefill is deprecated and "
                "will be removed in v0.22. Use "
                "--attention-config.mla_prefill_backend=TRTLLM_RAGGED "
                "instead."
            )

        if self.disable_flashinfer_prefill:
            if self.mla_prefill_backend is None:
                self.mla_prefill_backend = MLAPrefillBackendEnum.FLASH_ATTN
            logger.warning_once(
                "disable_flashinfer_prefill is deprecated and will be removed "
                "in v0.22. Use --attention-config.mla_prefill_backend="
                "FLASH_ATTN instead."
            )
