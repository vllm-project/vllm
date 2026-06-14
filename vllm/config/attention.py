# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Any, Literal

from pydantic import field_validator

from vllm.config.utils import config
from vllm.v1.attention.backends.mla.prefill.registry import MLAPrefillBackendEnum
from vllm.v1.attention.backends.registry import AttentionBackendEnum


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

    use_trtllm_attention: bool | None = None
    """If set to True/False, use or don't use the TRTLLM attention backend
    in flashinfer. If None, auto-detect the attention backend in flashinfer."""

    disable_flashinfer_q_quantization: bool = False
    """If set, when using fp8 kv, do not quantize Q to fp8."""

    mla_prefill_backend: MLAPrefillBackendEnum | None = None
    """MLA prefill backend to use. If None, will be selected automatically.
    Valid options: FLASH_ATTN (FA3/FA4), FLASHINFER, TRTLLM_RAGGED."""

    use_prefill_query_quantization: bool = False
    """If set, quantize query for attention in prefill."""

    use_fp4_indexer_cache: bool = False
    """If set, use fp4 indexer cache for dsv32 family model (not support yet)"""

    use_non_causal: bool = False
    """Whether to use non-causal (bidirectional) attention."""

    flex_attn_block_m: int | None = None
    """Triton kernel BLOCK_M tile size for flex attention.
    Must be a power of 2 >= 16. If None and VLLM_BATCH_INVARIANT=1,
    defaults to 16."""

    flex_attn_block_n: int | None = None
    """Triton kernel BLOCK_N tile size for flex attention.
    Must be a power of 2 >= 16. If None and VLLM_BATCH_INVARIANT=1,
    defaults to 16."""

    flex_attn_q_block_size: int | None = None
    """Logical Q block size for the flex attention block mask.
    Must be a power of 2 and divisible by flex_attn_block_m.
    If None, uses the default (16 on PyTorch >= 2.9, 128 otherwise)."""

    flex_attn_kv_block_size: int | None = None
    """Logical KV block size for the flex attention block mask.
    Must be a power of 2 and divisible by flex_attn_block_n.
    If None, uses the default (kv_cache_block_size on PyTorch >= 2.9,
    128 otherwise)."""

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

    @field_validator("flash_attn_max_num_splits_for_cuda_graph", mode="after")
    @classmethod
    def _check_flash_attn_max_num_splits(cls, v: int) -> int:
        if v <= 0:
            raise ValueError(
                f"flash_attn_max_num_splits_for_cuda_graph must be "
                f"positive (> 0), got {v}."
            )
        return v

    @field_validator("tq_max_kv_splits_for_cuda_graph", mode="after")
    @classmethod
    def _check_tq_max_kv_splits(cls, v: int) -> int:
        if v <= 0:
            raise ValueError(
                f"tq_max_kv_splits_for_cuda_graph must be positive (> 0), got {v}."
            )
        return v

    @field_validator("flex_attn_block_m", "flex_attn_block_n", mode="after")
    @classmethod
    def _check_flex_attn_block_size(cls, v: int | None) -> int | None:
        if v is not None:
            if v < 16:
                raise ValueError(f"flex_attn block size must be >= 16, got {v}.")
            # Check if power of 2
            if v & (v - 1) != 0:
                raise ValueError(f"flex_attn block size must be a power of 2, got {v}.")
        return v

    @field_validator("flex_attn_q_block_size", "flex_attn_kv_block_size", mode="after")
    @classmethod
    def _check_flex_attn_logical_block_size(cls, v: int | None) -> int | None:
        if v is not None and v & (v - 1) != 0:
            raise ValueError(
                f"flex_attn logical block size must be a power of 2, got {v}."
            )
        return v
