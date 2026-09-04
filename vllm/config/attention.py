# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Any, Literal

from pydantic import field_validator

from vllm.config.utils import config
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

    use_cudnn_prefill: bool = False
    """Whether to use cudnn prefill."""

    use_trtllm_ragged_deepseek_prefill: bool = False
    """Whether to use TRTLLM ragged deepseek prefill."""

    use_trtllm_attention: bool | None = None
    """If set to True/False, use or don't use the TRTLLM attention backend
    in flashinfer. If None, auto-detect the attention backend in flashinfer."""

    disable_flashinfer_prefill: bool = True
    """Whether to disable flashinfer prefill."""

    disable_flashinfer_q_quantization: bool = False
    """If set, when using fp8 kv, do not quantize Q to fp8."""

    use_prefill_query_quantization: bool = False
    """If set, quantize query for attention in prefill."""

    use_fp4_indexer_cache: bool = False
    """If set, use fp4 indexer cache for dsv32 family model (not support yet)"""

    # ZoomKV sparse KV retrieval knobs (used when backend=ZOOMKV).
    # GPU-only by default. When zoomkv_enable_offload=True, completed K+V
    # blocks are mirrored to pinned CPU and cold GPU pages are released.
    zoomkv_sink_size: int = 64
    """Always-retained prefix tokens for ZoomKV decode attention."""

    zoomkv_local_size: int = 256
    """Always-retained recent local window for ZoomKV decode attention."""

    zoomkv_final_topk: int = 100
    """Final retrieved token budget for ZoomKV sparse decode."""

    zoomkv_chunk_size: int = 16
    """Retrieval chunk size; must match the KV cache block size."""

    zoomkv_chunk_candidates: int = 200
    """Number of child chunks retained by the q·chunk-mean Top-K."""

    zoomkv_dense_chunks: int = 60
    """Leading candidate chunks that retain the larger token budget."""

    zoomkv_dense_topk: int = 8
    """Per-chunk local token budget for dense chunks."""

    zoomkv_sparse_topk: int = 4
    """Per-chunk local token budget for sparse chunks."""

    zoomkv_full_attention_threshold: int = 2000
    """Use dense attention below this sequence length."""

    zoomkv_dense_fallback: bool = False
    """Force dense paged attention for numerical parity debugging."""

    zoomkv_strict_kernels: bool = False
    """Fail if production Quest/KIVI/TopK kernels are unavailable."""

    zoomkv_enable_offload: bool = False
    """Enable K+V CPU offload of completed retrieval-zone blocks."""

    zoomkv_cpu_bytes_per_rank: int = 8 * 1024**3
    """Pinned host K+V pool budget per worker rank."""

    zoomkv_offload_unit_tokens: int = 64
    """Logical D2H granularity. Must be a multiple of chunk_size (16).

    vLLM still pages at 16 tokens so Quest children, slot_mapping, and hybrid
    GDN layers stay aligned. Offload waits until a 64-token retrieval-zone
    unit is complete (4 child pages) before the async D2H, which matches
    sink_size and avoids bouncing the in-flight write page to CPU.
    """

    def _validate_zoomkv(self) -> None:
        if self.backend != AttentionBackendEnum.ZOOMKV:
            return
        if self.zoomkv_chunk_size != 16:
            raise ValueError("zoomkv_chunk_size must be 16")
        if self.zoomkv_sink_size < 0 or self.zoomkv_local_size < 0:
            raise ValueError("zoomkv sink/local sizes must be non-negative")
        if self.zoomkv_final_topk <= 0:
            raise ValueError("zoomkv_final_topk must be positive")
        if self.zoomkv_full_attention_threshold < 0:
            raise ValueError("zoomkv_full_attention_threshold must be non-negative")
        if self.zoomkv_sink_size % self.zoomkv_chunk_size != 0:
            raise ValueError("zoomkv_sink_size must be divisible by zoomkv_chunk_size")
        if self.zoomkv_local_size % self.zoomkv_chunk_size != 0:
            raise ValueError("zoomkv_local_size must be divisible by zoomkv_chunk_size")
        if not (1 <= self.zoomkv_dense_topk <= self.zoomkv_chunk_size):
            raise ValueError("zoomkv_dense_topk must be in [1, zoomkv_chunk_size]")
        if not (1 <= self.zoomkv_sparse_topk <= self.zoomkv_chunk_size):
            raise ValueError("zoomkv_sparse_topk must be in [1, zoomkv_chunk_size]")
        if self.zoomkv_chunk_candidates <= 0:
            raise ValueError("zoomkv_chunk_candidates must be positive")
        if not (
            0
            < self.zoomkv_dense_chunks
            <= self.zoomkv_chunk_candidates
        ):
            raise ValueError(
                "zoomkv_dense_chunks must be in (0, zoomkv_chunk_candidates]"
            )
        if self.zoomkv_enable_offload and self.zoomkv_dense_fallback:
            raise ValueError("zoomkv_enable_offload cannot be combined with zoomkv_dense_fallback")
        if self.zoomkv_cpu_bytes_per_rank <= 0:
            raise ValueError("zoomkv_cpu_bytes_per_rank must be positive")
        if self.zoomkv_offload_unit_tokens < self.zoomkv_chunk_size:
            raise ValueError("zoomkv_offload_unit_tokens must be >= zoomkv_chunk_size")
        if self.zoomkv_offload_unit_tokens % self.zoomkv_chunk_size != 0:
            raise ValueError(
                "zoomkv_offload_unit_tokens must be divisible by zoomkv_chunk_size"
            )

    def __post_init__(self) -> None:
        self._validate_zoomkv()

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
