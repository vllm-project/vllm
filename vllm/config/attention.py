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

    zoomkv_quest_chunk: int = 16
    """Child chunk size; must match the KV cache block size."""

    zoomkv_quest_large_chunk: int = 256
    """Parent chunk size for hierarchical Quest filtering."""

    zoomkv_quest_large_ratio: float = 0.8
    """Fraction of parent chunks retained by Stage-1 Quest."""

    zoomkv_quest_small_ratio: float = 0.5
    """Fraction of child chunks retained inside selected parents."""

    zoomkv_dense_ratio: float = 0.4
    """Fraction of candidate chunks treated as dense in CDS rerank."""

    zoomkv_dense_topk: int = 16
    """Per-chunk local token budget for dense chunks."""

    zoomkv_sparse_topk: int = 8
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

    zoomkv_per_query_head: bool = False
    """Retrieve with the strongest (max-L2-norm) query head per KV group
    instead of the GQA group mean.  The original ZoomKV implementation uses
    the group mean; measured Top-K recall is significantly higher with the
    mean, so this defaults to False."""

    def _validate_zoomkv(self) -> None:
        if self.backend != AttentionBackendEnum.ZOOMKV:
            return
        if self.zoomkv_quest_chunk != 16:
            raise ValueError("zoomkv_quest_chunk must be 16 in the first release")
        if self.zoomkv_sink_size < 0 or self.zoomkv_local_size < 0:
            raise ValueError("zoomkv sink/local sizes must be non-negative")
        if self.zoomkv_final_topk <= 0:
            raise ValueError("zoomkv_final_topk must be positive")
        if self.zoomkv_full_attention_threshold < 0:
            raise ValueError("zoomkv_full_attention_threshold must be non-negative")
        if self.zoomkv_sink_size % self.zoomkv_quest_chunk != 0:
            raise ValueError("zoomkv_sink_size must be divisible by quest_chunk")
        if self.zoomkv_local_size % self.zoomkv_quest_chunk != 0:
            raise ValueError("zoomkv_local_size must be divisible by quest_chunk")
        if self.zoomkv_quest_large_chunk % self.zoomkv_quest_chunk != 0:
            raise ValueError(
                "zoomkv_quest_large_chunk must be divisible by quest_chunk"
            )
        if not (0.0 < self.zoomkv_quest_large_ratio <= 1.0):
            raise ValueError("zoomkv_quest_large_ratio must be in (0, 1]")
        if not (0.0 < self.zoomkv_quest_small_ratio <= 1.0):
            raise ValueError("zoomkv_quest_small_ratio must be in (0, 1]")
        if not (0.0 < self.zoomkv_dense_ratio <= 1.0):
            raise ValueError("zoomkv_dense_ratio must be in (0, 1]")
        if not (1 <= self.zoomkv_dense_topk <= self.zoomkv_quest_chunk):
            raise ValueError("zoomkv_dense_topk must be in [1, zoomkv_quest_chunk]")
        if not (1 <= self.zoomkv_sparse_topk <= self.zoomkv_quest_chunk):
            raise ValueError("zoomkv_sparse_topk must be in [1, zoomkv_quest_chunk]")
        if self.zoomkv_enable_offload and self.zoomkv_dense_fallback:
            raise ValueError(
                "zoomkv_enable_offload cannot be combined with zoomkv_dense_fallback"
            )
        if self.zoomkv_cpu_bytes_per_rank <= 0:
            raise ValueError("zoomkv_cpu_bytes_per_rank must be positive")

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
