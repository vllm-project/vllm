# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import copy
from collections import Counter
from dataclasses import dataclass, fields, replace
from enum import Enum, IntEnum
from math import prod
from typing import TYPE_CHECKING

import torch
from typing_extensions import Self

from vllm.logger import init_logger
from vllm.utils.math_utils import cdiv, round_up
from vllm.utils.torch_utils import get_dtype_size, nvfp4_kv_cache_full_dim
from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum
from vllm.v1.kv_cache_spec_registry import KVCacheSpecRegistry

if TYPE_CHECKING:
    from vllm.config import ModelConfig, VllmConfig

logger = init_logger(__name__)

# Dimension indices for (B, H, N, C) descriptor meta tensors.
DIM_H = 1
DIM_C = 3

# ---------------------------------------------------------------------------
# KV cache quantization mode
# ---------------------------------------------------------------------------


class KVQuantMode(IntEnum):
    """KV cache quantization mode.

    Used by attention backends and kernels to dispatch quantization logic
    without string matching on ``kv_cache_dtype``.
    """

    NONE = 0
    FP8_PER_TENSOR = 1  # per-tensor scales (current fp8 path)
    INT8_PER_TOKEN_HEAD = 2  # per-token-head dynamic scales for int8
    FP8_PER_TOKEN_HEAD = 3  # per-token-head dynamic scales for fp8
    INT4_PER_TOKEN_HEAD = 4  # packed 2×int4/byte, RHT + asymmetric zp
    NVFP4 = 5  # packed fp4 data + fp8 block scales

    @property
    def is_per_token_head(self) -> bool:
        """True for any per-token-head quantization mode."""
        return self in (
            KVQuantMode.INT8_PER_TOKEN_HEAD,
            KVQuantMode.FP8_PER_TOKEN_HEAD,
            KVQuantMode.INT4_PER_TOKEN_HEAD,
        )

    @property
    def is_nvfp4(self) -> bool:
        """True for NVFP4 packed quantization mode."""
        return self == KVQuantMode.NVFP4


def get_kv_quant_mode(kv_cache_dtype: str) -> KVQuantMode:
    """Map a ``kv_cache_dtype`` string to a :class:`KVQuantMode`."""
    if kv_cache_dtype == "int4_per_token_head":
        return KVQuantMode.INT4_PER_TOKEN_HEAD
    if kv_cache_dtype == "int8_per_token_head":
        return KVQuantMode.INT8_PER_TOKEN_HEAD
    if kv_cache_dtype == "fp8_per_token_head":
        return KVQuantMode.FP8_PER_TOKEN_HEAD
    if kv_cache_dtype == "nvfp4":
        return KVQuantMode.NVFP4
    if isinstance(kv_cache_dtype, str) and kv_cache_dtype.startswith("fp8"):
        return KVQuantMode.FP8_PER_TENSOR
    return KVQuantMode.NONE


def is_quantized_kv_cache(kv_cache_dtype: str) -> bool:
    return get_kv_quant_mode(kv_cache_dtype) != KVQuantMode.NONE


def kv_cache_uses_per_token_head_scales(kv_cache_dtype: str) -> bool:
    """Return True if *kv_cache_dtype* needs per-token-head scales."""
    return get_kv_quant_mode(kv_cache_dtype).is_per_token_head


class KVCacheSpecKind(str, Enum):
    FULL_ATTENTION = "full_attention"
    MLA_ATTENTION = "mla_attention"
    SLIDING_WINDOW = "sliding_window"
    SLIDING_WINDOW_MLA = "sliding_window_mla"
    MAMBA = "mamba"
    CHUNKED_LOCAL_ATTENTION = "chunked_local_attention"
    SINK_FULL_ATTENTION = "sink_full_attention"
    ENCODER_ONLY_ATTENTION = "encoder_only_attention"
    CROSS_ATTENTION = "cross_attention"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class KVCacheSpec:
    """
    A base class for specifying the KV cache format of one layer.
    """

    # number of tokens in a block
    block_size: int

    @property
    def page_size_bytes(self) -> int:
        """
        The size of a page with `block_size` tokens in bytes.

        Returns:
            The page size
        """
        raise NotImplementedError

    @property
    def storage_block_size(self) -> int:
        return self.block_size

    @property
    def tokens_per_state(self) -> int:
        """Tokens consumed per state slot.

        -1 = infinite/recurrent (Mamba), 1 = standard, N = compressed (MLA).
        """
        raise NotImplementedError

    def compute_transfer_shape(
        self,
        region_content_bytes: int,
        block_size: int,
        virtually_split: bool,
    ) -> tuple[int, int, int]:
        """Compute (num_heads, N, C) per-block shape for a descriptor meta tensor.

        The full meta tensor shape is (B, num_heads, N, C) where B = num_blocks
        is provided by the caller.
        Subclasses must override.
        """
        raise NotImplementedError

    def slice_for_tp_transfer(
        self,
        tensor: torch.Tensor,
        my_tp: int,
        my_rank: int,
        other_tp: int,
        other_rank: int,
        model_config: ModelConfig,
        virtually_split: bool = False,
    ) -> list[torch.Tensor]:
        """Decompose and narrow a region meta tensor for transfer.

        This is the single dispatch point for all descriptor construction.
        It handles both:
          1. Component decomposition (K/V split, Mamba sub-projections)
          2. TP head/channel narrowing

        For local descriptors, call with my_tp==other_tp, my_rank==other_rank.
        For remote descriptors, call with the remote's TP params.

        Args:
            tensor: (B, H, N, C) meta tensor for the full region.
            virtually_split: True when K/V are co-located in same region
                (FlashInfer layout) and need separate descriptor streams.

        Returns:
            List of narrowed/decomposed meta tensors. Each becomes one
            descriptor stream in the NIXL transfer.
        """
        if virtually_split:
            B, H, N, C = tensor.shape
            v_offset = H * N * C
            k_view = tensor
            v_view = torch.as_strided(
                tensor,
                (B, H, N, C),
                stride=tensor.stride(),
                storage_offset=tensor.storage_offset() + v_offset,
            )
            return [k_view, v_view]
        return [tensor]

    def max_memory_usage_bytes(self, vllm_config: VllmConfig) -> int:
        """
        The maximum possible memory usage of this KV cache in bytes.

        Returns:
            The KV cache size in bytes
        """
        raise NotImplementedError

    def copy_with_new_block_size(self, block_size: int) -> Self:
        """
        Create a new KVCacheSpec from self but replacing the block size.
        """
        return replace(self, block_size=block_size)

    @classmethod
    def merge(cls, specs: list[Self]) -> Self:
        """
        Merge a list of KVCacheSpec objects into a single KVCacheSpec object.
        """
        assert all(spec == specs[0] for spec in specs[1:]), (
            "All layers in the same KV cache group must be the same."
        )
        return copy.deepcopy(specs[0])

    def is_uniform_with_collection(
        self, kv_cache_specs: dict[str, KVCacheSpec]
    ) -> bool:
        """
        Whether this KVCacheSpec is uniform with all specs of all layers.
        """
        uniform_type_base_spec = KVCacheSpecRegistry.get_uniform_type_base_spec(self)
        assert uniform_type_base_spec is not None, (
            f"Unsupported KV cache spec type: {type(self)}. "
            "Please register it using @register_kv_cache_spec decorator."
        )
        return all(
            isinstance(spec, uniform_type_base_spec) for spec in kv_cache_specs.values()
        )


@dataclass(frozen=True, kw_only=True)
class AttentionSpec(KVCacheSpec):
    num_kv_heads: int
    head_size: int
    dtype: torch.dtype
    kv_quant_mode: KVQuantMode = KVQuantMode.NONE
    page_size_padded: int | None = None
    indexes_kv_by_block_stride: bool = False

    @property
    def tokens_per_state(self) -> int:
        return 1

    def compute_transfer_shape(
        self,
        region_content_bytes: int,
        block_size: int,
        virtually_split: bool,
    ) -> tuple[int, int, int]:
        elem = get_dtype_size(self.dtype)
        N = block_size
        if self.kv_quant_mode.is_nvfp4:
            head_dim = nvfp4_kv_cache_full_dim(self.head_size)
        elif self.kv_quant_mode == KVQuantMode.INT4_PER_TOKEN_HEAD:
            head_dim = self.head_size // 2
        else:
            head_dim = self.head_size

        if virtually_split:
            H = region_content_bytes // (2 * N * head_dim * elem)
        else:
            H = region_content_bytes // (N * head_dim * elem)
        return (H, N, head_dim)

    def slice_for_tp_transfer(
        self,
        tensor: torch.Tensor,
        my_tp: int,
        my_rank: int,
        other_tp: int,
        other_rank: int,
        model_config: ModelConfig,
        virtually_split: bool = False,
    ) -> list[torch.Tensor]:
        # tensor shape: (B, H, N, C) where H = heads, C = head_size

        # Split K/V if virtually_split. FlashInfer layout stores K and V
        # as two contiguous blocks [K_all | V_all], so V starts at offset
        # H * N * C from the block base.
        if virtually_split:
            B, H, N, C = tensor.shape
            v_offset = H * N * C
            k_view = tensor
            v_view = torch.as_strided(
                tensor,
                (B, H, N, C),
                stride=tensor.stride(),
                storage_offset=tensor.storage_offset() + v_offset,
            )
            parts = [k_view, v_view]
        else:
            parts = [tensor]

        if my_tp == other_tp and my_rank == other_rank:
            return parts

        # Narrow heads for TP overlap.
        total_kv = model_config.get_total_num_kv_heads()

        if total_kv >= my_tp:
            my_start = my_rank * total_kv // my_tp
            my_end = (my_rank + 1) * total_kv // my_tp
        else:
            my_start = my_rank * total_kv // my_tp
            my_end = my_start + 1

        if total_kv >= other_tp:
            other_start = other_rank * total_kv // other_tp
            other_end = (other_rank + 1) * total_kv // other_tp
        else:
            other_start = other_rank * total_kv // other_tp
            other_end = other_start + 1

        overlap_start = max(my_start, other_start)
        overlap_end = min(my_end, other_end)
        assert overlap_start < overlap_end, (
            f"No head overlap between local rank {my_rank}/{my_tp} "
            f"[{my_start},{my_end}) and remote rank {other_rank}/{other_tp} "
            f"[{other_start},{other_end}) with total_kv={total_kv}."
        )

        h_start = overlap_start - other_start
        h_len = overlap_end - overlap_start

        return [part.narrow(DIM_H, h_start, h_len) for part in parts]

    @property
    def page_size_bytes(self) -> int:
        real_page_size = self.real_page_size_bytes
        # Per-token-head scales are stored in separate tensors managed
        # by the attention backend, but the memory is carved from the
        # raw KV cache allocation so it must be budgeted here.
        if self.kv_quant_mode.is_per_token_head:
            real_page_size += (
                2 * self.block_size * self.num_kv_heads * get_dtype_size(torch.float32)
            )
        if self.page_size_padded is not None:
            assert self.page_size_padded >= real_page_size
            return self.page_size_padded
        return real_page_size

    @property
    def real_page_size_bytes(self) -> int:
        if self.kv_quant_mode.is_nvfp4:
            # Packed layout: fp4 data + fp8 block scales per head.
            head_dim = nvfp4_kv_cache_full_dim(self.head_size)
        elif self.kv_quant_mode == KVQuantMode.INT4_PER_TOKEN_HEAD:
            head_dim = self.head_size // 2
        else:
            head_dim = self.head_size
        return (
            2
            * self.block_size
            * self.num_kv_heads
            * head_dim
            * get_dtype_size(self.dtype)
        )


@dataclass(frozen=True, kw_only=True)
class FullAttentionSpec(AttentionSpec):
    """
    When hybrid allocator is disabled and the model contains both full
    attention layers and sliding window attention layers, sliding
    window attention are regarded as full attention in KV cache manager
    (blocks are allocated for all tokens), while computed as sliding window
    attention in model runner.
    In this case, we use FullAttentionSpec and record the sliding window size.
    """

    head_size_v: int = None  # type: ignore[assignment]

    sliding_window: int | None = None
    """
    Default to None for not using sliding window attention.
    """
    attention_chunk_size: int | None = None

    non_causal: bool = False
    """
    Whether the layer attends non-causally (e.g. Prefix LM). Carried on the
    spec so the engine core, which collects specs from all workers before the
    scheduler is built, can adjust scheduling policy (chunked prefill / prefix
    caching) regardless of tensor-parallel layout. It does not affect the KV
    cache layout itself.
    """

    def __post_init__(self):
        if self.head_size_v is None:
            object.__setattr__(self, "head_size_v", self.head_size)

    def max_memory_usage_bytes(self, vllm_config: VllmConfig) -> int:
        max_model_len = vllm_config.model_config.max_model_len
        dcp_world_size = vllm_config.parallel_config.decode_context_parallel_size
        pcp_world_size = vllm_config.parallel_config.prefill_context_parallel_size
        # Note(hc): each dcp rank only need save
        # (max_model_len//dcp_world_size) tokens locally.
        if dcp_world_size * pcp_world_size > 1:
            max_model_len = cdiv(max_model_len, dcp_world_size * pcp_world_size)
        return cdiv(max_model_len, self.block_size) * self.page_size_bytes

    @classmethod
    def merge_window_sizes(cls, window_sizes: set[int]) -> int | None:
        if len(window_sizes) == 0:
            return None
        elif len(window_sizes) == 1:
            return window_sizes.pop()
        else:
            raise ValueError(
                "All attention layers in the same KV cache group must have the "
                "same window size."
            )

    @classmethod
    def merge(cls, specs: list[Self]) -> Self:
        """
        Merge a list of FullAttentionSpec objects into a single
        FullAttentionSpec object.
        """
        assert all(isinstance(spec, FullAttentionSpec) for spec in specs), (
            "All attention layers in the same KV cache group must be FullAttentionSpec."
        )

        sliding_window = set(
            spec.sliding_window for spec in specs if spec.sliding_window is not None
        )
        attention_chunk_size = set(
            spec.attention_chunk_size
            for spec in specs
            if spec.attention_chunk_size is not None
        )
        assert not any(isinstance(spec, MLAAttentionSpec) for spec in specs), (
            "MLAAttentionSpec should be merged in MLAAttentionSpec.merge"
        )
        merged_spec = cls(
            block_size=specs[0].block_size,
            num_kv_heads=specs[0].num_kv_heads,
            head_size=specs[0].head_size,
            head_size_v=specs[0].head_size_v,
            dtype=specs[0].dtype,
            kv_quant_mode=specs[0].kv_quant_mode,
            page_size_padded=specs[0].page_size_padded,
            indexes_kv_by_block_stride=specs[0].indexes_kv_by_block_stride,
            sliding_window=cls.merge_window_sizes(sliding_window),
            attention_chunk_size=cls.merge_window_sizes(attention_chunk_size),
            # If any layer in the group is non-causal, treat the group as
            # non-causal so the engine core disables incompatible scheduling.
            non_causal=any(spec.non_causal for spec in specs),
        )
        for spec in specs:
            for f in fields(AttentionSpec):
                assert getattr(spec, f.name) == getattr(merged_spec, f.name), (
                    "All attention layers in the same KV cache group must have "
                    "the same attention spec."
                )
        assert (merged_spec.sliding_window is not None) + (
            merged_spec.attention_chunk_size is not None
        ) <= 1, (
            "Model with both sliding window layers and chunked local attention "
            "layers is not supported."
        )
        return merged_spec

    @property
    def real_page_size_bytes(self) -> int:
        if self.kv_quant_mode.is_nvfp4:
            # Packed layout per head: fp4 data + fp8 block scales.
            # fp4 data: head_size//2 bytes (2 fp4 values per byte)
            # fp8 block scale: head_size//16 bytes (1 scale per 16 elements)
            last_dim = nvfp4_kv_cache_full_dim(
                self.head_size
            ) + nvfp4_kv_cache_full_dim(self.head_size_v)
        elif self.kv_quant_mode == KVQuantMode.INT4_PER_TOKEN_HEAD:
            last_dim = self.head_size // 2 + self.head_size_v // 2
        else:
            last_dim = self.head_size + self.head_size_v
        return (
            self.block_size * self.num_kv_heads * last_dim * get_dtype_size(self.dtype)
        )


def _apply_alignment_padding(spec: MLAAttentionSpec | SlidingWindowMLASpec):
    if spec.alignment is None:
        return
    actual_page_size = spec.real_page_size_bytes
    padded_page_size = round_up(actual_page_size, spec.alignment)
    if padded_page_size != actual_page_size:
        object.__setattr__(spec, "page_size_padded", padded_page_size)


@dataclass(frozen=True, kw_only=True)
class TQFullAttentionSpec(FullAttentionSpec):
    """FullAttentionSpec with TQ-aware page size.

    Python equivalent of the C++ TQ4FullAttentionSpec. Overrides
    real_page_size_bytes to use TQ slot bytes instead of the raw
    head_size * dtype formula.
    """

    tq_slot_size: int = 0

    @property
    def real_page_size_bytes(self) -> int:
        if self.tq_slot_size > 0:
            return self.block_size * self.num_kv_heads * self.tq_slot_size
        return super().real_page_size_bytes

    @classmethod
    def merge(cls, specs: list[Self]) -> Self:
        merged = super().merge(specs)
        assert all(s.tq_slot_size == specs[0].tq_slot_size for s in specs), (
            "All TQ layers in the same KV cache group must use the same tq_slot_size."
        )
        return replace(merged, tq_slot_size=specs[0].tq_slot_size)


@dataclass(frozen=True, kw_only=True)
class MLAAttentionSpec(FullAttentionSpec):
    # TODO(Lucas/Chen): less hacky way to do this
    cache_dtype_str: str | None = None
    # DeepseekV4 only fields. Non-DeepseekV4 MLA models leave these at defaults.
    alignment: int | None = None  # Default to None for no padding.
    compress_ratio: int = 1  # Default to 1 for no compression.
    model_version: str | None = None

    def __post_init__(self):
        super().__post_init__()
        _apply_alignment_padding(self)

    @property
    def tokens_per_state(self) -> int:
        return self.compress_ratio

    def slice_for_tp_transfer(
        self,
        tensor: torch.Tensor,
        my_tp: int,
        my_rank: int,
        other_tp: int,
        other_rank: int,
        model_config: ModelConfig,
        virtually_split: bool = False,
    ) -> list[torch.Tensor]:
        return [tensor]

    @property
    def storage_block_size(self) -> int:
        return self.block_size // self.compress_ratio

    @property
    def real_page_size_bytes(self) -> int:
        if self.cache_dtype_str == "fp8_ds_mla":
            if self.model_version == "deepseek_v4":
                # DeepseekV4: 448B NoPE + 128B RoPE + 8B fp8 scale = 584B per token.
                # head_size stays semantic (512); bytes are determined here.
                return self.storage_block_size * 584
            # V3.2 main MLA: 656-byte custom layout (kv_lora_rank=512 +
            # qk_rope_head_dim=64, head_size=576). See flashmla_sparse.py.
            return self.block_size * 656
        if self.kv_quant_mode == KVQuantMode.INT4_PER_TOKEN_HEAD:
            head_dim = self.head_size // 2
        else:
            head_dim = self.head_size
        return (
            self.storage_block_size
            * self.num_kv_heads
            * head_dim
            * get_dtype_size(self.dtype)
        )

    @classmethod
    def merge(cls, specs: list[Self]) -> Self:
        assert all(isinstance(spec, MLAAttentionSpec) for spec in specs), (
            "All attention layers in the same KV cache group must be MLAAttentionSpec."
        )
        cache_dtype_str_set = set(spec.cache_dtype_str for spec in specs)
        compress_ratio_set = set(spec.compress_ratio for spec in specs)
        model_version_set = set(spec.model_version for spec in specs)
        block_stride_set = set(spec.indexes_kv_by_block_stride for spec in specs)
        assert (
            len(cache_dtype_str_set) == 1
            and len(compress_ratio_set) == 1
            and len(model_version_set) == 1
            and len(block_stride_set) == 1
        ), (
            "All attention layers in the same KV cache group must use the same "
            "quantization method, compress ratio, model version, and KV block "
            "stride indexing."
        )
        return cls(
            block_size=specs[0].block_size,
            num_kv_heads=specs[0].num_kv_heads,
            head_size=specs[0].head_size,
            dtype=specs[0].dtype,
            kv_quant_mode=specs[0].kv_quant_mode,
            page_size_padded=specs[0].page_size_padded,
            indexes_kv_by_block_stride=block_stride_set.pop(),
            cache_dtype_str=cache_dtype_str_set.pop(),
            compress_ratio=compress_ratio_set.pop(),
            model_version=model_version_set.pop(),
        )


@dataclass(frozen=True, kw_only=True)
class HiddenStateCacheSpec(MLAAttentionSpec):
    """Marker for hidden-state cache layers used by extract_hidden_states."""

    pass


@dataclass(frozen=True, kw_only=True)
class RSWASpec(FullAttentionSpec):
    """KV cache spec for Reference Sliding Window Attention (R-SWA).

    Prefill (image + text prompt) tokens are always globally visible.
    Only the last ``rswa_window`` generated tokens are kept in the KV cache;
    gap blocks (between the prefill tail and the current decode window) are
    evicted during each decode step to bound memory at
    O(prefix_blocks + window_blocks).
    """

    rswa_window: int

    @classmethod
    def merge(cls, specs: list[RSWASpec]) -> RSWASpec:
        assert all(isinstance(spec, RSWASpec) for spec in specs), (
            "All attention layers in the same KV cache group must be RSWASpec."
        )
        rswa_windows = {spec.rswa_window for spec in specs}
        assert len(rswa_windows) == 1, (
            f"All R-SWA layers must share the same rswa_window, got {rswa_windows}"
        )
        # Delegate common field merging to the parent, then reattach rswa_window.
        base = FullAttentionSpec.merge(specs)  # type: ignore[arg-type]
        return cls(
            block_size=base.block_size,
            num_kv_heads=base.num_kv_heads,
            head_size=base.head_size,
            head_size_v=base.head_size_v,
            dtype=base.dtype,
            kv_quant_mode=base.kv_quant_mode,
            page_size_padded=base.page_size_padded,
            indexes_kv_by_block_stride=base.indexes_kv_by_block_stride,
            sliding_window=base.sliding_window,
            attention_chunk_size=base.attention_chunk_size,
            non_causal=base.non_causal,
            rswa_window=rswa_windows.pop(),
        )


@dataclass(frozen=True, kw_only=True)
class ChunkedLocalAttentionSpec(AttentionSpec):
    attention_chunk_size: int

    def max_admission_blocks_per_request(
        self, max_num_batched_tokens: int, max_model_len: int
    ) -> int:
        """Per-request admission cap, in blocks.

        Single source of truth for both startup pool sizing
        (`max_memory_usage_bytes`) and the runtime admission gate, so requests
        admitted by startup can also be admitted at runtime.
        """
        # During chunked prefill, we hold KV for at most one chunk window.
        num_tokens = min(
            self.attention_chunk_size + max_num_batched_tokens, max_model_len
        )
        return cdiv(num_tokens, self.block_size)

    def max_memory_usage_bytes(self, vllm_config: VllmConfig) -> int:
        max_model_len = vllm_config.model_config.max_model_len
        max_num_batched_tokens = vllm_config.scheduler_config.max_num_batched_tokens
        max_blocks = self.max_admission_blocks_per_request(
            max_num_batched_tokens=max_num_batched_tokens, max_model_len=max_model_len
        )
        return max_blocks * self.page_size_bytes

    def is_uniform_with_collection(
        self, kv_cache_specs: dict[str, KVCacheSpec]
    ) -> bool:
        return all(
            isinstance(spec, ChunkedLocalAttentionSpec)
            and spec.attention_chunk_size == self.attention_chunk_size
            for spec in kv_cache_specs.values()
        )


@dataclass(frozen=True, kw_only=True)
class SlidingWindowSpec(AttentionSpec):
    sliding_window: int
    head_size_v: int = None  # type: ignore[assignment]

    def __post_init__(self):
        if self.head_size_v is None:
            object.__setattr__(self, "head_size_v", self.head_size)

    @property
    def real_page_size_bytes(self) -> int:
        # Mirror ``FullAttentionSpec.real_page_size_bytes`` for NVFP4 KV cache.
        if self.kv_quant_mode.is_nvfp4:
            last_dim = nvfp4_kv_cache_full_dim(
                self.head_size
            ) + nvfp4_kv_cache_full_dim(self.head_size_v)
            return (
                self.block_size
                * self.num_kv_heads
                * last_dim
                * get_dtype_size(self.dtype)
            )
        return (
            self.block_size
            * self.num_kv_heads
            * (self.head_size + self.head_size_v)
            * get_dtype_size(self.dtype)
        )

    def max_admission_blocks_per_request(
        self, max_num_batched_tokens: int, max_model_len: int
    ) -> int:
        """Per-request admission cap, in blocks.

        Single source of truth for both startup pool sizing
        (`max_memory_usage_bytes`) and the runtime admission gate. Per-request
        real-held blocks plateau at this bound because
        `SlidingWindowManager.remove_skipped_blocks` runs from `allocate_slots`
        before each chunk's `get_num_blocks_to_allocate`.
        """
        # During chunked prefill, we hold KV for the last `sliding_window-1`
        # computed tokens plus the newly scheduled tokens, and never more
        # than `max_model_len`.
        num_tokens = min(
            self.sliding_window - 1 + max_num_batched_tokens, max_model_len
        )
        # +1 because the sliding window may not start from the beginning of
        # the block. E.g. block size 4 and num_token 4 needs two blocks
        # [XXCD][EF] to store the 6-token window [CDEF].
        return cdiv(num_tokens, self.block_size) + 1

    def max_memory_usage_bytes(self, vllm_config: VllmConfig) -> int:
        assert vllm_config.parallel_config.decode_context_parallel_size == 1, (
            "DCP not support sliding window."
        )
        max_model_len = vllm_config.model_config.max_model_len
        max_num_batched_tokens = vllm_config.scheduler_config.max_num_batched_tokens
        max_blocks = self.max_admission_blocks_per_request(
            max_num_batched_tokens=max_num_batched_tokens, max_model_len=max_model_len
        )
        return max_blocks * self.page_size_bytes

    def is_uniform_with_collection(
        self, kv_cache_specs: dict[str, KVCacheSpec]
    ) -> bool:
        return all(
            isinstance(spec, SlidingWindowSpec)
            and spec.sliding_window == self.sliding_window
            for spec in kv_cache_specs.values()
        )


@dataclass(frozen=True, kw_only=True)
class SlidingWindowMLASpec(SlidingWindowSpec):
    """Sliding window attention with MLA cache format."""

    cache_dtype_str: str | None = None
    # DeepseekV4-only: see MLAAttentionSpec.model_version.
    alignment: int | None = None  # Default to None for no padding.
    compress_ratio: int = 1
    model_version: str | None = None

    def __post_init__(self):
        _apply_alignment_padding(self)

    @property
    def tokens_per_state(self) -> int:
        return self.compress_ratio

    def slice_for_tp_transfer(
        self,
        tensor: torch.Tensor,
        my_tp: int,
        my_rank: int,
        other_tp: int,
        other_rank: int,
        model_config: ModelConfig,
        virtually_split: bool = False,
    ) -> list[torch.Tensor]:
        return [tensor]

    @property
    def storage_block_size(self) -> int:
        return self.block_size // self.compress_ratio

    @property
    def real_page_size_bytes(self) -> int:
        if self.model_version == "deepseek_v4" and self.cache_dtype_str == "fp8_ds_mla":
            # DeepseekV4 FlashMLA: 448B NoPE + 128B RoPE + 8B fp8 scale = 584B
            # per token. FlashInfer's contiguous bf16/fp8 cache falls through to
            # the element-size formula below.
            return self.storage_block_size * 584
        assert self.model_version in (None, "deepseek_v4"), (
            f"Unsupported model version: {self.model_version}"
        )
        return (
            self.storage_block_size
            * self.num_kv_heads
            * self.head_size
            * get_dtype_size(self.dtype)
        )

    @classmethod
    def merge(cls, specs: list[Self]) -> Self:
        assert all(isinstance(spec, SlidingWindowMLASpec) for spec in specs), (
            "All attention layers in the same KV cache group must be "
            "SlidingWindowMLASpec."
        )
        cache_dtype_str_set = set(spec.cache_dtype_str for spec in specs)
        compress_ratio_set = set(spec.compress_ratio for spec in specs)
        model_version_set = set(spec.model_version for spec in specs)
        sliding_window_set = set(spec.sliding_window for spec in specs)
        block_stride_set = set(spec.indexes_kv_by_block_stride for spec in specs)
        assert (
            len(cache_dtype_str_set) == 1
            and len(compress_ratio_set) == 1
            and len(model_version_set) == 1
            and len(sliding_window_set) == 1
            and len(block_stride_set) == 1
        ), (
            "All attention layers in the same KV cache group must use the same "
            "quantization method, compress ratio, model version, sliding "
            "window size, and KV block stride indexing."
        )
        return cls(
            block_size=specs[0].block_size,
            num_kv_heads=specs[0].num_kv_heads,
            head_size=specs[0].head_size,
            dtype=specs[0].dtype,
            page_size_padded=specs[0].page_size_padded,
            indexes_kv_by_block_stride=block_stride_set.pop(),
            sliding_window=sliding_window_set.pop(),
            cache_dtype_str=cache_dtype_str_set.pop(),
            compress_ratio=compress_ratio_set.pop(),
            model_version=model_version_set.pop(),
        )

    def is_uniform_with_collection(
        self, kv_cache_specs: dict[str, KVCacheSpec]
    ) -> bool:
        return all(
            isinstance(spec, SlidingWindowMLASpec)
            and spec.sliding_window == self.sliding_window
            for spec in kv_cache_specs.values()
        )


@dataclass(frozen=True)
class MambaSpec(KVCacheSpec):
    shapes: tuple[tuple[int, ...], ...]
    dtypes: tuple[torch.dtype]
    page_size_padded: int | None = None
    mamba_type: MambaAttentionBackendEnum = MambaAttentionBackendEnum.MAMBA2
    mamba_cache_mode: str = "none"
    num_speculative_blocks: int = 0

    @property
    def tokens_per_state(self) -> int:
        return -1

    @property
    def conv_rows(self) -> int:
        """Number of conv kernel rows (DS layout: shapes[0] = (dim, rows))."""
        return self.shapes[0][1]

    @property
    def conv_dtype_size(self) -> int:
        return get_dtype_size(self.dtypes[0])

    @property
    def conv_proj_dims(self) -> tuple[int, int, int]:
        """Per-rank column counts for the 3 conv sub-projections."""
        local_conv_dim = self.shapes[0][0]
        if self.mamba_type == MambaAttentionBackendEnum.MAMBA2:
            x_local = self.shapes[1][0] * self.shapes[1][1]
            remainder = local_conv_dim - x_local
            b_local = remainder // 2
            return (x_local, b_local, b_local)
        elif self.mamba_type == MambaAttentionBackendEnum.GDN_ATTN:
            value_dim_local = self.shapes[1][0] * self.shapes[1][1]
            key_dim_local = (local_conv_dim - value_dim_local) // 2
            return (key_dim_local, key_dim_local, value_dim_local)
        raise NotImplementedError(
            f"Conv proj dims not supported for {self.mamba_type!r}"
        )

    @property
    def conv_proj_bytes(self) -> tuple[int, int, int]:
        """Byte sizes of the 3 sub-projections for one rank."""
        row_bytes = self.conv_rows * self.conv_dtype_size
        d0, d1, d2 = self.conv_proj_dims
        return (d0 * row_bytes, d1 * row_bytes, d2 * row_bytes)

    @property
    def conv_state_bytes(self) -> int:
        return prod(self.shapes[0]) * self.conv_dtype_size

    @property
    def content_bytes(self) -> int:
        """Total content bytes per block (conv + SSM), excluding padding."""
        return sum(
            prod(s) * get_dtype_size(d) for s, d in zip(self.shapes, self.dtypes)
        )

    @property
    def ssm_num_heads(self) -> int:
        """Local SSM head count (already divided by TP)."""
        return self.shapes[1][0]

    @property
    def ssm_head_bytes(self) -> int:
        """Bytes per SSM head = head_dim * d_state * dtype_size."""
        return prod(self.shapes[1][1:]) * get_dtype_size(self.dtypes[-1])

    def compute_transfer_shape(
        self,
        region_content_bytes: int,
        block_size: int,
        virtually_split: bool,
    ) -> tuple[int, int, int]:
        return (1, 1, region_content_bytes)

    def slice_for_tp_transfer(
        self,
        tensor: torch.Tensor,
        my_tp: int,
        my_rank: int,
        other_tp: int,
        other_rank: int,
        model_config: ModelConfig,
        virtually_split: bool = False,
    ) -> list[torch.Tensor]:
        B, H_in, N, flat_C = tensor.shape
        assert H_in == 1 and N == 1, (
            f"Mamba meta tensor must be (B, 1, 1, flat_C), got {tensor.shape}"
        )

        # Always decompose into conv sub-projections + SSM to maintain a
        # fixed descriptor count (required by _compute_desc_ids indexing).
        total = self.content_bytes
        proj_bytes = self.conv_proj_bytes

        conv_sizes = [pb * flat_C // total for pb in proj_bytes]
        ssm_size = flat_C - sum(conv_sizes)

        block_stride = tensor.stride()[0]
        offset = tensor.storage_offset()

        # Conv parts: (B, 1, 1, proj_bytes) — sharded on DIM_C.
        conv_parts: list[torch.Tensor] = []
        for c_size in conv_sizes:
            part = torch.as_strided(
                tensor,
                (B, 1, 1, c_size),
                stride=(block_stride, c_size, c_size, 1),
                storage_offset=offset,
            )
            conv_parts.append(part)
            offset += c_size

        # SSM: (B, ssm_heads, 1, head_bytes) — sharded on DIM_H.
        ssm_heads = ssm_size // self.ssm_head_bytes
        head_bytes = self.ssm_head_bytes
        ssm_view = torch.as_strided(
            tensor,
            (B, ssm_heads, 1, head_bytes),
            stride=(block_stride, head_bytes, head_bytes, 1),
            storage_offset=offset,
        )

        # Same-TP: no narrowing needed, return full decomposition.
        if my_tp == other_tp and my_rank == other_rank:
            return conv_parts + [ssm_view]

        # TP narrowing for conv: simple ratio on DIM_C.
        if my_tp > other_tp:
            tp_ratio = my_tp // other_tp
            local_idx = my_rank % tp_ratio
            conv_parts = [
                part.narrow(
                    DIM_C,
                    local_idx * (part.shape[DIM_C] // tp_ratio),
                    part.shape[DIM_C] // tp_ratio,
                )
                for part in conv_parts
            ]

        # TP narrowing for SSM: head-overlap on DIM_H (like attention).
        total_heads = self.ssm_num_heads * my_tp
        my_start = my_rank * total_heads // my_tp
        my_end = (my_rank + 1) * total_heads // my_tp
        other_start = other_rank * total_heads // other_tp
        other_end = (other_rank + 1) * total_heads // other_tp

        overlap_start = max(my_start, other_start)
        overlap_end = min(my_end, other_end)
        assert overlap_start < overlap_end, (
            f"No SSM head overlap between local rank {my_rank}/{my_tp} "
            f"[{my_start},{my_end}) and remote rank {other_rank}/{other_tp} "
            f"[{other_start},{other_end}) with total_heads={total_heads}."
        )

        h_start = overlap_start - other_start
        h_len = overlap_end - overlap_start
        ssm_narrowed = ssm_view.narrow(DIM_H, h_start, h_len)

        return conv_parts + [ssm_narrowed]

    @property
    def page_size_bytes(self) -> int:
        page_size = sum(
            prod(shape) * get_dtype_size(dtype)
            for (shape, dtype) in zip(self.shapes, self.dtypes)
        )
        if self.page_size_padded is not None:
            assert self.page_size_padded >= page_size
            return self.page_size_padded
        return page_size

    def max_memory_usage_bytes(self, vllm_config: VllmConfig) -> int:
        if vllm_config.cache_config.mamba_cache_mode == "all":
            max_model_len = vllm_config.model_config.max_model_len
            return (
                cdiv(max_model_len, self.block_size) + self.num_speculative_blocks
            ) * self.page_size_bytes
        elif vllm_config.cache_config.mamba_cache_mode == "align":
            return self.page_size_bytes * (2 + self.num_speculative_blocks)
        else:
            return self.page_size_bytes * (1 + self.num_speculative_blocks)

    def is_uniform_with_collection(
        self, kv_cache_specs: dict[str, KVCacheSpec]
    ) -> bool:
        return all(
            isinstance(spec, MambaSpec)
            and spec.num_speculative_blocks == self.num_speculative_blocks
            for spec in kv_cache_specs.values()
        )


@dataclass(frozen=True)
class EncoderOnlyAttentionSpec(AttentionSpec):
    def max_memory_usage_bytes(self, vllm_config: VllmConfig) -> int:
        # Encoder-only layers do not need KV cache
        return 0


@dataclass(frozen=True)
class CrossAttentionSpec(AttentionSpec):
    """
    KV cache spec for cross-attention layers in encoder-decoder models.
    """

    def max_memory_usage_bytes(self, vllm_config: VllmConfig) -> int:
        # For cross-attention, we need to cache encoder states
        # Get encoder length (e.g., 1500 for Whisper).
        max_encoder_len = vllm_config.scheduler_config.max_num_encoder_input_tokens
        return cdiv(max_encoder_len, self.block_size) * self.page_size_bytes


@dataclass(frozen=True)
class SinkFullAttentionSpec(FullAttentionSpec):
    sink_len: int | None = None

    @classmethod
    def merge(cls, specs: list[Self]) -> Self:
        """
        Merge a list of FullAttentionSpec objects into a single
        FullAttentionSpec object.
        """
        assert all(isinstance(spec, FullAttentionSpec) for spec in specs), (
            "All attention layers in the same KV cache group must be FullAttentionSpec."
        )

        sliding_window = set(
            spec.sliding_window for spec in specs if spec.sliding_window is not None
        )
        attention_chunk_size = set(
            spec.attention_chunk_size
            for spec in specs
            if spec.attention_chunk_size is not None
        )
        assert not any(isinstance(spec, MLAAttentionSpec) for spec in specs), (
            "MLAAttentionSpec should be merged in MLAAttentionSpec.merge"
        )
        merged_spec = cls(
            block_size=specs[0].block_size,
            num_kv_heads=specs[0].num_kv_heads,
            head_size=specs[0].head_size,
            head_size_v=specs[0].head_size_v,
            sink_len=specs[0].sink_len,
            dtype=specs[0].dtype,
            kv_quant_mode=specs[0].kv_quant_mode,
            page_size_padded=specs[0].page_size_padded,
            indexes_kv_by_block_stride=specs[0].indexes_kv_by_block_stride,
            sliding_window=cls.merge_window_sizes(sliding_window),
            attention_chunk_size=cls.merge_window_sizes(attention_chunk_size),
            non_causal=any(spec.non_causal for spec in specs),
        )
        for spec in specs:
            for f in fields(AttentionSpec):
                assert getattr(spec, f.name) == getattr(merged_spec, f.name), (
                    "All attention layers in the same KV cache group must have "
                    "the same attention spec."
                )
        assert (merged_spec.sliding_window is not None) + (
            merged_spec.attention_chunk_size is not None
        ) <= 1, (
            "Model with both sliding window layers and chunked local attention "
            "layers is not supported."
        )
        return merged_spec


@dataclass(frozen=True)
class UniformTypeKVCacheSpecs(KVCacheSpec):
    """
    A KV cache spec for multiple layers with the same type of attention. Here,
    same types means always need the same number of token slots. For example,
    sliding window attentions with different window sizes are not the same type
    and should not be merged into one UniformTypeKVCacheSpecs.
    """

    kv_cache_specs: dict[str, KVCacheSpec]

    @property
    def page_size_bytes(self) -> int:
        return sum(spec.page_size_bytes for spec in self.kv_cache_specs.values())

    def max_memory_usage_bytes(self, vllm_config: VllmConfig) -> int:
        max_num_pages = max(
            cdiv(spec.max_memory_usage_bytes(vllm_config), spec.page_size_bytes)
            for spec in self.kv_cache_specs.values()
        )
        return max_num_pages * self.page_size_bytes

    @classmethod
    def is_uniform_type(cls, kv_cache_specs: dict[str, KVCacheSpec]) -> bool:
        """
        Whether all layers have the same type of KV cache spec.

        Uses the registry to determine grouping base classes, so custom specs
        that inherit from FullAttentionSpec are treated as full attention.
        """
        block_sizes = set(spec.block_size for spec in kv_cache_specs.values())
        if len(block_sizes) > 1:
            # Different block sizes, not uniform.
            return False
        first_spec = next(iter(kv_cache_specs.values()))
        return first_spec.is_uniform_with_collection(kv_cache_specs)

    @classmethod
    def from_specs(cls, kv_cache_specs: dict[str, KVCacheSpec]) -> Self | None:
        """
        Return a SameTypeKVCacheSpecs object if all layers have the same type
        of KV cache spec. Return None if not.
        """
        if cls.is_uniform_type(kv_cache_specs):
            block_size = next(iter(kv_cache_specs.values())).block_size
            return cls(block_size=block_size, kv_cache_specs=kv_cache_specs)
        else:
            return None

    # NOTE: below util functions are only used by DeepseekV4 for now.
    def get_page_sizes(self) -> list[int]:
        return list(set(spec.page_size_bytes for spec in self.kv_cache_specs.values()))

    def get_num_layer_tuples(self) -> int:
        return Counter(
            spec.page_size_bytes for spec in self.kv_cache_specs.values()
        ).most_common(1)[0][1]

    def max_memory_usage_pages(self, vllm_config: VllmConfig) -> int:
        return max(
            cdiv(spec.max_memory_usage_bytes(vllm_config), spec.page_size_bytes)
            for spec in self.kv_cache_specs.values()
        )


def get_kv_cache_spec_kind(kv_cache_spec: KVCacheSpec) -> KVCacheSpecKind:
    if isinstance(kv_cache_spec, UniformTypeKVCacheSpecs):
        inner_kinds = {
            get_kv_cache_spec_kind(spec)
            for spec in kv_cache_spec.kv_cache_specs.values()
        }
        if len(inner_kinds) == 1:
            return next(iter(inner_kinds))
        return KVCacheSpecKind.UNKNOWN
    # Keep subclass checks before base classes so specialized specs keep their
    # more precise kind.
    if isinstance(kv_cache_spec, SlidingWindowMLASpec):
        return KVCacheSpecKind.SLIDING_WINDOW_MLA
    if isinstance(kv_cache_spec, MLAAttentionSpec):
        return KVCacheSpecKind.MLA_ATTENTION
    if isinstance(kv_cache_spec, SinkFullAttentionSpec):
        return KVCacheSpecKind.SINK_FULL_ATTENTION
    if isinstance(kv_cache_spec, FullAttentionSpec):
        return KVCacheSpecKind.FULL_ATTENTION
    if isinstance(kv_cache_spec, ChunkedLocalAttentionSpec):
        return KVCacheSpecKind.CHUNKED_LOCAL_ATTENTION
    if isinstance(kv_cache_spec, SlidingWindowSpec):
        return KVCacheSpecKind.SLIDING_WINDOW
    if isinstance(kv_cache_spec, MambaSpec):
        return KVCacheSpecKind.MAMBA
    if isinstance(kv_cache_spec, EncoderOnlyAttentionSpec):
        return KVCacheSpecKind.ENCODER_ONLY_ATTENTION
    if isinstance(kv_cache_spec, CrossAttentionSpec):
        return KVCacheSpecKind.CROSS_ATTENTION
    return KVCacheSpecKind.UNKNOWN


def get_kv_cache_spec_sliding_window(kv_cache_spec: KVCacheSpec) -> int | None:
    if isinstance(kv_cache_spec, UniformTypeKVCacheSpecs):
        inner_windows = {
            get_kv_cache_spec_sliding_window(spec)
            for spec in kv_cache_spec.kv_cache_specs.values()
        }
        return next(iter(inner_windows)) if len(inner_windows) == 1 else None
    if isinstance(kv_cache_spec, SlidingWindowSpec):
        return kv_cache_spec.sliding_window
    return None


@dataclass
class KVCacheTensor:
    """
    A class for specifying how the workers should initialize the KV cache.
    """

    size: int  # size of the KV cache tensor in bytes
    shared_by: list[str]  # layer names that share the same KV cache tensor
    offset: int = 0  # byte offset of this layer within a contiguous block
    block_stride: int = 0  # total bytes per block in a packed layout (0 = not packed)


@dataclass
class KVCacheGroupSpec:
    """
    Represents a group of model layers that share the same KV cache block table.
    These layers are regarded as one layer in the KV cache manager.
    """

    # The names of model layers in this group
    layer_names: list[str]
    # The KV cache spec of this manager layer
    kv_cache_spec: KVCacheSpec
    # Whether this group contains EAGLE/MTP draft attention layers.
    is_eagle_group: bool = False


@dataclass
class KVCacheConfig:
    """
    The KV cache configuration of a model.
    """

    num_blocks: int
    """The number of KV cache blocks"""
    kv_cache_tensors: list[KVCacheTensor]
    """How should model runner initialize the KV cache tensors for each layer"""
    kv_cache_groups: list[KVCacheGroupSpec]
    """
    The kv cache groups of the model.
    For models with only one type of attention, there is only one group that
    contains all layers.
    For models with multiple types of attention, there will be multiple groups,
    see `_get_kv_cache_config_uniform_page_size` for more details.
    """

    @property
    def has_mamba_layers(self) -> bool:
        return any(isinstance(g.kv_cache_spec, MambaSpec) for g in self.kv_cache_groups)

    @property
    def needs_kv_cache_zeroing(self) -> bool:
        return self.has_mamba_layers
