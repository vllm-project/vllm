# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import copy
from dataclasses import dataclass, fields, replace
from enum import IntEnum
from math import prod

import torch
from typing_extensions import Self

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.utils.math_utils import cdiv
from vllm.utils.torch_utils import get_dtype_size

logger = init_logger(__name__)


# ---------------------------------------------------------------------------
# KV cache quantization mode
# ---------------------------------------------------------------------------


class KVQuantMode(IntEnum):
    """KV cache quantization mode.

    Describes the *scale granularity*, not the storage dtype.  The storage
    dtype (int8, fp8_e4m3, …) is orthogonal and carried by
    ``AttentionSpec.dtype`` / the ``kv_cache_dtype`` string.

    Used by attention backends and kernels to dispatch quantization logic
    without string matching on ``kv_cache_dtype``.
    """

    NONE = 0
    FP8 = 1  # per-tensor scales (current fp8 path)
    PER_TOKEN = 2  # per-(token, head) dynamic scales (e.g. int8, fp8)
    PER_TOKEN_GROUP = 3  # per-(token, head, group) scales (e.g. fp8 1×128)
    NVFP4 = 4  # packed 4-bit KV + fp8 blockscales + per-token global scale


# Default group size for PER_TOKEN_GROUP quantization (e.g. deepseek-style
# block fp8).  Backends may override via AttentionSpec.quant_group_size.
DEFAULT_QUANT_GROUP_SIZE = 128

# Default block-scale group size for NVFP4 (1×16 groups).
NVFP4_BLOCKSCALE_GROUP_SIZE = 16


def get_kv_quant_mode(kv_cache_dtype: str) -> KVQuantMode:
    """Map a ``kv_cache_dtype`` string to a :class:`KVQuantMode`.

    Ordering matters: more specific suffixes are checked first so that
    e.g. ``"fp8_per_token"`` maps to ``PER_TOKEN``, not ``FP8``.
    """
    if kv_cache_dtype == "nvfp4":
        return KVQuantMode.NVFP4
    if kv_cache_dtype.endswith("_per_group"):
        return KVQuantMode.PER_TOKEN_GROUP
    if kv_cache_dtype.endswith("_per_token"):
        return KVQuantMode.PER_TOKEN
    if kv_cache_dtype.startswith("fp8"):
        return KVQuantMode.FP8
    return KVQuantMode.NONE


def is_quantized_kv_cache(kv_cache_dtype: str) -> bool:
    return get_kv_quant_mode(kv_cache_dtype) != KVQuantMode.NONE


def kv_cache_uses_per_token_scales(kv_cache_dtype: str) -> bool:
    """Return True if *kv_cache_dtype* needs per-token scales."""
    return get_kv_quant_mode(kv_cache_dtype) == KVQuantMode.PER_TOKEN


@dataclass(frozen=True)
class AuxBufferSpec:
    """Describes one auxiliary buffer allocated alongside the KV cache.

    The model runner uses these specs to pre-allocate tensors of shape
    ``(num_blocks, *shape_per_block)`` and bind them to the attention
    implementation via ``AttentionImpl.bind_auxiliary_buffers()``.
    """

    name: str  # e.g. "k_scale_cache"
    dtype: torch.dtype  # e.g. torch.float32
    shape_per_block: tuple[int, ...]  # e.g. (block_size,)


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

    def max_memory_usage_bytes(self, vllm_config: VllmConfig) -> int:
        """
        The maximum possible memory usage of this KV cache in bytes.

        Returns:
            The KV cache size in bytes
        """
        raise NotImplementedError

    @property
    def auxiliary_buffer_specs(self) -> list[AuxBufferSpec]:
        """Structured descriptions of auxiliary buffers for this cache format.

        Override in subclasses that need auxiliary buffers (e.g.
        per-token quantization scale caches).  The model runner
        allocates tensors of shape ``(num_blocks, *spec.shape_per_block)``
        for each entry and binds them to the attention implementation.
        """
        return []

    @property
    def auxiliary_memory_per_block(self) -> int:
        """Extra per-block memory not stored in the KV cache tensor itself.

        Derived from :attr:`auxiliary_buffer_specs`.  Factored into the
        block count calculation but NOT into page_size_bytes.
        """
        return sum(
            prod(s.shape_per_block) * get_dtype_size(s.dtype)
            for s in self.auxiliary_buffer_specs
        )

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


@dataclass(frozen=True, kw_only=True)
class AttentionSpec(KVCacheSpec):
    num_kv_heads: int
    head_size: int
    dtype: torch.dtype
    kv_quant_mode: KVQuantMode = KVQuantMode.NONE
    quant_group_size: int | None = None
    page_size_padded: int | None = None

    @property
    def page_size_bytes(self) -> int:
        real_page_size = self.real_page_size_bytes
        if self.page_size_padded is not None:
            assert self.page_size_padded >= real_page_size
            return self.page_size_padded
        return real_page_size

    @property
    def auxiliary_buffer_specs(self) -> list[AuxBufferSpec]:
        """Auxiliary buffers for quantized KV cache formats.

        Each quantization mode defines its own set of auxiliary buffers.
        The model runner allocates these and binds them to the attention
        implementation via ``bind_auxiliary_buffers()``.
        """
        mode = self.kv_quant_mode

        if mode == KVQuantMode.PER_TOKEN:
            # One float32 scale per token (shared across all KV heads).
            # Works for int8_per_token, fp8_per_token, etc.
            shape = (self.block_size,)
            return [
                AuxBufferSpec("k_scale_cache", torch.float32, shape),
                AuxBufferSpec("v_scale_cache", torch.float32, shape),
            ]

        if mode == KVQuantMode.PER_TOKEN_GROUP:
            # Per-(token, head, group) scales for block-quantised formats
            # (e.g. deepseek-style fp8 with 1×128 groups).
            gs = self.quant_group_size or DEFAULT_QUANT_GROUP_SIZE
            num_groups = (self.head_size + gs - 1) // gs
            shape = (self.block_size, self.num_kv_heads, num_groups)
            return [
                AuxBufferSpec("k_scale_cache", torch.float32, shape),
                AuxBufferSpec("v_scale_cache", torch.float32, shape),
            ]

        if mode == KVQuantMode.NVFP4:
            # NVFP4: packed KV (2 elements/byte in the main cache tensor)
            # + fp8 blockscales (1×16 groups) + per-token global scale.
            gs = NVFP4_BLOCKSCALE_GROUP_SIZE
            num_groups = (self.head_size + gs - 1) // gs
            blockscale_shape = (self.block_size, self.num_kv_heads, num_groups)
            global_scale_shape = (self.block_size,)
            return [
                AuxBufferSpec("k_blockscale", torch.float8_e4m3fn, blockscale_shape),
                AuxBufferSpec("v_blockscale", torch.float8_e4m3fn, blockscale_shape),
                AuxBufferSpec("k_global_scale", torch.float32, global_scale_shape),
                AuxBufferSpec("v_global_scale", torch.float32, global_scale_shape),
            ]

        return []

    @property
    def real_page_size_bytes(self) -> int:
        return (
            2
            * self.block_size
            * self.num_kv_heads
            * self.head_size
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
            quant_group_size=specs[0].quant_group_size,
            page_size_padded=specs[0].page_size_padded,
            sliding_window=cls.merge_window_sizes(sliding_window),
            attention_chunk_size=cls.merge_window_sizes(attention_chunk_size),
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
        return (
            self.block_size
            * self.num_kv_heads
            * (self.head_size + self.head_size_v)
            * get_dtype_size(self.dtype)
        )


@dataclass(frozen=True, kw_only=True)
class MLAAttentionSpec(FullAttentionSpec):
    # TODO(Lucas/Chen): less hacky way to do this
    cache_dtype_str: str | None = None

    @property
    def real_page_size_bytes(self) -> int:
        if self.cache_dtype_str == "fp8_ds_mla":
            # See `vllm/v1/attention/backends/mla/flashmla_sparse.py`
            #  for details.
            return self.block_size * 656
        return (
            self.block_size
            * self.num_kv_heads
            * self.head_size
            * get_dtype_size(self.dtype)
        )

    @classmethod
    def merge(cls, specs: list[Self]) -> Self:
        assert all(isinstance(spec, MLAAttentionSpec) for spec in specs), (
            "All attention layers in the same KV cache group must be MLAAttentionSpec."
        )
        cache_dtype_str_set = set(spec.cache_dtype_str for spec in specs)
        assert len(cache_dtype_str_set) == 1, (
            "All attention layers in the same KV cache group must use the same "
            "quantization method."
        )
        return cls(
            block_size=specs[0].block_size,
            num_kv_heads=specs[0].num_kv_heads,
            head_size=specs[0].head_size,
            dtype=specs[0].dtype,
            kv_quant_mode=specs[0].kv_quant_mode,
            quant_group_size=specs[0].quant_group_size,
            page_size_padded=specs[0].page_size_padded,
            cache_dtype_str=cache_dtype_str_set.pop(),
        )


@dataclass(frozen=True, kw_only=True)
class ChunkedLocalAttentionSpec(AttentionSpec):
    attention_chunk_size: int

    def max_memory_usage_bytes(self, vllm_config: VllmConfig) -> int:
        max_model_len = vllm_config.model_config.max_model_len
        max_num_batched_tokens = vllm_config.scheduler_config.max_num_batched_tokens

        # During chunked prefill, we allocate KV cache for at most
        # `self.attention_chunk_size` computed tokens plus the newly scheduled
        # tokens. And we won't allocate KV cache for more than `max_model_len`
        # tokens.
        num_tokens = min(
            self.attention_chunk_size + max_num_batched_tokens, max_model_len
        )

        return cdiv(num_tokens, self.block_size) * self.page_size_bytes


@dataclass(frozen=True, kw_only=True)
class SlidingWindowSpec(AttentionSpec):
    sliding_window: int

    def max_memory_usage_bytes(self, vllm_config: VllmConfig) -> int:
        assert vllm_config.parallel_config.decode_context_parallel_size == 1, (
            "DCP not support sliding window."
        )
        max_model_len = vllm_config.model_config.max_model_len
        max_num_batched_tokens = vllm_config.scheduler_config.max_num_batched_tokens

        # During chunked prefill, we allocate KV cache for the last
        # `self.sliding_window-1` computed tokens plus the newly scheduled
        # tokens. And we won't allocate KV cache for more than `max_model_len`
        # tokens.
        num_tokens = min(
            self.sliding_window - 1 + max_num_batched_tokens, max_model_len
        )

        # +1 here because the sliding window may not start from the beginning
        # of the block. For example, if the block size is 4 and num_token
        # is 4, we need two blocks [XXCD] [EF] to store the sliding
        # window [CDEF] of 6 tokens.
        return (cdiv(num_tokens, self.block_size) + 1) * self.page_size_bytes


@dataclass(frozen=True)
class MambaSpec(KVCacheSpec):
    shapes: tuple[tuple[int, ...], ...]
    dtypes: tuple[torch.dtype]
    page_size_padded: int | None = None
    mamba_type: str = "mamba2"
    mamba_cache_mode: str = "none"
    num_speculative_blocks: int = 0

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
            return cdiv(max_model_len, self.block_size) * self.page_size_bytes
        elif vllm_config.cache_config.mamba_cache_mode == "align":
            return self.page_size_bytes * (2 + self.num_speculative_blocks)
        else:
            return self.page_size_bytes * (1 + self.num_speculative_blocks)


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
            quant_group_size=specs[0].quant_group_size,
            page_size_padded=specs[0].page_size_padded,
            sliding_window=cls.merge_window_sizes(sliding_window),
            attention_chunk_size=cls.merge_window_sizes(attention_chunk_size),
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
        """
        block_sizes = set(spec.block_size for spec in kv_cache_specs.values())
        if len(block_sizes) > 1:
            # Different block sizes, not uniform.
            return False
        one_spec = next(iter(kv_cache_specs.values()))
        if isinstance(one_spec, FullAttentionSpec):
            return all(
                isinstance(spec, FullAttentionSpec) for spec in kv_cache_specs.values()
            )
        elif isinstance(one_spec, CrossAttentionSpec):
            return all(
                isinstance(spec, CrossAttentionSpec) for spec in kv_cache_specs.values()
            )
        elif isinstance(one_spec, SlidingWindowSpec):
            return all(
                isinstance(spec, SlidingWindowSpec)
                and spec.sliding_window == one_spec.sliding_window
                for spec in kv_cache_specs.values()
            )
        elif isinstance(one_spec, ChunkedLocalAttentionSpec):
            return all(
                isinstance(spec, ChunkedLocalAttentionSpec)
                and spec.attention_chunk_size == one_spec.attention_chunk_size
                for spec in kv_cache_specs.values()
            )
        elif isinstance(one_spec, MambaSpec):
            return all(
                isinstance(spec, MambaSpec)
                and spec.num_speculative_blocks == one_spec.num_speculative_blocks
                for spec in kv_cache_specs.values()
            )
        else:
            # NOTE(Chen): Please add new branches for new KV cache spec types.
            raise NotImplementedError(
                f"Unsupported KV cache spec type: {type(one_spec)}"
            )

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


@dataclass
class KVCacheTensor:
    """
    A class for specifying how the workers should initialize the KV cache.
    """

    size: int  # size of the KV cache tensor in bytes
    shared_by: list[str]  # layer names that share the same KV cache tensor


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
