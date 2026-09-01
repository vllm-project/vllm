# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import copy
from collections import Counter
from collections.abc import Collection, Sequence
from dataclasses import dataclass, fields, replace
from enum import Enum, IntEnum
from fractions import Fraction
from functools import cached_property
from math import prod
from typing import TYPE_CHECKING, TypeVar

import torch
from typing_extensions import Self

from vllm.logger import init_logger
from vllm.utils.math_utils import cdiv, round_up
from vllm.utils.torch_utils import get_dtype_size
from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum
from vllm.v1.kv_cache_layout import _DIM_B, _DIM_L, KVCacheLayout
from vllm.v1.kv_cache_spec_registry import KVCacheSpecRegistry

if TYPE_CHECKING:
    from vllm.config import VllmConfig

logger = init_logger(__name__)

_SpecT = TypeVar("_SpecT", bound="KVCacheSpec")


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
    # Hadamard-rotated Lloyd-Max quant, packed K+V per slot.
    TURBOQUANT_K8V4 = 6
    TURBOQUANT_4BIT_NC = 7
    TURBOQUANT_K3V4_NC = 8
    TURBOQUANT_3BIT_NC = 9
    NVFP4_DS_MLA = 10  # opaque-bytes NVFP4 DS-MLA layouts (FlashMLA sparse)

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

    @property
    def is_turboquant(self) -> bool:
        """True for any turboquant quantization mode."""
        return self in (
            KVQuantMode.TURBOQUANT_K8V4,
            KVQuantMode.TURBOQUANT_4BIT_NC,
            KVQuantMode.TURBOQUANT_K3V4_NC,
            KVQuantMode.TURBOQUANT_3BIT_NC,
        )


def get_kv_quant_mode(kv_cache_dtype: str) -> KVQuantMode:
    """Map a ``kv_cache_dtype`` string to a :class:`KVQuantMode`."""
    if kv_cache_dtype == "int4_per_token_head":
        return KVQuantMode.INT4_PER_TOKEN_HEAD
    if kv_cache_dtype == "int8_per_token_head":
        return KVQuantMode.INT8_PER_TOKEN_HEAD
    if kv_cache_dtype == "fp8_per_token_head":
        return KVQuantMode.FP8_PER_TOKEN_HEAD
    # Must precede the ``nvfp4`` prefix test below, which would otherwise match.
    if kv_cache_dtype == "nvfp4_ds_mla":
        # Page size is keyed on cache_dtype_str in the MLA specs, not
        # nvfp4_kv_cache_full_dim.
        return KVQuantMode.NVFP4_DS_MLA
    if kv_cache_dtype.startswith("nvfp4"):
        return KVQuantMode.NVFP4
    if isinstance(kv_cache_dtype, str) and kv_cache_dtype.startswith("turboquant_"):
        return KVQuantMode[kv_cache_dtype.upper()]
    if isinstance(kv_cache_dtype, str) and kv_cache_dtype.startswith("fp8"):
        return KVQuantMode.FP8_PER_TENSOR
    return KVQuantMode.NONE


def is_quantized_kv_cache(kv_cache_dtype: str) -> bool:
    return get_kv_quant_mode(kv_cache_dtype) != KVQuantMode.NONE


def replace_as(
    spec: KVCacheSpec,
    target_cls: type[_SpecT],
    *,
    drop: Collection[str] = (),
    **changes,
) -> _SpecT:
    """``dataclasses.replace``, but rebuilding *spec* as *target_cls*
      e.g. ``SlidingWindowSpec`` -> ``FullAttentionSpec``

    Every field of *spec* must exist on *target_cls* unless named in *drop*;
    fields only *target_cls* has keep their default values.
    """
    kwargs = {
        f.name: getattr(spec, f.name)
        for f in fields(spec)
        if f.init and f.name not in drop
    }
    kwargs.update(changes)
    return target_cls(**kwargs)


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
    def prefix_cacheable(self) -> bool:
        return True

    @property
    def num_heads(self) -> int:
        raise NotImplementedError

    @property
    def tokens_per_state(self) -> int | Fraction:
        raise NotImplementedError

    @property
    def state_content_size_bytes(self) -> int:
        raise NotImplementedError

    @property
    def page_size_bytes(self) -> int:
        """
        The size of a page with `block_size` tokens in bytes.

        Returns:
            The page size
        """
        raise NotImplementedError

    @property
    def num_states(self) -> int:
        return self.get_num_kernel_states(self.block_size)

    def get_num_kernel_states(self, kernel_block_size: int) -> int:
        if self.tokens_per_state > 0:
            return kernel_block_size // self.tokens_per_state
        return 1

    def max_memory_usage_bytes(self, vllm_config: VllmConfig) -> int:
        """
        The maximum possible memory usage of this KV cache in bytes.

        Returns:
            The KV cache size in bytes
        """
        raise NotImplementedError

    def max_num_blocks_per_req(self, vllm_config: VllmConfig, max_len: int) -> int:
        """
        The number of block table entries needed per request, i.e. the row
        length of the worker-side block table for this cache group.

        Args:
            vllm_config: The vllm config.
            max_len: The maximum sequence length to size for, including the
                encoder length for encoder-decoder models.
        """
        return cdiv(max_len, self.block_size)

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


def group_kernel_blocks(cache: torch.Tensor, num_blocks: int) -> torch.Tensor:
    """View a kernel-block-granular layer cache with manager blocks as dim 0.

    Kernel block splitting subdivides each manager block into uniformly strided
    kernel blocks, so grouping is a pure view: ``(num_blocks * ratio, ...)``
    """
    if cache.shape[0] == num_blocks:
        return cache
    assert cache.shape[0] % num_blocks == 0
    return cache.unflatten(0, (num_blocks, -1))


def compute_layer_kv_cache_shape_bytes(
    spec: KVCacheSpec,
    num_blocks: int,
    kernel_block_size: int | None = None,
) -> tuple[int, ...]:
    """Return the 4D logical shape ``(B, H, N, C)`` where C is in bytes."""
    bs = kernel_block_size if kernel_block_size is not None else spec.block_size
    assert spec.block_size % bs == 0, (
        f"Kernel block size {bs} must divide KV cache block size {spec.block_size}."
    )
    blocks_per_page = spec.block_size // bs
    return (
        num_blocks * blocks_per_page,
        spec.num_heads,
        spec.get_num_kernel_states(bs),
        spec.state_content_size_bytes,
    )


def compute_layout_strides(
    spec: KVCacheSpec,
    num_blocks: int,
    num_layers: int,
    layout: KVCacheLayout,
    kernel_block_size: int | None = None,
    fixed_strides: tuple[int | None, ...] = (None,) * 5,
) -> tuple[int, ...]:
    """Byte strides in logical ``[L, B, H, N, C]`` axis order."""
    assert len(fixed_strides) == 5
    assert all(stride is None or stride > 0 for stride in fixed_strides)
    shape = (
        num_layers,
        *compute_layer_kv_cache_shape_bytes(spec, num_blocks, kernel_block_size),
    )
    order = layout.stride_order
    padded_page_size = getattr(spec, "page_size_padded", None)
    if padded_page_size is not None:
        assert kernel_block_size is None or kernel_block_size == spec.block_size, (
            "Padded KV pages do not support kernel block splitting."
        )
        page_grid_end = max(order.index(_DIM_L), order.index(_DIM_B)) + 1
        page_grid_shape = tuple(shape[dim] for dim in order[:page_grid_end])
        assert prod(page_grid_shape) == num_layers * num_blocks, (
            "Page padding requires dimensions outside the page tail to be L, B, "
            f"or singleton; got {layout.name} with shape {shape}."
        )

    strides = [0] * 5
    current_stride = 1
    for physical_idx, dim in reversed(tuple(enumerate(order))):
        if padded_page_size is not None and physical_idx == page_grid_end - 1:
            current_stride = max(current_stride, padded_page_size)
            assert current_stride % padded_page_size == 0
        strides[dim] = fixed_strides[dim] or current_stride
        current_stride = strides[dim] * shape[dim]
    return tuple(strides)


def create_kv_cache_views(
    raw: torch.Tensor,
    spec: KVCacheSpec,
    num_blocks: int,
    layout: KVCacheLayout,
    kv_cache_tensor: KVCacheTensor,
    kernel_block_size: int | None = None,
) -> list[torch.Tensor]:
    """View a flat int8 buffer as one 4D ``[B, H, N, C]`` view per layer.

    Block ``b`` of layer ``l`` starts at the tensor offset plus its layer and
    block stride contributions.
    """
    num_layers = len(kv_cache_tensor.layers)
    layer_stride = kv_cache_tensor.layer_stride
    block_stride = kv_cache_tensor.block_stride
    shape_bytes = compute_layer_kv_cache_shape_bytes(
        spec, num_blocks, kernel_block_size
    )
    ratio = shape_bytes[0] // num_blocks
    if ratio > 1:
        # Kernel blocks subdivide a manager block into `ratio` equal pieces, so
        # they sit a constant stride apart only if a block is one dense page: no
        # padding at its end, and no other layer's page before the next block.
        dense_page_size = prod(compute_layer_kv_cache_shape_bytes(spec, 1)[1:])
        if block_stride != dense_page_size:
            raise ValueError(
                f"The resolved KV cache layout ({layout.name}) does not store "
                "blocks as dense, unpadded pages (block stride "
                f"{block_stride} != page {dense_page_size}), so a manager "
                f"block cannot be split into {ratio} kernel blocks of "
                f"{kernel_block_size} tokens. Reduce --block-size to "
                f"{kernel_block_size} or set VLLM_KV_CACHE_LAYOUT to a "
                "layer-compact layout (e.g. LBNHC)."
            )
        assert block_stride % ratio == 0, (
            f"Block stride {block_stride} must divide into {ratio} equal kernel blocks."
        )
        block_stride //= ratio

    logical_shape = (num_layers, *shape_bytes)
    strides = compute_layout_strides(
        spec,
        num_blocks,
        num_layers,
        layout,
        kernel_block_size,
        fixed_strides=(layer_stride, block_stride, None, None, None),
    )
    dtype = getattr(spec, "dtype", None)

    view_5d = torch.as_strided(
        raw,
        size=logical_shape,
        stride=strides,
        storage_offset=raw.storage_offset() + kv_cache_tensor.offset,
    )

    views = []
    for layer_idx in range(num_layers):
        cache_logical = view_5d[layer_idx]
        if dtype is not None:
            cache_logical = cache_logical.view(dtype)
        views.append(cache_logical)
    return views


@dataclass(frozen=True, kw_only=True)
class AttentionSpec(KVCacheSpec):
    num_kv_heads: int
    head_size: int
    dtype: torch.dtype
    head_size_v: int = None  # type: ignore[assignment]
    kv_quant_mode: KVQuantMode = KVQuantMode.NONE
    page_size_padded: int | None = None
    num_head_slots: int | None = None
    """H of the logical ``[B, H, N, C]`` page when packing diverges from one
    slot per KV head. None means one slot per KV head. Published by the backend.
    """
    state_content_bytes: int | None = None
    """C in bytes when packed; None means dense K/V content."""
    tokens_per_state: int | Fraction = 1
    """Tokens covered by one stored state. Ints > 1 compress multiple tokens
    into one state (DSv4 sparse MLA); fractions < 1 store multiple states per
    token (Whisper block pooling: ``Fraction(1, block_pool_size)``)."""

    def __post_init__(self):
        if self.head_size_v is None:
            object.__setattr__(self, "head_size_v", self.head_size)

    @property
    def num_heads(self) -> int:
        if self.num_head_slots is not None:
            return self.num_head_slots
        return self.num_kv_heads

    @property
    def state_content_size_bytes(self) -> int:
        """Bytes per (head slot, stored state) cell of the page."""
        if self.state_content_bytes is not None:
            return self.state_content_bytes
        return (self.head_size + self.head_size_v) * get_dtype_size(self.dtype)

    @property
    def unpadded_page_size_bytes(self) -> int:
        return self.num_heads * self.num_states * self.state_content_size_bytes

    @property
    def page_size_bytes(self) -> int:
        if self.page_size_padded is not None:
            assert self.page_size_padded >= self.unpadded_page_size_bytes
            return self.page_size_padded
        return self.unpadded_page_size_bytes

    @property
    def real_page_size_bytes(self) -> int:
        """Alias of ``unpadded_page_size_bytes``
        TODO(lucas): follow up with TPU backend to see if we can remove this property.
        """
        return self.unpadded_page_size_bytes

    def max_num_blocks_per_req(self, vllm_config: VllmConfig, max_len: int) -> int:
        parallel_config = vllm_config.parallel_config
        kv_shard_count = parallel_config.decode_context_parallel_size
        return cdiv(max_len, self.block_size * kv_shard_count)


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

    def max_memory_usage_bytes(self, vllm_config: VllmConfig) -> int:
        max_model_len = vllm_config.model_config.max_model_len
        dcp_world_size = vllm_config.parallel_config.decode_context_parallel_size
        if dcp_world_size > 1:
            max_model_len = cdiv(max_model_len, dcp_world_size)
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
            num_head_slots=specs[0].num_head_slots,
            state_content_bytes=specs[0].state_content_bytes,
            tokens_per_state=specs[0].tokens_per_state,
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


def _apply_alignment_padding(spec: MLAAttentionSpec | SlidingWindowMLASpec):
    if spec.alignment is None:
        return
    actual_page_size = spec.real_page_size_bytes
    padded_page_size = round_up(actual_page_size, spec.alignment)
    if padded_page_size != actual_page_size:
        object.__setattr__(spec, "page_size_padded", padded_page_size)


@dataclass(frozen=True, kw_only=True)
class MLAAttentionSpec(FullAttentionSpec):
    # TODO(Lucas/Chen): less hacky way to do this
    cache_dtype_str: str | None = None
    # DeepseekV4 only fields. Non-DeepseekV4 MLA models leave these at defaults.
    alignment: int | None = None  # Default to None for no padding.
    model_version: str | None = None
    # Marks draft groups that flatten a non-causal query block into decode rows.
    non_causal_multi_token_decode: bool = False
    # MLA stores a single latent vector per state; there is no separate V.
    head_size_v: int = 0

    def __post_init__(self):
        super().__post_init__()
        _apply_alignment_padding(self)

    @classmethod
    def merge(cls, specs: list[Self]) -> Self:
        assert all(isinstance(spec, MLAAttentionSpec) for spec in specs), (
            "All attention layers in the same KV cache group must be MLAAttentionSpec."
        )
        cache_dtype_str_set = set(spec.cache_dtype_str for spec in specs)
        tokens_per_state_set = set(spec.tokens_per_state for spec in specs)
        model_version_set = set(spec.model_version for spec in specs)
        assert (
            len(cache_dtype_str_set) == 1
            and len(tokens_per_state_set) == 1
            and len(model_version_set) == 1
        ), (
            "All attention layers in the same KV cache group must use the same "
            "quantization method, tokens per state, and model version."
        )
        non_causal_mtd_set = {spec.non_causal_multi_token_decode for spec in specs}
        assert len(non_causal_mtd_set) == 1, (
            "All attention layers in the same KV cache group must agree on "
            "non_causal_multi_token_decode."
        )
        merged_spec = cls(
            block_size=specs[0].block_size,
            num_kv_heads=specs[0].num_kv_heads,
            head_size=specs[0].head_size,
            dtype=specs[0].dtype,
            kv_quant_mode=specs[0].kv_quant_mode,
            page_size_padded=specs[0].page_size_padded,
            num_head_slots=specs[0].num_head_slots,
            state_content_bytes=specs[0].state_content_bytes,
            cache_dtype_str=cache_dtype_str_set.pop(),
            tokens_per_state=tokens_per_state_set.pop(),
            model_version=model_version_set.pop(),
            non_causal_multi_token_decode=non_causal_mtd_set.pop(),
        )
        for spec in specs:
            for f in fields(AttentionSpec):
                assert getattr(spec, f.name) == getattr(merged_spec, f.name), (
                    "All attention layers in the same KV cache group must have "
                    "the same attention spec."
                )
        return merged_spec


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
            num_head_slots=base.num_head_slots,
            state_content_bytes=base.state_content_bytes,
            tokens_per_state=base.tokens_per_state,
            sliding_window=base.sliding_window,
            attention_chunk_size=base.attention_chunk_size,
            non_causal=base.non_causal,
            rswa_window=rswa_windows.pop(),
        )


@dataclass(frozen=True, kw_only=True)
class ChunkedLocalAttentionSpec(AttentionSpec):
    attention_chunk_size: int

    def max_admission_blocks_per_request(
        self, max_in_flight_tokens: int, max_model_len: int
    ) -> int:
        """Per-request admission cap, in blocks.

        Single source of truth for both startup pool sizing
        (`max_memory_usage_bytes`) and the runtime admission gate, so requests
        admitted by startup can also be admitted at runtime.

        `max_in_flight_tokens` is the max tokens scheduled but not yet settled
        (one batch per concurrent step); see `VllmConfig.max_in_flight_tokens`.
        """
        # During chunked prefill, we hold KV for at most one chunk window plus
        # the in-flight tokens, since frees happen on the processed-token basis.
        num_tokens = min(
            self.attention_chunk_size + max_in_flight_tokens, max_model_len
        )
        return cdiv(num_tokens, self.block_size)

    def max_memory_usage_bytes(self, vllm_config: VllmConfig) -> int:
        max_blocks = self.max_admission_blocks_per_request(
            max_in_flight_tokens=vllm_config.max_in_flight_tokens,
            max_model_len=vllm_config.model_config.max_model_len,
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
    # The trailing edge of the window is extended by ``extra_retained_tokens``
    # so that those extra trailing tokens' blocks are retained (but not
    # attended). This is needed for multi-module spec decoding which can
    # re-prefill the last num_spec_prefill_tokens - 1 tokens from the end
    # of the sequence, and thus needs to delay freeing/caching of blocks.
    extra_retained_tokens: int = 0

    def max_admission_blocks_per_request(
        self, max_in_flight_tokens: int, max_model_len: int
    ) -> int:
        """Per-request admission cap, in blocks.

        Single source of truth for both startup pool sizing
        (`max_memory_usage_bytes`) and the runtime admission gate. Per-request
        real-held blocks plateau at this bound because
        `SlidingWindowManager.remove_skipped_blocks` runs from `allocate_slots`
        before each chunk's `get_num_blocks_to_allocate`.

        `max_in_flight_tokens` is the max tokens scheduled but not yet settled
        (one batch per concurrent step); see `VllmConfig.max_in_flight_tokens`.
        """
        # During chunked prefill, we hold KV for the last `sliding_window-1`
        # computed tokens plus the in-flight tokens (frees happen on the
        # processed-token basis); never more than `max_model_len`. An additional
        # `extra_retained_tokens` trailing tokens are kept alive below the
        # window for multi-module spec decoding, and must be accounted here too.
        num_tokens = min(
            self.sliding_window - 1 + self.extra_retained_tokens + max_in_flight_tokens,
            max_model_len,
        )
        # +1 because the sliding window may not start from the beginning of
        # the block. E.g. block size 4 and num_token 4 needs two blocks
        # [XXCD][EF] to store the 6-token window [CDEF].
        return cdiv(num_tokens, self.block_size) + 1

    def max_memory_usage_bytes(self, vllm_config: VllmConfig) -> int:
        assert vllm_config.parallel_config.decode_context_parallel_size == 1, (
            "DCP not support sliding window."
        )
        max_blocks = self.max_admission_blocks_per_request(
            max_in_flight_tokens=vllm_config.max_in_flight_tokens,
            max_model_len=vllm_config.model_config.max_model_len,
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
class CircularBufferSpec(AttentionSpec):
    """One block per request holding the raw keys of the token group that
    is still being compressed.

    ``block_size`` is the ring capacity. It must exceed the compression ratio
    by the speculative lookahead: a speculative step stores all of its rows,
    drafts included, before acceptance is known, while the next step still
    reads the open group's committed keys from the ring.
    """

    def max_memory_usage_bytes(self, vllm_config: VllmConfig) -> int:
        # The ring occupies one block per request for its whole lifetime.
        del vllm_config
        return self.page_size_bytes

    def max_num_blocks_per_req(self, vllm_config: VllmConfig, max_len: int) -> int:
        del vllm_config, max_len
        return 1

    def is_uniform_with_collection(
        self, kv_cache_specs: dict[str, KVCacheSpec]
    ) -> bool:
        return all(
            isinstance(spec, CircularBufferSpec) for spec in kv_cache_specs.values()
        )

    @property
    def prefix_cacheable(self) -> bool:
        return False


@dataclass(frozen=True, kw_only=True)
class SlidingWindowMLASpec(SlidingWindowSpec):
    """Sliding window attention with MLA cache format."""

    cache_dtype_str: str | None = None
    # DeepseekV4-only: see MLAAttentionSpec.model_version.
    alignment: int | None = None  # Default to None for no padding.
    model_version: str | None = None

    # MLA stores a single latent vector per state; there is no separate V.
    head_size_v: int = 0

    def __post_init__(self):
        assert self.model_version in (None, "deepseek_v4"), (
            f"Unsupported model version: {self.model_version}"
        )
        super().__post_init__()
        _apply_alignment_padding(self)

    @classmethod
    def merge(cls, specs: list[Self]) -> Self:
        assert all(isinstance(spec, SlidingWindowMLASpec) for spec in specs), (
            "All attention layers in the same KV cache group must be "
            "SlidingWindowMLASpec."
        )
        cache_dtype_str_set = set(spec.cache_dtype_str for spec in specs)
        tokens_per_state_set = set(spec.tokens_per_state for spec in specs)
        model_version_set = set(spec.model_version for spec in specs)
        sliding_window_set = set(spec.sliding_window for spec in specs)
        extra_retained_set = set(spec.extra_retained_tokens for spec in specs)
        assert (
            len(cache_dtype_str_set) == 1
            and len(tokens_per_state_set) == 1
            and len(model_version_set) == 1
            and len(sliding_window_set) == 1
            and len(extra_retained_set) == 1
        ), (
            "All attention layers in the same KV cache group must use the same "
            "quantization method, tokens per state, model version, sliding "
            "window size, and retained token count."
        )
        return cls(
            block_size=specs[0].block_size,
            num_kv_heads=specs[0].num_kv_heads,
            head_size=specs[0].head_size,
            dtype=specs[0].dtype,
            page_size_padded=specs[0].page_size_padded,
            num_head_slots=specs[0].num_head_slots,
            state_content_bytes=specs[0].state_content_bytes,
            sliding_window=sliding_window_set.pop(),
            extra_retained_tokens=extra_retained_set.pop(),
            cache_dtype_str=cache_dtype_str_set.pop(),
            tokens_per_state=tokens_per_state_set.pop(),
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
    dtypes: tuple[torch.dtype, ...]
    page_size_padded: int | None = None
    mamba_type: MambaAttentionBackendEnum = MambaAttentionBackendEnum.MAMBA2
    mamba_cache_mode: str = "none"
    num_speculative_blocks: int = 0
    num_prefill_checkpoint_blocks: int = 0
    num_heads: int = 1
    tokens_per_state: int = -1
    # False: the state is sharded across TP ranks (e.g. GDN). True: every TP
    # rank holds the full state (e.g. the replicated PLE conv state).
    tp_replicated: bool = False

    @property
    def state_content_size_bytes(self) -> int:
        return sum(
            prod(shape) * get_dtype_size(dtype)
            for (shape, dtype) in zip(self.shapes, self.dtypes)
        )

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
            return self.page_size_bytes * (
                2 + self.num_speculative_blocks + self.num_prefill_checkpoint_blocks
            )
        else:
            return self.page_size_bytes * (1 + self.num_speculative_blocks)

    def max_num_blocks_per_req(self, vllm_config: VllmConfig, max_len: int) -> int:
        # Mamba state is replicated across DCP/PCP ranks, never sharded, so
        # no CP scaling applies.
        if vllm_config.cache_config.mamba_cache_mode == "align":
            # Block table rows are position-indexed over the full sequence
            # even though only 2 + num_speculative_blocks state blocks are
            # resident at a time (earlier states are nulled out by
            # remove_skipped_blocks), so the row length must cover max_len
            # rather than max_memory_usage_bytes.
            return cdiv(max_len, self.block_size) + self.num_speculative_blocks
        return cdiv(self.max_memory_usage_bytes(vllm_config), self.page_size_bytes)

    def is_uniform_with_collection(
        self, kv_cache_specs: dict[str, KVCacheSpec]
    ) -> bool:
        return all(
            isinstance(spec, MambaSpec)
            and spec.num_speculative_blocks == self.num_speculative_blocks
            and spec.num_prefill_checkpoint_blocks == self.num_prefill_checkpoint_blocks
            and spec.page_size_bytes == self.page_size_bytes
            and spec.tp_replicated == self.tp_replicated
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
            num_head_slots=specs[0].num_head_slots,
            state_content_bytes=specs[0].state_content_bytes,
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
    def prefix_cacheable(self) -> bool:
        return all(spec.prefix_cacheable for spec in self.kv_cache_specs.values())

    @property
    def first_spec(self) -> KVCacheSpec:
        """Return the first spec in the group."""
        return next(iter(self.kv_cache_specs.values()))

    @property
    def page_size_bytes(self) -> int:
        return sum(spec.page_size_bytes for spec in self.kv_cache_specs.values())

    def max_memory_usage_bytes(self, vllm_config: VllmConfig) -> int:
        max_num_pages = max(
            cdiv(spec.max_memory_usage_bytes(vllm_config), spec.page_size_bytes)
            for spec in self.kv_cache_specs.values()
        )
        return max_num_pages * self.page_size_bytes

    def max_num_blocks_per_req(self, vllm_config: VllmConfig, max_len: int) -> int:
        # Metadata builders are constructed from the per-layer spec, so the base
        # cdiv(max_len, block_size) would drop its DCP sharding and size the
        # block table wider than those builders expect.
        widths = {
            spec.max_num_blocks_per_req(vllm_config, max_len)
            for spec in self.kv_cache_specs.values()
        }
        assert len(widths) == 1, (
            "All layers in the same KV cache group must need the same number "
            f"of block table entries, got {sorted(widths)}."
        )
        return next(iter(widths))

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

    def get_max_layers_per_page_size(self) -> int:
        """Max number of layers sharing a page size. For a balanced bucket
        this equals the number of repetitions of the layer pattern."""
        return Counter(
            spec.page_size_bytes for spec in self.kv_cache_specs.values()
        ).most_common(1)[0][1]

    def max_memory_usage_pages(self, vllm_config: VllmConfig) -> int:
        return max(
            cdiv(spec.max_memory_usage_bytes(vllm_config), spec.page_size_bytes)
            for spec in self.kv_cache_specs.values()
        )


def iter_layer_specs(kv_cache_spec: KVCacheSpec) -> Collection[KVCacheSpec]:
    """The per-layer specs a KV cache group spec covers.

    ``UniformTypeKVCacheSpecs`` groups keep one spec per layer; every other
    spec describes its group on its own. Returns the layer specs either way so
    callers do not have to special-case the wrapper.
    """
    if isinstance(kv_cache_spec, UniformTypeKVCacheSpecs):
        return kv_cache_spec.kv_cache_specs.values()
    return (kv_cache_spec,)


def is_full_attention_spec(kv_cache_spec: KVCacheSpec) -> bool:
    """Whether a KV cache group spec is (or wraps) full attention.

    ``UniformTypeKVCacheSpecs`` is not itself a ``FullAttentionSpec``, so a bare
    isinstance check misses groups that carry the wrapper -- DeepSeek-V4's MLA
    layers, or any model taking the ``UniformTypeKVCacheSpecs.from_specs`` path.

    Every layer must be full attention: a group holding a recycling
    (sliding-window) layer has no stable slot layout, so callers that key data
    by slot cannot use it.
    """
    layer_specs = iter_layer_specs(kv_cache_spec)
    return len(layer_specs) > 0 and all(
        isinstance(spec, FullAttentionSpec) for spec in layer_specs
    )


def get_kv_cache_spec_kind(kv_cache_spec: KVCacheSpec) -> KVCacheSpecKind:
    if isinstance(kv_cache_spec, UniformTypeKVCacheSpecs):
        inner_kinds = {
            get_kv_cache_spec_kind(spec)
            for spec in kv_cache_spec.kv_cache_specs.values()
        }
        if len(inner_kinds) == 1:
            return next(iter(inner_kinds))
        # A group is only formed when all members share one registered
        # uniform_type_base_spec, so UNKNOWN would discard what the merge
        # already established.
        base_specs = {
            KVCacheSpecRegistry.get_uniform_type_base_spec(spec)
            for spec in kv_cache_spec.kv_cache_specs.values()
        }
        if len(base_specs) == 1 and next(iter(base_specs)) is FullAttentionSpec:
            return KVCacheSpecKind.FULL_ATTENTION
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

    Placement of a set of same-shaped layers in the KV cache allocation.
    Layer ``layers[l]``'s page for block ``b`` starts at
    ``offset + l * layer_stride + b * block_stride`` bytes into the backing
    allocation of ``size`` bytes. Layer-outermost layouts give each layer a
    contiguous region (``layer_stride = page * num_blocks``,
    ``block_stride = page``); block-outermost layouts make each block a
    block of all layers' pages (``layer_stride = page``, ``block_stride`` =
    the packed block). Tensors whose address ranges overlap
    alias the same bytes: cache groups overlay each other, which is sound
    because a block ID is owned by one group at a time.
    """

    size: int  # total size of the backing allocation in bytes
    layers: list[str]  # layer names in L order
    layer_stride: int
    block_stride: int
    offset: int = 0  # byte offset of layers[0]'s block 0


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
    # Whether this group is part of the externally transferable KV state.
    enable_kv_transfer: bool = True


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
    prefix_cache_retention_interval: int | None = None
    """Resolved retention policy for local prefix-cache checkpoints."""
    kv_cache_layout: str | None = None
    """The KV cache layout resolved by the engine core, adopted by all workers."""

    @cached_property
    def transfer_group_ids(self) -> tuple[int, ...]:
        """IDs of cache groups that participate in external KV transfer."""
        return tuple(
            group_id
            for group_id, group in enumerate(self.kv_cache_groups)
            if group.enable_kv_transfer
        )

    @cached_property
    def transfer_groups(self) -> tuple[KVCacheGroupSpec, ...]:
        """Cache groups that participate in external KV transfer."""
        return tuple(
            self.kv_cache_groups[group_id] for group_id in self.transfer_group_ids
        )

    @cached_property
    def transfer_group_index_by_layer(self) -> dict[str, int]:
        """Transfer-group tuple index for each participating layer."""
        return {
            layer_name: group_index
            for group_index, group in enumerate(self.transfer_groups)
            for layer_name in group.layer_names
        }

    def select_transfer_block_ids(
        self, block_ids: Sequence[list[int]]
    ) -> tuple[list[int], ...]:
        """Select block IDs for externally transferable cache groups."""
        if len(block_ids) != len(self.kv_cache_groups):
            raise ValueError(
                f"Expected {len(self.kv_cache_groups)} KV cache groups, "
                f"got {len(block_ids)}."
            )
        return tuple(block_ids[group_id] for group_id in self.transfer_group_ids)

    @property
    def has_mamba_layers(self) -> bool:
        return any(
            isinstance(spec, MambaSpec)
            for group in self.kv_cache_groups
            for spec in iter_layer_specs(group.kv_cache_spec)
        )

    @property
    def has_mixed_precision_kv_cache(self) -> bool:
        """Whether attention groups store their KV cache at more than one precision."""
        kv_cache_precisions: set[tuple[torch.dtype, KVQuantMode]] = set()
        for group in self.kv_cache_groups:
            kv_cache_precisions.update(
                (spec.dtype, spec.kv_quant_mode)
                for spec in iter_layer_specs(group.kv_cache_spec)
                if isinstance(spec, AttentionSpec)
            )
        return len(kv_cache_precisions) > 1

    @property
    def needs_kv_cache_zeroing(self) -> bool:
        """Whether newly allocated KV cache blocks must be zeroed before use.

        Required for Mamba layers, whose state is read before it is fully written
        (#35219), and for mixed-precision caches, where a block reused across
        groups can be reinterpreted under a different precision and decode stale
        bytes to NaN/Inf. Uniform-precision caches skip zeroing.
        """
        return self.has_mamba_layers or self.has_mixed_precision_kv_cache
