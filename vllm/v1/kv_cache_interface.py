# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from enum import Enum, IntEnum
from fractions import Fraction
from functools import cached_property
from math import prod
from typing import TYPE_CHECKING, ClassVar, Self

import torch

from vllm.logger import init_logger
from vllm.utils.math_utils import cdiv, round_up
from vllm.utils.torch_utils import get_dtype_size
from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum
from vllm.v1.kv_cache_layout import _DIM_B, _DIM_L, KVCacheLayout

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.attention.backend import MultipleOf
    from vllm.v1.core.single_type_kv_cache_manager import SingleTypeKVCacheManager

logger = init_logger(__name__)

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
    ULTRAQUANT_4BIT = 11  # packed FP4 values + UE8M0 group scales

    # Backend self-manages kernel dispatch. Reads as quantized but matches no
    # generic kernel mode (is_per_token_head / is_nvfp4 / is_turboquant all
    # False). Registered handlers return this when they own kernel selection
    # rather than reusing an upstream kernel path. See KVCacheDTypeHandler.
    CUSTOM = 12

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

    @property
    def is_ultraquant(self) -> bool:
        """True for the UltraQuant 4-bit KV-cache format."""
        return self == KVQuantMode.ULTRAQUANT_4BIT

    @property
    def is_custom_mode(self) -> bool:
        """True when a platform backend fully self-manages kernel dispatch.

        The attention selector should defer to the backend's own kernel
        selection rather than a generic per-token-head / fp8 / turboquant
        path. Distinct from NONE (which means non-quantized): CUSTOM is
        quantized, just with no generic kernel mode.
        """
        return self == KVQuantMode.CUSTOM


def get_kv_quant_mode(kv_cache_dtype: str) -> KVQuantMode:
    """Map a ``kv_cache_dtype`` string to a :class:`KVQuantMode`."""
    from vllm.config.cache import get_kv_cache_dtype_handler

    handler = get_kv_cache_dtype_handler(kv_cache_dtype)
    if handler is not None:
        return handler.quant_mode()
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
    if kv_cache_dtype == "ultraquant_4bit":
        return KVQuantMode.ULTRAQUANT_4BIT
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


class SparseCacheRole(str, Enum):
    SPARSE = "sparse"
    INDEXER = "indexer"


@dataclass(frozen=True)
class ManagerConfig:
    """Allocation policy of a ``SingleTypeKVCacheManager``.

    Each manager subclasses this as its nested ``Config``, so a config also
    names its manager. Layers whose configs compare equal may share a block
    table.
    """

    manager_cls: ClassVar[type[SingleTypeKVCacheManager]]


@dataclass(frozen=True)
class KVCacheSpec:
    """The tensor layout of one layer's KV cache.

    Allocation policy (windows, recycling, prefix caching) lives on the
    layer's ``SingleTypeKVCacheManager.Config``, see ``KVCacheLayerConfig``.
    """

    # number of tokens in a block
    block_size: int
    dcp_sharded: bool = field(default=False, kw_only=True)
    """Whether DCP shards this cache's token positions across ranks."""

    block_stride_alignment: int | MultipleOf | None = field(default=None, kw_only=True)
    """Required byte alignment between physical blocks, including packed layers.
    ``MultipleOf(n)``: whole kernel blocks, when blocks are split into kernel
    blocks of up to ``n`` tokens."""

    def __post_init__(self):
        alignment = self.block_stride_alignment
        if isinstance(alignment, int) and alignment <= 0:
            raise ValueError("block_stride_alignment must be positive")

    def get_block_stride_alignment(self) -> int:
        """Bytes the stride between physical blocks must be a multiple of."""
        from vllm.v1.attention.backend import MultipleOf

        alignment = self.block_stride_alignment
        if not isinstance(alignment, MultipleOf):
            return alignment or 1
        kernel_block_size = math.gcd(alignment.base, self.block_size)
        if kernel_block_size == self.block_size:
            return 1
        return self.copy_with_new_block_size(kernel_block_size).page_size_bytes

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
        """The size of a page with `block_size` tokens in bytes.

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

    def copy_with_new_block_size(self, block_size: int) -> Self:
        """Create a new KVCacheSpec from self but replacing the block size."""
        return replace(self, block_size=block_size)

    @property
    def has_layer_views(self) -> bool:
        """Whether generic allocation creates per-layer cache views.

        Specs without a per-layer shape keep the raw backing tensor and lay it
        out themselves. They also have no attention module of their own, so
        they take no part in attention-backend or kernel-block selection.
        """
        return True


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
class HiSparsePageSpec(KVCacheSpec):
    """Raw pages HiSparse lays out itself (hot and GPU-resident caches)."""

    page_size: int

    @property
    def page_size_bytes(self) -> int:
        return self.page_size

    @property
    def has_layer_views(self) -> bool:
        return False


@dataclass(frozen=True, kw_only=True)
class AttentionSpec(KVCacheSpec):
    dcp_sharded: bool = True
    max_tp_shards: int | None = None
    """Distinct shards this cache splits into across TP; TP ranks beyond this
    hold replicas. None: unknown, treated as not replicated."""
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
        super().__post_init__()
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


def _apply_alignment_padding(spec: MLAAttentionSpec):
    if spec.alignment is None:
        return
    actual_page_size = spec.real_page_size_bytes
    padded_page_size = round_up(actual_page_size, spec.alignment)
    if padded_page_size != actual_page_size:
        object.__setattr__(spec, "page_size_padded", padded_page_size)


@dataclass(frozen=True, kw_only=True)
class MLAAttentionSpec(AttentionSpec):
    # TODO(Lucas/Chen): less hacky way to do this
    cache_dtype_str: str | None = None
    topk_buffer_width: int | None = None
    """Width of the model's sparse top-k indices buffer, including padding."""
    # DeepseekV4 only fields. Non-DeepseekV4 MLA models leave these at defaults.
    alignment: int | None = None  # Default to None for no padding.
    model_version: str | None = None
    cache_role: SparseCacheRole = SparseCacheRole.SPARSE
    is_index_group_leader: bool = False
    # Group capability enabled when any member flattens a non-causal query block
    # into decode rows. Runtime metadata still selects causal vs. non-causal mode.
    non_causal_multi_token_decode: bool = False
    # MLA stores a single latent vector per state; there is no separate V.
    head_size_v: int = 0

    def __post_init__(self):
        assert self.model_version in (None, "deepseek_v4"), (
            f"Unsupported model version: {self.model_version}"
        )
        super().__post_init__()
        _apply_alignment_padding(self)


@dataclass(frozen=True, kw_only=True)
class HiddenStateCacheSpec(MLAAttentionSpec):
    """Marker for hidden-state cache layers used by extract_hidden_states."""

    pass


@dataclass(frozen=True)
class MambaSpec(KVCacheSpec):
    shapes: tuple[tuple[int, ...], ...]
    dtypes: tuple[torch.dtype, ...]
    page_size_padded: int | None = None
    mamba_type: MambaAttentionBackendEnum = MambaAttentionBackendEnum.MAMBA2
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
    def real_page_size_bytes(self) -> int:
        return self.state_content_size_bytes

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


def get_mamba_prefill_checkpoint_position(
    num_tokens: int,
    hash_block_size: int,
    drop_eagle_block: bool,
) -> int:
    """Return the reusable Mamba checkpoint boundary for a prefill."""
    # Without EAGLE, leave one token to recompute on resend.
    # EAGLE's block drop already leaves tokens to recompute.
    proof_limit = num_tokens if drop_eagle_block else num_tokens - 1
    checkpoint_position = proof_limit // hash_block_size * hash_block_size
    if drop_eagle_block:
        checkpoint_position -= hash_block_size
    return max(checkpoint_position, 0)


def is_mamba_prefill_checkpoint_valid(
    query_start: int,
    query_end: int,
    checkpoint_position: int,
    hash_block_size: int,
    mamba_block_size: int,
    checkpoint_alignment: int | None,
) -> bool:
    """Whether a backend can export the checkpoint in this query."""
    if checkpoint_alignment is None:
        return False
    assert checkpoint_alignment > 0

    initial_state_col = (query_start - 1) // mamba_block_size
    checkpoint_col = cdiv(query_end, mamba_block_size) - 2
    return (
        query_start % hash_block_size == 0
        and checkpoint_col > initial_state_col
        and query_start + hash_block_size <= checkpoint_position
        and query_start < checkpoint_position < query_end
        and (checkpoint_position - query_start) % checkpoint_alignment == 0
    )


@dataclass
class KVCacheTensor:
    """A class for specifying how the workers should initialize the KV cache.

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
    host_resident: bool = False


class KVCacheGroupRole(str, Enum):
    DEFAULT = "default"
    HISPARSE_INDEXER = "hisparse_indexer"


@dataclass(frozen=True)
class KVCacheLayerConfig:
    """What one layer needs from the KV cache: the manager that allocates its
    blocks, that manager's policy, and the layer's tensor layout."""

    manager_config: ManagerConfig
    spec: KVCacheSpec
    is_draft: bool = field(default=False, kw_only=True, compare=False)
    """Whether a draft model owns this layer; excluded from equality so it never
    affects grouping."""

    @property
    def manager_cls(self) -> type[SingleTypeKVCacheManager]:
        return self.manager_config.manager_cls

    def replace_spec(self, **changes) -> KVCacheLayerConfig:
        return replace(self, spec=replace(self.spec, **changes))

    def max_memory_usage_bytes(self, vllm_config: VllmConfig) -> int:
        blocks = self.manager_cls.max_blocks_per_request(
            self.manager_config,
            self.spec.block_size,
            self.spec.dcp_sharded,
            vllm_config,
        )
        return blocks * self.spec.page_size_bytes


@dataclass
class KVCacheGroupSpec:
    """A group of model layers that share one KV cache block table.

    The layers share a manager and its policy, the block size and the DCP
    sharding, and are regarded as one layer by the KV cache manager. Their
    cache layouts may differ.
    """

    # The cache layout of each model layer in the group
    kv_cache_specs: dict[str, KVCacheSpec]
    # The allocation policy, which names the manager allocating the blocks
    manager_config: ManagerConfig
    # Tokens per block of the group's block table
    block_size: int
    # Whether DCP shards the group's token positions across ranks
    dcp_sharded: bool = False
    # Whether this group holds draft layers and takes the EAGLE last-block drop.
    is_eagle_group: bool = False
    # Host groups use the dedicated HiSparse pool; others share the device pool.
    host_resident: bool = False
    # Whether this group is part of the externally transferable KV state.
    # Ephemeral accelerator-side replicas are rebuilt from their transferable
    # source group after a connector load.
    enable_kv_transfer: bool = True
    role: KVCacheGroupRole = KVCacheGroupRole.DEFAULT

    @classmethod
    def from_layers(
        cls,
        layer_configs: Mapping[str, KVCacheLayerConfig],
        **kwargs,
    ) -> KVCacheGroupSpec:
        """A group of layers sharing a manager policy and block size."""
        first = next(iter(layer_configs.values()))
        assert all(
            lc.manager_config == first.manager_config
            and lc.spec.block_size == first.spec.block_size
            and lc.spec.dcp_sharded == first.spec.dcp_sharded
            for lc in layer_configs.values()
        ), "Layers in a KV cache group must share a manager policy and block size."
        return cls(
            {name: lc.spec for name, lc in layer_configs.items()},
            first.manager_config,
            first.spec.block_size,
            first.spec.dcp_sharded,
            **kwargs,
        )

    @property
    def layer_names(self) -> list[str]:
        return list(self.kv_cache_specs)

    @property
    def page_size_bytes(self) -> int:
        """Bytes one block of the group occupies across all its layers."""
        return sum(spec.page_size_bytes for spec in self.kv_cache_specs.values())

    @property
    def manager_cls(self) -> type[SingleTypeKVCacheManager]:
        return self.manager_config.manager_cls

    @property
    def block_table_key(self) -> tuple:
        """Groups with equal keys allocate blocks identically and can be looked
        up together."""
        return self.manager_config, self.block_size, self.dcp_sharded

    @property
    def prefix_cacheable(self) -> bool:
        return self.manager_cls.prefix_cacheable(self.manager_config)

    @property
    def prefix_replay_tokens(self) -> int:
        return self.manager_cls.prefix_replay_tokens(self.manager_config)

    @property
    def uses_slot_mapping(self) -> bool:
        return self.manager_cls.uses_slot_mapping

    @property
    def block_table_token_alignment(self) -> int | None:
        return self.manager_cls.block_table_token_alignment

    @property
    def is_full_attention(self) -> bool:
        """Whether every token keeps its KV for the request's lifetime."""
        return self.manager_cls.is_full_attention

    @property
    def sliding_window(self) -> int | None:
        return self.manager_cls.attention_window(self.manager_config)

    @property
    def kind(self) -> KVCacheSpecKind:
        return self.manager_cls.kv_cache_kind(self.kv_cache_specs.values())

    def max_blocks_per_request(self, vllm_config: VllmConfig) -> int:
        """Blocks one request at ``max_model_len`` holds in this group."""
        return self.manager_cls.max_blocks_per_request(
            self.manager_config, self.block_size, self.dcp_sharded, vllm_config
        )

    def max_memory_usage_bytes(self, vllm_config: VllmConfig) -> int:
        return self.max_blocks_per_request(vllm_config) * self.page_size_bytes

    def max_num_blocks_per_req(
        self, vllm_config: VllmConfig, max_len: int | None = None
    ) -> int:
        """The row length of the worker-side block table for this group, sized
        for ``max_len`` tokens (default: ``max_model_len``)."""
        if max_len is None:
            max_len = vllm_config.model_config.max_model_len
        return self.manager_cls.max_num_blocks_per_req(
            self.manager_config, self.block_size, self.dcp_sharded, vllm_config, max_len
        )


@dataclass
class KVCacheConfig:
    """The KV cache configuration of a model."""

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
    see `_get_kv_cache_groups_uniform_page_size` for more details.
    """
    prefix_cache_retention_interval: int | None = None
    """Resolved retention policy for local prefix-cache checkpoints."""
    kv_cache_layout: str | None = None
    """The KV cache layout resolved by the engine core, adopted by all workers."""
    hash_block_size: int | None = None
    """Tokens per prefix-cache block hash, resolved by the engine core."""
    cache_hit_alignment_tokens: int | None = None
    """Token granularity of prefix-cache hits, resolved by the engine core."""
    hisparse_host_num_blocks: int | None = None
    """Capacity of the dedicated HiSparse host-block manager, when enabled."""

    hisparse_host_block_stride: int | None = None
    """Physical bytes between consecutive HiSparse host blocks."""

    hisparse_shared_host_pool: bool = False
    """Whether local TP ranks share one physical HiSparse host pool."""

    kv_tp_replicas: int = 1
    """Consecutive TP ranks holding identical KV for every layer (1: none)."""

    @cached_property
    def transfer_group_ids(self) -> tuple[int, ...]:
        """IDs of cache groups that participate in external KV transfer."""
        return tuple(
            group_id
            for group_id, group in enumerate(self.kv_cache_groups)
            if group.enable_kv_transfer
        )

    @cached_property
    def prefix_cacheable_group_ids(self) -> tuple[int, ...]:
        """IDs of transferable groups eligible for hash-addressed stores."""
        return tuple(
            group_id
            for group_id in self.transfer_group_ids
            if self.kv_cache_groups[group_id].prefix_cacheable
        )

    @cached_property
    def prefix_cacheable_groups(self) -> tuple[KVCacheGroupSpec, ...]:
        """Transferable groups eligible for hash-addressed stores."""
        return tuple(
            self.kv_cache_groups[group_id]
            for group_id in self.prefix_cacheable_group_ids
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
    def host_group_ids(self) -> tuple[int, ...]:
        return tuple(
            group_id
            for group_id, group in enumerate(self.kv_cache_groups)
            if group.host_resident
        )

    def num_blocks_of(self, tensor: KVCacheTensor) -> int:
        """Number of blocks addressable by the pool backing ``tensor``."""
        if not tensor.host_resident:
            return self.num_blocks
        assert self.hisparse_host_num_blocks is not None
        return self.hisparse_host_num_blocks

    @property
    def has_mamba_layers(self) -> bool:
        return any(
            isinstance(spec, MambaSpec)
            for group in self.kv_cache_groups
            for spec in group.kv_cache_specs.values()
        )

    @property
    def has_mixed_precision_kv_cache(self) -> bool:
        """Whether device attention caches use more than one precision."""
        kv_cache_precisions: set[tuple[torch.dtype, KVQuantMode]] = set()
        for group in self.kv_cache_groups:
            if group.host_resident:
                continue
            kv_cache_precisions.update(
                (spec.dtype, spec.kv_quant_mode)
                for spec in group.kv_cache_specs.values()
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
