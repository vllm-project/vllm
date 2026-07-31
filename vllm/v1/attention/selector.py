# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from functools import cache
from typing import TYPE_CHECKING, NamedTuple, cast, get_args

import torch

import vllm.envs as envs
from vllm.config.cache import CacheDType
from vllm.logger import init_logger
from vllm.utils.import_utils import resolve_obj_by_qualname
from vllm.v1.attention.backend import AttentionBackend, AttentionType
from vllm.v1.attention.backends.registry import (
    MambaAttentionBackendEnum,
)

if TYPE_CHECKING:
    from vllm.v1.kv_cache_interface import KVCacheSpecKind

logger = init_logger(__name__)


class AttentionSelectorConfig(NamedTuple):
    head_size: int
    dtype: torch.dtype
    kv_cache_dtype: CacheDType | None
    block_size: int | None
    use_mla: bool = False
    has_sink: bool = False
    use_sparse: bool = False
    use_mm_prefix: bool = False
    use_per_head_quant_scales: bool = False
    attn_type: str = AttentionType.DECODER
    has_sliding_window: bool = False
    use_non_causal: bool = False
    use_batch_invariant: bool = False
    use_kv_connector: bool = False
    use_pcp: bool = False

    def __repr__(self):
        return (
            f"AttentionSelectorConfig(head_size={self.head_size}, "
            f"dtype={self.dtype}, "
            f"kv_cache_dtype={self.kv_cache_dtype}, "
            f"block_size={self.block_size}, "
            f"use_mla={self.use_mla}, "
            f"has_sink={self.has_sink}, "
            f"use_sparse={self.use_sparse}, "
            f"use_mm_prefix={self.use_mm_prefix}, "
            f"use_per_head_quant_scales={self.use_per_head_quant_scales}, "
            f"attn_type={self.attn_type}, "
            f"has_sliding_window={self.has_sliding_window}, "
            f"use_non_causal={self.use_non_causal}, "
            f"use_batch_invariant={self.use_batch_invariant}, "
            f"use_kv_connector={self.use_kv_connector}, "
            f"use_pcp={self.use_pcp})"
        )


def get_attn_spec_kind(
    use_mla: bool,
    has_sliding_window: bool,
    attn_type: str,
) -> "KVCacheSpecKind":
    """Derive the KV-cache group kind a layer belongs to from its signals.

    Mirrors ``get_kv_cache_spec_kind`` (which derives the kind from the
    produced ``KVCacheSpec``) so users can target groups by kind when
    setting ``AttentionConfig.backend_per_kind``.

    ``SINK_FULL_ATTENTION`` is intentionally not derived here: it is produced
    only by the ``StaticSinkAttention`` layer, whereas a plain ``Attention``
    layer with attention sinks (e.g. gpt-oss) still yields a
    ``FullAttentionSpec``/``SlidingWindowSpec``. Sinks therefore do not change
    the kind.

    Args:
        use_mla: Whether the layer uses multi-head latent attention.
        has_sliding_window: Whether the layer applies a sliding window.
        attn_type: The layer's ``AttentionType``.

    Returns:
        The ``KVCacheSpecKind`` the layer maps to.
    """
    from vllm.v1.kv_cache_interface import KVCacheSpecKind

    if attn_type == AttentionType.ENCODER_ONLY:
        return KVCacheSpecKind.ENCODER_ONLY_ATTENTION
    if attn_type == AttentionType.ENCODER_DECODER:
        return KVCacheSpecKind.CROSS_ATTENTION
    if use_mla:
        if has_sliding_window:
            return KVCacheSpecKind.SLIDING_WINDOW_MLA
        return KVCacheSpecKind.MLA_ATTENTION
    if has_sliding_window:
        return KVCacheSpecKind.SLIDING_WINDOW
    return KVCacheSpecKind.FULL_ATTENTION


def get_attn_backend(
    head_size: int,
    dtype: torch.dtype,
    kv_cache_dtype: str | None,
    use_mla: bool = False,
    has_sink: bool = False,
    use_sparse: bool = False,
    use_mm_prefix: bool = False,
    use_per_head_quant_scales: bool = False,
    attn_type: str | None = None,
    num_heads: int | None = None,
    has_sliding_window: bool = False,
) -> type[AttentionBackend]:
    """Selects which attention backend to use and lazily imports it."""

    if kv_cache_dtype is not None:
        valid_cache_dtypes = get_args(CacheDType)
        assert kv_cache_dtype in valid_cache_dtypes, (
            f"Invalid kv_cache_dtype: {kv_cache_dtype}. "
            f"Valid values are: {valid_cache_dtypes}"
        )

    from vllm.config import get_current_vllm_config

    vllm_config = get_current_vllm_config()

    cache_config = vllm_config.cache_config
    block_size: int | None
    if cache_config is not None and cache_config.user_specified_block_size:
        block_size = cache_config.block_size
    else:
        block_size = None

    kv_transfer_config = vllm_config.kv_transfer_config
    use_kv_connector = (
        kv_transfer_config is not None and kv_transfer_config.is_kv_transfer_instance
    )

    attn_type = attn_type or AttentionType.DECODER
    attn_selector_config = AttentionSelectorConfig(
        head_size=head_size,
        dtype=dtype,
        kv_cache_dtype=cast(CacheDType | None, kv_cache_dtype),
        block_size=block_size,
        use_mla=use_mla,
        has_sink=has_sink,
        use_sparse=use_sparse,
        use_mm_prefix=use_mm_prefix,
        use_per_head_quant_scales=use_per_head_quant_scales,
        attn_type=attn_type,
        has_sliding_window=has_sliding_window,
        use_non_causal=vllm_config.attention_config.use_non_causal,
        use_batch_invariant=envs.VLLM_BATCH_INVARIANT,
        use_kv_connector=use_kv_connector,
        use_pcp=vllm_config.parallel_config.prefill_context_parallel_size > 1,
    )

    # A per-KV-group override (keyed by KVCacheSpecKind) takes precedence over
    # the global backend; kinds not present in the map fall back to it.
    attention_config = vllm_config.attention_config
    backend = attention_config.backend
    if attention_config.backend_per_kind:
        kind = get_attn_spec_kind(
            use_mla=use_mla,
            has_sliding_window=has_sliding_window,
            attn_type=attn_type,
        )
        backend = attention_config.backend_per_kind.get(kind.value, backend)

    return _cached_get_attn_backend(
        backend=backend,
        attn_selector_config=attn_selector_config,
        num_heads=num_heads,
    )


@cache
def _cached_get_attn_backend(
    backend,
    attn_selector_config: AttentionSelectorConfig,
    num_heads: int | None = None,
) -> type[AttentionBackend]:
    from vllm.platforms import current_platform

    attention_cls = current_platform.get_attn_backend_cls(
        backend,
        attn_selector_config=attn_selector_config,
        num_heads=num_heads,
    )
    if not attention_cls:
        raise ValueError(
            f"Invalid attention backend for {current_platform.device_name}"
        )
    backend = resolve_obj_by_qualname(attention_cls)

    # Adjust kv cache layout if the selected backend requires a specific one
    required_layout = backend.get_required_kv_cache_layout()
    if required_layout is not None:
        from vllm.v1.attention.backends.utils import set_kv_cache_layout

        set_kv_cache_layout(required_layout)
        logger.info(
            "Using %s KV cache layout for %s backend.",
            required_layout,
            backend.get_name(),
        )

    return backend


def get_mamba_attn_backend(
    mamba_type: MambaAttentionBackendEnum,
) -> type[AttentionBackend]:
    """Select which mamba attention backend to use and lazily import it."""
    return _cached_get_mamba_attn_backend(mamba_type)


@cache
def _cached_get_mamba_attn_backend(
    mamba_type: MambaAttentionBackendEnum,
) -> type[AttentionBackend]:
    assert mamba_type and isinstance(mamba_type, MambaAttentionBackendEnum)

    mamba_attn_backend = mamba_type.get_class()
    if envs.VLLM_BATCH_INVARIANT and not mamba_attn_backend.supports_batch_invariance():
        raise RuntimeError(
            "VLLM batch_invariant mode is not supported for "
            f"{mamba_attn_backend.get_name()}."
        )
    return mamba_attn_backend
