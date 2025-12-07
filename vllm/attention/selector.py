# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import inspect
from functools import cache
from typing import cast, get_args

import torch

from vllm.attention.backends.abstract import AttentionBackend
from vllm.attention.backends.registry import (
    MAMBA_TYPE_TO_BACKEND_MAP,
    MambaAttentionBackendEnum,
)
from vllm.config.cache import CacheDType
from vllm.logger import init_logger
from vllm.utils.import_utils import resolve_obj_by_qualname

logger = init_logger(__name__)


def get_attn_backend(
    head_size: int,
    dtype: torch.dtype,
    kv_cache_dtype: str | None,
    block_size: int | None,
    use_mla: bool = False,
    has_sink: bool = False,
    use_sparse: bool = False,
    attn_type: str | None = None,
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
    backend_enum = vllm_config.attention_config.backend

    return _cached_get_attn_backend(
        backend=backend_enum,
        head_size=head_size,
        dtype=dtype,
        kv_cache_dtype=cast(CacheDType | None, kv_cache_dtype),
        block_size=block_size,
        use_mla=use_mla,
        has_sink=has_sink,
        use_sparse=use_sparse,
        attn_type=attn_type,
    )


@cache
def _cached_get_attn_backend(
    backend,
    head_size: int,
    dtype: torch.dtype,
    kv_cache_dtype: CacheDType | None,
    block_size: int | None,
    use_mla: bool = False,
    has_sink: bool = False,
    use_sparse: bool = False,
    attn_type: str | None = None,
) -> type[AttentionBackend]:
    from vllm.platforms import current_platform

    sig = inspect.signature(current_platform.get_attn_backend_cls)
    if "use_v1" in sig.parameters:
        logger.warning_once(
            "use_v1 parameter for get_attn_backend_cls is deprecated and will "
            "be removed in v0.13.0 or v1.0.0, whichever is soonest. Please "
            "remove it from your plugin code."
        )
        attention_cls = current_platform.get_attn_backend_cls(
            backend,
            head_size,
            dtype,
            kv_cache_dtype,
            block_size,
            True,  # use_v1
            use_mla,
            has_sink,
            use_sparse,
            attn_type,
        )
    else:
        attention_cls = current_platform.get_attn_backend_cls(
            backend,
            head_size,
            dtype,
            kv_cache_dtype,
            block_size,
            use_mla,
            has_sink,
            use_sparse,
            attn_type,
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
    mamba_type: str,
) -> type[AttentionBackend]:
    """Select which mamba attention backend to use and lazily import it."""
    return _cached_get_mamba_attn_backend(mamba_type)


@cache
def _cached_get_mamba_attn_backend(
    mamba_type: str,
) -> type[AttentionBackend]:
    assert mamba_type and isinstance(mamba_type, str)

    selected_backend = None
    try:
        backend_name = MAMBA_TYPE_TO_BACKEND_MAP[mamba_type]
        selected_backend = MambaAttentionBackendEnum[backend_name]
    except KeyError as e:
        raise ValueError(
            f"Invalid mamba attention backend type: '{backend_name}'. Valid "
            f"backends are: {list(MambaAttentionBackendEnum.__members__.keys())}"
        ) from e

    mamba_attn_backend = selected_backend.get_class()
    return mamba_attn_backend
