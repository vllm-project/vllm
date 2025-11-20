# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import inspect
import os
from collections.abc import Generator
from contextlib import contextmanager
from functools import cache
from typing import cast, get_args

import torch

import vllm.envs as envs
from vllm.attention.backends.abstract import AttentionBackend
from vllm.attention.backends.registry import (
    MAMBA_TYPE_TO_BACKEND_MAP,
    AttentionBackendEnum,
    MambaAttentionBackendEnum,
)
from vllm.config.cache import CacheDType
from vllm.logger import init_logger
from vllm.utils import STR_BACKEND_ENV_VAR
from vllm.utils.import_utils import resolve_obj_by_qualname

logger = init_logger(__name__)


def get_env_variable_attn_backend() -> AttentionBackendEnum | None:
    """
    Get the backend override specified by the vLLM attention
    backend environment variable, if one is specified.

    Returns:

    * AttentionBackendEnum value if an override is specified
    * None otherwise
    """
    backend_name = os.environ.get(STR_BACKEND_ENV_VAR)
    return None if backend_name is None else AttentionBackendEnum[backend_name]


# Global state allows a particular choice of backend
# to be forced, overriding the logic which auto-selects
# a backend based on system & workload configuration
# (default behavior if this variable is None)
#
# THIS SELECTION TAKES PRECEDENCE OVER THE
# VLLM_ATTENTION_BACKEND ENVIRONMENT VARIABLE
forced_attn_backend: AttentionBackendEnum | None = None


def global_force_attn_backend(attn_backend: AttentionBackendEnum | None) -> None:
    """
    Force all attention operations to use a specified backend.

    Passing `None` for the argument re-enables automatic
    backend selection.,

    Arguments:

    * attn_backend: backend selection (None to revert to auto)
    """
    global forced_attn_backend
    forced_attn_backend = attn_backend


def get_global_forced_attn_backend() -> AttentionBackendEnum | None:
    """
    Get the currently-forced choice of attention backend,
    or None if auto-selection is currently enabled.
    """
    return forced_attn_backend


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

    return _cached_get_attn_backend(
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
    head_size: int,
    dtype: torch.dtype,
    kv_cache_dtype: CacheDType | None,
    block_size: int | None,
    use_mla: bool = False,
    has_sink: bool = False,
    use_sparse: bool = False,
    attn_type: str | None = None,
) -> type[AttentionBackend]:
    # Check whether a particular choice of backend was
    # previously forced.
    #
    # THIS SELECTION OVERRIDES THE VLLM_ATTENTION_BACKEND
    # ENVIRONMENT VARIABLE.
    selected_backend = None
    backend_by_global_setting: AttentionBackendEnum | None = (
        get_global_forced_attn_backend()
    )
    if backend_by_global_setting is not None:
        selected_backend = backend_by_global_setting
    else:
        # Check the environment variable and override if specified
        backend_by_env_var: str | None = envs.VLLM_ATTENTION_BACKEND
        if backend_by_env_var is not None:
            if backend_by_env_var.endswith("_VLLM_V1"):
                logger.warning(
                    "The suffix '_VLLM_V1' in the environment variable "
                    "%s is no longer necessary as V0 backends have been "
                    "deprecated. Please remove this suffix from your "
                    "environment variable setting.",
                    STR_BACKEND_ENV_VAR,
                )
                backend_by_env_var = backend_by_env_var.removesuffix("_VLLM_V1")
            try:
                selected_backend = AttentionBackendEnum[backend_by_env_var]
            except KeyError as e:
                raise ValueError(
                    f"Invalid attention backend: '{backend_by_env_var}'. Valid "
                    f"backends are: {list(AttentionBackendEnum.__members__.keys())}"
                ) from e

    # get device-specific attn_backend
    from vllm.platforms import current_platform

    sig = inspect.signature(current_platform.get_attn_backend_cls)
    if "use_v1" in sig.parameters:
        logger.warning_once(
            "use_v1 parameter for get_attn_backend_cls is deprecated and will "
            "be removed in v0.13.0 or v1.0.0, whichever is soonest. Please "
            "remove it from your plugin code."
        )
        attention_cls = current_platform.get_attn_backend_cls(
            selected_backend,
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
            selected_backend,
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


@contextmanager
def global_force_attn_backend_context_manager(
    attn_backend: AttentionBackendEnum,
) -> Generator[None, None, None]:
    """
    Globally force a vLLM attention backend override within a
    context manager, reverting the global attention backend
    override to its prior state upon exiting the context
    manager.

    Arguments:

    * attn_backend: attention backend to force

    Returns:

    * Generator
    """

    # Save the current state of the global backend override (if any)
    original_value = get_global_forced_attn_backend()

    # Globally force the new backend override
    global_force_attn_backend(attn_backend)

    # Yield control back to the enclosed code block
    try:
        yield
    finally:
        # Revert the original global backend override, if any
        global_force_attn_backend(original_value)
        _cached_get_attn_backend.cache_clear()
