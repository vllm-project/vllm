# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
from collections.abc import Generator
from contextlib import contextmanager
from functools import cache

import torch

import vllm.envs as envs
from vllm.attention.backends.abstract import AttentionBackend
from vllm.attention.backends.registry import _Backend
from vllm.logger import init_logger
from vllm.utils import STR_BACKEND_ENV_VAR
from vllm.utils.import_utils import resolve_obj_by_qualname

logger = init_logger(__name__)


def get_env_variable_attn_backend() -> _Backend | None:
    """
    Get the backend override specified by the vLLM attention
    backend environment variable, if one is specified.

    Returns:

    * _Backend enum value if an override is specified
    * None otherwise
    """
    backend_name = os.environ.get(STR_BACKEND_ENV_VAR)
    return None if backend_name is None else _Backend[backend_name]


# Global state allows a particular choice of backend
# to be forced, overriding the logic which auto-selects
# a backend based on system & workload configuration
# (default behavior if this variable is None)
#
# THIS SELECTION TAKES PRECEDENCE OVER THE
# VLLM_ATTENTION_BACKEND ENVIRONMENT VARIABLE
forced_attn_backend: _Backend | None = None


def global_force_attn_backend(attn_backend: _Backend | None) -> None:
    """
    Force all attention operations to use a specified backend.

    Passing `None` for the argument re-enables automatic
    backend selection.,

    Arguments:

    * attn_backend: backend selection (None to revert to auto)
    """
    global forced_attn_backend
    forced_attn_backend = attn_backend


def get_global_forced_attn_backend() -> _Backend | None:
    """
    Get the currently-forced choice of attention backend,
    or None if auto-selection is currently enabled.
    """
    return forced_attn_backend


def get_attn_backend(
    head_size: int,
    dtype: torch.dtype,
    kv_cache_dtype: str | None,
    block_size: int,
    use_mla: bool = False,
    has_sink: bool = False,
    use_sparse: bool = False,
) -> type[AttentionBackend]:
    """Selects which attention backend to use and lazily imports it."""
    # Accessing envs.* behind an @lru_cache decorator can cause the wrong
    # value to be returned from the cache if the value changes between calls.
    # To avoid this, we read envs.VLLM_USE_V1 here and pass it explicitly to the
    # private function.
    return _cached_get_attn_backend(
        head_size=head_size,
        dtype=dtype,
        kv_cache_dtype=kv_cache_dtype,
        block_size=block_size,
        use_v1=envs.VLLM_USE_V1,
        use_mla=use_mla,
        has_sink=has_sink,
        use_sparse=use_sparse,
    )


@cache
def _cached_get_attn_backend(
    head_size: int,
    dtype: torch.dtype,
    kv_cache_dtype: str | None,
    block_size: int,
    use_v1: bool = False,
    use_mla: bool = False,
    has_sink: bool = False,
    use_sparse: bool = False,
) -> type[AttentionBackend]:
    # Check whether a particular choice of backend was
    # previously forced.
    #
    # THIS SELECTION OVERRIDES THE VLLM_ATTENTION_BACKEND
    # ENVIRONMENT VARIABLE.
    selected_backend = None
    backend_by_global_setting: _Backend | None = get_global_forced_attn_backend()
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
                selected_backend = _Backend[backend_by_env_var]
            except KeyError as e:
                raise ValueError(
                    f"Invalid attention backend: '{backend_by_env_var}'. "
                    f"Valid backends are: {list(_Backend.__members__.keys())}"
                ) from e

    # get device-specific attn_backend
    from vllm.platforms import current_platform

    attention_cls = current_platform.get_attn_backend_cls(
        selected_backend,
        head_size,
        dtype,
        kv_cache_dtype,
        block_size,
        use_v1,
        use_mla,
        has_sink,
        use_sparse,
    )
    if not attention_cls:
        raise ValueError(
            f"Invalid attention backend for {current_platform.device_name}"
        )
    backend = resolve_obj_by_qualname(attention_cls)

    # Adjust block size if the selected backend doesn't support it
    # TODO: per-layer block size configuration
    if not backend.supports_block_size(block_size):
        from vllm.config import get_current_vllm_config

        vllm_config = get_current_vllm_config()
        if vllm_config and vllm_config.cache_config:
            new_block_size = backend.get_supported_block_sizes()[0]
            logger.info(
                "Adjusting kv cache block size from %d to %d for %s backend.",
                block_size,
                new_block_size,
                backend.get_name(),
            )
            vllm_config.cache_config.block_size = new_block_size

    # Adjust kv cache layout if the selected backend requires a specific one
    device_capability = current_platform.get_device_capability()
    required_layout = backend.get_required_kv_cache_layout(device_capability)
    if required_layout is not None:
        from vllm.v1.attention.backends.utils import set_kv_cache_layout

        set_kv_cache_layout(required_layout)
        logger.info(
            "Using %s KV cache layout for %s backend.",
            required_layout,
            backend.get_name(),
        )

    return backend


@contextmanager
def global_force_attn_backend_context_manager(
    attn_backend: _Backend,
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
