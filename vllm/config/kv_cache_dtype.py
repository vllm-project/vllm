# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from typing import Protocol, get_args, runtime_checkable

import torch

from vllm.config.cache import CacheDType
from vllm.logger import init_logger
from vllm.platforms import current_platform, is_current_platform_resolved
from vllm.utils.torch_utils import STR_DTYPE_TO_TORCH_DTYPE
from vllm.v1.kv_cache_interface import KVQuantMode

logger = init_logger(__name__)

# Mutable mirror of the CacheDType Literal. Custom dtypes registered by
# platform backends are appended here at registration time. Never snapshot
# this list (no list(...), tuple(...), or import-time key capture) —
# later-registered entries must stay visible to all consumers.
KV_CACHE_DTYPES: list[str] = list(get_args(CacheDType))

# Custom dtype handlers: name -> handler instance.
_KV_CACHE_DTYPE_HANDLERS: dict[str, KVCacheDTypeHandler] = {}


@runtime_checkable
class KVCacheDTypeHandler(Protocol):
    """Contract for a custom ``--kv-cache-dtype`` value.

    A platform backend implements this protocol to answer the questions vLLM
    currently answers with closed tables (``STR_DTYPE_TO_TORCH_DTYPE`` /
    ``get_kv_quant_mode`` / ``is_quantized_kv_cache``). A single handler may
    be device-aware (e.g., resolving different torch dtypes per device
    generation).
    """

    name: str

    def torch_dtype(self) -> torch.dtype:
        """Storage dtype for KV cache tensors.

        Called once at registration time; the result is injected into the
        shared ``STR_DTYPE_TO_TORCH_DTYPE`` dict so existing consumers need no
        changes. The device must already be resolvable (registration runs at
        platform activation time).
        """
        ...

    def is_quantized(self) -> bool:
        """True if the cache is stored in a quantized (non-native) format."""
        ...

    def quant_mode(self) -> KVQuantMode:
        """The :class:`KVQuantMode` used by generic kernels.

        Return an existing mode to reuse generic kernel paths, or
        ``KVQuantMode.BACKEND`` when the backend fully self-manages kernel
        dispatch.
        """
        ...

    def supports_platform(self, platform) -> bool:
        """Whether this dtype is supported on the given platform."""
        ...


def register_kv_cache_dtype(name: str):
    """Register a custom ``--kv-cache-dtype`` value (eager decorator).

    Runs at platform activation time (first ``current_platform`` access),
    which is after ``--help`` exits and before any KV cache tensor is
    allocated. During decoration the handler will:

    - append ``name`` to the mutable ``KV_CACHE_DTYPES`` list (driving
      Pydantic + CLI validation), and to the active platform's
      ``supported_kv_cache_dtypes`` when it enforces a whitelist;
    - resolve ``torch_dtype()`` once and inject it into
      ``STR_DTYPE_TO_TORCH_DTYPE``, so all existing ``dict[name]`` /
      ``name in dict`` sites need zero changes;
    - be stored for later ``quant_mode()`` / ``is_quantized()`` queries.

    Examples:
        >>> @register_kv_cache_dtype("int8")
        ... class Int8Handler:
        ...     name = "int8"
        ...
        ...     def torch_dtype(self):
        ...         return torch.int8
        ...
        ...     def is_quantized(self):
        ...         return True
        ...
        ...     def quant_mode(self):
        ...         return KVQuantMode.BACKEND
        ...
        ...     def supports_platform(self, platform):
        ...         return platform.__class__.__name__ == "NPUPlatform"
    """

    def _decorate(cls):
        if name in KV_CACHE_DTYPES:
            logger.warning(
                "The kv-cache-dtype '%s' already exists and will be "
                "overwritten by handler %s.",
                name,
                cls,
            )
        else:
            KV_CACHE_DTYPES.append(name)
            # Auto-register support on the active platform, mirroring
            # register_quantization_config. Registration may run while the
            # platform is still resolving (OOT backends register from their
            # platform __init__), where accessing current_platform would
            # recurse; platforms enforcing a whitelist hard-code it instead.
            if is_current_platform_resolved() and (
                skd := current_platform.supported_kv_cache_dtypes
            ):
                skd.append(name)
        inst = cls()
        # Inject into the shared dtype dict in-place so every existing
        # direct [] / `in` access site sees the new entry.
        STR_DTYPE_TO_TORCH_DTYPE[name] = inst.torch_dtype()
        _KV_CACHE_DTYPE_HANDLERS[name] = inst
        return cls

    return _decorate


def get_kv_cache_dtype_handler(name: str) -> KVCacheDTypeHandler | None:
    """Return the handler for ``name``.

    Returns None for upstream dtypes that have no handler (the dtype table
    already covers these).
    """
    return _KV_CACHE_DTYPE_HANDLERS.get(name)


def is_known_kv_cache_dtype(name: str) -> bool:
    """Whether ``name`` is a valid ``--kv-cache-dtype`` value."""
    return name in KV_CACHE_DTYPES
