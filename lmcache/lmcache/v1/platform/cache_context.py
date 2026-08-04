# SPDX-License-Identifier: Apache-2.0
"""Platform-agnostic cache-context factory.

The concrete implementations live in their respective sub-packages:

* :class:`~lmcache.v1.platform.cuda.cache_context.GPUCacheContext` --
  CUDA-backed.
* :class:`~lmcache.v1.platform.cpu.cache_context.CPUCacheContext` --
  CPU-only fallback (POSIX-SHM-backed KV tensors).

:func:`create_cache_context` keeps the dispatch out of the call site
in :mod:`lmcache.v1.multiprocess.server`. Selection is data-driven
and delegated to the :class:`~lmcache.v1.platform.base.device_spec.
DeviceSpec` registry maintained by :mod:`lmcache.v1.platform`: each
backend sub-package ships a ``DeviceSpec`` subclass whose
``create_cache_context`` hook lazy-imports and instantiates the
matching :class:`~lmcache.v1.platform.base.cache_context.
BaseCacheContext`. Adding a new accelerator therefore requires
*zero* edits to this module -- just implement
``DeviceSpec.create_cache_context`` in the sub-package's
``__init__.py``.
"""

# Future
from __future__ import annotations

# Standard
from collections.abc import Sequence
from typing import TYPE_CHECKING

# First Party
from lmcache.logging import init_logger
from lmcache.utils import EngineType
from lmcache.v1.gpu_connector.utils import LayoutHints
from lmcache.v1.multiprocess.custom_types import KVCache
from lmcache.v1.platform import get_device_spec
from lmcache.v1.platform.base.cache_context import BaseCacheContext

if TYPE_CHECKING:
    # First Party
    from lmcache.v1.multiprocess.group_view import EngineGroupInfo

logger = init_logger(__name__)


def _detect_device_type(kv_caches: KVCache) -> str:
    """Return the ``torch.device.type`` describing *kv_caches*.

    All wrappers in *kv_caches* must materialize tensors on the same
    device type; mixed-device batches are not supported by any
    downstream cache-context implementation.
    """
    device_types = {w.to_tensor().device.type for w in kv_caches}
    if len(device_types) != 1:
        raise ValueError(
            "create_cache_context requires all kv_caches to share one "
            "device type, got %r" % sorted(device_types)
        )
    return next(iter(device_types))


def create_cache_context(
    kv_caches: KVCache,
    lmcache_tokens_per_chunk: int = 256,
    layout_hints: LayoutHints | None = None,
    engine_group_infos: "Sequence[EngineGroupInfo]" = (),
    engine_type: EngineType = EngineType.VLLM,
    separate_object_groups: bool = True,
    full_sw_kv: bool = False,
) -> BaseCacheContext:
    """Create the appropriate cache context for *kv_caches*.

    The signature mirrors :class:`GPUCacheContext` so callers can
    forward their kwargs verbatim and stay agnostic of the active
    backend.

    Selection is driven by ``tensor.device.type`` of *kv_caches*:
    the platform ``DeviceSpec`` registry is consulted and the
    matching spec's ``create_cache_context`` hook is invoked.
    ``"cuda"``, ``"cpu"``, future ``"xpu"`` ... all resolve through
    the same code path -- no ``isinstance`` / ``if-elif`` chain.

    Args:
        kv_caches: KV cache tensor wrappers from the serving engine.
            Must be non-empty.
        lmcache_tokens_per_chunk: Number of tokens per LMCache chunk.
        layout_hints: Optional hints for KV format detection.
            Forwarded verbatim to the concrete context constructor.
        engine_group_infos: Engine-neutral KV cache group metadata.
        engine_type: Which serving engine produced the caches.
        separate_object_groups: When True (default), split kernel groups into
            one object group per sliding-window size; when False, a single
            full-attention object group.
        full_sw_kv: When True, sliding-window groups store/transfer full
            per-chunk KV (no sub-chunk window cutting) so chunks stay valid for
            reuse at any position; see
            :meth:`KVLayerGroupsManager.enable_full_sw_kv`.

    Returns:
        A concrete cache context instance.

    Raises:
        ValueError: If *kv_caches* is empty, mixes device types, or
            targets a device type with no registered backend.
    """
    if not kv_caches:
        raise ValueError("create_cache_context requires a non-empty kv_caches list")

    device_type = _detect_device_type(kv_caches)
    spec = get_device_spec(device_type)
    if spec is None:
        raise ValueError(
            "No cache-context class registered for device type %r. "
            "Make sure ``lmcache.v1.platform.<backend>.__init__`` "
            "ships a DeviceSpec subclass whose ``create_cache_context`` "
            "hook returns a BaseCacheContext instance." % device_type
        )
    return spec.create_cache_context(
        kv_caches,
        lmcache_tokens_per_chunk,
        layout_hints,
        engine_group_infos,
        engine_type,
        separate_object_groups,
        full_sw_kv,
    )
