# SPDX-License-Identifier: Apache-2.0
"""Device-agnostic helpers that wrap worker KV caches for IPC transport.

These helpers used to live under ``lmcache.integration.vllm`` for historical
reasons, but they are engine-neutral: dispatch happens purely via
:func:`resolve_kv_wrapper_factory` on ``tensor.device.type``. Keeping them
here lets core transfer contexts (e.g. ``LMCacheDrivenTransferContext``) use
them without importing the vLLM integration package.
"""

# Future
from __future__ import annotations

# Standard
from typing import Any

# Third Party
import torch

# First Party
from lmcache.logging import init_logger
from lmcache.v1.multiprocess.custom_types import KVCache
from lmcache.v1.platform import resolve_kv_wrapper_factory

logger = init_logger(__name__)


def wrap_one_kv_cache(tensor: torch.Tensor) -> Any:
    """Dispatch by ``tensor.device.type`` via the platform registry.

    Concrete factories are auto-discovered from ``DeviceIPCWrapper``
    subclasses under ``lmcache.v1.platform``, so this call site stays
    free of if/elif chains and new accelerators plug in by shipping a
    sibling wrapper class.
    """
    return resolve_kv_wrapper_factory(tensor.device.type)(tensor)


def wrap_kv_caches(kv_caches: dict[str, torch.Tensor]) -> KVCache:
    """Wrap every KV cache tensor for IPC transport.

    Args:
        kv_caches: Mapping from layer name to worker-owned KV cache tensor.

    Returns:
        The list of per-tensor IPC wrappers, ready for the msgspec wire.
    """
    # Emit a per-layer (name, shape, dtype) summary so the operator can
    # verify the exact layer set & tensor geometry being shipped to the
    # LMCache server, then the low-noise count of handles being wrapped.
    kept_summary = [
        (name, tuple(tensor.shape), str(tensor.dtype))
        for name, tensor in kv_caches.items()
    ]
    logger.debug(
        "KV cache transfer keeping %d layer(s) (name, shape, dtype):\n%s",
        len(kept_summary),
        "\n".join(
            f"  [{i}] {name}  shape={shape}  dtype={dtype}"
            for i, (name, shape, dtype) in enumerate(kept_summary)
        ),
    )
    logger.info("Wrapping %d KV cache tensors for IPC", len(kv_caches))
    # Per-iteration resource management: if wrapping the N-th tensor
    # raises, ``shm_unlink`` whatever earlier iterations already
    # registered with POSIX SHM so the named segments do not outlive
    # the failed batch. CUDA wrappers do not own a named segment and
    # are skipped via the duck-typed ``shm_name`` check.
    wrappers: KVCache = []
    try:
        for tensor in kv_caches.values():
            wrappers.append(wrap_one_kv_cache(tensor))
    except BaseException:
        _release_partial_kv_wrappers(wrappers)
        raise
    return wrappers


def _release_partial_kv_wrappers(wrappers: list[Any]) -> None:
    """Best-effort unlink of SHM segments owned by partially built wrappers.

    Used by :func:`wrap_kv_caches` to roll back a half-finished batch
    when a later iteration raises. Only POSIX-SHM-backed wrappers carry
    a ``shm_name`` attribute, so other wrapper kinds (e.g. CUDA-IPC)
    are silently skipped.
    """
    # First Party
    from lmcache.v1.multiprocess.posix_shm import shm_unlink

    for w in wrappers:
        name = getattr(w, "shm_name", None)
        if name is None:
            continue
        try:
            shm_unlink(name)
        except Exception:  # pragma: no cover - best effort
            logger.debug("shm_unlink failed during rollback", exc_info=True)
