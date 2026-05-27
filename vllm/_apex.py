# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""APEX hint-API client for vLLM (Demo 1).

This module is a thin ctypes wrapper around libapex_hints.so. It exposes
Python helpers used by the model loader, block manager, and MoE router
to emit ExpertPrefetch / KvBlockPrefetch / Register* hints to the apexd
daemon over its Unix-domain socket.

Behaviour:
  * The whole module is a no-op unless the env var ``VLLM_APEX_HINTS=1``
    is set. This keeps the patch fully opt-in: vLLM built with this
    module behaves identically to upstream unless the user opts in.
  * If ``VLLM_APEX_HINTS=1`` but ``libapex_hints.so`` cannot be loaded
    (e.g. APEX is not installed), every helper logs once and degrades
    to a no-op.
  * All functions are exception-safe — they never raise into the hot
    inference path.

Env vars
  VLLM_APEX_HINTS=1      master switch
  APEX_SOCKET=<path>     override Unix socket (default /run/apex/apexd.sock,
                         consumed inside libapex_hints itself)

Wire types follow ``crates/apex-hints/include/apex_hints.h``::

    int apex_init(void);
    int apex_hint_experts(const uint32_t *ids, uint32_t count, uint32_t top_k);
    int apex_hint_kv_blocks(const uint64_t *ids, uint32_t count, uint64_t seq_id);
    int apex_hint_register_expert(uint32_t id, uint64_t addr, uint64_t len);
    int apex_hint_register_kv_block(uint64_t block_id, uint64_t addr, uint64_t len);
"""

from __future__ import annotations

import ctypes
import logging
import os
import threading
from typing import Iterable

logger = logging.getLogger("vllm.apex")

_ENABLED = os.environ.get("VLLM_APEX_HINTS", "0") == "1"
_lib: ctypes.CDLL | None = None
_init_lock = threading.Lock()
_init_done = False
_load_failed = False

# Diagnostic counters; cheap to update, useful in the demo dashboard.
_stats = {
    "experts_emitted": 0,
    "kv_blocks_emitted": 0,
    "experts_registered": 0,
    "kv_blocks_registered": 0,
    "errors": 0,
}


def enabled() -> bool:
    """True iff vLLM should emit APEX hints in this process."""
    return _ENABLED and not _load_failed


def _load() -> ctypes.CDLL | None:
    """Load libapex_hints.so once and wire ctypes prototypes."""
    global _lib, _init_done, _load_failed

    if _load_failed:
        return None
    if _lib is not None:
        return _lib

    with _init_lock:
        if _lib is not None:
            return _lib
        if _load_failed:
            return None

        try:
            lib = ctypes.CDLL("libapex_hints.so")
        except OSError as exc:
            logger.warning(
                "VLLM_APEX_HINTS=1 but libapex_hints.so could not be "
                "loaded (%s). Hints will be no-ops.",
                exc,
            )
            _load_failed = True
            return None

        lib.apex_init.restype = ctypes.c_int
        lib.apex_init.argtypes = []

        lib.apex_shutdown.restype = None
        lib.apex_shutdown.argtypes = []

        lib.apex_hint_experts.restype = ctypes.c_int
        lib.apex_hint_experts.argtypes = [
            ctypes.POINTER(ctypes.c_uint32),
            ctypes.c_uint32,
            ctypes.c_uint32,
        ]

        lib.apex_hint_kv_blocks.restype = ctypes.c_int
        lib.apex_hint_kv_blocks.argtypes = [
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.c_uint32,
            ctypes.c_uint64,
        ]

        lib.apex_hint_register_expert.restype = ctypes.c_int
        lib.apex_hint_register_expert.argtypes = [
            ctypes.c_uint32,
            ctypes.c_uint64,
            ctypes.c_uint64,
        ]

        lib.apex_hint_register_kv_block.restype = ctypes.c_int
        lib.apex_hint_register_kv_block.argtypes = [
            ctypes.c_uint64,
            ctypes.c_uint64,
            ctypes.c_uint64,
        ]

        if not _init_done:
            rc = lib.apex_init()
            _init_done = True
            if rc != 0:
                logger.warning("apex_init returned %d; hints may not work", rc)

        _lib = lib
        logger.info("APEX hint client loaded; emitting hints from vLLM")
        return _lib


def register_expert(global_id: int, addr: int, length: int) -> bool:
    """Register one MoE expert's weight region with apexd.

    ``global_id`` is opaque to apexd — use any stable encoding (e.g. layer
    index * num_experts + expert_index). Pair this with
    :func:`prefetch_experts` using the same encoding.
    """
    if not enabled():
        return False
    lib = _load()
    if lib is None:
        return False
    try:
        rc = lib.apex_hint_register_expert(
            ctypes.c_uint32(global_id & 0xFFFFFFFF),
            ctypes.c_uint64(addr),
            ctypes.c_uint64(length),
        )
        if rc == 0:
            _stats["experts_registered"] += 1
            return True
        _stats["errors"] += 1
        return False
    except Exception:
        _stats["errors"] += 1
        return False


def register_kv_block(block_id: int, addr: int, length: int) -> bool:
    """Register one KV-cache block's physical region with apexd."""
    if not enabled():
        return False
    lib = _load()
    if lib is None:
        return False
    try:
        rc = lib.apex_hint_register_kv_block(
            ctypes.c_uint64(block_id),
            ctypes.c_uint64(addr),
            ctypes.c_uint64(length),
        )
        if rc == 0:
            _stats["kv_blocks_registered"] += 1
            return True
        _stats["errors"] += 1
        return False
    except Exception:
        _stats["errors"] += 1
        return False


def register_kv_cache_tensor(
    tensor,
    *,
    block_size_bytes: int,
    layer_id: int,
    num_layers: int,
) -> int:
    """Register every block inside one layer's contiguous KV-cache tensor.

    Block ids are encoded so they're globally unique across layers:
        global_block_id = layer_id * (num_total_blocks_per_layer) + local_idx
    Returns the number of successfully registered blocks.

    Safe to call before vLLM starts serving — runs once at init.
    """
    if not enabled():
        return 0
    if block_size_bytes <= 0:
        return 0

    base = int(tensor.data_ptr())
    total = int(tensor.nbytes)
    n_blocks = total // block_size_bytes
    if n_blocks <= 0:
        return 0

    layer_stride = 1_000_000  # generous: assume <1M blocks per layer
    success = 0
    for i in range(n_blocks):
        global_id = layer_id * layer_stride + i
        if register_kv_block(global_id, base + i * block_size_bytes, block_size_bytes):
            success += 1
    logger.info(
        "apex: registered %d/%d KV blocks for layer %d (%.1f MiB)",
        success,
        n_blocks,
        layer_id,
        total / (1024 * 1024),
    )
    return success


def prefetch_experts(global_ids: Iterable[int], top_k: int = 8) -> bool:
    """Emit an ExpertPrefetch hint listing the predicted expert ids."""
    if not enabled():
        return False
    lib = _load()
    if lib is None:
        return False
    try:
        ids = [int(x) & 0xFFFFFFFF for x in global_ids]
        if not ids:
            return False
        arr = (ctypes.c_uint32 * len(ids))(*ids)
        rc = lib.apex_hint_experts(arr, ctypes.c_uint32(len(ids)), ctypes.c_uint32(top_k))
        if rc >= 0:
            _stats["experts_emitted"] += 1
            return True
        _stats["errors"] += 1
        return False
    except Exception:
        _stats["errors"] += 1
        return False


def prefetch_blocks(block_ids: Iterable[int], sequence_id: int = 0) -> bool:
    """Emit a KvBlockPrefetch hint listing the upcoming block ids."""
    if not enabled():
        return False
    lib = _load()
    if lib is None:
        return False
    try:
        ids = [int(x) for x in block_ids]
        if not ids:
            return False
        arr = (ctypes.c_uint64 * len(ids))(*ids)
        rc = lib.apex_hint_kv_blocks(
            arr, ctypes.c_uint32(len(ids)), ctypes.c_uint64(sequence_id)
        )
        if rc >= 0:
            _stats["kv_blocks_emitted"] += 1
            return True
        _stats["errors"] += 1
        return False
    except Exception:
        _stats["errors"] += 1
        return False


def get_stats() -> dict:
    """Return a copy of the diagnostic counters."""
    return dict(_stats)


# Convenience encoder for the (layer, expert) → u32 mapping used by both
# the vLLM patches and apex-vmem's phase3 speculative prefetcher. Keep
# this in sync with the encoding in phase3/apex_speculative_prefetch.py.
def encode_expert_id(layer_idx: int, expert_idx: int, n_experts_per_layer: int) -> int:
    """Stable encoding for a (layer, expert) pair into a single u32 id.

    Reserves the lowest bit for w13 (0) vs w2 (1) so a single hint can
    address both halves of an expert.
    """
    return (layer_idx * n_experts_per_layer + expert_idx) * 2
