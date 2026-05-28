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
import weakref
from typing import Iterable

logger = logging.getLogger("vllm.apex")

_ENABLED = os.environ.get("VLLM_APEX_HINTS", "0") == "1"
# When VLLM_APEX_USE_CUSTOM_OP=1, route hint emission through a
# ``torch.library.custom_op`` instead of calling ``_emit_next_layer_hint``
# directly from the forward hook. The custom op is opaque to torch.compile,
# which lets the hook body stay traceable in vLLM's fullgraph mode.
# Default off so the green --enforce-eager path stays the safe default.
_USE_CUSTOM_OP = os.environ.get("VLLM_APEX_USE_CUSTOM_OP", "0") == "1"

# Phase-A perf knob: when the model + KV cache fit comfortably in HBM,
# the runtime cost of emitting hints (Python hook + ctypes + Unix socket
# + optional sysfs write) is pure overhead — there's nothing to prefetch.
# We let the harness signal "real tier hierarchy in play" via APEX_OFFLOAD_GB
# (or APEX_HAS_OFFLOAD=1 for non-numeric scenarios). With the default
# APEX_AUTO_NOOP_IF_FITS=1, registrations + counters still happen (cheap
# and useful for verification), but the *per-forward* emit path becomes a
# fast no-op when no offload was declared. Set
# APEX_AUTO_NOOP_IF_FITS=0 to force-emit for benchmarking the overhead
# itself.
def _auto_noop_active() -> bool:
    if os.environ.get("APEX_AUTO_NOOP_IF_FITS", "1") != "1":
        return False
    try:
        offload_gb = float(os.environ.get("APEX_OFFLOAD_GB", "0") or "0")
    except ValueError:
        offload_gb = 0.0
    has_offload = os.environ.get("APEX_HAS_OFFLOAD", "0") == "1"
    return offload_gb <= 0.0 and not has_offload

_EMIT_NOOP = _auto_noop_active()

# Phase-A perf knob: rate-cap per-token emissions. The forward hook fires
# 26× per token on DeepSeek-V2-Lite; that's a lot of ctypes round-trips.
# When set to N > 0, we emit at most one hint per N microseconds across
# the whole process — the predictor degrades gracefully (next layer's
# prediction comes from the *last* observed top-k, not the current one),
# but ctypes call frequency drops by orders of magnitude. 0 disables the
# rate-cap and emits on every layer.
try:
    _EMIT_MIN_INTERVAL_US = int(os.environ.get("APEX_EMIT_MIN_INTERVAL_US", "0"))
except ValueError:
    _EMIT_MIN_INTERVAL_US = 0
_last_emit_ns = [0]  # mutable holder so hook can update without `global`
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
    "warm_calls": 0,        # in-process torch prefetch calls (Fix b)
    "warm_bytes": 0,        # bytes scheduled for prefetch
    "warm_cpu_hits": 0,     # how often the predicted expert was on CPU
    "warm_gpu_hits": 0,     # already on GPU — no copy needed
}

# Registry built during ``_apex_register_moe_experts``: maps layer_idx
# → FusedMoE module. Used by ``_apex_warm_next_layer`` (Fix b) to look
# up the next layer's expert tensors and prefetch them on a side stream
# while the current layer is still computing.
_moe_by_layer: dict[int, object] = {}

# Side-channel CUDA stream for prefetch (Fix b). Created lazily on first
# use because torch.cuda may not be initialised at module import time.
# Honour APEX_DISABLE_WARM=1 to keep the registration/hint path active
# but disable the actual H2D copy (useful for measuring the cost of the
# emit path itself vs the emit+prefetch path).
_warm_stream: object | None = None
_warm_disabled = os.environ.get("APEX_DISABLE_WARM", "0") == "1"

# --- async emit worker -----------------------------------------------------
#
# The hot-path forward hook MUST NOT do anything that forces a CUDA sync —
# that stalls the decode pipeline by tens of ms per emit. Instead we
# enqueue (router_logits.detach(), metadata) onto a thread-safe queue
# and a single background thread does the topk + ctypes round-trip
# asynchronously. The hint arrives at apexd a few ms late but that's
# perfectly fine for next-layer prefetch — the layer-to-layer routing
# correlation is steady-state and apexd's policy maintains its own
# sliding window anyway.
import collections as _collections

_emit_queue: "_collections.deque[tuple]" = _collections.deque(maxlen=64)
_emit_queue_lock = threading.Lock()
_emit_queue_event = threading.Event()
_emit_thread: threading.Thread | None = None
_emit_thread_started = False


def _enqueue_for_async_emit(
    router_logits, layer_idx: int, top_k: int, n_experts: int
) -> None:
    """Stash the router_logits tensor (not a copy — just a reference) for
    the worker thread to consume. Detach so we don't hold autograd refs.
    The maxlen=64 deque drops old entries when the worker can't keep up,
    which is fine; we'd rather lose a few hints than stall the hot path.
    """
    try:
        if router_logits is None:
            return
        # Detach is essentially free; it shares storage with the original.
        _emit_queue.append(
            (router_logits.detach(), int(layer_idx), int(top_k), int(n_experts))
        )
        _emit_queue_event.set()
    except Exception:
        _stats["errors"] += 1


def _emit_worker() -> None:
    """Background thread that drains :data:`_emit_queue`, performs the
    GPU→CPU sync, picks topk, calls the ctypes emit, and (if enabled)
    kicks off the in-process prefetch. Loops until ``_emit_thread_stop``.
    Errors are counted but never crash the thread.
    """
    import torch  # safe; we got here only after torch was already loaded

    while True:
        if not _emit_queue:
            _emit_queue_event.wait(timeout=0.5)
            _emit_queue_event.clear()
        try:
            item = _emit_queue.popleft()
        except IndexError:
            continue
        rl, layer_idx, top_k, n_experts = item
        try:
            if rl.dim() != 2 or n_experts <= 0:
                continue
            if rl.shape[0] > 32:
                rl = rl[-32:]
            avg = rl.mean(dim=0)
            k = min(top_k, avg.numel())
            if k <= 0:
                continue
            _, top_ids = torch.topk(avg, k)
            local_ids = top_ids.cpu().tolist()  # the one sync, off hot path

            next_layer = layer_idx + 1
            global_ids: list[int] = []
            for eid in local_ids:
                base = encode_expert_id(next_layer, int(eid), n_experts)
                global_ids.append(base)
                global_ids.append(base + 1)
            prefetch_experts(global_ids, top_k=top_k)

            if not _warm_disabled:
                _apex_warm_next_layer(next_layer, [int(e) for e in local_ids])
        except Exception as exc:
            _stats["errors"] += 1
            if _stats["errors"] < 4:
                logger.warning("apex: async-emit worker error: %s", exc)


def _start_emit_worker_once() -> None:
    """Spawn the singleton background emit worker. Idempotent and safe
    to call from inside a forward hook (cheap fast-path after the first
    call)."""
    global _emit_thread, _emit_thread_started
    if _emit_thread_started:
        return
    _emit_thread_started = True
    t = threading.Thread(
        target=_emit_worker, name="apex-async-emit", daemon=True
    )
    t.start()
    _emit_thread = t
    logger.info("apex: async-emit worker thread started")

# When set, _apex periodically dumps `_stats` to this path so external
# verification scripts (e.g. demo1/run_benchmark.sh) can read counters
# from outside the vLLM process. Persistence is best-effort: a background
# thread writes the file every APEX_STATS_INTERVAL_MS milliseconds.
_stats_file_path = os.environ.get("APEX_STATS_FILE", "")
try:
    _stats_interval_ms = int(os.environ.get("APEX_STATS_INTERVAL_MS", "500"))
except ValueError:
    _stats_interval_ms = 500
_stats_thread: threading.Thread | None = None
_stats_stop = threading.Event()


def _persist_stats_once() -> None:
    if not _stats_file_path:
        return
    try:
        import json

        tmp = _stats_file_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(_stats, f)
        os.replace(tmp, _stats_file_path)
    except Exception:
        pass


def _start_stats_persistence() -> None:
    """Spawn a background thread that periodically dumps _stats to a JSON
    file. Idempotent. No-op if APEX_STATS_FILE is unset."""
    global _stats_thread
    if not _stats_file_path or _stats_thread is not None:
        return

    def _run():
        while not _stats_stop.wait(_stats_interval_ms / 1000.0):
            _persist_stats_once()
        _persist_stats_once()

    t = threading.Thread(target=_run, name="apex-stats-persist", daemon=True)
    t.start()
    _stats_thread = t
    try:
        import atexit

        atexit.register(_persist_stats_once)
    except Exception:
        pass


def enabled() -> bool:
    """True iff vLLM should emit APEX hints in this process."""
    return _ENABLED and not _load_failed


def emit_active() -> bool:
    """True iff *per-forward* hint emission should pay its cost.

    Independent from :func:`enabled` — registration hints, KV-cache
    block hints, and the diagnostic counters still run when APEX is
    enabled but emit is "noop'd". This is the chokepoint the
    forward-hook fast-path consults; if it returns False the
    expert-prefetch hook returns in a couple of nanoseconds.
    """
    return enabled() and not _EMIT_NOOP


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
        _start_stats_persistence()
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
    """Register blocks inside one layer's contiguous KV-cache tensor.

    Block ids are encoded so they're globally unique across layers:
        global_block_id = layer_id * (num_total_blocks_per_layer) + local_idx

    For very large KV caches (~1.8M tokens with 16-token blocks → ~113k
    blocks per layer × N layers = millions of socket round-trips at
    startup), we cap registrations per layer to the first
    ``APEX_KV_BLOCKS_PER_LAYER`` blocks (default 256). Bumped to
    something much larger for production runs that actually want full
    block-level prefetch granularity. The rest of the cache still gets
    prefetch hints by id (the daemon treats unknown ids as best-effort).

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

    try:
        cap = int(os.environ.get("APEX_KV_BLOCKS_PER_LAYER", "256"))
    except ValueError:
        cap = 256
    if cap > 0:
        n_to_register = min(n_blocks, cap)
    else:
        n_to_register = n_blocks

    layer_stride = 1_000_000  # generous: assume <1M blocks per layer
    success = 0
    for i in range(n_to_register):
        global_id = layer_id * layer_stride + i
        if register_kv_block(global_id, base + i * block_size_bytes, block_size_bytes):
            success += 1
    logger.info(
        "apex: registered %d/%d KV blocks for layer %d (%.1f MiB, cap=%d)",
        success,
        n_blocks,
        layer_id,
        total / (1024 * 1024),
        cap,
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


# ---------------------------------------------------------------------------
# Runtime monkey-patch overlay
# ---------------------------------------------------------------------------
#
# When this file is dropped into the installed vLLM package (rather than
# applied as a git patch on top of our fork's source tree), we still need
# to register MoE expert weights, KV-cache blocks, and emit prefetch
# hints at the right spots inside vLLM's model loader, block pool, and
# DeepSeek MoE forward(). Doing it as monkey-patches lets the overlay
# work across different vLLM revisions without editing those files.
#
# install_hooks() is idempotent; safe to call multiple times.
#
# All hooks are wrapped in try/except — they NEVER crash inference for a
# prefetch failure.

_hooks_installed = False


def _extract_layer_index(qualified_name: str) -> int:
    """Pull the integer right after ``.layers.`` in a module path like
    ``model.layers.17.mlp``. Returns 0 if not found."""
    marker = ".layers."
    if marker not in qualified_name:
        return 0
    try:
        rest = qualified_name.split(marker, 1)[1]
        return int(rest.split(".", 1)[0])
    except (ValueError, IndexError):
        return 0


def _hook_block_pool() -> bool:
    """Wrap ``BlockPool.get_new_blocks`` so freshly-allocated blocks are
    announced via :func:`prefetch_blocks` for apexd to warm into DRAM."""
    try:
        from vllm.v1.core.block_pool import BlockPool
    except Exception as exc:
        logger.warning("apex: cannot import BlockPool (%s); skipping hook", exc)
        return False

    if getattr(BlockPool.get_new_blocks, "_apex_wrapped", False):
        return True

    original = BlockPool.get_new_blocks

    def wrapped(self, *args, **kwargs):
        ret = original(self, *args, **kwargs)
        try:
            if enabled() and ret:
                block_ids = []
                for b in ret:
                    bid = getattr(b, "block_id", None)
                    if bid is not None:
                        block_ids.append(int(bid))
                if block_ids:
                    prefetch_blocks(block_ids, sequence_id=0)
        except Exception:
            _stats["errors"] += 1
        return ret

    wrapped._apex_wrapped = True  # type: ignore[attr-defined]
    BlockPool.get_new_blocks = wrapped

    def apex_prefetch_request_blocks(
        self, block_ids, sequence_id: int = 0
    ) -> None:
        if not enabled() or not block_ids:
            return
        prefetch_blocks(block_ids, sequence_id=sequence_id)

    BlockPool.apex_prefetch_request_blocks = apex_prefetch_request_blocks
    return True


def _apex_register_moe_experts(runner) -> None:
    """Walk every FusedMoE module in the loaded model and register each
    expert's (w13, w2) regions with apexd.

    Also installs a per-instance forward hook on the FusedMoE itself, so
    the hot-path expert-prediction emit fires for *any* model that uses
    vLLM's :class:`FusedMoE` — Mixtral, Qwen-MoE, DBRX, GLM-MoE, … —
    not just DeepSeek-V2 (which is also hooked via its enclosing
    DeepseekV2MoE wrapper, but that wrapper is model-specific).

    The hook reads ``router_logits`` from positional arg 1 or
    ``kwargs["router_logits"]``, which matches every vLLM FusedMoE
    caller pattern observed in tree (Mixtral, Qwen, DBRX, GLM-MoE all
    call ``self.experts(hidden_states, router_logits)`` — see
    ``vllm.model_executor.models.mixtral.MixtralMoE.forward`` and
    siblings). For Mixtral the call site is positional;
    ``_on_experts_forward`` already handles both.
    """
    try:
        from vllm.model_executor.layers.fused_moe.layer import FusedMoE
    except Exception as exc:
        logger.warning("apex: cannot import FusedMoE (%s); skipping registration", exc)
        return

    model = getattr(runner, "model", None)
    if model is None or not hasattr(model, "named_modules"):
        logger.info("apex: runner.model not available; skipping MoE registration")
        return

    moe_modules: list[tuple[int, "FusedMoE"]] = []
    for name, module in model.named_modules():
        if isinstance(module, FusedMoE):
            moe_modules.append((_extract_layer_index(name), module))

    if not moe_modules:
        logger.info("apex: no FusedMoE modules found — nothing to register")
        return

    n_experts_per_layer = 0
    for _, m in moe_modules:
        n = getattr(m, "logical_num_experts", 0) or getattr(m, "global_num_experts", 0)
        if n and n > n_experts_per_layer:
            n_experts_per_layer = int(n)
    if n_experts_per_layer <= 0:
        # Last-resort: derive from w13_weight.shape[0]
        for _, m in moe_modules:
            w = getattr(m, "w13_weight", None)
            if w is not None and w.dim() >= 1:
                n_experts_per_layer = max(n_experts_per_layer, int(w.shape[0]))
    if n_experts_per_layer <= 0:
        logger.warning("apex: cannot determine num_experts; skipping registration")
        return

    total = 0
    for layer_idx, moe in moe_modules:
        w13 = getattr(moe, "w13_weight", None)
        w2 = getattr(moe, "w2_weight", None)
        if w13 is None or w2 is None:
            continue
        local_n = min(int(w13.shape[0]), int(w2.shape[0]))
        for eid in range(local_n):
            w13_slice = w13[eid]
            w2_slice = w2[eid]
            base = encode_expert_id(layer_idx, eid, n_experts_per_layer)
            if register_expert(
                base,
                int(w13_slice.data_ptr()),
                int(w13_slice.nelement() * w13_slice.element_size()),
            ):
                total += 1
            if register_expert(
                base + 1,
                int(w2_slice.data_ptr()),
                int(w2_slice.nelement() * w2_slice.element_size()),
            ):
                total += 1

    # --- model-agnostic hot-path hook -------------------------------------
    # Stamp per-instance APEX metadata + register a forward hook on each
    # FusedMoE so the next-layer expert-prediction fires for every model
    # (not just DeepSeek-V2). Also build a layer_idx → FusedMoE registry
    # used by ``_apex_warm_next_layer`` (the in-process torch prefetch
    # path, Fix (b)).
    _moe_by_layer.clear()
    n_hooked = 0
    for layer_idx, moe in moe_modules:
        try:
            moe._apex_layer_idx = int(layer_idx)
            moe._apex_topk = int(
                getattr(moe, "top_k", 0)
                or getattr(moe, "_apex_topk", 0)
                or 8
            )
            moe._apex_n_routed_experts = int(
                getattr(moe, "logical_num_experts", 0)
                or getattr(moe, "global_num_experts", 0)
                or n_experts_per_layer
            )
            _moe_by_layer[int(layer_idx)] = moe
            # If a hook is already registered (e.g. via _hook_deepseek_v2_moe
            # for DSv2), don't double-register.
            if getattr(moe, "_apex_hook_handle", None) is None:
                try:
                    handle = moe.register_forward_hook(
                        _on_experts_forward, with_kwargs=True
                    )
                except TypeError:
                    handle = moe.register_forward_hook(
                        lambda m, a, o: _on_experts_forward(m, a, {}, o)
                    )
                moe._apex_hook_handle = handle
                n_hooked += 1
        except Exception as exc:
            logger.warning(
                "apex: failed to hook FusedMoE at layer %d: %s", layer_idx, exc
            )
            _stats["errors"] += 1

    logger.info(
        "apex: registered %d expert regions across %d FusedMoE modules "
        "(n_experts_per_layer=%d, n_hooked=%d)",
        total,
        len(moe_modules),
        n_experts_per_layer,
        n_hooked,
    )


def _apex_register_kv_cache_tensors(runner, kv_cache_raw_tensors, kv_cache_config) -> None:
    """Register each layer's raw KV cache tensor with apexd, broken into
    page-sized blocks."""
    layer_to_page: dict[str, int] = {}
    try:
        groups = getattr(kv_cache_config, "kv_cache_groups", None) or []
        for group in groups:
            page = getattr(group.kv_cache_spec, "page_size_bytes", 0)
            if page <= 0:
                continue
            for layer_name in getattr(group, "layer_names", []):
                layer_to_page[layer_name] = int(page)
    except Exception as exc:
        logger.warning("apex: failed to read kv_cache_groups (%s)", exc)
        return

    already_done: set[int] = set()
    items = sorted(kv_cache_raw_tensors.items())
    for layer_idx, (layer_name, tensor) in enumerate(items):
        page = layer_to_page.get(layer_name)
        if page is None or page <= 0:
            continue
        tensor_id = int(tensor.data_ptr())
        if tensor_id in already_done:
            continue
        already_done.add(tensor_id)
        register_kv_cache_tensor(
            tensor,
            block_size_bytes=page,
            layer_id=layer_idx,
            num_layers=len(items),
        )


def _hook_gpu_model_runner() -> bool:
    """Wrap ``GPUModelRunner.load_model`` and ``_allocate_kv_cache_tensors``
    so expert weights and KV-cache blocks get registered with apexd at the
    right moments. Tolerant to either method being absent in this revision."""
    try:
        from vllm.v1.worker.gpu_model_runner import GPUModelRunner
    except Exception as exc:
        logger.warning(
            "apex: cannot import GPUModelRunner (%s); skipping hook", exc
        )
        return False

    if not getattr(GPUModelRunner.load_model, "_apex_wrapped", False):
        original_load_model = GPUModelRunner.load_model

        def wrapped_load_model(self, *args, **kwargs):
            result = original_load_model(self, *args, **kwargs)
            try:
                if enabled():
                    _apex_register_moe_experts(self)
            except Exception as exc:
                logger.warning("apex: MoE expert registration failed: %s", exc)
                _stats["errors"] += 1
            return result

        wrapped_load_model._apex_wrapped = True  # type: ignore[attr-defined]
        GPUModelRunner.load_model = wrapped_load_model

    alloc = getattr(GPUModelRunner, "_allocate_kv_cache_tensors", None)
    if alloc is not None and not getattr(alloc, "_apex_wrapped", False):
        original_alloc = alloc

        def wrapped_alloc(self, kv_cache_config, *args, **kwargs):
            result = original_alloc(self, kv_cache_config, *args, **kwargs)
            try:
                if enabled() and isinstance(result, dict):
                    _apex_register_kv_cache_tensors(self, result, kv_cache_config)
            except Exception as exc:
                logger.warning("apex: KV-cache registration failed: %s", exc)
                _stats["errors"] += 1
            return result

        wrapped_alloc._apex_wrapped = True  # type: ignore[attr-defined]
        GPUModelRunner._allocate_kv_cache_tensors = wrapped_alloc

    return True


_emit_log_counter = [0]
_emit_error_log_counter = [0]


def _on_experts_forward(module, args, kwargs, output):
    """Forward hook fired AFTER ``module.forward`` returns.

    Two emission paths:

    * **Direct (default)** — call :func:`_emit_next_layer_hint` from
      pure Python. Requires ``--enforce-eager`` because vLLM's fullgraph
      ``torch.compile`` mode inlines this hook into the traced graph
      and refuses ctypes calls.
    * **Custom-op (`VLLM_APEX_USE_CUSTOM_OP=1`)** — dispatch through
      ``torch.ops.apex.hint_experts_for_next_layer``, which Dynamo
      treats as opaque so compile-mode runs don't blow up.

    Either way the hook body itself is small and uses only attributes
    stamped at init time, so it stays Dynamo-traceable.
    """
    if not emit_active():
        return
    # Rate-cap: avoid 26× ctypes round-trips per token when most of them
    # will predict the same set of experts. Uses time.monotonic_ns which
    # is cheap and lock-free.
    if _EMIT_MIN_INTERVAL_US > 0:
        import time
        now_ns = time.monotonic_ns()
        last_ns = _last_emit_ns[0]
        if last_ns and (now_ns - last_ns) < (_EMIT_MIN_INTERVAL_US * 1000):
            return
        _last_emit_ns[0] = now_ns
    layer_idx = getattr(module, "_apex_layer_idx", -1)
    top_k = getattr(module, "_apex_topk", 0)
    n_experts = getattr(module, "_apex_n_routed_experts", 0)
    if layer_idx < 0 or n_experts <= 0:
        return
    rl = kwargs.get("router_logits") if kwargs else None
    if rl is None and len(args) >= 2:
        rl = args[1]
    if rl is None:
        return
    # No try/except, no logger.info — both confuse Dynamo. Errors are
    # caught & counted inside _do_emit_next_layer_hint / the custom op
    # implementation. The "hook fired" debug message is emitted from
    # _do_emit_next_layer_hint on its first call instead.
    if _USE_CUSTOM_OP and _custom_op_registered:
        import torch as _torch
        _torch.ops.apex.hint_experts_for_next_layer(
            rl, int(layer_idx), int(top_k), int(n_experts)
        )
    else:
        _do_emit_next_layer_hint(
            rl, int(layer_idx), int(top_k), int(n_experts)
        )


def _hook_deepseek_v2_moe() -> bool:
    """Wrap ``DeepseekV2MoE.__init__`` to register a forward hook on
    ``self.experts`` and stash the per-instance APEX metadata.

    Crucially we do NOT replace ``DeepseekV2MoE.forward`` — that would
    drag our ctypes-based emit path inside the torch.compile graph, and
    vLLM's fullgraph mode bails on any ``torch.compiler.disable``-wrapped
    call. Module forward hooks are explicitly handled by Dynamo and run
    outside the compiled graph.
    """
    try:
        from vllm.model_executor.models.deepseek_v2 import DeepseekV2MoE
    except Exception as exc:
        logger.info("apex: deepseek_v2 not importable (%s); skipping hook", exc)
        return False

    if getattr(DeepseekV2MoE.__init__, "_apex_wrapped", False):
        return True

    original_init = DeepseekV2MoE.__init__

    def wrapped_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        try:
            prefix = kwargs.get("prefix", "")
            self._apex_layer_idx = _extract_layer_index(prefix or "")
            config = kwargs.get("config", None)
            if config is None and args:
                config = args[0]
            self._apex_topk = int(getattr(config, "num_experts_per_tok", 8) or 8)
            self._apex_n_routed_experts = int(
                getattr(config, "n_routed_experts", 0) or 0
            )
            experts = getattr(self, "experts", None)
            if experts is not None:
                # Stamp the per-layer APEX metadata directly on the experts
                # module. Plain ints / dead weakrefs don't trip
                # ``nn.Module.__setattr__``'s submodule auto-registration,
                # so we don't introduce a cycle in ``_modules`` (which would
                # blow the stack on ``model.eval()`` → recursive ``.train()``).
                experts._apex_layer_idx = self._apex_layer_idx
                experts._apex_topk = self._apex_topk
                experts._apex_n_routed_experts = self._apex_n_routed_experts
                # Keep the weakref too in case external callers want to walk
                # back from experts → parent MoE.
                experts._apex_parent_moe_ref = weakref.ref(self)
                try:
                    experts.register_forward_hook(
                        _on_experts_forward, with_kwargs=True
                    )
                except TypeError:
                    # Older torch (< 2.0) doesn't support with_kwargs.
                    experts.register_forward_hook(
                        lambda m, a, o: _on_experts_forward(m, a, {}, o)
                    )
        except Exception as exc:
            logger.warning("apex: deepseek init hook failed: %s", exc)
            _stats["errors"] += 1

    wrapped_init._apex_wrapped = True  # type: ignore[attr-defined]
    DeepseekV2MoE.__init__ = wrapped_init
    return True


def _emit_next_layer_hint(self, router_logits) -> None:
    """Compute top-K predicted experts from this layer's gate logits and
    emit an ExpertPrefetch hint targeting the *next* MoE layer (relies
    on the ~85% layer-to-layer expert-id correlation in DeepSeek-V3 /
    Kimi-K2).

    Always called from a PyTorch forward hook (``register_forward_hook``),
    which Dynamo handles by graph-breaking around ``__call__`` — so this
    function never executes inside a compiled graph and is free to call
    into ctypes / Python heap allocations.
    """
    topk_attr = int(getattr(self, "_apex_topk", 8) or 8)
    n_experts = int(
        getattr(self, "_apex_n_routed_experts", 0)
        or getattr(self, "n_routed_experts", 0)
        or 0
    )
    layer_idx = int(getattr(self, "_apex_layer_idx", 0) or 0)
    _do_emit_next_layer_hint(router_logits, layer_idx, topk_attr, n_experts)


def _do_emit_next_layer_hint(
    router_logits, layer_idx: int, top_k: int, n_experts: int
) -> None:
    """Pure-function form of :func:`_emit_next_layer_hint` — takes the
    per-instance metadata as explicit args so it can be wrapped by a
    ``torch.library.custom_op`` (which only accepts plain tensors and
    primitive types in its signature).
    """
    if _emit_log_counter[0] < 1:
        _emit_log_counter[0] += 1
        try:
            logger.info(
                "apex: experts forward-hook fired (layer=%d, top_k=%d, "
                "n_experts=%d, custom_op=%s)",
                int(layer_idx), int(top_k), int(n_experts),
                bool(_USE_CUSTOM_OP and _custom_op_registered),
            )
        except Exception:
            pass
    # The OLD path computed topk + .tolist() inline in the hot path.
    # That .tolist() forces a CUDA sync — waits for the entire GPU to
    # drain before the Python ints land. On a busy decode pipeline each
    # sync stalls ~tens of ms, and at 200 µs emit cap (≈300 emits/sec)
    # the cumulative cost dominates TPOT.
    #
    # Iteration measurements (DSV2-Lite, --cpu-offload-gb 16):
    #   hook off, emit off              :  -1.6% TPOT (noise)
    #   hook on, emit ON@200µs, warm off: -15.4% TPOT
    #   hook on, emit ON@200µs, warm on : -33.5% TPOT
    #
    # New path: enqueue (router_logits, metadata) to a background worker
    # thread. The worker does the GPU sync, topk, and ctypes round-trip
    # asynchronously. The hot path never blocks the decode pipeline.
    try:
        _enqueue_for_async_emit(router_logits, layer_idx, top_k, n_experts)
    except Exception as exc:
        _stats["errors"] += 1
        if _emit_error_log_counter[0] < 1:
            _emit_error_log_counter[0] += 1
            import sys
            import traceback
            msg = (
                f"apex: first emit error ({type(exc).__name__}): {exc!r}\n"
                f"{traceback.format_exc()}"
            )
            try:
                logger.warning(msg)
            except Exception:
                pass
            try:
                sys.__stderr__.write("[apex] " + msg + "\n")
                sys.__stderr__.flush()
            except Exception:
                pass
            try:
                if _stats_file_path:
                    err_path = _stats_file_path + ".firsterror"
                    with open(err_path, "w") as f:
                        f.write(msg)
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Fix (b): in-process torch.cuda.Stream-based prefetch of next-layer experts.
# ---------------------------------------------------------------------------
#
# This is the actual "do the work" path. The hint emitted to apexd is a
# *signal* that the runtime can route through any backend (kernel-mode
# page migration, rocm-xio P2P, …); on the standalone backend apexd's
# only available action is ``process_madvise`` which doesn't accelerate
# host→HBM streaming. Until apexd grows a HIP-side backend (per
# ``apex-rocm-xio-integration.md``, Phase 2) we get the real win in-
# process here: issue the H2D copy of the predicted expert tensors on a
# side CUDA stream while the current MoE layer is still computing. The
# copy overlaps with compute, the torch caching allocator holds the
# resulting buffer warm, and the next forward pass finds the data on GPU.
#
# Caveats (all instrumented via _stats so the dashboard can show what's
# happening):
#   * If both tensors are already on GPU (no offload, or already warm),
#     this is essentially a free pointer-check + counter increment.
#   * If torch.cuda is not initialised yet (early in startup) we skip
#     and try again on the next emit.
#   * We don't synchronise on the side stream — that's the whole point.
#     The next layer's forward will block on the data when it actually
#     touches it via cudaStreamWait or by sharing the default stream.


def _ensure_warm_stream():
    """Return the side CUDA stream (lazily created). None if cuda is not
    initialised yet or unavailable."""
    global _warm_stream
    if _warm_stream is not None:
        return _warm_stream
    try:
        import torch
        if not torch.cuda.is_available():
            return None
        if not torch.cuda.is_initialized():
            # Don't force initialisation here — wait until vLLM has
            # already done it.
            return None
        _warm_stream = torch.cuda.Stream(priority=-1)
        logger.info("apex: created side prefetch stream (priority=-1)")
        return _warm_stream
    except Exception as exc:
        logger.warning("apex: could not create warm stream: %s", exc)
        return None


def _apex_warm_next_layer(next_layer_idx: int, local_expert_ids: list[int]) -> None:
    """Issue async H2D prefetch for the predicted experts of the next
    MoE layer. Idempotent and best-effort — any failure is counted and
    swallowed so it can never break the inference path.

    Strategy:
      - Look up the next layer's FusedMoE via ``_moe_by_layer`` (populated
        in ``_apex_register_moe_experts``).
      - For each predicted expert id, take its ``w13_weight[eid]`` and
        ``w2_weight[eid]`` views.
      - If the view's storage is on CPU (i.e. cpu-offloaded), issue an
        async copy to GPU on the side stream. The resulting buffer lives
        in the torch caching allocator until vLLM next references it.
      - If already on GPU, just bump warm_gpu_hits — no work needed.
    """
    moe = _moe_by_layer.get(int(next_layer_idx))
    if moe is None:
        return
    stream = _ensure_warm_stream()
    if stream is None:
        return
    try:
        import torch
        w13 = getattr(moe, "w13_weight", None)
        w2 = getattr(moe, "w2_weight", None)
        if w13 is None or w2 is None:
            return
        device = torch.cuda.current_device()
        # Cap the number of experts we prefetch per call. Real Mixtral
        # top_k is 2; DeepSeek-V2 top_k is 6. The list we get is already
        # top-K; bound it again as a defensive cap so a buggy router
        # logits tensor can't make us copy 64 experts.
        ids = local_expert_ids[: max(1, min(len(local_expert_ids), 8))]
        with torch.cuda.stream(stream):
            for eid in ids:
                if eid < 0 or eid >= w13.shape[0]:
                    continue
                w13_e = w13[eid]
                w2_e = w2[eid]
                # ``w13`` may itself live on CPU after the UVA offloader
                # has flipped p.data → cpu tensor. ``.device`` is the
                # ground truth.
                if w13_e.device.type == "cpu":
                    # non_blocking is honoured only when the source is
                    # pinned. vLLM pins by default (UVAOffloader.pin_memory
                    # = is_pin_memory_available()), so this is async.
                    w13_e.to(device, non_blocking=True)
                    _stats["warm_bytes"] += int(w13_e.numel() * w13_e.element_size())
                    _stats["warm_cpu_hits"] += 1
                else:
                    _stats["warm_gpu_hits"] += 1
                if w2_e.device.type == "cpu":
                    w2_e.to(device, non_blocking=True)
                    _stats["warm_bytes"] += int(w2_e.numel() * w2_e.element_size())
                    _stats["warm_cpu_hits"] += 1
                else:
                    _stats["warm_gpu_hits"] += 1
        _stats["warm_calls"] += 1
    except Exception as exc:
        _stats["errors"] += 1
        if _stats["errors"] < 4:  # log first few only
            logger.warning("apex: warm-next-layer failed: %s", exc)


# Lazy-registered ``torch.library`` custom op. Wraps the side-effect-only
# emit path so torch.compile can see it as an opaque call (Dynamo never
# tries to trace into a registered custom op). Enabled by
# ``VLLM_APEX_USE_CUSTOM_OP=1``; otherwise we fall back to the direct
# Python call which only works with ``--enforce-eager``.
_custom_op_registered = False
_apex_lib = None  # type: ignore[var-annotated]


def _register_custom_op() -> bool:
    """Register ``torch.ops.apex.hint_experts_for_next_layer`` if torch
    is importable. Idempotent; safe to call before any forward."""
    global _custom_op_registered, _apex_lib
    if _custom_op_registered:
        return True
    try:
        import torch
        # Use the low-level Library API — it lets us declare a side-effect
        # op with a void return type, which ``torch.library.custom_op`` (the
        # high-level decorator) does not directly support.
        _apex_lib = torch.library.Library("apex", "FRAGMENT")
        # The (a!) annotation on router_logits declares "this op may mutate
        # its first argument". The op doesn't actually mutate anything, but
        # the declaration is what stops Inductor from dead-code-eliminating
        # the call (a void-return op with no declared side effects is
        # treated as pure and DCE'd, which would defeat the whole point).
        _apex_lib.define(
            "hint_experts_for_next_layer("
            "Tensor(a!) router_logits, int layer_idx, int top_k, int n_experts"
            ") -> ()"
        )

        def _impl(router_logits, layer_idx, top_k, n_experts):
            _do_emit_next_layer_hint(
                router_logits, int(layer_idx), int(top_k), int(n_experts)
            )

        def _meta_impl(router_logits, layer_idx, top_k, n_experts):
            # FakeTensor / meta-device pass executed by Dynamo during
            # tracing. We must NOT call .tolist() / .topk() on the input
            # here because it carries no data. The op has no return value,
            # so doing nothing is correct.
            return

        # Real impl: runs on CUDA *and* CPU. We don't bother with a CUDA
        # kernel — the .tolist() inside _do_emit_next_layer_hint moves
        # data to CPU explicitly.
        _apex_lib.impl(
            "hint_experts_for_next_layer", _impl, "CUDA"
        )
        _apex_lib.impl(
            "hint_experts_for_next_layer", _impl, "CPU"
        )
        _apex_lib.impl(
            "hint_experts_for_next_layer", _meta_impl, "Meta"
        )
        _custom_op_registered = True
        logger.info("apex: torch.ops.apex.hint_experts_for_next_layer registered")
        return True
    except Exception as exc:
        logger.warning("apex: custom op registration failed (%s)", exc)
        return False


def install_hooks() -> bool:
    """Install the monkey-patches that wire APEX hint emission into the
    surrounding vLLM. Returns True iff at least one hook was installed.

    Safe to call multiple times; each hook is idempotent.
    """
    global _hooks_installed
    if _hooks_installed:
        return True

    if not _ENABLED:
        # We still want to be importable when VLLM_APEX_HINTS=0; just
        # skip the hook installation in that case so vanilla vLLM is
        # bit-for-bit identical to upstream.
        return False

    if _USE_CUSTOM_OP:
        _register_custom_op()

    any_installed = False
    try:
        any_installed |= _hook_block_pool()
    except Exception as exc:
        logger.warning("apex: block_pool hook failed: %s", exc)
    try:
        any_installed |= _hook_gpu_model_runner()
    except Exception as exc:
        logger.warning("apex: gpu_model_runner hook failed: %s", exc)
    try:
        any_installed |= _hook_deepseek_v2_moe()
    except Exception as exc:
        logger.info("apex: deepseek_v2 hook not applicable: %s", exc)

    _hooks_installed = any_installed
    if any_installed:
        # Start the background emit-worker so the hot path can enqueue
        # without ever blocking. Cheap if already running.
        try:
            _start_emit_worker_once()
        except Exception as exc:
            logger.warning("apex: emit-worker start failed: %s", exc)
        logger.info(
            "apex: runtime hooks installed (emit_active=%s, "
            "emit_min_interval_us=%d, auto_noop=%s, offload_gb=%s)",
            emit_active(),
            _EMIT_MIN_INTERVAL_US,
            _EMIT_NOOP,
            os.environ.get("APEX_OFFLOAD_GB", "0"),
        )
    return any_installed
