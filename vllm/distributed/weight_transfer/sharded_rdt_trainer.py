# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Trainer-side engine for the sharded-RDT (pull-based NIXL) backend.

Symmetric to the NCCL/IPC trainer engines, but RDT is *pull-based*: the vLLM
workers initiate every transfer, dialing the trainer's Ray actors and pulling the
exact slice each worker consumes. So unlike NCCL (which broadcasts) this engine
pushes nothing from `send_weights`; instead it

  * owns a per-rank **producer server** -- an internal Ray actor exposing the NIXL
    serve surface (`rdt_produce_weights_batched`, `free_gather`,
    `reserve_serve_arena`) the worker engine calls by name, and
  * on each `send_weights`, gathers this rank's weights group-by-group from the
    `WeightSource`, shares each group into the server over CUDA IPC, and (on the
    sender) drives the inference-side `start/update/finish` handshake -- the single
    empty `update_weights` unblocks the workers to pull.

All serve-side state lives on the server actor, so trainer processes need no
mixin, no named actors and no `enable_tensor_transport` / `max_concurrency` actor
options: any process that can reach Ray and (for multi-rank)
`torch.distributed` works.

See docs/training/weight_transfer/sharded_rdt.md for the publish -> serve ->
free_gather -> release lifecycle, the ownership model, and the known concurrency
rough edges.
"""

import contextlib
import threading
import time
import uuid
from collections.abc import Callable, Collection
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from typing import Any, ClassVar

import ray
import torch
from torch.multiprocessing.reductions import rebuild_cuda_tensor, reduce_tensor
from typing_extensions import Self

from vllm.distributed.weight_transfer.base import (
    ParamMeta,
    TrainerInitInfo,
    TrainerWeightTransferEngine,
    VLLMWeightSyncClient,
    WeightSource,
    layerwise_groups,
)
from vllm.distributed.weight_transfer.sharded_rdt_common import (
    ALLOWED_OPS,
    RdtRouter,
    arena_alloc_bytes,
)
from vllm.logger import init_logger

logger = init_logger(__name__)

# How many gathered groups may be resident (and served) at once before the
# gather loop blocks. Bounds resident gathered groups on the trainer.
DEFAULT_GATHER_LOOKAHEAD = 2

# [RDT-STALL-WATCHDOG] Seconds of no progress at all — no publish, no produce, no
# free — before the producer declares the sync dead.
#
# A consumer that dies mid-sync never sends its ``free_gather``, so the group is
# never released and the three waits below block forever. That stops this rank
# iterating its WeightSource, which is a collective, which wedges every other
# trainer rank — with no exception surfacing anywhere until an unrelated NCCL
# watchdog kills training. The bound converts that into one real error.
#
# Liveness backstop, not a latency target: nominal inter-progress gaps are
# sub-second (the bake runs at init), so this is a ~100x margin even at 235B. It
# arrives on the init info rather than from the environment because the producer
# is a Ray actor and does not inherit the trainer process's environment.
DEFAULT_STALL_TIMEOUT_S = 300.0
# How often a blocked waiter wakes to re-check the progress stamp. Short enough
# that the reported stall time is accurate, long enough to be free.
_STALL_POLL_S = 5.0

# The actor method name the worker engine dials for the NIXL pull. Fixed by
# contract (ShardedRDTWeightTransferInitInfo.produce_method_name default).
PRODUCE_METHOD_NAME = "rdt_produce_weights_batched"


@dataclass
class ShardedRDTTrainerInitInfo(TrainerInitInfo):
    """Trainer init info for the sharded-RDT backend.

    Identical on every trainer rank except `rank` (kw-only, from the base;
    rank 0 is the sender). The trainer no longer supplies actor names — the
    engine generates a server-actor name per rank and all-gathers them across
    ranks — so this only carries the must-agree wire params, which the sender
    forwards verbatim onto the worker-side init info so the two sides can't
    drift.
    """

    backend: ClassVar[str] = "sharded_rdt"

    num_consumers: int
    """Total inference-worker (consumer) count across the whole fleet (TP*DP),
    for the M:N block assignment / free ref-count."""
    trainer_actor_namespace: str | None = None
    """Ray namespace the engine spawns its serve actors in. The inference
    workers (which run in their own EngineCore subprocess with its own
    ``ray.init``) resolve those actors by name, so this must be the namespace
    they can see. Forwarded to the worker-side init info."""
    num_rdt_buffers: int = 2
    """Serve/receive ring depth K (must match the worker)."""
    layerwise_split: int = 1
    """Chunk split S (forwarded to the worker; the producer mirrors its
    packed layout)."""
    arena_presize_gb: float = 0.0
    """Serve-arena pre-size floor in GiB (avoids NIXL desc-cache churn)."""
    nosync: bool = False
    """Scoped-sync serve: pack on a dedicated stream gated on gather events
    instead of a whole-device sync."""
    pack_check: bool = False
    """Emit per-blob checksums to /tmp/rdt_profile for offline diffing."""
    gather_lookahead: int = DEFAULT_GATHER_LOOKAHEAD
    """Resident gathered groups before the gather loop blocks."""
    stall_timeout_s: float = DEFAULT_STALL_TIMEOUT_S
    """Seconds of no publish/serve/free progress before the producer fails the
    sync (see ``DEFAULT_STALL_TIMEOUT_S``)."""


class _RDTProducerServer:
    """Per-rank NIXL serve surface for the sharded-RDT backend.

    Spawned by the engine as an internal Ray actor sharing the trainer rank's
    GPU (via CUDA IPC). Holds a gather cache of rebuilt IPC tensors, per-consumer
    serve rings, free ref-counting, and the byte-exact packed serve. The engine
    feeds it gathered weights with `publish_group`; the workers pull with
    `rdt_produce_weights_batched` and free with `free_gather`.

    This is a plain class; the engine wraps it with `ray.remote(...)` at spawn
    so the actor options (name / tensor transport / concurrency / GPU pinning)
    live in one place.
    """

    def __init__(
        self,
        *,
        num_rdt_buffers: int,
        arena_presize_gb: float,
        nosync: bool,
        pack_check: bool,
        gather_lookahead: int,
        served_names: list[str] | None = None,
        stall_timeout_s: float = DEFAULT_STALL_TIMEOUT_S,
    ) -> None:
        import gc

        self._device_index = torch.accelerator.current_device_index()
        # name -> rebuilt CUDA-IPC tensor (or view); guarded by _cache_cond.
        self._cache: dict[str, torch.Tensor] = {}
        self._cache_cond = threading.Condition()
        self._gather_error: BaseException | None = None

        # [RDT-STALL-WATCHDOG] Monotonic stamp of the last forward step of the
        # publish -> serve -> free credit loop, bumped under ``_cache_cond`` by
        # begin_sync / publish / produce completion / free. The three waits below
        # poll it instead of blocking forever, so a consumer that dies mid-sync
        # fails the sync with a real error rather than wedging every trainer rank.
        self._stall_timeout = float(stall_timeout_s)
        self._last_progress = time.monotonic()

        # Names this producer publishes, so a misrouted pull fails loudly
        # instead of blocking forever in the cache wait. None = serve anything.
        self._served_names = set(served_names) if served_names is not None else None

        # [RDT-FREE-REFCOUNT] Each consumer routed to this producer for a group
        # fires free_gather once; the group is actually freed (and reported back
        # to the engine) on the last of them. The target is per group — under
        # per-layer routing a producer serves different consumer counts for
        # different groups — and arrives with the group's publish.
        self._free_targets: dict[tuple, int] = {}
        self._free_counts: dict[tuple, int] = {}

        # [RDT-BACKPRESSURE] Published-but-not-yet-freed group keys. publish_group
        # blocks while len(...) >= gather_lookahead; free_gather (the consumer
        # back-edge) drains it. Freed keys are handed back to the engine so it
        # drops its trainer-side refs to the shared storage.
        self._lookahead = max(1, gather_lookahead)
        self._inflight_keys: list[tuple] = []
        self._freed_pending: list[tuple] = []

        # [RDT-RING] Per-consumer ring of packed serve arenas, rotated per pull.
        self._nring = max(1, num_rdt_buffers)
        self._serve_rings: dict[int, list[torch.Tensor | None]] = {}
        self._serve_idx: dict[int, int] = {}
        self._serve_lock = threading.Lock()
        # registerMem on a shared NIXL agent is not concurrency-safe; serialize.
        self._reg_lock = threading.Lock()
        self._arena_presize = int(arena_presize_gb * (1 << 30))

        # [RDT-NOSYNC] Scoped-sync serve stream + per-name completion events.
        # The stream's presence IS the mode: everything downstream branches on
        # ``self._serve_stream is not None``.
        self._serve_stream = torch.cuda.Stream() if nosync else None
        self._cache_event: dict[str, torch.cuda.Event] = {}

        self._pack_check = pack_check
        # [RDT-PACK-DSTS] (consumer_id, ring idx, packed layout) ->
        # (arena data_ptr, destination views). Keyed on the arena pointer too, so
        # a ring regrow invalidates rather than writing into a freed buffer. See
        # the serve path for why the layout, not the spec names, is the key.
        self._pack_dsts: dict[tuple, tuple[int, list[torch.Tensor]]] = {}

        # profiling counters
        self._timing_lock = threading.Lock()
        self._produce_calls = self._produce_specs = self._produce_bytes = 0
        self._produce_wait_seconds = self._produce_slice_seconds = 0.0
        self._produce_method_seconds = 0.0

        from vllm.distributed.weight_transfer._nixl_profile import (
            install_nixl_timing,
        )

        install_nixl_timing()  # fail-soft inside

        # Freeze the static post-init object graph so gen-2 GC never stops the
        # world mid-serve (measured straggler fix in the old producer).
        gc.collect()
        gc.freeze()

    def ping(self) -> int:
        return self._device_index

    # ---------------- stall watchdog ----------------

    def _note_progress_locked(self) -> None:
        """Record that the credit loop moved. Caller holds ``_cache_cond``."""
        self._last_progress = time.monotonic()

    def _wait_for(self, blocked: Callable[[], bool], what: str) -> None:
        """``_cache_cond.wait()`` with a liveness bound.

        Waits while ``blocked()`` holds, returning early on a gather error exactly
        as the unbounded waits it replaces did. If nothing anywhere on this
        producer makes progress for ``_stall_timeout``, self-fires the existing
        ``set_gather_error`` channel — which wakes every other waiter here and is
        already checked by all three loops — so the whole rank unwinds through one
        path and the driver gets a real exception.

        The stamp is global to the producer rather than per-waiter on purpose: a
        consumer that dies is not the only thing that stops, and a waiter that is
        merely slow is kept alive by its peers' progress. Caller holds
        ``_cache_cond``.
        """
        while blocked():
            if self._gather_error is not None:
                return
            self._cache_cond.wait(_STALL_POLL_S)
            if not blocked() or self._gather_error is not None:
                return
            stalled = time.monotonic() - self._last_progress
            if stalled >= self._stall_timeout:
                msg = (
                    f"RDT stall: no progress for {stalled:.0f}s while waiting for "
                    f"{what} (timeout {self._stall_timeout:.0f}s). A consumer most "
                    f"likely died mid-sync: {len(self._inflight_keys)} group(s) "
                    f"published and unfreed."
                )
                logger.error("[rdt-stall] %s", msg)
                # Same channel a gather failure uses: wakes every waiter here, and
                # each raises rather than returning a half-served result.
                self._gather_error = RuntimeError(msg)
                self._cache_cond.notify_all()
                return

    def warmup_nixl(self) -> None:
        """Create this server's NIXL agent NOW, while the rank's GPU is quiet.

        The engine calls this at spawn time, before the server-name all-gather
        barrier -- i.e. before any trainer rank can be spinning in a collective.
        Creating the agent lazily instead deadlocks on EFA-class fabrics; see the
        cycle in docs/training/weight_transfer/sharded_rdt.md ("warmup_nixl breaks
        a startup deadlock"). The warmup buffer stays registered and alive so the
        agent and its CUDA-HMEM path stay initialized.
        """
        from ray.experimental import register_nixl_memory

        self._nixl_warmup_buf = torch.zeros(1 << 20, dtype=torch.uint8, device="cuda")
        with self._reg_lock:
            register_nixl_memory(self._nixl_warmup_buf)

    # ---------------- engine-facing (per sync) ----------------

    def begin_sync(self) -> None:
        """Reset per-sync free/backpressure state. The driver awaits the
        previous sync's finish (which drains every consumer's frees) before the
        next begins, so nothing is in flight here.

        The packed-destination cache deliberately SURVIVES: the layout repeats
        every sync, which is what makes caching it worthwhile.
        """
        with self._cache_cond:
            self._gather_error = None
            self._inflight_keys.clear()
            self._freed_pending.clear()
            self._free_counts.clear()
            self._free_targets.clear()
            self._note_progress_locked()

    def _release_group_locked(self, key: tuple) -> None:
        """Drop a freed group's cache entries and release its backpressure slot.

        The group key IS its name tuple, so no name -> key map is needed: a free
        whose names do not match a published group cannot silently release a
        different group's slot.

        Caller must hold ``_cache_cond``.
        """
        for name in key:
            self._cache.pop(name, None)
            self._cache_event.pop(name, None)
        self._free_counts.pop(key, None)
        self._free_targets.pop(key, None)
        if key in self._inflight_keys:
            self._inflight_keys.remove(key)
            self._freed_pending.append(key)

    def publish_group(
        self, group_key: tuple, entries: tuple, free_target: int
    ) -> list[tuple]:
        """Rebuild one gather group's CUDA-IPC tensors and publish to the serve
        cache. Blocks while `gather_lookahead` groups are already in flight so
        trainer memory stays bounded (the consumer's `free_gather` drains it).
        Returns the group keys freed since the last call so the engine can drop
        its refs to the shared storage.

        ``entries`` is ``(storages, views)``: ``storages`` maps a storage id to
        the ``reduce_tensor`` args of a whole-storage uint8 view (ONE CUDA-IPC
        export per storage), ``views`` maps each served name to
        ``(sid, dtype_name, shape, stride, storage_offset)`` -- rebuilt here as
        ``as_strided`` views. One export per storage rather than per name; see the
        doc for the cost this replaced.

        ``free_target`` is how many consumers are routed to this producer for this
        group (>= 1; the engine skips publishing a group no consumer pulls from
        it). Frees can arrive BEFORE the publish they belong to -- a consumer that
        pulls nothing of a group frees it as soon as its plan starts -- so a group
        whose frees have already all landed is released here rather than waiting
        for one that will never come.
        """
        with self._cache_cond:
            self._wait_for(
                lambda: len(self._inflight_keys) >= self._lookahead,
                "a lookahead credit",
            )

        storages, views = entries
        bases: dict[int, torch.Tensor] = {}
        for sid, reduce_args in storages.items():
            list_args = list(reduce_args)
            # Index 6 of reduce_tensor's args is the exporter's device index;
            # rebuild on this server's device (same physical GPU as the rank).
            list_args[6] = self._device_index
            bases[sid] = rebuild_cuda_tensor(*list_args)
        rebuilt: dict[str, torch.Tensor] = {}
        for name, (sid, dtype_name, shape, stride, storage_offset) in views.items():
            typed = bases[sid].view(getattr(torch, dtype_name))
            rebuilt[name] = torch.as_strided(typed, shape, stride, storage_offset)
        del bases

        ev = None
        if self._serve_stream is not None:
            ev = torch.cuda.Event()
            ev.record()
        with self._cache_cond:
            self._cache.update(rebuilt)
            if ev is not None:
                for n in rebuilt:
                    self._cache_event[n] = ev
            self._inflight_keys.append(group_key)
            self._free_targets[group_key] = max(1, int(free_target))
            if self._free_counts.get(group_key, 0) >= self._free_targets[group_key]:
                self._release_group_locked(group_key)
            freed = self._freed_pending
            self._freed_pending = []
            self._note_progress_locked()
            self._cache_cond.notify_all()
        return freed

    def end_sync(self) -> list[tuple]:
        """Block until every published group has been freed by its consumers;
        return the remaining freed keys so the engine drops its last refs."""
        with self._cache_cond:
            self._wait_for(
                lambda: bool(self._inflight_keys),
                "every published group to be freed",
            )
            freed = self._freed_pending
            self._freed_pending = []
            return freed

    def set_gather_error(self, message: str) -> None:
        """Record a trainer-side gather failure so blocked serves / publishes
        stop waiting and surface it."""
        with self._cache_cond:
            self._gather_error = RuntimeError(message)
            self._cache_cond.notify_all()

    # ---------------- consumer-facing (called by name over Ray) ----------------

    def free_gather(self, names: list[str]) -> None:
        """Consumer back-edge: one consumer finished pulling this group.

        Ref-counts to the group's ``free_target``; on the last free, drops the
        cache entries, releases one backpressure slot, and records the freed key
        for the engine. A free that arrives before its publish is only counted —
        ``publish_group`` completes it.
        """
        key = tuple(names)
        with self._cache_cond:
            count = self._free_counts.get(key, 0) + 1
            self._free_counts[key] = count
            target = self._free_targets.get(key)
            if target is not None and count >= target:
                self._release_group_locked(key)
            self._note_progress_locked()
            self._cache_cond.notify_all()

    def reserve_serve_arena(self, consumer_id: int, nbytes: int) -> None:
        """Pre-allocate + NIXL-register this consumer's serve ring before any
        pull, while the fabric is idle (avoids registration races during the
        sync-0 RDMA churn under M:N fan-in). Idempotent; grows only if needed."""
        from ray.experimental import register_nixl_memory

        alloc = arena_alloc_bytes(nbytes, self._arena_presize)
        with self._serve_lock:
            rings = self._serve_rings.setdefault(consumer_id, [None] * self._nring)
            self._serve_idx.setdefault(consumer_id, 0)
        for i in range(self._nring):
            slot = rings[i]
            if slot is None or slot.numel() < alloc:
                t = torch.empty(alloc, dtype=torch.uint8, device="cuda:0")
                with self._reg_lock:
                    register_nixl_memory(t)
                rings[i] = t

    @ray.method(tensor_transport="nixl")
    def rdt_produce_weights_batched(self, specs: list, consumer_id: int = 0):
        """Serve one batched slice request over NIXL.

        Waits until the specs' names are cached, replays each spec's op chain
        (pure views into cached tensors, guarded by ALLOWED_OPS), byte-packs the
        slices 16B-aligned into this consumer's ring slot (mirroring the
        consumer's identical layout), and returns the one packed blob. Callers
        that want a single slice pass one spec and read the blob back with that
        slice's dtype/shape.
        """
        t_m0 = time.perf_counter()
        needed = sorted({n for n, _ in specs})
        if self._served_names is not None:
            unserved = [n for n in needed if n not in self._served_names]
            if unserved:
                # Without this the cache wait below would block forever: this
                # producer never gathers these names, so they can never arrive.
                raise RuntimeError(
                    f"pull routed to the wrong producer: {unserved[:3]} "
                    f"({len(unserved)} names) are not served here"
                )
        t_w0 = time.perf_counter()
        with self._cache_cond:
            self._wait_for(
                lambda: not all(n in self._cache for n in needed),
                f"{len(needed)} name(s) to be published",
            )
            if not all(n in self._cache for n in needed):
                # _wait_for only gives up on a gather error (its own stall
                # included), so the names are never coming.
                raise RuntimeError(
                    f"gather errored before {needed}: {self._gather_error!r}"
                )
        wait_s = time.perf_counter() - t_w0

        t_s0 = time.perf_counter()
        sliced: list = []  # (byte_off, tensor)
        pack_cur = 0
        nbytes = 0
        for name, chain in specs:
            t = self._cache[name]
            for op, args, kw in chain:
                if op not in ALLOWED_OPS:
                    raise ValueError(f"{name!r}: disallowed op {op!r}")
                t = getattr(t, op)(*args, **dict(kw))
            off = (pack_cur + 15) & ~15
            pack_cur = off + t.numel() * t.element_size()
            sliced.append((off, t))
            nbytes += t.numel() * t.element_size()

        with self._serve_lock:
            rings = self._serve_rings.setdefault(consumer_id, [None] * self._nring)
            idx = self._serve_idx.get(consumer_id, 0)
            self._serve_idx[consumer_id] = (idx + 1) % self._nring
        arena = rings[idx]
        if arena is None or arena.numel() < pack_cur:
            from ray.experimental import register_nixl_memory

            alloc = arena_alloc_bytes(pack_cur, self._arena_presize)
            arena = torch.empty(alloc, dtype=torch.uint8, device="cuda:0")
            with self._reg_lock:
                register_nixl_memory(arena)
            rings[idx] = arena

        ss = self._serve_stream
        if ss is not None:
            for ev in {
                id(e): e
                for e in (self._cache_event.get(n) for n in needed)
                if e is not None
            }.values():
                ss.wait_event(ev)
        # [RDT-PACK-DSTS] The destination views are a pure function of this
        # consumer's packed layout, which is byte-identical every sync (its plan
        # is static), so build them ONCE per (consumer, ring slot, layout) and
        # reuse. Rebuilding them per call cost 5.2ms of the 7.5ms measured for a
        # 384-spec 235B group — three Python ops per spec, all redundant.
        # _foreach_copy_ then issues the copies in one dispatch (a further
        # 0.6ms; verified byte-identical to the per-view loop).
        #
        # The key is the LAYOUT the views were carved for — each slice's packed
        # offset, dtype and shape — not the spec names. Names alone do not
        # identify it: a name can appear in two requests with different op chains
        # (the same source sliced differently, which layerwise_split > 1 produces
        # when one name's copies land in separate chunks), and serving the second
        # through the first's views would write the wrong bytes with nothing
        # downstream to catch it. Building the signature costs ~1.5% of the pack.
        dst_key = (
            consumer_id,
            idx,
            tuple((off, t.dtype, t.shape) for off, t in sliced),
        )
        cached = self._pack_dsts.get(dst_key)
        if cached is None or cached[0] != arena.data_ptr():
            dsts = []
            for off, t in sliced:
                nb = t.numel() * t.element_size()
                dsts.append(arena[off : off + nb].view(t.dtype).reshape(t.shape))
            self._pack_dsts[dst_key] = (arena.data_ptr(), dsts)
        else:
            dsts = cached[1]
        with torch.cuda.stream(ss):
            torch._foreach_copy_(dsts, [t for _off, t in sliced])
        if ss is not None:
            ss.synchronize()

        blob = arena[:pack_cur]
        if self._pack_check:
            self._log_pack_check(blob, pack_cur)
        self._bump_timing(t_m0, wait_s, t_s0, len(specs), nbytes)
        return [blob]

    # ---------------- profiling ----------------

    def _bump_timing(self, t_m0, wait_s, t_s0, nspecs, nbytes) -> None:
        slice_s = time.perf_counter() - t_s0
        # A served pull is the third of the four progress signals the stall
        # watchdog reads (publish / produce / free / begin_sync): a long sync whose
        # consumers pull steadily but slowly must never trip it.
        with self._cache_cond:
            self._note_progress_locked()
        with self._timing_lock:
            self._produce_calls += 1
            self._produce_specs += nspecs
            self._produce_wait_seconds += wait_s
            self._produce_slice_seconds += slice_s
            self._produce_bytes += nbytes
            self._produce_method_seconds += time.perf_counter() - t_m0

    def _log_pack_check(self, blob: torch.Tensor, pack_cur: int) -> None:
        import json
        import os

        s = 0
        w = 32 << 20
        for i in range(0, pack_cur, w):
            s += int(blob[i : min(i + w, pack_cur)].sum(dtype=torch.int64))
        os.makedirs("/tmp/rdt_profile", exist_ok=True)
        with open("/tmp/rdt_profile/packcheck_prod.jsonl", "a") as f:
            f.write(
                json.dumps({"pid": os.getpid(), "bytes": pack_cur, "sum": s}) + "\n"
            )

    def get_produce_timing(self) -> dict:
        with self._timing_lock:
            return dict(
                calls=self._produce_calls,
                specs=self._produce_specs,
                wait_seconds=self._produce_wait_seconds,
                slice_seconds=self._produce_slice_seconds,
                bytes=self._produce_bytes,
                method_seconds=self._produce_method_seconds,
            )

    def reset_produce_timing(self) -> None:
        with self._timing_lock:
            self._produce_calls = self._produce_specs = self._produce_bytes = 0
            self._produce_wait_seconds = self._produce_slice_seconds = 0.0
            self._produce_method_seconds = 0.0

    def get_nixl_timing(self) -> dict:
        from vllm.distributed.weight_transfer import _nixl_profile

        return _nixl_profile.snapshot()

    def reset_nixl_timing(self) -> None:
        from vllm.distributed.weight_transfer import _nixl_profile

        _nixl_profile.reset()

    def shutdown(self) -> None:
        with self._cache_cond:
            self._cache.clear()
            self._cache_event.clear()
        with self._serve_lock:
            self._serve_rings.clear()
            # Must go with the rings: these are views INTO them, and the
            # data_ptr guard that normally invalidates them cannot tell a freed
            # arena from a new one recycled at the same address.
            self._pack_dsts.clear()


class ShardedRDTTrainerWeightTransferEngine(
    TrainerWeightTransferEngine[ShardedRDTTrainerInitInfo]
):
    """Trainer-side engine for the pull-based sharded-RDT backend.

    Lives on every trainer rank. Owns a per-rank `_RDTProducerServer` actor
    (the NIXL serve surface). `send_weights` gathers this rank's weights
    group-by-group from the `WeightSource`, shares each group into the server
    over CUDA IPC, and — on the sender — drives the inference-side handshake so
    the workers pull. Non-sender ranks only gather (staying in the collective).
    """

    init_info_cls = ShardedRDTTrainerInitInfo

    def __init__(
        self,
        *,
        client: VLLMWeightSyncClient,
        source: WeightSource,
        is_sender: bool = True,
        init_info: ShardedRDTTrainerInitInfo,
    ) -> None:
        super().__init__(client=client, source=source, is_sender=is_sender)
        self._init_info = init_info
        self._server: Any = None  # the _RDTProducerServer actor handle
        self._server_name: str | None = None
        # Group-major metadata / partition, computed at trainer_init.
        self._meta: list[ParamMeta] = []
        self._groups: list[list[str]] = []
        # Per-group routing, resolved at trainer_init from the source's ownership.
        self._router: RdtRouter | None = None
        self._group_owners: list[list[int]] = []
        self._owned_idx: list[int] = []
        self._free_targets: dict[int, int] = {}
        # Strong refs to gathered tensors we've shared into the server, keyed by
        # group key. CUDA-IPC exports must outlive the importer, so we hold them
        # until the server reports the group freed. See send_weights.
        self._inflight: dict[tuple, dict[str, torch.Tensor]] = {}
        self._sync_timing: dict[str, float] = {}

    def _rpc(self, method: str, *args: Any) -> Any:
        """Call one of the server actor's methods and block for the result.
        The single seam through which the engine talks to its server, so tests
        can inject a local (non-Ray) fake server."""
        import ray

        return ray.get(getattr(self._server, method).remote(*args))

    def _publish_async(self, key, entries, free_target):
        """Fire publish_group WITHOUT blocking (the gather loop overlaps the
        publish with the next group's gather) and return a handle that
        ``_drop_when_ready`` resolves. Ray actor handle in production; a plain
        (non-Ray) fake server runs inline and returns its result directly."""
        method = self._server.publish_group
        remote = getattr(method, "remote", None)
        if remote is not None:
            return remote(key, entries, free_target)
        return method(key, entries, free_target)

    def _drop_when_ready(self, ref) -> None:
        import ray

        freed = ray.get(ref) if isinstance(ref, ray.ObjectRef) else ref
        self._drop_inflight(freed)

    # ---------------- construction ----------------

    @classmethod
    def trainer_init(
        cls,
        init_info: ShardedRDTTrainerInitInfo,
        *,
        client: VLLMWeightSyncClient,
        source: WeightSource | None = None,
    ) -> Self:
        if source is None:
            raise ValueError(
                "Sharded RDT trainer weight transfer requires a WeightSource."
            )
        engine = cls(
            client=client,
            source=source,
            is_sender=init_info.is_sender,
            init_info=init_info,
        )

        # CUDA-IPC publish is the trainer engine's hot path, and the VMM-based
        # expandable_segments allocator makes IPC storage opens ~9x slower on both
        # the export and rebuild side (measured). Frameworks that enable it should
        # disable it around send_weights so the gathered tensors land in classic,
        # IPC-fast segments. This only sees the env var -- runtime
        # _set_allocator_settings changes are not introspectable.
        import os as _os

        if "expandable_segments:True" in _os.environ.get("PYTORCH_CUDA_ALLOC_CONF", ""):
            logger.warning(
                "Sharded-RDT trainer: PYTORCH_CUDA_ALLOC_CONF enables "
                "expandable_segments; CUDA-IPC weight publishing will be "
                "several times slower. Disable expandable segments around "
                "weight sync (allocate gather buffers in classic segments)."
            )

        engine._meta = list(source.metadata())
        names = [m.name for m in engine._meta]
        engine._groups = layerwise_groups(names)
        flat = [n for g in engine._groups for n in g]
        if flat != names:
            raise ValueError(
                "Sharded RDT requires a WeightSource whose metadata order is "
                "group-contiguous (pre / per-decoder-layer / post). Reorder the "
                "source so each model.layers.<N>.* block is contiguous."
            )

        world, rank = engine._world_and_rank()
        engine._build_router(world, rank)
        served = [
            n
            for gi in engine._owned_idx
            if engine._free_targets[gi] > 0
            for n in engine._groups[gi]
        ]
        engine._spawn_server(served)

        # Every rank's server must exist before the sender's init RPC (the worker
        # init calls reserve_serve_arena back on ALL producer servers). The
        # all-gather of server names doubles as that barrier.
        server_names = engine._all_gather_server_names(world, rank)

        if engine.is_sender:
            worker_init = engine._build_worker_init_info(server_names)
            engine.client.init_weight_transfer_engine(asdict(worker_init))
        return engine

    def _world_and_rank(self) -> tuple[int, int]:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            return torch.distributed.get_world_size(), torch.distributed.get_rank()
        return 1, self._init_info.rank

    def _build_router(self, world: int, rank: int) -> None:
        """Resolve per-group ownership and this rank's publish plan.

        A source may gather only part of the model — pipeline-parallel producers
        gather within a stage rather than to all ranks — so ownership is
        all-gathered here and shipped to the consumers in the worker init info.
        Every rank must agree on who serves each group: a consumer pulling from
        a producer that never gathered the group would block forever. Sources
        without ``owned_groups`` own everything, keeping the gather-to-all
        layout on its historical path.
        """
        assert self.source is not None  # guaranteed by trainer_init
        num_groups = len(self._groups)
        owned: list[int] | None = None
        declared = self.source.owned_groups()
        if declared is not None:
            owned = sorted({int(g) for g in declared})
            bad = [g for g in owned if not 0 <= g < num_groups]
            if bad:
                raise ValueError(
                    f"WeightSource.owned_groups() out of range for "
                    f"{num_groups} groups: {bad}"
                )
            if not owned:
                raise ValueError(
                    "WeightSource.owned_groups() is empty; a rank with nothing "
                    "to serve cannot take part in the gather"
                )

        # One collective carries both: the ownership lists and a digest of the
        # metadata they index into. The digest check is what makes partial
        # ownership safe — only the sender's metadata reaches the consumers, so a
        # rank that described just its own share would leave the rest of the
        # model silently un-transferred instead of failing.
        digest = self._meta_digest()
        per_rank = self._all_gather_owned(world, (digest, owned))
        mismatched = [r for r, (d, _o) in enumerate(per_rank) if d != digest]
        if mismatched:
            raise ValueError(
                f"WeightSource.metadata() disagrees across trainer ranks "
                f"(rank {rank} digest {digest}, differing ranks {mismatched[:4]}). "
                "Every rank must describe the WHOLE model, even when it owns "
                "only some groups."
            )
        owned_per_rank = [o for _d, o in per_rank]

        group_owners: list[list[int]] | None = None
        if any(o is not None for o in owned_per_rank):
            group_owners = [[] for _ in range(num_groups)]
            for r, rank_owned in enumerate(owned_per_rank):
                for gi in range(num_groups) if rank_owned is None else rank_owned:
                    group_owners[gi].append(r)

        router = RdtRouter(
            world, self._init_info.num_consumers, group_owners, num_groups
        )
        router.validate()
        self._router = router
        self._group_owners = group_owners or []
        self._owned_idx = owned if owned is not None else list(range(num_groups))
        self._free_targets = {
            gi: router.free_target(rank, gi) for gi in self._owned_idx
        }

    def _meta_digest(self) -> str:
        """Stable digest of this rank's metadata (name order + count)."""
        import hashlib

        h = hashlib.sha256()
        h.update(f"{len(self._meta)}\n".encode())
        for m in self._meta:
            h.update(m.name.encode())
            h.update(b"\n")
        return h.hexdigest()[:16]

    def _all_gather_owned(self, world: int, mine: tuple) -> list[tuple]:
        """All-gather each rank's (metadata digest, owned groups)."""
        if world <= 1 or not (
            torch.distributed.is_available() and torch.distributed.is_initialized()
        ):
            return [mine]
        gathered: list[Any] = [None] * world
        torch.distributed.all_gather_object(gathered, mine)
        return gathered

    def _all_gather_server_names(self, world: int, rank: int) -> list[str]:
        assert self._server_name is not None
        if world <= 1 or not (
            torch.distributed.is_available() and torch.distributed.is_initialized()
        ):
            return [self._server_name]
        gathered: list[str | None] = [None] * world
        torch.distributed.all_gather_object(gathered, self._server_name)
        return [n for n in gathered if n is not None]

    def _spawn_server(self, served_names: list[str]) -> None:
        import ray
        from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

        ii = self._init_info
        self._server_name = f"vllm_rdt_producer_{uuid.uuid4().hex[:12]}_rk{ii.rank}"
        node_id = ray.get_runtime_context().get_node_id()
        # Pin the server to this rank's physical GPU (num_gpus=0 so Ray doesn't
        # allocate a second one; CUDA_VISIBLE_DEVICES makes CUDA IPC to the
        # rank's gathered tensors possible — same device family as the IPC
        # backend). max_concurrency > 1: serves pulls while begin/publish/end
        # calls are in flight. enable_tensor_transport: NIXL serve.
        gpu_ids = ray.get_gpu_ids()
        # The server is the trainer rank's process twin: forward the env the
        # rank runs under (library paths etc.) so it imports torch/vllm the
        # same way, then pin it to the rank's physical GPU for CUDA IPC.
        import os

        env_vars = {
            k: os.environ[k]
            for k in (
                "LD_LIBRARY_PATH",
                "LD_PRELOAD",
                "NCCL_CUMEM_ENABLE",
                "VLLM_NCCL_SO_PATH",
                "PATH",
            )
            if k in os.environ
        }
        if gpu_ids:
            # The server is a num_gpus=0 actor (it SHARES the rank's GPU for
            # CUDA IPC, so it must not claim a second one). Ray would otherwise
            # set CUDA_VISIBLE_DEVICES="" and hide every GPU; tell it not to
            # touch the var (the pattern the weight-transfer tests use) and pin
            # the server to the rank's physical GPU ourselves.
            env_vars["RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES"] = "1"
            env_vars["CUDA_VISIBLE_DEVICES"] = str(gpu_ids[0])
        runtime_env = {"env_vars": env_vars} if env_vars else {}
        server_cls = ray.remote(_RDTProducerServer).options(
            name=self._server_name,
            namespace=ii.trainer_actor_namespace,
            num_cpus=0,
            num_gpus=0,
            # Thread budget: up to K produce calls sit BLOCKED in cache-wait
            # per bound consumer, plus one backpressure-blocked publish_group,
            # plus free_gather / begin/end_sync must still get a thread
            # promptly (a queued free_gather stalls the whole credit loop).
            max_concurrency=max(8, 2 * ii.num_rdt_buffers + 4),
            enable_tensor_transport=True,
            scheduling_strategy=NodeAffinitySchedulingStrategy(
                node_id=node_id, soft=False
            ),
            runtime_env=runtime_env,
        )
        self._server = server_cls.remote(
            served_names=served_names,
            num_rdt_buffers=ii.num_rdt_buffers,
            arena_presize_gb=ii.arena_presize_gb,
            nosync=ii.nosync,
            pack_check=ii.pack_check,
            gather_lookahead=ii.gather_lookahead,
            stall_timeout_s=ii.stall_timeout_s,
        )
        ray.get(self._server.ping.remote())
        # Pre-barrier NIXL warmup: must complete before the server-name
        # all-gather so no rank can be inside a collective yet (see warmup_nixl).
        ray.get(self._server.warmup_nixl.remote())

    def _build_worker_init_info(self, server_names: list[str]):
        from vllm.distributed.weight_transfer.sharded_rdt_engine import (
            ShardedRDTWeightTransferInitInfo,
        )

        group_lens = [len(g) for g in self._groups]
        names = [m.name for m in self._meta]
        dtype_names = [str(m.dtype).split(".")[-1] for m in self._meta]
        shapes = [list(m.shape) for m in self._meta]
        return ShardedRDTWeightTransferInitInfo(
            trainer_actor_names=server_names,
            trainer_actor_namespace=self._init_info.trainer_actor_namespace,
            produce_method_name=PRODUCE_METHOD_NAME,
            names=names,
            dtype_names=dtype_names,
            shapes=shapes,
            group_lens=group_lens,
            group_owners=self._group_owners,
            num_consumers=self._init_info.num_consumers,
            num_rdt_buffers=self._init_info.num_rdt_buffers,
            layerwise_split=self._init_info.layerwise_split,
            arena_presize_gb=self._init_info.arena_presize_gb,
            pack_check=self._init_info.pack_check,
        )

    # ---------------- per-round ----------------

    @contextlib.contextmanager
    def _live_consumers(self, live_consumer_ids: Collection[int] | None):
        """Scope this sync's free targets to the consumers still alive.

        The provisioned geometry — ``num_consumers``, the router, the ownership
        table, ``_owned_idx``, the ``served_names`` the producer registered — is
        FROZEN for the run and is untouched here. All that changes is how many
        consumers each owned group expects a ``free_gather`` from, which
        ``publish_group`` already takes as an argument on every call. So syncing
        to a degraded fleet is a per-sync recompute, not a protocol change.

        Groups whose live target falls to zero are gathered and dropped by the
        existing publish loop: the gather is a collective across the group's
        owners and must run on every rank regardless, but publishing a group
        nobody will free would park a backpressure slot until ``end_sync`` waited
        forever.

        Every rank must be given the SAME live set — they are all inside the same
        gather collectives — so the caller has to compute it once and dispatch it
        to all of them.
        """
        if live_consumer_ids is None:
            yield
            return
        assert self._router is not None  # set by trainer_init
        rank = self._world_and_rank()[1]
        provisioned = self._free_targets
        live = sorted(set(live_consumer_ids))
        self._free_targets = {
            gi: self._router.free_target(rank, gi, live) for gi in self._owned_idx
        }
        dropped = sum(
            1
            for gi in self._owned_idx
            if self._free_targets[gi] <= 0 < provisioned.get(gi, 0)
        )
        logger.warning(
            "[rdt-degraded] serving %d/%d live consumers; %d of this rank's %d "
            "owned groups become gather-and-drop",
            len(live),
            self._init_info.num_consumers,
            dropped,
            len(self._owned_idx),
        )
        try:
            yield
        finally:
            self._free_targets = provisioned

    def send_weights(self, live_consumer_ids: Collection[int] | None = None) -> None:
        """Gather this rank's weights and publish them for the consumers to pull.

        ``live_consumer_ids`` restricts the sync to the consumers still alive;
        ``None`` (the default) serves the whole provisioned set. See
        ``_live_consumers``.
        """
        assert self.source is not None
        with self._live_consumers(live_consumer_ids):
            self._send_weights_inner()

    def _send_weights_inner(self) -> None:
        if not self.is_sender:
            self._run_gather_loop(update_future=None)
            return

        wall0 = time.perf_counter()
        t0 = time.perf_counter()
        self.client.start_weight_update()
        self._sync_timing["start_seconds"] = time.perf_counter() - t0

        from vllm.distributed.weight_transfer.sharded_rdt_engine import (
            ShardedRDTWeightTransferUpdateInfo,
        )

        empty_update = asdict(ShardedRDTWeightTransferUpdateInfo())
        with ThreadPoolExecutor(max_workers=1) as exe:
            # The workers block inside update_weights until they've pulled every
            # group, so it runs concurrently with the gather/publish loop.
            tu0 = time.perf_counter()
            future = exe.submit(self.client.update_weights, empty_update)
            self._run_gather_loop(update_future=future)
            future.result()  # surface inference-side errors
            self._sync_timing["update_weights_seconds"] = time.perf_counter() - tu0

        tf0 = time.perf_counter()
        self.client.finish_weight_update()
        self._sync_timing["finish_seconds"] = time.perf_counter() - tf0
        self._sync_timing["wall_seconds"] = time.perf_counter() - wall0

    def _run_gather_loop(self, update_future) -> None:
        """Gather this rank's weights group-by-group and publish each into the
        server over CUDA IPC. `publish_group` blocks when the lookahead is full,
        so the loop self-paces to the consumers' pull rate. Runs on every rank;
        only the sender has an `update_future` to fail fast on."""
        gather0 = time.perf_counter()
        assert self.source is not None  # guaranteed by trainer_init
        self._rpc("begin_sync")
        # One generator resume per GROUP, not per tensor: `iter_groups` yields
        # (names, tensors) for each group this rank owns, in metadata order.
        # Sources that can materialize a whole group at once override it; the
        # base default batches `__iter__` and checks the order as it goes. Every
        # owner of a group must reach it in the same order or their shared gather
        # collective mismatches.
        groups = self.source.iter_groups()
        # Publishes are fired without an inline ray.get and harvested this
        # window deep, so the publish RPC + server-side rebuild overlap the
        # NEXT group's gather/export instead of serializing the loop. The
        # server's lookahead backpressure still bounds resident groups (a
        # queued publish blocks server-side; worst case lookahead + window - 1
        # groups are live at once).
        _PUBLISH_WINDOW = 2
        pending_publish: list = []

        try:
            for gi in self._owned_idx:
                group = self._groups[gi]
                key = tuple(group)
                names, tensors = next(groups)
                if list(names) != list(group):
                    raise RuntimeError(
                        f"WeightSource group yielded {len(names)} names starting "
                        f"{names[:2]!r} but expected {len(group)} starting "
                        f"{group[:2]!r}; iteration order must match metadata."
                    )
                free_target = self._free_targets.get(gi, 0)
                if free_target <= 0:
                    # Gathered (the group's collective spans every owner) but no
                    # consumer pulls it from this rank. Publishing it would park
                    # a group nobody frees, holding a backpressure slot until
                    # end_sync waits forever.
                    names, tensors = [], []
                    continue
                # Share each unique STORAGE once (one cudaIpc export instead of
                # one per name) and describe every name as an as_strided view
                # spec relative to its storage.
                storages: dict[int, tuple] = {}
                views: dict[str, tuple] = {}
                refs: dict[str, torch.Tensor] = {}
                for name, tensor in zip(names, tensors):
                    tensor = tensor.detach()
                    if not tensor.is_cuda:
                        tensor = tensor.cuda()
                    tensor = tensor.contiguous()
                    refs[name] = tensor  # keep the export alive
                    ust = tensor.untyped_storage()
                    sid = ust.data_ptr()
                    if sid not in storages:
                        base = torch.empty(0, dtype=torch.uint8, device=tensor.device)
                        base.set_(ust, 0, (ust.nbytes(),))
                        _rebuild, reduce_args = reduce_tensor(base)
                        storages[sid] = reduce_args
                    views[name] = (
                        sid,
                        str(tensor.dtype).split(".")[-1],
                        list(tensor.shape),
                        list(tensor.stride()),
                        tensor.storage_offset(),
                    )
                # Hold our refs before publishing; drop them only when the
                # server reports the group freed (IPC export must outlive import).
                self._inflight[key] = refs
                pending_publish.append(
                    self._publish_async(key, (storages, views), free_target)
                )
                while len(pending_publish) >= _PUBLISH_WINDOW:
                    self._drop_when_ready(pending_publish.pop(0))
                if update_future is not None and update_future.done():
                    # update_weights returned/failed early — surface now instead
                    # of blocking further publishes.
                    update_future.result()
            while pending_publish:
                self._drop_when_ready(pending_publish.pop(0))
            freed = self._rpc("end_sync")
            self._drop_inflight(freed)
        except BaseException as e:
            with contextlib.suppress(Exception):
                self._rpc("set_gather_error", repr(e))
            self._inflight.clear()
            raise
        finally:
            self._sync_timing["gather_seconds"] = time.perf_counter() - gather0

    def _drop_inflight(self, freed_keys: list) -> None:
        for k in freed_keys:
            self._inflight.pop(tuple(k), None)

    # ---------------- misc ----------------

    def get_sync_timing(self) -> dict:
        """Coarse per-round timing (start / gather / update_weights / finish /
        wall seconds) — the replacement for the example CriticalPathProfiler's
        driver buckets. Producer/NIXL counters live on the server."""
        return dict(self._sync_timing)

    def get_produce_timing(self) -> dict:
        return self._rpc("get_produce_timing")

    def reset_produce_timing(self) -> None:
        self._rpc("reset_produce_timing")

    def get_nixl_timing(self) -> dict:
        return self._rpc("get_nixl_timing")

    def reset_nixl_timing(self) -> None:
        self._rpc("reset_nixl_timing")

    def shutdown(self) -> None:
        if self._server is None:
            return
        import ray

        with contextlib.suppress(Exception):
            ray.get(self._server.shutdown.remote())
            ray.kill(self._server)
        self._server = None
        self._inflight.clear()
