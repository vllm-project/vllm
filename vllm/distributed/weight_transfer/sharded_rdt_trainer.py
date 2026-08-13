# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Trainer-side engine for the sharded-RDT (pull-based NIXL) backend.

Symmetric to the NCCL/IPC trainer engines, but RDT is *pull-based*: the vLLM
workers initiate every transfer, dialing the trainer's Ray actors and pulling the
exact slice each worker consumes. So unlike NCCL (which broadcasts) this engine
pushes nothing from `send_weights`; instead it

  * owns a per-rank **producer server** -- an internal Ray actor exposing the NIXL
    serve surface (`rdt_produce_weights_batched`, `free_group`,
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
free_group -> release lifecycle, the ownership model, and the known concurrency
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

# How many gathered-but-unfreed groups the gather loop may run AHEAD of the
# consumers before it stops gathering. A gathered group is published (made
# serveable) IMMEDIATELY — publish_group never blocks — and the loop gates
# before gathering further while more than this many groups are unfreed, so at
# most ``lookahead + 1`` groups are ever resident on the trainer (~0.6
# GiB/group/rank at 235B post-EP-local). That +1 bound is the contract larger
# models rely on; the tests pin it.
# 1: while the consumers pull group N, group N+1 is already gathered AND
# published, so the boundary's free-barrier latency chain hides behind live
# pulls. (The old publish-gated pipeline at lookahead=1 kept N+1 gathered but
# UNSERVEABLE until N freed — every boundary drained the consumers' pull
# pipeline, ~2.5-3s of 235B sync wall — while still holding ~3 groups through
# the parked publish + its window. This credits the gather instead: old
# lookahead=2 overlap at 2 groups resident.) Raise it only if one group's
# gather is slower than its pulls.
DEFAULT_GATHER_LOOKAHEAD = 1

# [RDT-STALL-WATCHDOG] Seconds of no progress at all — no publish, no produce, no
# free — before the producer declares the sync dead.
#
# A consumer that dies mid-sync never sends its ``free_group`` signals, so the
# group is never released and the three waits below block forever. That stops this rank
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
    arena_presize_gb: float = 0.0
    """Serve-arena pre-size floor in GiB (avoids NIXL desc-cache churn)."""
    gather_lookahead: int = DEFAULT_GATHER_LOOKAHEAD
    """Gathered-but-unfreed groups the gather loop may run ahead by. Bounds
    trainer-resident memory at ``gather_lookahead + 1`` groups."""
    stall_timeout_s: float = DEFAULT_STALL_TIMEOUT_S
    """Seconds of no publish/serve/free progress before the producer fails the
    sync (see ``DEFAULT_STALL_TIMEOUT_S``)."""


class _RDTProducerServer:
    """Per-rank NIXL serve surface for the sharded-RDT backend.

    Spawned by the engine as an internal Ray actor sharing the trainer rank's
    GPU (via CUDA IPC). Holds a gather cache of rebuilt IPC tensors, per-consumer
    serve rings, the per-group free barrier, and the byte-exact packed serve. The
    engine feeds it gathered weights with `publish_group`; the workers pull with
    `rdt_produce_weights_batched` and signal completion with `free_group`.

    This is a plain class; the engine wraps it with `ray.remote(...)` at spawn
    so the actor options (name / tensor transport / concurrency / GPU pinning)
    live in one place.
    """

    def __init__(
        self,
        *,
        num_rdt_buffers: int,
        arena_presize_gb: float,
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

        # [RDT-FREE-BARRIER] Every LIVE consumer signals free_group(gi) at every
        # owner of gi, exactly once per sync; the group is freed (and reported
        # back to the engine) when the count reaches the live consumer total
        # handed to begin_sync. One uniform integer target — no routed
        # per-producer targets. Signals may arrive BEFORE the publish (a
        # consumer that pulls nothing from a group signals at sync start);
        # publish_group completes an already-satisfied group.
        self._live_count = 1
        self._free_counts: dict[int, int] = {}
        # gi -> the names this producer published for it (what release drops).
        self._group_names: dict[int, list[str]] = {}

        # [RDT-GATHER-CREDIT] Published-but-not-yet-freed group indices, plus
        # the freed indices not yet handed back to the engine. The memory gate
        # lives in the ENGINE's gather loop — it stops GATHERING (never
        # publishing; publish_group does not block) while more than
        # gather_lookahead groups are unfreed. This side only accounts:
        # free_group (the consumer back-edge) moves a group from
        # _inflight_groups to _freed_pending, and the engine collects
        # _freed_pending through wait_freed / end_sync to drop its refs to the
        # shared storage. _lookahead is engine-enforced; unused here.
        self._lookahead = max(1, gather_lookahead)
        self._inflight_groups: list[int] = []
        self._freed_pending: list[int] = []

        # [RDT-RING] Per-consumer ring of packed serve arenas, rotated per pull.
        self._nring = max(1, num_rdt_buffers)
        self._serve_rings: dict[int, list[torch.Tensor | None]] = {}
        self._serve_idx: dict[int, int] = {}
        self._serve_lock = threading.Lock()
        # registerMem on a shared NIXL agent is not concurrency-safe; serialize.
        self._reg_lock = threading.Lock()
        self._arena_presize = int(arena_presize_gb * (1 << 30))

        # [RDT-PACK-DSTS] (consumer_id, ring idx, packed layout) ->
        # (arena data_ptr, destination views). Keyed on the arena pointer too, so
        # a ring regrow invalidates rather than writing into a freed buffer. See
        # the serve path for why the layout, not the spec names, is the key.
        self._pack_dsts: dict[tuple, tuple[int, list[torch.Tensor]]] = {}

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
                    f"likely died mid-sync: {len(self._inflight_groups)} group(s) "
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

    def begin_sync(self, live_count: int) -> None:
        """Reset per-sync free/backpressure state and set this sync's barrier
        target. ``live_count`` is how many consumers take part in THIS sync (the
        whole provisioned fleet outside degraded syncs). REQUIRED, no default:
        a forgotten argument that silently set the target to 1 would free
        groups after the FIRST signal while other consumers still pull —
        use-after-free, not an error message. The driver awaits the
        previous sync's finish (which drains every consumer's signals) before
        the next begins, so nothing is in flight here — and a straggler signal
        from the previous sync would otherwise credit the wrong sync's group,
        which is why the consumer drains its fired signals before finishing.

        The packed-destination cache deliberately SURVIVES: the layout repeats
        every sync, which is what makes caching it worthwhile.
        """
        with self._cache_cond:
            self._gather_error = None
            self._live_count = max(1, int(live_count))
            self._inflight_groups.clear()
            self._freed_pending.clear()
            self._free_counts.clear()
            self._group_names.clear()
            self._note_progress_locked()

    def _release_group_locked(self, group_idx: int) -> None:
        """Drop a freed group's cache entries and queue it for the engine
        (whose gather-credit gate blocks in ``wait_freed`` on exactly this).

        Shared by the last ``free_group`` and by ``publish_group`` completing an
        early-signaled group. Caller must hold ``_cache_cond``.
        """
        for name in self._group_names.pop(group_idx, ()):
            self._cache.pop(name, None)
        self._free_counts.pop(group_idx, None)
        if group_idx in self._inflight_groups:
            self._inflight_groups.remove(group_idx)
            self._freed_pending.append(group_idx)

    def publish_group(self, group_idx: int, entries: tuple) -> None:
        """Rebuild one gather group's CUDA-IPC tensors and publish to the serve
        cache. NEVER blocks: a gathered group is serveable immediately, so the
        consumers can start on it the moment they finish the previous one. The
        memory bound lives in the engine's gather loop, which stops GATHERING
        (not publishing) while more than ``gather_lookahead`` groups are
        unfreed — see [RDT-GATHER-CREDIT].

        ``entries`` is ``(storages, views)``: ``storages`` maps a storage id to
        the ``reduce_tensor`` args of a whole-storage uint8 view (ONE CUDA-IPC
        export per storage), ``views`` maps each served name to
        ``(sid, dtype_name, shape, stride, storage_offset)`` -- rebuilt here as
        ``as_strided`` views. One export per storage rather than per name; see the
        doc for the cost this replaced. The names are exactly what this rank
        holds for the group (its replicated names + its own EP coordinate's
        experts).

        Signals can arrive BEFORE the publish they belong to -- a consumer that
        pulls nothing of a group signals it as soon as its plan starts -- so a
        group whose barrier is already satisfied is released here rather than
        waiting for a signal that will never come. Freed groups reach the
        engine ONLY through ``wait_freed`` / ``end_sync``, never a publish
        return: a freed notice riding an unharvested async publish result
        while the engine blocks in ``wait_freed`` would wedge the loop.
        """
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

        with self._cache_cond:
            self._cache.update(rebuilt)
            self._inflight_groups.append(group_idx)
            self._group_names[group_idx] = list(rebuilt)
            if self._free_counts.get(group_idx, 0) >= self._live_count:
                self._release_group_locked(group_idx)
            self._note_progress_locked()
            self._cache_cond.notify_all()

    def wait_freed(self) -> list[int]:
        """Block until at least one published group has been freed; return and
        clear the freed-group backlog. The engine's gather-credit gate calls
        this when its loop is ``gather_lookahead`` gathered groups ahead of the
        consumers — so THIS wait is where the trainer paces to the consumers'
        pull rate (the old pipeline parked publish_group on the same credit,
        which kept the next group gathered but unserveable across every
        boundary).

        Raises — rather than returning empty — when the sync errors or stalls
        out (watchdog-covered): the engine is blocked on this call inside its
        gather loop, and an empty return would only spin it straight back in.
        """
        with self._cache_cond:
            self._wait_for(lambda: not self._freed_pending, "a freed-group credit")
            if not self._freed_pending:
                raise RuntimeError(
                    "gather errored while waiting for a freed-group credit: "
                    f"{self._gather_error!r}"
                )
            freed = self._freed_pending
            self._freed_pending = []
            return freed

    def end_sync(self) -> list[int]:
        """Block until every published group has been freed by its consumers;
        return the remaining freed keys so the engine drops its last refs."""
        with self._cache_cond:
            self._wait_for(
                lambda: bool(self._inflight_groups),
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

    def free_group(self, group_idx: int) -> None:
        """Consumer back-edge: one consumer is done with gather group
        ``group_idx`` — its last chunk of the group has landed, or it had
        nothing to pull from it and signaled at sync start.

        The per-group barrier: counts one signal per live consumer against the
        ``begin_sync`` live count (every consumer signals EVERY owner of the
        group, so the target is the same uniform integer on all owners). On the
        last signal, drops the cache entries and queues the freed group for the
        engine — the gather credit its loop may be blocked on in ``wait_freed``.
        A signal that arrives before its publish is only counted —
        ``publish_group`` completes it.
        """
        gi = int(group_idx)
        with self._cache_cond:
            count = self._free_counts.get(gi, 0) + 1
            self._free_counts[gi] = count
            if gi in self._group_names and count >= self._live_count:
                self._release_group_locked(gi)
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

        sliced: list = []  # (byte_off, tensor)
        pack_cur = 0
        for name, chain in specs:
            t = self._cache[name]
            for op, args, kw in chain:
                if op not in ALLOWED_OPS:
                    raise ValueError(f"{name!r}: disallowed op {op!r}")
                t = getattr(t, op)(*args, **dict(kw))
            off = (pack_cur + 15) & ~15
            pack_cur = off + t.numel() * t.element_size()
            sliced.append((off, t))

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
        # (the same source sliced differently, which per-ep_rank chunking produces
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
        torch._foreach_copy_(dsts, [t for _off, t in sliced])

        blob = arena[:pack_cur]
        # A served pull is the third of the four progress signals the stall
        # watchdog reads (publish / produce / free / begin_sync): a long sync whose
        # consumers pull steadily but slowly must never trip it.
        with self._cache_cond:
            self._note_progress_locked()
        return [blob]

    def shutdown(self) -> None:
        with self._cache_cond:
            self._cache.clear()
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
        # Expert stamps (see WeightSource.expert_ownership), resolved at
        # trainer_init: name_ep_rank stamps the metadata names, my_ep_rank is
        # this rank's coordinate, producer_ep_ranks the all-gathered coords.
        # None/-1 = no expert sharding declared.
        self._name_ep_rank: list[int] | None = None
        self._producer_ep_ranks: list[int] | None = None
        self._my_ep_rank: int = -1
        # Strong refs to gathered tensors we've shared into the server, keyed by
        # group index. CUDA-IPC exports must outlive the importer, so we hold
        # them until the server reports the group freed. See send_weights.
        self._inflight: dict[int, dict[str, torch.Tensor]] = {}

    def _rpc(self, method: str, *args: Any) -> Any:
        """Call one of the server actor's methods and block for the result.
        The single seam through which the engine talks to its server, so tests
        can inject a local (non-Ray) fake server."""
        import ray

        return ray.get(getattr(self._server, method).remote(*args))

    def _publish_async(self, group_idx: int, entries):
        """Fire publish_group WITHOUT waiting on the RPC (the gather loop
        overlaps the publish's server-side rebuild with the next group's
        gather) and return a handle that ``_await_publish`` resolves. Ray actor
        handle in production; a plain (non-Ray) fake server runs inline."""
        method = self._server.publish_group
        remote = getattr(method, "remote", None)
        if remote is not None:
            return remote(group_idx, entries)
        return method(group_idx, entries)

    def _await_publish(self, ref) -> None:
        """Resolve one async publish so a server-side rebuild error surfaces at
        window depth instead of at end_sync. Publishes carry nothing back —
        freed groups flow only through wait_freed/end_sync (one channel; see
        publish_group)."""
        import ray

        if isinstance(ref, ray.ObjectRef):
            ray.get(ref)

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
        engine._spawn_server(engine._served_names())

        # Every rank's server must exist before the sender's init RPC (the worker
        # init calls reserve_serve_arena back on ALL producer servers). The
        # all-gather of server names doubles as that barrier.
        server_names = engine._all_gather_server_names(world, rank)
        # Retained (rather than consumed and dropped) so a consumer that RESTARTS can
        # be re-initialized later without another all-gather: see
        # `get_worker_init_payload`. The names are uuid actor names that stay valid for
        # the run, because producers never restart.
        engine._server_names = server_names

        if engine.is_sender:
            worker_init = engine._build_worker_init_info(server_names)
            engine.client.init_weight_transfer_engine(asdict(worker_init))
        return engine

    def get_worker_init_payload(self) -> dict:
        """The consumer-side init payload, rebuilt on demand. PURE -- no collective.

        This is what lets a restarted inference engine rejoin at a sync boundary
        instead of failing the run: ``_build_worker_init_info`` reads only retained
        state (``_meta``, ``_groups``, ``_group_owners``, ``_init_info`` and the cached
        ``_server_names``), so the driver can ask any time -- including mid-run, with
        every rank sitting in its own training step -- and hand the result to exactly
        the one engine that came back.

        A collective here would deadlock: the caller is the driver, and the ranks are
        not at a matching collective.

        Raises:
            RuntimeError: called before ``trainer_init`` cached the server names.
        """
        if getattr(self, "_server_names", None) is None:
            raise RuntimeError(
                "get_worker_init_payload requires trainer_init to have run (the producer "
                "server names are gathered there)."
            )
        return asdict(self._build_worker_init_info(self._server_names))

    def _world_and_rank(self) -> tuple[int, int]:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            return torch.distributed.get_world_size(), torch.distributed.get_rank()
        return 1, self._init_info.rank

    def _build_router(self, world: int, rank: int) -> None:
        """Resolve per-group ownership, expert stamps, and this rank's publish
        plan.

        A source may gather only part of the model — pipeline-parallel producers
        gather within a stage rather than to all ranks, expert-parallel ranks
        hold only their coordinate's experts — so both ownership declarations
        are all-gathered here and shipped to the consumers in the worker init
        info. Every rank must agree on who serves what: a consumer pulling from
        a producer that never gathered the name trips its served-names guard.
        Sources without ``owned_groups``/``expert_ownership`` own everything,
        keeping the gather-to-all layout on its historical path.
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
        # Resolved AFTER owned_groups: its shared-group discovery can demote a
        # shard-aware source to the naive path, and the two declarations flip
        # together.
        ownership = self.source.expert_ownership()
        name_ep_rank: list[int] | None = None
        my_ep_rank = -1
        if ownership is not None:
            name_ep_rank, my_ep_rank = list(ownership[0]), int(ownership[1])
            if len(name_ep_rank) != len(self._meta):
                raise ValueError(
                    f"WeightSource.expert_ownership() stamped "
                    f"{len(name_ep_rank)} names for {len(self._meta)} metadata "
                    "entries."
                )

        # One collective carries all of it: the ownership lists and digests of
        # the metadata + stamps they index into. The digest checks are what make
        # partial ownership safe — only the sender's metadata/stamps reach the
        # consumers, so a rank that described just its own share (or stamped
        # differently) would leave the model silently mis-served instead of
        # failing.
        digest = self._meta_digest()
        ep_digest = self._stamps_digest(name_ep_rank)
        per_rank = self._all_gather_owned(world, (digest, owned, my_ep_rank, ep_digest))
        mismatched = [r for r, (d, *_rest) in enumerate(per_rank) if d != digest]
        if mismatched:
            raise ValueError(
                f"WeightSource.metadata() disagrees across trainer ranks "
                f"(rank {rank} digest {digest}, differing ranks {mismatched[:4]}). "
                "Every rank must describe the WHOLE model, even when it owns "
                "only some groups."
            )
        ep_mismatched = [
            r for r, (_d, _o, _ep, epd) in enumerate(per_rank) if epd != ep_digest
        ]
        if ep_mismatched:
            raise ValueError(
                "WeightSource.expert_ownership() name stamps disagree across "
                f"trainer ranks (differing ranks {ep_mismatched[:4]}). Every rank "
                "must derive the identical name_ep_rank list (it is a pure "
                "function of the metadata names)."
            )
        owned_per_rank = [o for _d, o, _ep, _epd in per_rank]
        ep_coords = [int(ep) for _d, _o, ep, _epd in per_rank]

        group_owners: list[list[int]] | None = None
        if any(o is not None for o in owned_per_rank):
            group_owners = [[] for _ in range(num_groups)]
            for r, rank_owned in enumerate(owned_per_rank):
                for gi in range(num_groups) if rank_owned is None else rank_owned:
                    group_owners[gi].append(r)

        producer_ep_ranks = ep_coords if name_ep_rank is not None else None
        router = RdtRouter(
            world,
            self._init_info.num_consumers,
            group_owners,
            num_groups,
            producer_ep_ranks=producer_ep_ranks,
        )
        router.validate()
        self._router = router
        self._group_owners = group_owners or []
        self._owned_idx = owned if owned is not None else list(range(num_groups))
        self._name_ep_rank = name_ep_rank
        self._producer_ep_ranks = producer_ep_ranks
        self._my_ep_rank = my_ep_rank
        # The names this rank is REQUIRED to yield real tensors for (and the
        # only ones it may), per the stamps. None = unstamped source, no check.
        self._held_names = (
            set(self._served_names()) if name_ep_rank is not None else None
        )

    def _validate_stamped_yields(self, gi: int, names, tensors) -> None:
        """Stamps must be truthful against yields (the ABC contract's first
        invariant), and this is the one place both sit side by side. Without
        this check, a source that stamps a name as held but yields ``None`` for
        it produces a 300s stall-watchdog death (consumers route pulls here,
        the pull passes the served-names guard, and the cache wait never
        completes) instead of an immediate, named error. It also discharges
        the interchangeability invariant: if every rank's yields match the
        rank-identical stamps, same-coordinate ranks hold identical sets by
        construction. Cost: one set lookup per name per sync."""
        if self._held_names is None:
            return
        for name, tensor in zip(names, tensors):
            if (tensor is None) == (name in self._held_names):
                claim = "does not hold" if tensor is not None else "holds"
                have = "a real tensor" if tensor is not None else "None"
                raise RuntimeError(
                    f"expert_ownership stamps disagree with the yielded tensors: "
                    f"group {gi} name {name!r} yielded {have} but the stamps say "
                    f"this rank {claim} it. The WeightSource's stamps must be "
                    "truthful against its iteration (see "
                    "WeightSource.expert_ownership)."
                )

    def _served_names(self) -> list[str]:
        """The names this rank actually publishes: for every owned group, its
        replicated (``-1``) names plus its own EP coordinate's expert names.
        This is the sidecar's misroute guard — a pull for any other name fails
        loudly instead of blocking forever in the cache wait."""
        if self._name_ep_rank is None:
            return [n for gi in self._owned_idx for n in self._groups[gi]]
        stamp_of = {m.name: er for m, er in zip(self._meta, self._name_ep_rank)}
        return [
            n
            for gi in self._owned_idx
            for n in self._groups[gi]
            if stamp_of[n] < 0 or stamp_of[n] == self._my_ep_rank
        ]

    def _meta_digest(self) -> str:
        """Stable digest of this rank's metadata (name order + count)."""
        import hashlib

        h = hashlib.sha256()
        h.update(f"{len(self._meta)}\n".encode())
        for m in self._meta:
            h.update(m.name.encode())
            h.update(b"\n")
        return h.hexdigest()[:16]

    @staticmethod
    def _stamps_digest(name_ep_rank: "list[int] | None") -> str:
        """Stable digest of the expert stamp list (``"none"`` when undeclared —
        a rank WITH stamps then mismatches a rank without, which is the point:
        mixed declarations are as wrong as differing ones)."""
        if name_ep_rank is None:
            return "none"
        import hashlib

        h = hashlib.sha256()
        h.update(",".join(map(str, name_ep_rank)).encode())
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
            # Thread budget: under EP-local routing EVERY consumer pulls from
            # every producer, and with issue-ahead each consumer can park up to
            # K produce calls in this actor's cache-wait — C*K blocked calls.
            # publish_group / free_group / wait_freed / begin/end_sync queue
            # BEHIND them if the pool is smaller, which is a guaranteed
            # deadlock: the parked produces wait for a publish that can never
            # get a thread (observed as the sync-2 wedge at 235B tp8: 16 parked
            # produces vs 8 threads, "0 groups published"). Size the pool to
            # the worst case plus slack for the control plane (at most one
            # parked wait_freed + two async publishes + one begin/end at once).
            max_concurrency=ii.num_consumers * ii.num_rdt_buffers + 4,
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
            name_ep_rank=self._name_ep_rank or [],
            producer_ep_ranks=self._producer_ep_ranks or [],
            num_consumers=self._init_info.num_consumers,
            num_rdt_buffers=self._init_info.num_rdt_buffers,
            arena_presize_gb=self._init_info.arena_presize_gb,
        )

    # ---------------- per-round ----------------

    def send_weights(self, live_consumer_ids: Collection[int] | None = None) -> None:
        """Gather this rank's weights and publish them for the consumers to pull.

        ``live_consumer_ids`` restricts the sync to the consumers still alive;
        ``None`` (the default) serves the whole provisioned set. The provisioned
        geometry — ``num_consumers``, the router, the ownership tables,
        ``_owned_idx``, the ``served_names`` the sidecar registered — is FROZEN
        for the run: a degraded sync only lowers the per-group free barrier's
        target, the live COUNT handed to ``begin_sync``. Every rank must be
        given the SAME live set — they are all inside the same gather
        collectives — so the caller has to compute it once and dispatch it to
        all of them.
        """
        assert self.source is not None
        if live_consumer_ids is None:
            live_count = self._init_info.num_consumers
        else:
            live_count = len(set(live_consumer_ids))
            logger.warning(
                "[rdt-degraded] serving %d/%d live consumers; every group's "
                "free barrier counts to the live total",
                live_count,
                self._init_info.num_consumers,
            )
        self._send_weights_inner(live_count)

    def _send_weights_inner(self, live_count: int) -> None:
        if not self.is_sender:
            self._run_gather_loop(update_future=None, live_count=live_count)
            return

        self.client.start_weight_update()

        from vllm.distributed.weight_transfer.sharded_rdt_engine import (
            ShardedRDTWeightTransferUpdateInfo,
        )

        empty_update = asdict(ShardedRDTWeightTransferUpdateInfo())
        with ThreadPoolExecutor(max_workers=1) as exe:
            # The workers block inside update_weights until they've pulled every
            # group, so it runs concurrently with the gather/publish loop.
            future = exe.submit(self.client.update_weights, empty_update)
            self._run_gather_loop(update_future=future, live_count=live_count)
            future.result()  # surface inference-side errors

        self.client.finish_weight_update()

    def _run_gather_loop(self, update_future, live_count: int) -> None:
        """Gather this rank's weights group-by-group and publish each into the
        server over CUDA IPC. A gathered group is published — serveable —
        immediately; the loop gates BEFORE the next gather while more than
        `gather_lookahead` groups are unfreed (the per-group free barrier: a
        credit releases when every live consumer has signaled the group). So
        the loop self-paces to the consumers' pull rate with at most
        `gather_lookahead + 1` groups resident. Runs on every rank; only the
        sender has an `update_future` to fail fast on."""
        assert self.source is not None  # guaranteed by trainer_init
        self._rpc("begin_sync", live_count)
        # One generator resume per GROUP, not per tensor: `iter_groups` yields
        # (names, tensors) for each group this rank owns, in metadata order.
        # Sources that can materialize a whole group at once override it; the
        # base default batches `__iter__` and checks the order as it goes. Every
        # owner of a group must reach it in the same order or their shared gather
        # collective mismatches.
        groups = self.source.iter_groups()
        # Publishes are fired without an inline ray.get and harvested this
        # window deep, so the publish RPC + server-side rebuild overlap the
        # NEXT group's gather/export instead of serializing the loop.
        # publish_group never blocks (memory is bounded by the gather credit
        # gate below, not by parking publishes), so a harvest costs only the
        # rebuild; the window is about overlapping that and surfacing a
        # server-side error at depth 2 instead of at end_sync.
        _PUBLISH_WINDOW = 2
        pending_publish: list = []
        # [RDT-GATHER-CREDIT] The memory bound: gate BEFORE gathering while
        # more than `bound` groups are gathered-but-unfreed, so at most
        # bound + 1 groups are ever resident — `_inflight` gains its entry
        # right after each gather and only wait_freed/end_sync returns shrink
        # it, so the count here is exact (a group freed server-side but not yet
        # collected still holds its refs, and still counts). The publish window
        # is drained before blocking so every resident group's publish has
        # LANDED — the consumers can be pulling all of them while we wait.
        bound = max(1, self._init_info.gather_lookahead)

        try:
            for gi in self._owned_idx:
                group = self._groups[gi]
                while len(self._inflight) > bound:
                    while pending_publish:
                        self._await_publish(pending_publish.pop(0))
                    self._drop_inflight(self._rpc("wait_freed"))
                names, tensors = next(groups)
                if list(names) != list(group):
                    raise RuntimeError(
                        f"WeightSource group yielded {len(names)} names starting "
                        f"{names[:2]!r} but expected {len(group)} starting "
                        f"{group[:2]!r}; iteration order must match metadata."
                    )
                self._validate_stamped_yields(gi, names, tensors)
                # Share each unique STORAGE once (one cudaIpc export instead of
                # one per name) and describe every name as an as_strided view
                # spec relative to its storage. ``None`` tensors are names this
                # rank does not hold (foreign experts under shard-aware
                # serving) — the source keeps them in the name list so the
                # order check above stays rank-uniform, and they are dropped
                # here, before the IPC export, matching the sidecar's
                # served_names.
                storages: dict[int, tuple] = {}
                views: dict[str, tuple] = {}
                refs: dict[str, torch.Tensor] = {}
                for name, tensor in zip(names, tensors):
                    if tensor is None:
                        continue
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
                del tensors
                if not refs:
                    # A group with nothing held here cannot occur (every group
                    # carries replicated names), but publishing an empty group
                    # would park a credit nobody's pull is waiting on; skip.
                    continue
                # Hold our refs before publishing; drop them only when the
                # server reports the group freed (IPC export must outlive import).
                self._inflight[gi] = refs
                pending_publish.append(self._publish_async(gi, (storages, views)))
                while len(pending_publish) >= _PUBLISH_WINDOW:
                    self._await_publish(pending_publish.pop(0))
                if update_future is not None and update_future.done():
                    # update_weights returned/failed early — surface now instead
                    # of blocking further gathers.
                    update_future.result()
            while pending_publish:
                self._await_publish(pending_publish.pop(0))
            freed = self._rpc("end_sync")
            self._drop_inflight(freed)
        except BaseException as e:
            with contextlib.suppress(Exception):
                self._rpc("set_gather_error", repr(e))
            self._inflight.clear()
            raise

    def _drop_inflight(self, freed_keys: list) -> None:
        for k in freed_keys:
            self._inflight.pop(int(k), None)

    # ---------------- misc ----------------

    def shutdown(self) -> None:
        if self._server is None:
            return
        import ray

        with contextlib.suppress(Exception):
            ray.get(self._server.shutdown.remote())
            ray.kill(self._server)
        self._server = None
        self._inflight.clear()
