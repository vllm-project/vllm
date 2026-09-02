# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Trainer-side engine for the sharded-RDT (pull-based NIXL) backend.

RDT is pull-based, so unlike NCCL this engine broadcasts nothing. It owns a
per-rank producer server (an internal Ray actor exposing the NIXL serve surface
the worker engine dials by name), and on each ``send_weights`` gathers this
rank's weights group-by-group from the ``WeightSource``, shares each group into
the server over CUDA IPC, and — on the sender — drives the inference-side
start/update/finish handshake, whose single empty ``update_weights`` unblocks the
workers to pull.

All serve-side state lives on the server actor, so trainer processes need no
mixin, no named actors and no special actor options.

See docs/training/weight_transfer/sharded_rdt.md for the publish -> serve ->
free_group -> release lifecycle and the ownership model.
"""

import contextlib
import threading
import time
import uuid
from collections.abc import Callable, Collection
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from typing import Any, ClassVar

import ray
import torch
from torch.multiprocessing.reductions import (
    StorageWeakRef,
    rebuild_cuda_tensor,
    reduce_tensor,
)
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
    buffer_alloc_bytes,
    check_ray_rdt_version,
)
from vllm.logger import init_logger

logger = init_logger(__name__)

# Gathered-but-unfreed groups the gather loop may run ahead by before it stops
# gathering. A gathered group is published immediately (publish_group never
# blocks), so at most ``lookahead + 1`` groups are resident on the trainer — the
# memory bound larger models size against, pinned by the tests.
#
# At 1, group N+1 is gathered and serveable while the consumers pull N, so the
# boundary's free-barrier latency hides behind live pulls. Raise it only if one
# group's gather is slower than its pulls.
DEFAULT_GATHER_LOOKAHEAD = 1

# Seconds with no publish, produce or free before the
# producer declares the sync dead. A consumer that dies mid-sync never signals
# ``free_group``, so the group is never released and the waits below block
# forever — which stops this rank iterating its WeightSource, a collective, and
# wedges every other trainer rank with no exception anywhere. This converts that
# into one real error.
#
# A liveness backstop, not a latency target: nominal gaps are sub-second, so this
# is ~100x margin. It rides the init info because the producer is a Ray actor and
# does not inherit the trainer's environment.
DEFAULT_STALL_TIMEOUT_S = 300.0
# How often a blocked waiter re-checks the progress stamp.
_STALL_POLL_S = 5.0

# Tensors whose STORAGE is at most this are packed into a ring slot and exported
# as one CUDA-IPC handle; larger storages are exported directly. Batching the
# small ones is what keeps sync time steady -- a per-tensor handle for each of
# ~37k names dominated the variance -- and `_pack_group_for_export` is also the
# only publish path that orders the pack against the sidecar's IPC read, so the
# packing is not optional.
_EXPORT_RING_MAX_BYTES = 64 << 20

# The actor method the worker engine dials for the NIXL pull; fixed by contract.
PRODUCE_METHOD_NAME = "rdt_produce_weights_batched"


# Cross-deployment serve-slot sharing.
#
# Consumers whose ids differ by a multiple of ``workers_per_replica`` are the
# same worker of different inference deployments, so they share a baked plan and
# pull byte-identical chunks. A serve ring each would cost one full ring per
# deployment of the producer's GPU and repack the same bytes once per deployment.
#
# NIXL reads are one-sided, so R readers can read ONE registered slot; what the
# producer cannot see is when a reader has FINISHED. The release edge is
# therefore the consumer's ISSUE order, sent as ``seq``: the pipeline drains pull
# i before issuing i+K, so slot ``seq % K`` was last packed for ``seq - K``,
# whose read is over. A counter on THIS side would follow execution order
# instead, which is wrong -- Ray may start a consumer's K concurrent produce
# calls in any order, so a call that executes K-before another can be a pull
# still being read, and its slot gets repacked underneath the reader. Silent,
# showing up only as a logprob drift.
#
# Hence one rendezvous per generation, keyed by ``seq``: the group's live sharers
# meet there, the LAST to arrive packs, and all return that one blob. It carries
# the release edge too, since ``seq`` is packed only once every live sharer has
# arrived, so each has drained ``seq - K``. Mismatched plans would rendezvous on
# differing bytes, so they are rejected at init by ``reserve_serve_buffer``'s
# ``plan_digest``. A sharer that dies mid-sync stalls its group, exactly as a
# dead consumer already stalls the per-group free barrier; the next
# ``begin_sync`` narrows the live set.
#
# LIMITATION: the rendezvous synchronizes deployments that a pull-based transfer
# would otherwise leave independent -- per chunk every sharer waits for the
# slowest, so skew between their DP pause/resume brackets lands in every
# deployment's sync time. Nothing about P2P requires this. It is the price of
# serving out of GPU memory, which is too scarce for a ring per consumer, so
# slots must be reused, and reuse needs a release edge.
#
# TODO: explore staging packed chunks in HOST memory. Host buffers are cheap
# enough for a ring per consumer, which would drop the rendezvous, the
# cross-replica coupling and the slot-reuse hazard together. The cost is a D2H
# bounce on the serve path, which has been the bottleneck before, so it needs
# measuring rather than assuming.


@dataclass
class _SharedPack:
    """One generation of one slot-sharing group: the rendezvous state for a
    single chunk, identified by the consumers' issue index."""

    group: int
    slot: int
    """``seq % ring_depth``. Fixed at creation, so slot reuse follows the
    consumers' issue order rather than this side's execution order."""
    arrived: set[int] = field(default_factory=set)
    blob: Any = None
    packing: bool = False
    done: bool = False
    error: BaseException | None = None


@dataclass
class ShardedRDTTrainerInitInfo(TrainerInitInfo):
    """Trainer init info for the sharded-RDT backend.

    Identical on every rank except ``rank`` (rank 0 is the sender). Carries only
    the must-agree wire params; the sender forwards them verbatim onto the
    worker-side init info so the two cannot drift. Server-actor names are
    generated per rank and all-gathered by the engine, not supplied here.
    """

    backend: ClassVar[str] = "sharded_rdt"

    num_consumers: int
    """Total inference-worker (consumer) count across the whole fleet
    (DP*TP*PP*PCP), for the M:N block assignment / free ref-count."""
    workers_per_replica: int = 0
    """Consumers per inference DEPLOYMENT (``num_consumers // num_replicas``).

    Fixes the slot-sharing groups: consumers whose ids differ by a multiple of
    this are the same worker of different deployments, so they bake identical
    plans and pull byte-identical chunks, and ONE registered serve slot can
    serve all of them. 0 disables sharing, and so does the single-deployment
    value ``num_consumers``, which makes every group a singleton and the serve
    path identical to the unshared one.
    """
    trainer_actor_namespace: str | None = None
    """Ray namespace the engine spawns its serve actors in. The inference
    workers (which run in their own EngineCore subprocess with its own
    ``ray.init``) resolve those actors by name, so this must be the namespace
    they can see. Forwarded to the worker-side init info."""
    num_rdt_buffers: int = 2
    """Serve/receive ring depth K (must match the worker)."""
    buffer_presize_gb: float = 0.0
    """Serve-buffer pre-size floor in GiB (avoids NIXL desc-cache churn)."""
    gather_lookahead: int = DEFAULT_GATHER_LOOKAHEAD
    """Gathered-but-unfreed groups the gather loop may run ahead by. Bounds
    trainer-resident memory at ``gather_lookahead + 1`` groups."""
    stall_timeout_s: float = DEFAULT_STALL_TIMEOUT_S
    """Seconds of no publish/serve/free progress before the producer fails the
    sync (see ``DEFAULT_STALL_TIMEOUT_S``)."""


class _RDTProducerServer:
    """Per-rank NIXL serve surface: an internal Ray actor sharing the trainer
    rank's GPU over CUDA IPC. Holds the gather cache, per-consumer serve rings,
    the per-group free barrier and the packed serve. The engine feeds it with
    ``publish_group``; workers pull with ``rdt_produce_weights_batched`` and
    signal with ``free_group``.

    A plain class — the engine wraps it with ``ray.remote(...)`` at spawn so the
    actor options live in one place.
    """

    def __init__(
        self,
        *,
        num_rdt_buffers: int,
        buffer_presize_gb: float,
        gather_lookahead: int,
        served_names: list[str] | None = None,
        stall_timeout_s: float = DEFAULT_STALL_TIMEOUT_S,
        workers_per_replica: int = 0,
    ) -> None:
        import gc

        self._device_index = torch.accelerator.current_device_index()
        # name -> rebuilt CUDA-IPC tensor (or view); guarded by _cache_cond.
        self._cache: dict[str, torch.Tensor] = {}
        self._cache_cond = threading.Condition()
        self._gather_error: BaseException | None = None

        # Monotonic stamp of the last forward step of the
        # publish -> serve -> free loop. The waits below poll it instead of
        # blocking forever, so a dead consumer fails the sync with a real error.
        self._stall_timeout = float(stall_timeout_s)
        self._last_progress = time.monotonic()

        # Names this producer publishes, so a misrouted pull fails loudly
        # instead of blocking forever in the cache wait. None = serve anything.
        self._served_names = set(served_names) if served_names is not None else None

        # Every live consumer signals free_group(gi) at every
        # owner, once per sync; the group frees when the count reaches
        # begin_sync's live total — one uniform target, no routed per-producer
        # ones. Signals may precede the publish, which completes them.
        self._live_count = 1
        self._free_counts: dict[int, int] = {}
        # gi -> the names this producer published for it (what release drops).
        self._group_names: dict[int, list[str]] = {}

        # Published-but-unfreed groups, plus freed ones not
        # yet handed back. The memory gate lives in the ENGINE's gather loop; this
        # side only accounts — free_group moves a group from _inflight_groups to
        # _freed_pending, which the engine collects via wait_freed / end_sync to
        # drop its storage refs. _lookahead is engine-enforced, unused here.
        self._lookahead = max(0, gather_lookahead)
        self._inflight_groups: list[int] = []
        self._freed_pending: list[int] = []

        # Ring of packed serve buffers, one ring per SLOT-SHARING GROUP (one
        # consumer per group unless several deployments share); the slot within
        # it is the consumer's ``seq % nring``.
        self._nring = max(1, num_rdt_buffers)
        self._serve_rings: dict[int, list[torch.Tensor | None]] = {}
        self._serve_lock = threading.Lock()
        # registerMem on a shared NIXL agent is not concurrency-safe; serialize.
        self._reg_lock = threading.Lock()
        self._buffer_presize = int(buffer_presize_gb * (1 << 30))
        self._serve_device = torch.device("cuda", self._device_index)

        # Slot-sharing state, all guarded by _cache_cond, so the rendezvous
        # waits ride the stall watchdog like every other wait here.
        # ``_share_width`` = consumers per deployment; 0 means every group is a
        # singleton and this degenerates to the unshared serve path.
        self._share_width = max(0, int(workers_per_replica))
        self._live_ids: set[int] | None = None
        self._sharing_active = False
        self._sharers: dict[int, frozenset] = {}  # group -> live members
        self._gens: dict[int, dict] = {}  # group -> {seq: _SharedPack}
        # group -> {cid: plan digest}. Init-time state, not cleared per sync: it
        # is what proves the sharers of a group pull the same chunks.
        self._plan_digests: dict[int, dict[int, str]] = {}

        # (sharing group, ring idx, packed layout) ->
        # (buffer data_ptr, destination views). Keyed on the buffer pointer too, so
        # a ring regrow invalidates rather than writing into a freed buffer. See
        # the serve path for why the layout, not the spec names, is the key.
        self._pack_dsts: dict[tuple, tuple[int, list[torch.Tensor]]] = {}
        # [RDT-EXPORT-RING] handle -> rebuilt base. The trainer reuses buffers
        # across groups, so the same handle arrives on many publishes; without
        # this we reopen it every group. Cleared per sync so no IPC mapping into
        # trainer memory outlives the sync that made it.
        self._base_cache: dict[tuple, torch.Tensor] = {}

        # Freeze the static post-init object graph so gen-2 GC never stops the
        # world mid-serve, which shows up as serve stragglers.
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

        Waits while ``blocked()`` holds, returning early on a gather error. If
        nothing on this producer progresses for ``_stall_timeout``, self-fires
        ``set_gather_error``, which every waiter here already checks — so the rank
        unwinds through one path and the driver gets a real exception.

        The progress stamp is global to the producer, not per-waiter: a merely
        slow waiter is kept alive by its peers' progress. Caller holds
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
        """Create this server's NIXL agent now, while the rank's GPU is quiet.

        Called at spawn, before the server-name all-gather, so no rank can be
        spinning in a collective. Creating the agent lazily instead deadlocks on
        EFA-class fabrics (see "warmup_nixl breaks a startup deadlock" in the
        doc). The warmup buffer stays registered so the agent's CUDA-HMEM path
        stays initialized.
        """
        from ray.experimental import register_nixl_memory

        self._nixl_warmup_buf = torch.zeros(1 << 20, dtype=torch.uint8, device="cuda")
        with self._reg_lock:
            register_nixl_memory(self._nixl_warmup_buf)

    # ---------------- engine-facing (per sync) ----------------

    def begin_sync(
        self, live_count: int, live_consumer_ids: list | None = None
    ) -> None:
        """Reset per-sync free/backpressure state and set this sync's barrier
        target.

        ``live_count`` is how many consumers take part in THIS sync. Required, no
        default: a forgotten argument silently targeting 1 would free groups after
        the FIRST signal while others still pull — use-after-free, not an error.
        The driver awaits the previous sync's finish before the next begins, so
        nothing is in flight; a straggler signal would otherwise credit the wrong
        sync, which is why the consumer drains its signals before finishing.

        ``live_consumer_ids`` is that same live set enumerated, which is what
        sizes each slot-sharing group's rendezvous; a count cannot, since a group
        is a specific set of ids. ``None`` leaves every group a singleton.

        The packed-destination cache deliberately survives: the layout repeats
        every sync. So does ``_plan_digests``, which is init-time state.
        """
        with self._cache_cond:
            self._gather_error = None
            self._live_count = max(1, int(live_count))
            self._live_ids = (
                set(int(c) for c in live_consumer_ids)
                if live_consumer_ids is not None
                else None
            )
            # Sharing is on only when the geometry shows several deployments;
            # one deployment takes the singleton path.
            self._sharing_active = (
                self._share_width > 0
                and self._live_ids is not None
                and len(self._live_ids) > self._share_width
            )
            self._sharers.clear()
            self._gens.clear()
            self._inflight_groups.clear()
            self._freed_pending.clear()
            self._free_counts.clear()
            self._group_names.clear()
            self._base_cache.clear()
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
        """Rebuild one gather group's CUDA-IPC tensors into the serve cache.

        NEVER blocks: a gathered group is serveable immediately. The memory bound
        lives in the engine's gather loop, which stops GATHERING (not publishing)
        past ``gather_lookahead`` unfreed groups.

        ``entries`` is ``(storages, views)``: one CUDA-IPC export per storage,
        plus per-name ``(sid, dtype_name, shape, stride, storage_offset)`` rebuilt
        here as ``as_strided`` views.

        Signals can arrive BEFORE their publish — a consumer pulling nothing of a
        group signals it as its plan starts — so a group whose barrier is already
        satisfied is released here. Freed groups reach the engine only through
        ``wait_freed`` / ``end_sync``: a freed notice riding an unharvested async
        publish result would wedge the loop.
        """
        storages, views = entries
        bases: dict[int, torch.Tensor] = {}
        for sid, reduce_args in storages.items():
            cached = self._base_cache.get(reduce_args)
            if cached is not None:
                bases[sid] = cached
                continue
            list_args = list(reduce_args)
            # Index 6 of reduce_tensor's args is the exporter's device index;
            # rebuild on this server's device (same physical GPU as the rank).
            list_args[6] = self._device_index
            bases[sid] = self._base_cache[reduce_args] = rebuild_cuda_tensor(*list_args)
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
        clear the freed backlog. The engine's gather-credit gate calls this once
        its loop is ``gather_lookahead`` groups ahead, so this wait is where the
        trainer paces to the consumers' pull rate.

        Raises rather than returning empty when the sync errors or stalls: the
        engine is blocked here inside its gather loop, and an empty return would
        spin it straight back in.
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
            self._base_cache.clear()
            return freed

    def set_gather_error(self, message: str) -> None:
        """Record a trainer-side gather failure so blocked serves / publishes
        stop waiting and surface it."""
        with self._cache_cond:
            self._gather_error = RuntimeError(message)
            self._cache_cond.notify_all()

    # ---------------- consumer-facing (called by name over Ray) ----------------

    def free_group(self, group_idx: int) -> None:
        """Consumer back-edge: one consumer is done with group ``group_idx``,
        either because its last chunk landed or because it had nothing to pull.

        The per-group barrier counts one signal per live consumer against the
        ``begin_sync`` count; every consumer signals every owner, so the target is
        the same integer everywhere. The last signal drops the cache entries and
        queues the group as gather credit for ``wait_freed``. A signal arriving
        before its publish is only counted — ``publish_group`` completes it.
        """
        gi = int(group_idx)
        with self._cache_cond:
            count = self._free_counts.get(gi, 0) + 1
            self._free_counts[gi] = count
            if gi in self._group_names and count >= self._live_count:
                self._release_group_locked(gi)
            self._note_progress_locked()
            self._cache_cond.notify_all()

    def _new_serve_buffer(self, nbytes: int) -> torch.Tensor:
        """Allocate + NIXL-register one serve slot: the single allocation seam,
        so registration cannot be skipped on either of the two paths that make
        buffers (the init-time reservation and the serve-path backstop)."""
        from ray.experimental import register_nixl_memory

        t = torch.empty(nbytes, dtype=torch.uint8, device=self._serve_device)
        with self._reg_lock:
            register_nixl_memory(t)
        return t

    def reserve_serve_buffer(
        self, consumer_id: int, nbytes: int, plan_digest: str | None = None
    ) -> None:
        """Pre-allocate + NIXL-register this consumer's serve ring before any
        pull, while the fabric is idle (avoids registration races during the
        sync-0 RDMA churn under M:N fan-in). Idempotent; grows only if needed.

        The ring is keyed by SLOT-SHARING GROUP, so the sharers of one group
        reserve ONE ring between them. They all call this, same group and same
        size, so the whole body runs under ``_serve_lock``, or two concurrent
        first-callers would each allocate a ring and one would be dropped while
        still registered.

        ``plan_digest`` is the caller's ordered chunk plan for THIS producer.
        Sharers must agree on it, and this is where a disagreement is caught
        rather than at serve time, where it is a rendezvous nobody completes.

        Raises:
            RuntimeError: two consumers of one sharing group disagree on the
                chunks they pull from this producer.
        """
        sg = self._share_group(consumer_id)
        if plan_digest is not None:
            with self._cache_cond:
                seen = self._plan_digests.setdefault(sg, {})
                clash = next(
                    ((c, d) for c, d in seen.items() if d != plan_digest), None
                )
                if clash is None:
                    seen[consumer_id] = plan_digest
            if clash is not None:
                raise RuntimeError(
                    f"consumers {clash[0]} and {consumer_id} are in slot-sharing "
                    f"group {sg} but do not pull the same chunks from this "
                    f"producer ({clash[1]} != {plan_digest}). Sharing a serve "
                    f"slot requires the deployments to be identical; check that "
                    f"workers_per_replica matches the inference fleet's "
                    f"geometry."
                )
        # This is where the producer's NIXL serve memory is committed: one ring
        # of K slots per SHARING GROUP that pulls from this rank -- with a single
        # deployment, one ring per consumer, since every group is a singleton.
        # Routing only sends a consumer here for names this rank owns, so the
        # count is the fan-in, not the fleet.
        #
        # Every slot is sized by the LARGEST chunk that consumer pulls from us,
        # rounded up to a 256 MB multiple, so the whole ring pays for the worst
        # chunk in the plan while most chunks are far smaller.
        #
        # TODO: that over-allocates. Sizing slots per chunk instead of per plan,
        # or capping chunk bytes so no chunk is an outlier, would shrink it; so
        # would dropping the 256 MB round-up, which is slack once a buffer never
        # regrows. Row-splitting the vocab matrix shrinks the outlier itself.
        # The host-staging TODO above would remove these rings altogether.
        alloc = buffer_alloc_bytes(nbytes, self._buffer_presize)
        with self._serve_lock:
            rings = self._serve_rings.setdefault(sg, [None] * self._nring)
            for i in range(self._nring):
                slot = rings[i]
                if slot is None or slot.numel() < alloc:
                    rings[i] = self._new_serve_buffer(alloc)

    def _share_group(self, consumer_id: int) -> int:
        """The slot-sharing group ``consumer_id`` belongs to: its index within
        its own deployment, since that is what fixes the plan. Without a width
        the group is the consumer itself and nothing is shared."""
        return consumer_id % self._share_width if self._share_width > 0 else consumer_id

    def _sharers_of(self, sg: int) -> frozenset:
        """The LIVE consumers of group ``sg``: the rendezvous width. Derived from
        ``begin_sync``'s live set, so a degraded sync narrows the barrier instead
        of waiting forever on a dead deployment. Caller holds ``_cache_cond``."""
        cached = self._sharers.get(sg)
        if cached is None:
            if not self._sharing_active or self._live_ids is None:
                cached = frozenset((sg,))
            else:
                cached = frozenset(
                    c for c in self._live_ids if c % self._share_width == sg
                ) or frozenset((sg,))
            self._sharers[sg] = cached
        return cached

    def _join_share_group(self, consumer_id: int, seq: int) -> tuple:
        """Register this call's arrival in its group's rendezvous for ``seq`` and
        return ``(generation, is_packer)``.

        Exactly one caller per generation gets ``is_packer=True``, whichever
        completes the arrival set, and it is the only one that touches the GPU.
        The slot is ``seq % nring``: fixed by the consumers' issue order, so it
        needs no release accounting on this side.
        """
        sg = self._share_group(consumer_id)
        with self._cache_cond:
            gens = self._gens.setdefault(sg, {})
            gen = gens.get(seq)
            if gen is None:
                gen = gens[seq] = _SharedPack(group=sg, slot=seq % self._nring)
            gen.arrived.add(consumer_id)
            if gen.packing or gen.done or not gen.arrived >= self._sharers_of(sg):
                return gen, False
            gen.packing = True
            return gen, True

    def _await_shared_pack(self, gen: _SharedPack) -> torch.Tensor:
        """Block until this generation's packer published its blob, and return
        it. A failed pack is re-raised here, so every sharer of a bad pack fails
        instead of reading a half-written slot."""
        with self._cache_cond:
            self._wait_for(
                lambda: not gen.done and gen.error is None,
                "a co-replica's shared serve pack",
            )
            if gen.error is not None:
                raise RuntimeError(
                    f"the sharer packing this chunk failed: {gen.error!r}"
                ) from gen.error
            if not gen.done:
                raise RuntimeError(
                    f"gather errored while awaiting a shared pack: "
                    f"{self._gather_error!r}"
                )
            return gen.blob

    def _serve_slot(self, sg: int, idx: int, need: int) -> torch.Tensor:
        """Group ``sg``'s ring slot ``idx``, grown if this chunk outgrew the
        reservation. Growing registers memory while the fabric is busy, the
        hazard ``reserve_serve_buffer`` exists to avoid, so it is a backstop: the
        reservation is sized from the same static plan as this pack."""
        with self._serve_lock:
            buffer = self._serve_rings.setdefault(sg, [None] * self._nring)[idx]
        if buffer is not None and buffer.numel() >= need:
            return buffer
        buffer = self._new_serve_buffer(buffer_alloc_bytes(need, self._buffer_presize))
        with self._serve_lock:
            self._serve_rings.setdefault(sg, [None] * self._nring)[idx] = buffer
        return buffer

    def _fail_shared_pack(self, gen: _SharedPack, exc: BaseException) -> None:
        """Publish a pack failure so waiting sharers raise instead of hanging on
        a generation that will never complete."""
        with self._cache_cond:
            gen.error = exc
            self._cache_cond.notify_all()

    @ray.method(tensor_transport="nixl")
    def rdt_produce_weights_batched(
        self, specs: list, consumer_id: int = 0, seq: int = -1
    ):
        """Serve one batched slice request over NIXL.

        Waits until the specs' names are cached, then rendezvouses with the
        other live sharers of this consumer's slot-sharing group. The last to
        arrive replays each spec's op chain (pure views into cached tensors,
        guarded by ALLOWED_OPS) and byte-packs the slices 16B-aligned into the
        group's ring slot ``seq % nring``, mirroring the consumer's identical
        layout; every sharer then returns that one packed blob, so R deployments
        cost one pack and one slot instead of R. With a single deployment every
        group is a singleton, so the arriving call is always its own packer and
        this is the unshared serve path exactly.

        ``seq`` is the caller's index in its pull stream to this producer, which
        is what fixes the slot. Callers that want a single slice pass one spec
        and read the blob back with that slice's dtype/shape.

        Raises:
            ValueError: ``seq`` was not supplied.
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

        if seq < 0:
            # The slot is derived from it, so a caller that does not send one
            # cannot be served safely.
            raise ValueError(
                "rdt_produce_weights_batched requires the consumer's issue index `seq`"
            )
        gen, is_packer = self._join_share_group(consumer_id, seq)
        if not is_packer:
            blob = self._await_shared_pack(gen)
        else:
            try:
                blob = self._pack_shared(gen, specs)
            except BaseException as e:
                self._fail_shared_pack(gen, e)
                raise
        # A served pull is the third of the four progress signals the stall
        # watchdog reads (publish / produce / free / begin_sync): a long sync whose
        # consumers pull steadily but slowly must never trip it.
        with self._cache_cond:
            self._note_progress_locked()
        return [blob]

    def _pack_shared(self, gen: _SharedPack, specs: list) -> torch.Tensor:
        """Replay the op chains into this generation's slot and publish the blob
        to the sharers waiting on it.

        Runs on exactly one call per generation and OUTSIDE ``_cache_cond``: the
        pack is GPU work and must not block publishes, frees or other groups.
        """
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

        buffer = self._serve_slot(gen.group, gen.slot, pack_cur)

        # The destination views are a pure function of the packed
        # layout, which is byte-identical every sync, so build them once per
        # (sharing group, ring slot, layout) and reuse — rebuilding per call cost
        # 5.2ms of a 7.5ms 384-spec 235B group.
        #
        # The key is the LAYOUT, not the spec names: a name can appear in two
        # requests with different op chains (one name's copies can split across
        # owner-class chunks), and serving the second through the first's views
        # would write the wrong bytes with nothing downstream to catch it.
        dst_key = (
            gen.group,
            gen.slot,
            tuple((off, t.dtype, t.shape) for off, t in sliced),
        )
        cached = self._pack_dsts.get(dst_key)
        if cached is None or cached[0] != buffer.data_ptr():
            dsts = []
            for off, t in sliced:
                nb = t.numel() * t.element_size()
                dsts.append(buffer[off : off + nb].view(t.dtype).reshape(t.shape))
            self._pack_dsts[dst_key] = (buffer.data_ptr(), dsts)
        else:
            dsts = cached[1]
        torch._foreach_copy_(dsts, [t for _off, t in sliced])

        blob = buffer[:pack_cur]
        with self._cache_cond:
            gen.blob = blob
            gen.done = True
            self._note_progress_locked()
            self._cache_cond.notify_all()
        return blob

    def shutdown(self) -> None:
        with self._cache_cond:
            self._cache.clear()
        with self._cache_cond:
            # Generations hold blob views into the rings, so they go first.
            self._gens.clear()
            self._sharers.clear()
        with self._serve_lock:
            self._serve_rings.clear()
            # Must go with the rings: these are views INTO them, and the
            # data_ptr guard that normally invalidates them cannot tell a freed
            # buffer from a new one recycled at the same address.
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
        # Ownership resolved at trainer_init from the fleet's held names: the
        # distinct owner sets, the per-name index into them, this rank's groups
        # and its held set. Routing itself is consumer-side; the trainer only
        # ships the table.
        self._owner_sets: list[list[int]] = []
        self._name_owner_class: list[int] = []
        self._owned_idx: list[int] = []
        self._held_names: set[str] | None = None
        # Strong refs to gathered tensors we've shared into the server, keyed by
        # group index. CUDA-IPC exports must outlive the importer, so we hold
        # them until the server reports the group freed. See send_weights.
        self._inflight: dict[int, dict[str, torch.Tensor]] = {}
        # [RDT-EXPORT-RING] Reusable packed slots, one per resident group. See
        # _pack_group_for_export. Persistent across syncs: freeing and
        # reallocating them each sync fragmented the caching allocator badly
        # enough to grow reserved memory ~1.5 GB per sync.
        self._export_ring: list[torch.Tensor | None] = []
        self._export_ring_args: list[tuple | None] = []
        # Which live group holds which slot, and the slots free to hand out. A
        # modular counter is unsafe here: the credit gate bounds the NUMBER of
        # unfreed groups, not their order, so it can hand a later group the slot
        # a still-live group is being served from.
        self._slot_of_group: dict[int, int] = {}
        self._free_slots: list[int] = []
        # Set by trainer_init from the server-name all-gather; retained so a
        # restarted consumer can be re-initialized without another collective.
        self._server_names: list[str] | None = None

    def _rpc(self, method: str, *args: Any) -> Any:
        """Call one of the server actor's methods and block for the result.
        The single seam through which the engine talks to its server, so tests
        can inject a local (non-Ray) fake server."""
        import ray

        return ray.get(getattr(self._server, method).remote(*args))

    @staticmethod
    def _contig_stride(shape) -> tuple:
        stride: list[int] = []
        acc = 1
        for s in reversed(list(shape)):
            stride.append(acc)
            acc *= int(s)
        return tuple(reversed(stride))

    def _pack_group_for_export(self, held: list, slot_idx: int) -> tuple:
        """Copy ``held`` into ring slot ``slot_idx``; return ``(storages, views,
        refs)`` in the same shape the per-storage path returns, but with ONE
        storage. Views are built exactly as ``publish_group`` rebuilds them, so
        the two sides cannot disagree about layout.

        Slot safety is not the credit gate alone: that bounds how MANY groups
        are unfreed, not the order they free in, and barriers do complete out of
        order. The caller takes ``slot_idx`` from a pool keyed by live group
        (``_slot_of_group`` / ``_free_slots``); ``_drop_inflight`` returns a slot
        only once its group is freed everywhere.
        """
        offsets: list[int] = []
        cur = 0
        for _name, t in held:
            cur = (cur + 15) & ~15  # 16B keeps every element size we ship aligned
            offsets.append(cur)
            cur += t.numel() * t.element_size()
        need = max(16, (cur + 15) & ~15)

        slot = self._export_ring[slot_idx]
        device = held[0][1].device
        if slot is None or slot.numel() < need or slot.device != device:
            self._export_ring[slot_idx] = self._export_ring_args[slot_idx] = None
            slot = torch.empty(need, dtype=torch.uint8, device=device)
            self._export_ring[slot_idx] = slot

        ust = slot.untyped_storage()
        sid = ust.data_ptr()
        reduce_args = self._export_ring_args[slot_idx]
        if reduce_args is None:
            base = torch.empty(0, dtype=torch.uint8, device=device)
            base.set_(ust, 0, (ust.nbytes(),))
            _rebuild, reduce_args = reduce_tensor(base)
            self._export_ring_args[slot_idx] = reduce_args

        views: dict[str, tuple] = {}
        refs: dict[str, torch.Tensor] = {}
        for (name, t), off in zip(held, offsets):
            shape, esz = tuple(t.shape), t.element_size()
            stride = self._contig_stride(shape)
            dst = torch.as_strided(slot.view(t.dtype), shape, stride, off // esz)
            dst.copy_(t)
            refs[name] = dst
            views[name] = (
                sid,
                str(t.dtype).split(".")[-1],
                list(shape),
                list(stride),
                off // esz,
            )
        # The producer sidecar reads this slot from another process over CUDA
        # IPC, while these copies are merely enqueued on this rank's stream;
        # nothing orders the two. Without this sync it packs bytes the copy has
        # not written yet -- silent corruption, and only for ring-packed tensors.
        if slot.is_cuda:
            torch.cuda.current_stream(slot.device).synchronize()
        return {sid: reduce_args}, views, refs

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

        # The VMM-based expandable_segments allocator makes IPC storage opens ~9x
        # slower on both export and rebuild, and CUDA-IPC publish is this engine's
        # hot path. Frameworks enabling it should disable it around send_weights.
        # Only the env var is visible; runtime changes are not introspectable.
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
                "group-contiguous. Reorder the source so all names sharing a "
                "layer index are adjacent."
            )

        world, rank = engine._world_and_rank()
        engine._resolve_ownership(world, rank)
        engine._spawn_server(sorted(engine._held_names or []))

        # Every rank's server must exist before the sender's init RPC (the worker
        # init calls reserve_serve_buffer back on ALL producer servers). The
        # all-gather of server names doubles as that barrier.
        server_names = engine._all_gather_server_names(world, rank)
        # Retained so a RESTARTED consumer can be re-initialized without another
        # all-gather (see get_worker_init_payload). The uuid actor names stay valid
        # for the run, because producers never restart.
        engine._server_names = server_names

        if engine.is_sender:
            worker_init = engine._build_worker_init_info(server_names)
            engine.client.init_weight_transfer_engine(asdict(worker_init))
        return engine

    def get_worker_init_payload(self) -> dict:
        """The consumer-side init payload, rebuilt on demand. Pure — no collective.

        Reads only retained state, so a restarted inference engine can rejoin at a
        sync boundary: the driver can ask any time, including mid-run with every
        rank in its own training step. A collective here would deadlock, since the
        ranks are not at a matching one.

        Raises:
            RuntimeError: called before ``trainer_init`` cached the server names.
        """
        if self._server_names is None:
            raise RuntimeError(
                "get_worker_init_payload requires trainer_init to have run (the "
                "producer server names are gathered there)."
            )
        return asdict(self._build_worker_init_info(self._server_names))

    def _world_and_rank(self) -> tuple[int, int]:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            return torch.distributed.get_world_size(), torch.distributed.get_rank()
        return 1, self._init_info.rank

    def _resolve_ownership(self, world: int, rank: int) -> None:
        """Resolve which rank holds which name, and this rank's publish plan.

        A source may hold only part of the model — pipeline stages, expert
        parallelism, or any mix — so each rank declares its held names and the
        fleet all-gathers them. The consumers route per name, so the wire carries
        the transposed result: the distinct owner sets, and a per-name index into
        them.

        The masks are positional over metadata order, which is why the metadata
        digest is checked first: a rank whose names disagree would transpose into
        the wrong owners entirely.
        """
        assert self.source is not None  # guaranteed by trainer_init
        num_groups = len(self._groups)
        names = [m.name for m in self._meta]
        held = self.source.held_names()
        if held is None:
            held_set = set(names)
        else:
            held_set = {str(n) for n in held}
            unknown = sorted(held_set - set(names))
            if unknown:
                raise ValueError(
                    f"WeightSource.held_names() lists {len(unknown)} name(s) not "
                    f"in metadata(), e.g. {unknown[:3]}."
                )
            if not held_set:
                raise ValueError(
                    "WeightSource.held_names() is empty; a rank with nothing to "
                    "serve cannot take part in the gather"
                )

        # One collective carries this rank's holdings as a bitmask over metadata
        # order, plus the digest of the metadata it indexes into. The digest is
        # what makes partial ownership safe: only the sender's metadata reaches
        # the consumers, so a rank describing just its own share would silently
        # serve the wrong weights.
        digest = self._meta_digest()
        mask = bytearray((len(names) + 7) // 8)
        for i, n in enumerate(names):
            if n in held_set:
                mask[i >> 3] |= 1 << (i & 7)
        per_rank = self._all_gather_owned(world, (digest, bytes(mask)))
        mismatched = [r for r, (d, *_rest) in enumerate(per_rank) if d != digest]
        if mismatched:
            raise ValueError(
                f"WeightSource.metadata() disagrees across trainer ranks "
                f"(rank {rank} digest {digest}, differing ranks {mismatched[:4]}). "
                "Every rank must describe the WHOLE model, even when it holds "
                "only some of it."
            )
        masks = [m for _d, m in per_rank]

        # Transpose to per-name owners, then dedup into classes. Numbering by
        # FIRST APPEARANCE in metadata order keeps it a pure function of
        # rank-identical inputs, which is what lets a rejoining consumer rebuild
        # the identical table from get_worker_init_payload.
        owner_sets: list[list[int]] = []
        class_of_owners: dict[tuple[int, ...], int] = {}
        name_owner_class: list[int] = []
        for i, n in enumerate(names):
            owners = tuple(r for r, m in enumerate(masks) if m[i >> 3] & (1 << (i & 7)))
            if not owners:
                raise ValueError(
                    f"no trainer rank holds {n!r}; every name in metadata() must "
                    "be held by at least one rank or it can never be served."
                )
            ci = class_of_owners.get(owners)
            if ci is None:
                ci = len(owner_sets)
                class_of_owners[owners] = ci
                owner_sets.append(list(owners))
            name_owner_class.append(ci)

        self._owner_sets = owner_sets
        self._name_owner_class = name_owner_class
        # Groups holding anything here: exactly what iteration must cover.
        self._owned_idx = [
            gi
            for gi in range(num_groups)
            if any(n in held_set for n in self._groups[gi])
        ]
        self._held_names = held_set

    def _validate_held_yields(self, gi: int, names, tensors) -> None:
        """Check ``held_names()`` against what the source actually yields — the
        one place both sit side by side.

        Without it, a source that claims a name but yields ``None`` for it dies
        by stall watchdog 300s later: consumers route pulls here, the pull
        passes the served-names guard, and the cache wait never completes. With
        it, that is an immediate error naming the weight. One set lookup per
        name per sync."""
        if self._held_names is None:
            return
        for name, tensor in zip(names, tensors):
            if (tensor is None) == (name in self._held_names):
                claim = "does not hold" if tensor is not None else "holds"
                have = "a real tensor" if tensor is not None else "None"
                raise RuntimeError(
                    f"held_names() disagrees with the yielded tensors: group {gi} "
                    f"name {name!r} yielded {have} but this rank {claim} it. A "
                    "WeightSource must yield a real tensor for every held name "
                    "and None for the rest (see WeightSource.held_names)."
                )

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
        """All-gather each rank's (metadata digest, held-name bitmask)."""
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

        # The trainer is usually a separate install from the workers, so it gets
        # its own check rather than trusting the consumer side's.
        check_ray_rdt_version()

        ii = self._init_info
        self._server_name = f"vllm_rdt_producer_{uuid.uuid4().hex[:12]}_rk{ii.rank}"
        node_id = ray.get_runtime_context().get_node_id()
        # Pin the server to this rank's physical GPU: num_gpus=0 so Ray does not
        # allocate a second, CUDA_VISIBLE_DEVICES so CUDA IPC to the rank's
        # gathered tensors works. max_concurrency > 1 serves pulls while control
        # calls are in flight; enable_tensor_transport gives the NIXL serve.
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
            # A num_gpus=0 actor sharing the rank's GPU: Ray would set
            # CUDA_VISIBLE_DEVICES="" and hide every GPU, so tell it not to touch
            # the var and pin the device ourselves.
            env_vars["RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES"] = "1"
            env_vars["CUDA_VISIBLE_DEVICES"] = str(gpu_ids[0])
        runtime_env = {"env_vars": env_vars} if env_vars else {}
        server_cls = ray.remote(_RDTProducerServer).options(
            name=self._server_name,
            namespace=ii.trainer_actor_namespace,
            num_cpus=0,
            num_gpus=0,
            # Thread budget: under EP-local routing every consumer pulls from every
            # producer, and with issue-ahead each can park K produce calls in the
            # cache-wait — C*K blocked calls. A smaller pool queues the control
            # plane behind them and deadlocks: the parked produces wait for a
            # publish that can never get a thread (the sync-2 wedge at 235B tp8,
            # 16 parked vs 8 threads). Size for the worst case plus control-plane
            # slack.
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
            buffer_presize_gb=ii.buffer_presize_gb,
            gather_lookahead=ii.gather_lookahead,
            stall_timeout_s=ii.stall_timeout_s,
            workers_per_replica=ii.workers_per_replica,
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
            owner_sets=self._owner_sets,
            name_owner_class=self._name_owner_class,
            num_consumers=self._init_info.num_consumers,
            num_rdt_buffers=self._init_info.num_rdt_buffers,
            buffer_presize_gb=self._init_info.buffer_presize_gb,
        )

    # ---------------- per-round ----------------

    def send_weights(self, live_consumer_ids: Collection[int] | None = None) -> None:
        """Gather this rank's weights and publish them for the consumers to pull.

        ``live_consumer_ids`` restricts the sync to the consumers still alive;
        ``None`` serves the whole provisioned set. The provisioned geometry is
        FROZEN for the run — a degraded sync only lowers the live count handed to
        ``begin_sync``. Every rank must get the SAME live set, since they share
        the gather collectives, so the caller computes it once for all of them.
        """
        assert self.source is not None
        if live_consumer_ids is None:
            live_ids = list(range(self._init_info.num_consumers))
            live_count = len(live_ids)
        else:
            live_ids = sorted(set(int(c) for c in live_consumer_ids))
            live_count = len(live_ids)
            logger.warning(
                "[rdt-degraded] serving %d/%d live consumers; every group's "
                "free barrier counts to the live total",
                live_count,
                self._init_info.num_consumers,
            )
        self._send_weights_inner(live_count, live_ids)

    def _send_weights_inner(self, live_count: int, live_ids: list[int]) -> None:
        if not self.is_sender:
            self._run_gather_loop(
                update_future=None, live_count=live_count, live_ids=live_ids
            )
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
            self._run_gather_loop(
                update_future=future, live_count=live_count, live_ids=live_ids
            )
            future.result()  # surface inference-side errors

        self.client.finish_weight_update()

    def _run_gather_loop(
        self, update_future, live_count: int, live_ids: list[int] | None = None
    ) -> None:
        """Gather this rank's weights group-by-group and publish each into the
        server over CUDA IPC. A gathered group is published — serveable —
        immediately; the loop gates BEFORE the next gather while more than
        `gather_lookahead` groups are unfreed (the per-group free barrier: a
        credit releases when every live consumer has signaled the group). So
        the loop self-paces to the consumers' pull rate with at most
        `gather_lookahead + 1` groups resident. Runs on every rank; only the
        sender has an `update_future` to fail fast on."""
        assert self.source is not None  # guaranteed by trainer_init
        # The live IDS ride along with the count: the free barrier needs only
        # the count, but the slot-sharing rendezvous needs to know WHICH
        # consumers are live.
        self._rpc("begin_sync", live_count, live_ids)
        # One generator resume per GROUP: `iter_groups` yields (names, tensors)
        # per owned group in metadata order. Every owner must reach a group in the
        # same order or their shared gather collective mismatches.
        groups = self.source.iter_groups()
        # Publishes fire without an inline ray.get and are harvested this window
        # deep, so the RPC and server-side rebuild overlap the NEXT group's
        # gather/export. The window also surfaces a server-side error at depth 2
        # rather than at end_sync.
        _PUBLISH_WINDOW = 2
        pending_publish: list = []
        # The memory bound: gate BEFORE gathering while more
        # than `bound` groups are unfreed, so at most bound + 1 are resident. The
        # count is exact — `_inflight` gains its entry right after each gather and
        # only wait_freed/end_sync shrinks it, so a group freed server-side but not
        # yet collected still holds its refs and still counts. The publish window
        # drains first, so every resident group is pullable while we wait.
        #
        # 0 is legal and means NO pipelining: the loop waits for group N to be
        # freed by every live consumer before gathering N+1, so exactly one group
        # is resident and the sync costs the full serialized
        # gather + publish + pull + free RTT per group. 1 (the default) overlaps
        # N+1's gather with N's pulls.
        bound = max(0, self._init_info.gather_lookahead)

        # [RDT-EXPORT-RING] `bound + 1` slots is exactly the residency the credit
        # gate enforces. Small storages are packed into a slot and exported once
        # per slot instead of once per storage (518 -> 7 cudaIpcGetMemHandle
        # calls per rank per sync at 235B, each 0.5-1.5ms).
        nslots = bound + 1
        # data_ptr -> (storage weakref, reduce_args), this sync only. `storages`
        # is per-group, so a source that reuses a buffer across groups would
        # otherwise re-export it every group. The weakref is load-bearing: a
        # freed allocation can be reissued at the same address, and serving that
        # from a stale handle is silently wrong bytes.
        handle_cache: dict[int, tuple] = {}
        if len(getattr(self, "_export_ring", [])) != nslots:
            self._export_ring = [None] * nslots
            self._export_ring_args = [None] * nslots
        # Slots are handed out per group and returned by _drop_inflight, so the
        # pool starts full every sync.
        self._slot_of_group = {}
        self._free_slots = list(range(nslots))

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
                self._validate_held_yields(gi, names, tensors)
                # Share each unique STORAGE once and describe every name as an
                # as_strided view onto it. ``None`` means a name this rank does not
                # hold (a foreign expert); the source keeps it in the list so the
                # order check stays rank-uniform, and it is dropped here, before
                # the IPC export, matching the sidecar's served_names.
                storages: dict[int, tuple] = {}
                views: dict[str, tuple] = {}
                refs: dict[str, torch.Tensor] = {}
                small: list = []
                for name, tensor in zip(names, tensors):
                    if tensor is None:
                        continue
                    tensor = tensor.detach()
                    if not tensor.is_cuda:
                        tensor = tensor.cuda()
                    # Threshold on the STORAGE, not the tensor: a source may hand
                    # out many small views of one large stacked buffer, and a
                    # per-tensor test would pack the whole thing, costing +1
                    # group of peak memory for a storage already coalesced.
                    if tensor.untyped_storage().nbytes() <= _EXPORT_RING_MAX_BYTES:
                        small.append((name, tensor))
                        continue
                    tensor = tensor.contiguous()
                    refs[name] = tensor  # keep the export alive
                    ust = tensor.untyped_storage()
                    sid = ust.data_ptr()
                    if sid not in storages:
                        cached = handle_cache.get(sid)
                        if cached is not None and not cached[0].expired():
                            reduce_args = cached[1]
                        else:
                            base = torch.empty(
                                0, dtype=torch.uint8, device=tensor.device
                            )
                            base.set_(ust, 0, (ust.nbytes(),))
                            _rebuild, reduce_args = reduce_tensor(base)
                            handle_cache[sid] = (StorageWeakRef(ust), reduce_args)
                        storages[sid] = reduce_args
                    views[name] = (
                        sid,
                        str(tensor.dtype).split(".")[-1],
                        list(tensor.shape),
                        list(tensor.stride()),
                        tensor.storage_offset(),
                    )
                if small:
                    # Take a slot no live group holds. The gate above already
                    # guarantees one exists, so this is a backstop -- but it
                    # must drain publishes first, like the credit gate.
                    while not self._free_slots:
                        while pending_publish:
                            self._await_publish(pending_publish.pop(0))
                        self._drop_inflight(self._rpc("wait_freed"))
                    slot_idx = self._free_slots.pop(0)
                    self._slot_of_group[gi] = slot_idx
                    st, vw, rf = self._pack_group_for_export(small, slot_idx)
                    storages.update(st)
                    views.update(vw)
                    refs.update(rf)
                del small
                del tensors
                # The loop leaves its last `tensor` / `ust` / `base` bound
                # through the credit gate and into the next gather, and `base`
                # is a whole-storage view -- so one tensor and its entire
                # allocation stay alive past `_drop_inflight`, unseen by the
                # `_inflight` accounting the residency bound rests on.
                # Assignment, not `del`: unbound if no direct-path tensor.
                tensor = ust = base = None
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
        finally:
            # Keep the slots, drop the handles: the sidecar's base cache is
            # cleared per sync, so a handle must not outlive the sync either.
            self._export_ring_args = [None] * len(getattr(self, "_export_ring", []))

    def _drop_inflight(self, freed_keys: list) -> None:
        """Release a freed group's refs and return its export-ring slot."""
        free_slots = getattr(self, "_free_slots", None)
        slot_of_group = getattr(self, "_slot_of_group", None)
        for k in freed_keys:
            gi = int(k)
            self._inflight.pop(gi, None)
            if slot_of_group is None:
                continue
            slot = slot_of_group.pop(gi, None)
            if slot is not None and free_slots is not None and slot not in free_slots:
                free_slots.append(slot)

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
