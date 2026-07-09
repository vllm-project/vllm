# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared trainer-side producer for the sharded-RDT weight-transfer engine.

One implementation of the producer half of the engine's packed single-call
contract (see ``ShardedRDTWeightTransferInitInfo.produce_method_name``), used by
every sharded-RDT example. The tricky invariant — the byte-exact mirror of the
consumer's packed layout — lives here and only here.

A trainer actor mixes in :class:`RDTShardedProducer`, calls
``init_rdt_producer()`` once its CUDA context is up, and either:

- implements ``rdt_gather_group(names)`` (publish gathered tensors via
  ``rdt_publish_gathered``) when weights must be collectively gathered per
  group — the driver kicks ``run_gather_plan(groups)`` once per sync and the
  engine frees each group with ``free_gather`` as its chunks finish; or
- pre-populates ``self._cache`` with live tensors at init when everything is
  always resident (no gather; ``free_gather`` is then a no-op).

Serve path (``rdt_produce_weights_batched``): wait until the specs' names are
cached, replay each spec's op chain (pure views), byte-pack every slice into a
uint8 serve arena (16B-aligned, specs order — the consumer computes the
IDENTICAL layout and carves dtype views back out), and return the ONE packed
blob. Arenas form a ring of NUM_RDT_BUFFERS slots rotated per call, so the
serve of chunk i+1 overwrites nothing the consumer is still reading (the
engine keeps at most NUM_RDT_BUFFERS pulls outstanding and issues produce
#(i+K) only after read #i completed). ``pack=False`` serves one contiguous
tensor per spec instead — the engine's rare residual/unbaked path.

Env knobs (read on the trainer): NUM_RDT_BUFFERS (ring depth, match the
consumer), RDT_ARENA_PRESIZE_GB (arena pre-size; avoids regrowth churn that
can false-hit Ray's data_ptr-keyed NIXL desc cache), RDT_NOSYNC (scoped-sync
serve on a dedicated stream, pairs with the patched Ray extract),
RDT_PACK_CHECK (per-call blob checksums to /tmp/rdt_profile/).
"""
import os
import threading
import time

import ray
import torch

# Op chains the consumer's baked plan may request. The producer refuses any op
# outside this set so a misbehaving / spoofed consumer cannot execute
# arbitrary methods on cached tensors.
ALLOWED_OPS = frozenset(
    (
        "narrow",
        "view",
        "reshape",
        "transpose",
        "permute",
        "contiguous",
        "squeeze",
        "unsqueeze",
        "__getitem__",
        "to",
        "chunk",
        "split",
        "select",
        "flatten",
    )
)

GATHER_LOOKAHEAD = 2  # gathered groups resident (memory bound, matches history)


def layerwise_groups(names: list[str]) -> list[list[str]]:
    """Partition flat parameter names into pre / per-decoder-layer / post gather
    groups (keys on ``model.layers.<N>.``). Shared by every example: the group
    is the unit of gathering, freeing, AND the packed pull's chunk budget —
    without it a whole model becomes one chunk and the receive/serve arenas
    balloon to the full per-worker share."""
    pre: list[str] = []
    layers: dict[int, list[str]] = {}
    post: list[str] = []
    seen = False
    for n in names:
        if n.startswith("model.layers."):
            seen = True
            idx = int(n[len("model.layers."):].split(".", 1)[0])
            layers.setdefault(idx, []).append(n)
        elif not seen:
            pre.append(n)
        else:
            post.append(n)
    groups: list[list[str]] = []
    if pre:
        groups.append(pre)
    for i in sorted(layers):
        groups.append(layers[i])
    if post:
        groups.append(post)
    return groups


class RDTShardedProducer:
    """Mixin implementing the sharded-RDT packed producer contract."""

    # ---------------- init ----------------

    def init_rdt_producer(self, num_consumers: int = 1) -> None:
        """Call once from the actor's __init__ after CUDA is initialized.

        ``num_consumers`` is the TOTAL inference-worker count. Under M:N block
        assignment this producer is bound by ``count_consumers(P, num_consumers,
        rank)`` of them (P = the producer-fleet size), and it must receive a
        ``free_gather`` from EACH before actually freeing a group (ref-counting) —
        otherwise it would drop a gather another consumer is still reading. It also
        keeps a PER-CONSUMER serve ring so interleaved pulls from different
        consumers never share a slot (the C>P fan-in regime); the split regime
        (P>C) has one consumer per producer and reduces to the old single ring."""
        # name -> gathered/live tensor (views fine); guarded by _cache_cond.
        self._cache: dict[str, torch.Tensor] = {}
        self._cache_cond = threading.Condition()
        self._gather_error: BaseException | None = None
        self._gather_sem: threading.Semaphore | None = None

        # [RDT-FREE-REFCOUNT] How many consumers free each group (block-assignment
        # inverse). Every assigned consumer fires free_gather for every group; the
        # group is actually freed (and its gather-lookahead permit released) only on
        # the Nth call. Reset per sync in run_gather_plan.
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            world = dist.get_world_size()
            rank = dist.get_rank()
        else:
            world = int(getattr(self, "world_size", 1) or 1)
            rank = int(getattr(self, "rank", 0) or 0)
        from vllm.distributed.weight_transfer.sharded_rdt_engine import (
            count_consumers,
        )

        self._free_target = count_consumers(world, max(1, num_consumers), rank)
        self._free_counts: dict[tuple, int] = {}
        self._free_lock = threading.Lock()

        # [RDT-RING] PER-CONSUMER ring of packed serve arenas rotated per produce
        # call so the serve of chunk i+1 can fill its slot while chunk i's slot is
        # still being RDMA-read. Keyed by consumer id: each consumer drains its own
        # pulls in order (≤K outstanding), so its own ring is reuse-safe, and two
        # consumers never collide on a slot. Rotation is atomic (_serve_lock).
        self._nring = max(1, int(os.environ.get("NUM_RDT_BUFFERS", "2")))
        self._serve_rings: dict[int, list[torch.Tensor | None]] = {}
        self._serve_idx: dict[int, int] = {}
        self._serve_lock = threading.Lock()
        # Serialize first-use NIXL memory registration across concurrent produce
        # calls. With M:N fan-in (C>P) several consumers pull from one producer,
        # so multiple produce calls run at once (max_concurrency) and each may
        # allocate+register a fresh per-consumer serve arena on first use.
        # ``registerMem`` on a shared NIXL agent is NOT concurrency-safe (observed
        # NIXL_ERR_BACKEND under simultaneous registration); this lock serializes
        # it. Steady state registers nothing, so the lock is uncontended then.
        self._reg_lock = threading.Lock()
        # Pre-size arenas (see the engine's arena_presize_gb docstring: sizing
        # ONCE avoids regrowth churn that can false-hit Ray's data_ptr-keyed
        # NIXL desc cache -> NIXL_ERR_NOT_FOUND / stale-MR writes).
        self._arena_presize = int(
            float(os.environ.get("RDT_ARENA_PRESIZE_GB", "0")) * (1 << 30)
        )

        # [RDT-NOSYNC] Scoped-sync serve: run the pack copies on a dedicated
        # stream that waits only on the served group's gather-completion events,
        # then sync that stream — the served bytes are materialized before
        # produce returns WITHOUT a whole-device sync (pairs with the patched
        # Ray extract that skips its device sync under the same env).
        self._scoped_sync = os.environ.get("RDT_NOSYNC", "0") == "1"
        self._serve_stream = torch.cuda.Stream() if self._scoped_sync else None
        self._cache_event: dict[str, torch.cuda.Event] = {}

        self._pack_check = os.environ.get("RDT_PACK_CHECK", "0") == "1"

        # profiling counters (rdt_profile.py's producer attribution)
        self._timing_lock = threading.Lock()
        self._produce_calls = self._produce_specs = self._produce_bytes = 0
        self._produce_wait_seconds = self._produce_slice_seconds = 0.0
        self._produce_method_seconds = 0.0

        # Freeze the (large, static) post-load object graph so gen-2 GC never
        # stop-the-world scans it mid-produce (measured straggler fix).
        import gc

        gc.collect()
        gc.freeze()

    # ---------------- gather orchestration ----------------

    def rdt_gather_group(self, names: list[str]) -> None:
        """Collectively gather one group and publish it via
        ``rdt_publish_gathered``. Implemented by trainers whose weights are not
        always resident. Every rank receives the IDENTICAL ordered plan, so the
        per-group collectives rendezvous safely."""
        raise NotImplementedError

    def rdt_free_group(self, names: list[str]) -> None:
        """Drop one group's cache entries (override to also free backing
        storage, e.g. gathered physical stacks)."""
        for name in names:
            self._cache.pop(name, None)
            self._cache_event.pop(name, None)

    def rdt_publish_gathered(self, entries: dict[str, torch.Tensor]) -> None:
        """Publish gathered tensors to the serve cache (with a completion event
        for the scoped-sync serve when enabled)."""
        ev = None
        if self._serve_stream is not None:
            ev = torch.cuda.Event()
            ev.record()
        with self._cache_cond:
            self._cache.update(entries)
            if ev is not None:
                for n in entries:
                    self._cache_event[n] = ev
            self._cache_cond.notify_all()

    def run_gather_plan(self, groups: list[list[str]]) -> int:
        """Self-paced gather loop for one sync iteration (driver-kicked).

        Lookahead is bounded by a GATHER_LOOKAHEAD-deep semaphore; the engine's
        ``free_gather`` releases it as each group's chunks finish (and drains
        its free refs before the sync ends, keeping the per-sync semaphore
        accounting balanced). Occupies one actor thread for the sync."""
        # Reset the per-group free ref-counts for this sync. Safe: the driver
        # awaits finish_weight_update (which drains every consumer's free refs) on
        # the previous sync before kicking this one, so all N_p frees for every
        # group have already fired and been counted to completion.
        with self._free_lock:
            self._free_counts.clear()
        self._gather_sem = threading.Semaphore(GATHER_LOOKAHEAD)
        for names in groups:
            self._gather_sem.acquire()
            try:
                self.rdt_gather_group(names)
            except BaseException as e:
                with self._cache_cond:
                    self._gather_error = e
                    self._cache_cond.notify_all()
                raise
        return len(groups)

    def free_gather(self, names: list[str]) -> None:
        """Engine-fired (fire-and-forget) free of one gather group: its chunks
        are fully read on the calling consumer, so its serves are done. No-op for
        gather-free producers (no plan ever ran).

        [RDT-FREE-REFCOUNT] Under M:N a group is shared by ``self._free_target``
        consumers; free (and release the lookahead permit) only after ALL of them
        have called — otherwise a slow consumer's still-in-flight read would find
        its gather buffer already dropped. Exactly one free + one ``sem.release()``
        per group per sync, so the semaphore accounting stays balanced."""
        sem = self._gather_sem
        if sem is None:
            return
        key = tuple(names)
        with self._free_lock:
            count = self._free_counts.get(key, 0) + 1
            self._free_counts[key] = count
            do_free = count >= self._free_target
            if do_free:
                del self._free_counts[key]
        if do_free:
            with self._cache_cond:
                self.rdt_free_group(names)
            sem.release()

    def reserve_serve_arena(self, consumer_id: int, nbytes: int) -> None:
        """Pre-allocate + register this consumer's serve ring AT INIT, before any
        weight sync runs. Called once per binding consumer by the engine's
        ``_preregister_at_init`` with the exact max bytes that consumer will pull
        from this producer (from its static plan). Registering here — while the
        fabric is idle — avoids the intermittent ``ibv_reg_mr: Bad address`` /
        NIXL_ERR_BACKEND that concurrent dma-buf GPUDirect registration hits when
        it coincides with the sync-0 RDMA churn (only reached under M:N fan-in,
        where several consumers pull one producer). Steady-state serving then
        reuses these arenas and registers nothing.

        ``nbytes`` is the max packed bytes for one chunk-run; each of the
        NUM_RDT_BUFFERS ring slots is sized to cover it (same round-up as the lazy
        path). Idempotent: grows a slot only if a later reserve needs more."""
        from ray.experimental import register_nixl_memory

        from vllm.distributed.weight_transfer.sharded_rdt_engine import (
            _arena_alloc_bytes,
        )

        alloc = _arena_alloc_bytes(nbytes, self._arena_presize)
        with self._serve_lock:
            rings = self._serve_rings.setdefault(consumer_id, [None] * self._nring)
            self._serve_idx.setdefault(consumer_id, 0)
        for i in range(self._nring):
            if rings[i] is None or rings[i].numel() < alloc:
                t = torch.empty(alloc, dtype=torch.uint8, device="cuda:0")
                with self._reg_lock:
                    register_nixl_memory(t)
                rings[i] = t

    # ---------------- serve ----------------

    @ray.method(tensor_transport="nixl")
    def rdt_produce_weights_batched(self, specs, pack: bool = True, consumer_id: int = 0):
        """Serve one batched slice request (see module docstring).

        ``consumer_id`` selects this caller's private serve ring so interleaved
        pulls from multiple consumers never share a slot (see init_rdt_producer)."""
        _t_m0 = time.perf_counter()
        needed = sorted({n for n, _ in specs})
        _t_w0 = time.perf_counter()
        with self._cache_cond:
            while not all(n in self._cache for n in needed):
                if self._gather_error is not None:
                    raise RuntimeError(
                        f"gather errored before {needed}: {self._gather_error!r}"
                    )
                self._cache_cond.wait()
        wait_s = time.perf_counter() - _t_w0

        _t_s0 = time.perf_counter()
        # Replay every spec's op chain (pure views into cached tensors) and
        # compute the packed byte layout — 16B-aligned offsets in specs order,
        # mirroring the consumer's computation exactly.
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

        if not pack:
            # Residual/unbaked slow path: one tensor per spec, detached from
            # the cache's lifetime. No arena, no registration (Ray default).
            out = [t.contiguous().clone() for _off, t in sliced]
            torch.cuda.synchronize()
            self._bump_timing(_t_m0, _t_w0, wait_s, _t_s0, len(specs), nbytes)
            return out

        # Rotate to this call's ring slot within THIS consumer's private ring
        # (atomic — overlapping produce calls must never share a slot) and size it
        # once with headroom.
        with self._serve_lock:
            rings = self._serve_rings.setdefault(
                consumer_id, [None] * self._nring
            )
            idx = self._serve_idx.get(consumer_id, 0)
            self._serve_idx[consumer_id] = (idx + 1) % self._nring
        arena = rings[idx]
        if arena is None or arena.numel() < pack_cur:
            from vllm.distributed.weight_transfer.sharded_rdt_engine import (
                _arena_alloc_bytes,
            )

            alloc = _arena_alloc_bytes(pack_cur, self._arena_presize)
            arena = torch.empty(alloc, dtype=torch.uint8, device="cuda:0")
            from ray.experimental import register_nixl_memory

            # Serialize registration: concurrent produce calls (M:N fan-in) must
            # not call registerMem on the shared NIXL agent simultaneously.
            with self._reg_lock:
                register_nixl_memory(arena)  # registered once, reused every call
            rings[idx] = arena

        # Copy the slices into the packed arena. Scoped sync: the copies run on
        # the serve stream after this group's gather events, then the stream is
        # synced — served bytes are materialized before we return.
        ss = self._serve_stream
        if ss is not None:
            for ev in {
                id(e): e
                for e in (self._cache_event.get(n) for n in needed)
                if e is not None
            }.values():
                ss.wait_event(ev)
        with torch.cuda.stream(ss):
            for off, t in sliced:
                nb = t.numel() * t.element_size()
                view = arena[off : off + nb].view(t.dtype).reshape(t.shape)
                view.copy_(t)
        if ss is not None:
            ss.synchronize()

        blob = arena[:pack_cur]
        if self._pack_check:
            self._log_pack_check(blob, pack_cur)
        self._bump_timing(_t_m0, _t_w0, wait_s, _t_s0, len(specs), nbytes)
        return [blob]

    def _bump_timing(self, t_m0, t_w0, wait_s, t_s0, nspecs, nbytes) -> None:
        slice_s = time.perf_counter() - t_s0
        with self._timing_lock:
            self._produce_calls += 1
            self._produce_specs += nspecs
            self._produce_wait_seconds += wait_s
            self._produce_slice_seconds += slice_s
            self._produce_bytes += nbytes
            self._produce_method_seconds += time.perf_counter() - t_m0

    def _log_pack_check(self, blob: torch.Tensor, pack_cur: int) -> None:
        # [RDT-PACK-CHECK] checksum what we serve; the consumer logs the
        # matching sum (compare offline per pull order). Chunked sums:
        # .sum(dtype=int64) upcasts its input 8x — a whole-blob sum OOMs.
        import json

        s = 0
        w = 32 << 20
        for i in range(0, pack_cur, w):
            s += int(blob[i : min(i + w, pack_cur)].sum(dtype=torch.int64))
        os.makedirs("/tmp/rdt_profile", exist_ok=True)
        with open("/tmp/rdt_profile/packcheck_prod.jsonl", "a") as f:
            f.write(json.dumps({"pid": os.getpid(), "bytes": pack_cur, "sum": s}) + "\n")

    # ---------------- profiling accessors (rdt_profile.py) ----------------

    def get_produce_timing(self):
        with self._timing_lock:
            return dict(
                calls=self._produce_calls,
                specs=self._produce_specs,
                wait_seconds=self._produce_wait_seconds,
                slice_seconds=self._produce_slice_seconds,
                bytes=self._produce_bytes,
                method_seconds=self._produce_method_seconds,
            )

    def reset_produce_timing(self):
        with self._timing_lock:
            self._produce_calls = self._produce_specs = self._produce_bytes = 0
            self._produce_wait_seconds = self._produce_slice_seconds = 0.0
            self._produce_method_seconds = 0.0

    def get_nixl_timing(self):
        from vllm.distributed.weight_transfer import _nixl_profile

        return _nixl_profile.snapshot()

    def reset_nixl_timing(self):
        from vllm.distributed.weight_transfer import _nixl_profile

        _nixl_profile.reset()
