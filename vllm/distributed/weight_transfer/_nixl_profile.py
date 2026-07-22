# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Lightweight, hardcoded profiling shims for the sharded-RDT NIXL path.

This module is intentionally dependency-free and side-effect-light so it can be
imported in *both* the trainer actor process (NIXL producer) and the vLLM worker
process (NIXL consumer). It does two things:

  1. ``install_nixl_timing()`` monkeypatches Ray's ``NixlTensorTransport`` so the
     lazily-created ``nixl_agent``'s low-level methods accumulate per-process
     timing into module-global counters. Because NIXL is one-sided, the producer
     process only ever calls ``register_memory`` while the consumer process calls
     ``register_memory`` (its receive buffers) *and* ``transfer``/
     ``check_xfer_state`` (the actual RDMA read). So the same patch, read in each
     process, cleanly separates registration from transfer. See
     ``sharded_rdt_engine.py``'s module docstring / the deep dive for the model.

  2. ``read_efa_counters()`` reads every per-device hardware counter EFA exposes
     under ``/sys/class/infiniband/<dev>/ports/<p>/hw_counters/`` (and the
     IB-standard ``counters/`` dir as a fallback), so the driver can diff them
     across a sync and confirm transfer is spread across NICs.

Nothing here is gated behind env vars; it is hardcoded benchmarking scaffolding,
not part of the final commit.
"""

import threading
import time
from collections import defaultdict
from contextlib import contextmanager


class PhaseTimer:
    """Accumulate wall time per named phase of the consumer's ``_replay`` process
    loop, syncing CUDA at each phase exit so async GPU work is charged to the
    phase that launched it (otherwise every phase reads ~0 and the trailing
    cuda.synchronize eats all the time).

    Benchmark scaffolding: the per-phase syncs serialize phases that would
    otherwise overlap, so the split SUMS to ``process`` but slightly inflates it
    vs the un-instrumented path (the doc measured this overhead as negligible).
    """

    def __init__(self, stream=None) -> None:
        self.t: dict = defaultdict(float)
        # When the process phase runs on a dedicated background stream (pull/
        # process pipelining), sync only THAT stream at each phase exit. A global
        # ``torch.cuda.synchronize()`` here would stall the RPC thread's
        # concurrent next-group pull and defeat the pipeline.
        self._stream = stream

    @contextmanager
    def phase(self, name: str):
        import torch

        t0 = time.perf_counter()
        try:
            yield
        finally:
            if self._stream is not None:
                self._stream.synchronize()
            else:
                torch.cuda.synchronize()
            self.t[name] += time.perf_counter() - t0


# ---- per-process NIXL timing accumulators ----
_lock = threading.Lock()
_local = threading.local()  # holds the in-flight transfer's start timestamp
_patched = False

_FIELDS = (
    "register_seconds",
    "register_calls",
    "deregister_seconds",
    "deregister_calls",
    "transfer_seconds",  # wall time from transfer() to the DONE state (incl. sleeps)
    "transfer_calls",
    "initialize_xfer_seconds",
    "descs_seconds",  # get/serialize/deserialize xfer descs + add_remote_agent
    # Producer-side only: total time inside Ray's extract_tensor_transport_metadata
    # (the post-return path that does cuda.synchronize + register + build descs).
    # Subtract register_seconds + descs_seconds from this to isolate the
    # per-RPC cuda.synchronize() the producer pays.
    "extract_seconds",
    "extract_calls",
    # Consumer-side remote-agent (re)binding — the suspected control-plane cost.
    # Currently UNtimed by Ray, so it lands in the "control-plane residual"
    # (pull - transfer - register - descs). Time it to split that residual.
    "add_remote_agent_seconds",
    "add_remote_agent_calls",
    "remove_remote_agent_seconds",  # >0 ⇒ producer version churned this pull
    "remove_remote_agent_calls",
    "check_calls",  # check_xfer_state invocations; ~(check_calls-1) poll sleeps/pull
    # DIRECT consumer-side split of pull (no cross-process inference):
    # produce_wait = time from pull-start (just before produce.remote) to the
    # NIXL read starting (recv_multiple_tensors entry) = producer execution
    # (slice+clone) + metadata transport + Ray dispatch, all blocked-on.
    # recv_wall = the NIXL read itself (transfer + descs + add). pull ≈ produce_wait + recv_wall.
    "produce_wait_seconds",
    "produce_wait_calls",
    "recv_wall_seconds",
    # Sub-split of produce_wait: time from the engine entering ray.get on the
    # drained chunk (mark_get_entry) to recv entry = the TRUE wait inside the
    # get (producer metadata not yet arrived + rdt store plumbing). The
    # remainder (produce_wait - meta_wait) is issue-side work between the last
    # produce.remote and the drain (set_target, loop Python, GIL).
    "meta_wait_seconds",
)

# Set by the engine just before it calls produce.remote(); read at recv entry.
# Pulls are serialized per consumer, so a plain holder (no thread-local) is safe.
_pull_t0 = [0.0]
_get_t0 = [0.0]


def mark_pull_start() -> None:
    _pull_t0[0] = time.perf_counter()


def mark_get_entry() -> None:
    """Stamp just before the engine's blocking ray.get on a pending pull."""
    _get_t0[0] = time.perf_counter()


def _zero() -> dict:
    return {f: 0.0 if f.endswith("seconds") else 0 for f in _FIELDS}


_stats: dict = _zero()


def snapshot() -> dict:
    """Return a copy of the current per-process NIXL counters."""
    with _lock:
        return dict(_stats)


def reset() -> None:
    with _lock:
        _stats.update(_zero())


def delta(before: dict, after: dict) -> dict:
    """after - before, field by field."""
    return {f: after[f] - before[f] for f in _FIELDS}


def _add(field: str, amount: float, calls: str | None = None) -> None:
    with _lock:
        _stats[field] += amount
        if calls is not None:
            _stats[calls] += 1


def _wrap_agent(agent) -> None:
    """Replace the agent's relevant bound methods with timed versions (once)."""
    if getattr(agent, "_rdt_timed", False):
        return

    orig_register = agent.register_memory
    orig_dereg = agent.deregister_memory
    orig_transfer = agent.transfer
    orig_check = agent.check_xfer_state
    orig_init = agent.initialize_xfer
    orig_get_descs = agent.get_xfer_descs
    orig_ser = agent.get_serialized_descs
    orig_deser = agent.deserialize_descs
    orig_add_remote = getattr(agent, "add_remote_agent", None)
    orig_rem_remote = getattr(agent, "remove_remote_agent", None)

    def register_memory(*a, **k):
        t0 = time.perf_counter()
        try:
            return orig_register(*a, **k)
        finally:
            _add("register_seconds", time.perf_counter() - t0, "register_calls")

    def deregister_memory(*a, **k):
        t0 = time.perf_counter()
        try:
            return orig_dereg(*a, **k)
        finally:
            _add("deregister_seconds", time.perf_counter() - t0, "deregister_calls")

    def initialize_xfer(*a, **k):
        t0 = time.perf_counter()
        try:
            return orig_init(*a, **k)
        finally:
            _add("initialize_xfer_seconds", time.perf_counter() - t0)

    def get_xfer_descs(*a, **k):
        t0 = time.perf_counter()
        try:
            return orig_get_descs(*a, **k)
        finally:
            _add("descs_seconds", time.perf_counter() - t0)

    def get_serialized_descs(*a, **k):
        t0 = time.perf_counter()
        try:
            return orig_ser(*a, **k)
        finally:
            _add("descs_seconds", time.perf_counter() - t0)

    def deserialize_descs(*a, **k):
        t0 = time.perf_counter()
        try:
            return orig_deser(*a, **k)
        finally:
            _add("descs_seconds", time.perf_counter() - t0)

    def transfer(*a, **k):
        # Stamp the start of the RDMA read window on this thread. recv runs the
        # whole transfer (transfer + poll) on one thread, so thread-local is safe.
        _local.t0 = time.perf_counter()
        state = orig_transfer(*a, **k)
        if state == "DONE":  # small transfers can complete synchronously
            _add("transfer_seconds", time.perf_counter() - _local.t0, "transfer_calls")
            _local.t0 = None
        return state

    def check_xfer_state(*a, **k):
        with _lock:
            _stats["check_calls"] += 1
        state = orig_check(*a, **k)
        if state == "DONE" and getattr(_local, "t0", None) is not None:
            _add("transfer_seconds", time.perf_counter() - _local.t0, "transfer_calls")
            _local.t0 = None
        return state

    def add_remote_agent(*a, **k):
        t0 = time.perf_counter()
        try:
            return orig_add_remote(*a, **k)
        finally:
            _add("add_remote_agent_seconds",
                 time.perf_counter() - t0, "add_remote_agent_calls")

    def remove_remote_agent(*a, **k):
        t0 = time.perf_counter()
        try:
            return orig_rem_remote(*a, **k)
        finally:
            _add("remove_remote_agent_seconds",
                 time.perf_counter() - t0, "remove_remote_agent_calls")

    agent.register_memory = register_memory
    agent.deregister_memory = deregister_memory
    agent.initialize_xfer = initialize_xfer
    agent.get_xfer_descs = get_xfer_descs
    agent.get_serialized_descs = get_serialized_descs
    agent.deserialize_descs = deserialize_descs
    agent.transfer = transfer
    agent.check_xfer_state = check_xfer_state
    if orig_add_remote is not None:
        agent.add_remote_agent = add_remote_agent
    if orig_rem_remote is not None:
        agent.remove_remote_agent = remove_remote_agent
    agent._rdt_timed = True


def install_nixl_timing() -> bool:
    """Patch NixlTensorTransport.get_nixl_agent to wrap its agent on first use.

    Idempotent and fail-soft: returns True if the patch is in place, False if Ray
    internals weren't found (so callers can degrade gracefully).
    """
    global _patched
    if _patched:
        return True
    try:
        from ray.experimental.rdt.nixl_tensor_transport import NixlTensorTransport
    except Exception:
        return False

    orig_get_agent = NixlTensorTransport.get_nixl_agent

    def get_nixl_agent(self):
        agent = orig_get_agent(self)
        try:
            _wrap_agent(agent)
        except Exception:
            pass
        return agent

    NixlTensorTransport.get_nixl_agent = get_nixl_agent

    # Time the producer-side metadata extraction (runs on the trainer after a
    # tensor_transport="nixl" method returns). This wraps the cuda.synchronize +
    # register + descriptor build; subtracting register/descs isolates the sync.
    orig_extract = NixlTensorTransport.extract_tensor_transport_metadata

    def extract_tensor_transport_metadata(self, *a, **k):
        t0 = time.perf_counter()
        try:
            return orig_extract(self, *a, **k)
        finally:
            _add("extract_seconds", time.perf_counter() - t0, "extract_calls")

    NixlTensorTransport.extract_tensor_transport_metadata = (
        extract_tensor_transport_metadata
    )

    # Consumer-side: split pull into produce_wait (blocked on producer execution +
    # metadata + dispatch) vs recv_wall (the NIXL read). recv_multiple_tensors is
    # invoked by Ray during the engine's ray.get, after produce returns.
    orig_recv = NixlTensorTransport.recv_multiple_tensors

    def recv_multiple_tensors(self, *a, **k):
        t_start = time.perf_counter()
        if _pull_t0[0]:
            _add("produce_wait_seconds", t_start - _pull_t0[0], "produce_wait_calls")
            _pull_t0[0] = 0.0
        if _get_t0[0]:
            # In-get portion: metadata wait + rdt store plumbing.
            _add("meta_wait_seconds", t_start - _get_t0[0])
            _get_t0[0] = 0.0
        try:
            return orig_recv(self, *a, **k)
        finally:
            _add("recv_wall_seconds", time.perf_counter() - t_start)

    NixlTensorTransport.recv_multiple_tensors = recv_multiple_tensors

    _patched = True
    return True


# ---- EFA / RDMA per-NIC hardware counters ----
import os  # noqa: E402

_IB_ROOT = "/sys/class/infiniband"


def read_efa_counters() -> dict:
    """Read every numeric per-device RDMA counter on this node.

    Returns ``{"<dev>:<port>:<counter>": int}`` for all hw_counters (EFA) and
    counters (IB/RoCE) files that parse as ints. Reading everything (rather than
    guessing names) means the driver can just diff and report whichever counters
    moved -- robust across EFA / mlx5 / RoCE.
    """
    out: dict = {}
    try:
        devices = os.listdir(_IB_ROOT)
    except OSError:
        return out
    for dev in devices:
        ports_dir = os.path.join(_IB_ROOT, dev, "ports")
        try:
            ports = os.listdir(ports_dir)
        except OSError:
            continue
        for port in ports:
            for sub in ("hw_counters", "counters"):
                cdir = os.path.join(ports_dir, port, sub)
                try:
                    files = os.listdir(cdir)
                except OSError:
                    continue
                for cname in files:
                    try:
                        with open(os.path.join(cdir, cname)) as f:
                            val = int(f.read().strip())
                    except (OSError, ValueError):
                        continue
                    out[f"{dev}:{port}:{cname}"] = val
    return out
