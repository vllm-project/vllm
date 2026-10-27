# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Keep non-blocking host<->device copies non-blocking under NVIDIA CC.

Under bounce-buffer Confidential Computing, cudaMemcpyAsync is
host-synchronous: it returns only after the copy *and all prior work on its
stream* complete. ``install()`` patches torch's non_blocking copy entry points
so that:

- H2D runs on an idle per-device staging stream, so the host waits only for
  the transfer. The current stream then waits on the staging stream. A copy
  into an existing tensor lands in a staging tensor and is moved by a D2D on
  the current stream, so it is ordered exactly like the original copy. This
  applies to every caller.
- D2H issued on a stream registered with ``register_d2h_stream()`` is issued
  by a worker thread once the source is ready. Event / stream / device
  synchronize additionally wait for the D2H copies submitted before the event
  was recorded (or before the call). Registering a stream asserts that its D2H
  sources aren't overwritten until the copy is synced (later compute-stream
  work is never ordered after a side stream anyway), and that their completion
  is only awaited through the patched torch.cuda / torch.Event methods. D2H on
  any other stream keeps the original, host-blocking copy, since e.g. a
  ``torch.accelerator.current_stream().synchronize()`` can't be patched.
  A registered stream is only ever ordered after a producer stream by
  ``wait_stream()`` and then carries D2H copies, so the events that mark its
  position (the copies' readiness, and events recorded on it such as the
  caller's copy event) are recorded on that producer stream instead. Under CC
  an event recorded on an otherwise idle stream behind a cross-stream wait was
  observed to fire only once the compute stream's *next* graph had finished,
  which delayed every sampled token by a full step; an event recorded on the
  producer stream fires on time.

Copies made while torch.compile is tracing or a CUDA graph is being captured
are left alone, as are H2D copies into an existing tensor that would need a
staging buffer larger than ``_MAX_STAGED_BYTES`` (e.g. KV cache swaps).
"""

import threading
import weakref
from concurrent.futures import Future, ThreadPoolExecutor
from functools import cache
from typing import Any

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)

_orig_copy_ = torch.Tensor.copy_
_orig_to = torch.Tensor.to
_orig_cuda = torch.Tensor.cuda
_orig_event_cls = torch.Event
_orig_cuda_event_record = torch.cuda.Event.record
_orig_cuda_event_synchronize = torch.cuda.Event.synchronize
_orig_cuda_event_query = torch.cuda.Event.query
_orig_stream_synchronize = torch.cuda.Stream.synchronize
_orig_stream_wait_stream = torch.cuda.Stream.wait_stream
_orig_stream_query = torch.cuda.Stream.query
_orig_cuda_synchronize = torch.cuda.synchronize
_orig_accel_synchronize = torch.accelerator.synchronize

# Single worker keeps D2H copies in submission order; created by install().
_d2h_executor: ThreadPoolExecutor | None = None
_pending_lock = threading.Lock()
_pending: list[Future] = []
# Raw handles of the streams registered via register_d2h_stream().
_d2h_streams: set[int] = set()
# For each registered stream, the producer stream it last waited on.
_producer_of: dict[int, torch.cuda.Stream] = {}
# Storage base addresses of the tensors registered via
# register_inplace_h2d_target(); views of them are matched too.
_inplace_h2d_storages: set[int] = set()
# Whether a non_blocking H2D from pinned memory has completed when
# cudaMemcpyAsync returns (bounce-buffer CC). Probed by install(). When it has,
# an in-place upload needs no compute-stream wait on the staging stream.
_pinned_h2d_host_sync = False
# Upper bound on the staging buffer an H2D copy into an existing tensor may use.
_MAX_STAGED_BYTES = 32 << 20


def register_d2h_stream(stream: torch.cuda.Stream) -> None:
    """Route non_blocking D2H copies issued on ``stream`` through the worker
    thread once ``install()`` has run (see the module docstring)."""
    _d2h_streams.add(stream.cuda_stream)


def register_inplace_h2d_target(tensor: torch.Tensor) -> None:
    """Let non_blocking H2D copies into ``tensor`` (or a view of it) land in
    place on the staging stream once ``install()`` has run: no staging tensor
    and no D2D. The caller guarantees that no GPU reader of ``tensor`` is still
    in flight whenever it is rewritten (e.g. a round-robin pool slot whose
    readers retired). The staging stream is ordered once after the current
    stream, so work that initialized ``tensor`` (e.g. the zero fill of a fresh
    allocation) completes before the first in-place upload. The registration
    lapses with ``tensor``, so a later tensor reusing its memory is not matched.
    """
    if not tensor.is_cuda:
        return
    ptr = tensor.untyped_storage().data_ptr()
    cur = torch.cuda.current_stream(tensor.device)
    _staging_stream(cur.device_index).wait_stream(cur)
    _inplace_h2d_storages.add(ptr)
    weakref.finalize(tensor, _inplace_h2d_storages.discard, ptr)


@cache
def _staging_stream(device_index: int) -> torch.cuda.Stream:
    return torch.cuda.Stream(device_index)


@cache
def _d2h_stream(device_index: int) -> torch.cuda.Stream:
    return torch.cuda.Stream(device_index)


def _nbytes(t: torch.Tensor) -> int:
    return t.numel() * t.element_size()


def _bypass() -> bool:
    return torch.compiler.is_compiling() or torch.cuda.is_current_stream_capturing()


def _h2d(
    src: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype | None = None,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    cur = torch.cuda.current_stream(device)
    staging = _staging_stream(cur.device_index)
    with torch.cuda.stream(staging):
        tmp = _orig_to(src, device=cur.device, dtype=dtype, non_blocking=True)
    cur.wait_stream(staging)
    tmp.record_stream(cur)
    return tmp if out is None else _orig_copy_(out, tmp, non_blocking=True)


def _probe_pinned_h2d_host_sync() -> bool:
    """True if a non_blocking H2D from pinned memory returns only after the
    work queued ahead of it on its stream has completed."""
    stream = torch.cuda.Stream()
    src = torch.ones(1 << 16, pin_memory=True)
    with torch.cuda.stream(stream):
        x = torch.randn(2048, 2048, device=stream.device)
        dst = torch.empty_like(src, device=stream.device)
        for _ in range(16):
            x = x @ x
        before = torch.cuda.Event()
        _orig_cuda_event_record(before, stream)
        _orig_copy_(dst, src, non_blocking=True)
        host_sync = bool(_orig_cuda_event_query(before))
    _orig_stream_synchronize(stream)
    del x, dst
    torch.cuda.empty_cache()
    return host_sync


def _h2d_inplace(out: torch.Tensor, src: torch.Tensor) -> torch.Tensor:
    cur = torch.cuda.current_stream(out.device)
    staging = _staging_stream(cur.device_index)
    with torch.cuda.stream(staging):
        _orig_copy_(out, src, non_blocking=True)
    if not _pinned_h2d_host_sync:
        cur.wait_stream(staging)
    return out


def _d2h(src: torch.Tensor, out: torch.Tensor | None = None) -> torch.Tensor | None:
    """Returns None if the copy should fall back to the original method."""
    cur = torch.cuda.current_stream(src.device)
    if cur.cuda_stream not in _d2h_streams:
        return None
    if out is None:
        out = torch.empty(src.shape, dtype=src.dtype, pin_memory=True)
    ready = torch.cuda.Event(blocking=True)
    _orig_cuda_event_record(ready, _producer_of.get(cur.cuda_stream, cur))

    def run() -> None:
        # Sleep until the producer is done, so the host-synchronous copy only
        # pays for its own transfer (and doesn't hold driver locks meanwhile).
        _orig_cuda_event_synchronize(ready)
        side = _d2h_stream(src.device.index)
        # Entering the stream also switches this thread to its device.
        # Inference mode is thread-local, and out may be an inference tensor;
        # an in-place copy into a normal tensor is allowed in inference mode.
        with torch.inference_mode(), torch.cuda.stream(side):
            _orig_copy_(out, src, non_blocking=True)
            # The closure keeps src alive until here; this keeps the caching
            # allocator from reusing it before the copy has run.
            src.record_stream(side)
        _orig_stream_synchronize(side)

    assert _d2h_executor is not None
    fut = _d2h_executor.submit(run)
    with _pending_lock:
        _pending.append(fut)
    return out


def _pending_snapshot() -> list[Future]:
    with _pending_lock:
        _pending[:] = [f for f in _pending if not f.done()]
        return list(_pending)


def _wait_pending() -> None:
    for f in _pending_snapshot():
        f.result()


def _copy_(self: torch.Tensor, src: torch.Tensor, non_blocking: bool = False):
    if non_blocking and not _bypass():
        if self.is_cuda and src.is_cpu:
            if (
                _inplace_h2d_storages
                and self.dtype == src.dtype
                and self.untyped_storage().data_ptr() in _inplace_h2d_storages
            ):
                return _h2d_inplace(self, src)
            if _nbytes(src) <= _MAX_STAGED_BYTES:
                return _h2d(src, self.device, out=self)
        if (
            self.is_cpu
            and src.is_cuda
            and self.shape == src.shape
            and self.dtype == src.dtype
            and (out := _d2h(src, out=self)) is not None
        ):
            return out
    return _orig_copy_(self, src, non_blocking)


def _to(self: torch.Tensor, *args, **kwargs):
    if not (kwargs.get("non_blocking") or any(a is True for a in args)) or _bypass():
        return _orig_to(self, *args, **kwargs)
    parse_kwargs = {k: v for k, v in kwargs.items() if k != "copy"}
    device, dtype, _, memory_format = torch._C._nn._parse_to(*args, **parse_kwargs)
    if device is not None and memory_format is None:
        if self.is_cpu and device.type == "cuda":
            return _h2d(self, device, dtype)
        if (
            self.is_cuda
            and device.type == "cpu"
            and dtype in (None, self.dtype)
            and (out := _d2h(self)) is not None
        ):
            return out
    return _orig_to(self, *args, **kwargs)


def _cuda(
    self: torch.Tensor,
    device=None,
    non_blocking: bool = False,
    memory_format: torch.memory_format = torch.preserve_format,
):
    if (
        non_blocking
        and self.is_cpu
        and memory_format == torch.preserve_format
        and not _bypass()
    ):
        if device is None:
            device = torch.accelerator.current_device_index()
        if isinstance(device, int):
            device = torch.device("cuda", device)
        return _h2d(self, torch.device(device))
    return _orig_cuda(self, device, non_blocking, memory_format)


def _patch_event_methods(cls: Any) -> dict[str, Any]:
    """Wrap ``cls``'s record/synchronize/query so a recorded event also covers
    the D2H copies submitted before it was recorded."""
    orig_record, orig_synchronize, orig_query = (
        cls.record,
        cls.synchronize,
        cls.query,
    )

    def record(self, stream=None):
        self._cc_pending = _pending_snapshot()
        stream = _record_target(stream)
        if stream is None:
            return orig_record(self)
        return orig_record(self, stream)

    def synchronize(self):
        for f in getattr(self, "_cc_pending", ()):
            f.result()
        return orig_synchronize(self)

    def query(self):
        pending = getattr(self, "_cc_pending", ())
        return all(f.done() for f in pending) and orig_query(self)

    return {"record": record, "synchronize": synchronize, "query": query}


def _stream_wait_stream(self, stream):
    _orig_stream_wait_stream(self, stream)
    if self.cuda_stream in _d2h_streams:
        _producer_of[self.cuda_stream] = stream


def _record_target(stream):
    """Where an event recorded on ``stream`` (None: the current stream) goes:
    the producer stream of a registered D2H stream, otherwise ``stream``."""
    s = stream if stream is not None else torch.cuda.current_stream()
    if s.cuda_stream in _d2h_streams:
        return _producer_of.get(s.cuda_stream, stream)
    return stream


def _stream_synchronize(self):
    _wait_pending()
    return _orig_stream_synchronize(self)


def _stream_query(self):
    return all(f.done() for f in _pending_snapshot()) and _orig_stream_query(self)


def _cuda_synchronize(device=None):
    _wait_pending()
    return _orig_cuda_synchronize(device)


def _accel_synchronize(device_index=None):
    _wait_pending()
    return _orig_accel_synchronize(device_index)


def install() -> None:
    """Install the patches. Call once per worker process, only under CC.

    torch.Event is an immutable C type, so it is replaced with a subclass:
    events created through ``torch.Event(...)`` afterwards are covered, but
    events created in C (e.g. ``torch.Stream.record_event()``) are not, and
    ``isinstance(e, torch.Event)`` becomes False for those and for
    ``torch.cuda.Event`` instances (which subclass the original). Only D2H
    copies on registered streams depend on these patches.
    """
    global _d2h_executor, _pinned_h2d_host_sync
    if _d2h_executor is not None:
        return
    _d2h_executor = ThreadPoolExecutor(1, thread_name_prefix="vllm-cc-d2h")
    _pinned_h2d_host_sync = _probe_pinned_h2d_host_sync()
    logger.info(
        "NVIDIA Confidential Computing: issuing non-blocking H2D copies on a "
        "staging stream and D2H copies from a worker thread."
    )
    torch.Tensor.copy_ = _copy_
    torch.Tensor.to = _to
    torch.Tensor.cuda = _cuda
    for name, fn in _patch_event_methods(torch.cuda.Event).items():
        setattr(torch.cuda.Event, name, fn)
    torch.Event = type(
        "Event", (_orig_event_cls,), _patch_event_methods(_orig_event_cls)
    )
    torch.cuda.Stream.synchronize = _stream_synchronize
    torch.cuda.Stream.wait_stream = _stream_wait_stream
    torch.cuda.Stream.query = _stream_query
    torch.cuda.synchronize = _cuda_synchronize
    torch.accelerator.synchronize = _accel_synchronize
