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

Copies made while torch.compile is tracing or a CUDA graph is being captured
are left alone.
"""

import threading
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
_orig_stream_synchronize = torch.cuda.Stream.synchronize
_orig_stream_query = torch.cuda.Stream.query
_orig_cuda_synchronize = torch.cuda.synchronize
_orig_accel_synchronize = torch.accelerator.synchronize

# Single worker keeps D2H copies in submission order; created by install().
_d2h_executor: ThreadPoolExecutor | None = None
_pending_lock = threading.Lock()
_pending: list[Future] = []
# Raw handles of the streams registered via register_d2h_stream().
_d2h_streams: set[int] = set()


def register_d2h_stream(stream: torch.cuda.Stream) -> None:
    """Route non_blocking D2H copies issued on ``stream`` through the worker
    thread once ``install()`` has run (see the module docstring)."""
    _d2h_streams.add(stream.cuda_stream)


@cache
def _staging_stream(device_index: int) -> torch.cuda.Stream:
    return torch.cuda.Stream(device_index)


@cache
def _d2h_stream(device_index: int) -> torch.cuda.Stream:
    return torch.cuda.Stream(device_index)


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


def _d2h(src: torch.Tensor, out: torch.Tensor | None = None) -> torch.Tensor | None:
    """Returns None if the copy should fall back to the original method."""
    cur = torch.cuda.current_stream(src.device)
    if cur.cuda_stream not in _d2h_streams:
        return None
    if out is None:
        out = torch.empty(src.shape, dtype=src.dtype, pin_memory=True)
    ready = torch.cuda.Event(blocking=True)
    _orig_cuda_event_record(ready, cur)

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
    global _d2h_executor
    if _d2h_executor is not None:
        return
    _d2h_executor = ThreadPoolExecutor(1, thread_name_prefix="vllm-cc-d2h")
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
    torch.cuda.Stream.query = _stream_query
    torch.cuda.synchronize = _cuda_synchronize
    torch.accelerator.synchronize = _accel_synchronize
