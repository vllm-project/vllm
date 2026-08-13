# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Copy helpers for NVIDIA Confidential Computing.

Under bounce-buffer Confidential Computing a host<->device
``cudaMemcpyAsync`` is forced host-SYNCHRONOUS: the issuing thread blocks
until the copy -- and everything already queued on its stream -- completes.
Two consequences for the engine:

* An H2D issued on the compute stream (which has the in-flight forward
  queued) blocks the scheduler for ~one forward per copy, starving the GPU.
* The per-step D2H token readback blocks the thread that issues it, delaying
  the next step's CUDA graph launch by ~one decode step.

Two mitigations, both gated on ``confidential_compute_enabled()`` and no-ops
outside Confidential Computing:

* Staged H2D (``StagedH2DCopier`` / ``prep_stream_ctx``): issue the H2D into
  a device staging buffer on an idle prep stream, so it only pays its own
  transfer (~tens of us) instead of the forward drain; then a D2D
  staging->dst on the compute stream. The D2D is genuinely asynchronous under
  Confidential Computing and is ordered after the forward's read of the
  reused graph-input buffer, so there is no host block and no
  write-after-read race. Correctness relies on the pinned H2D being
  host-synchronous under Confidential Computing: staging is fully populated
  by the time the copy call returns, so the D2D enqueued on another stream
  reads valid data without a cross-stream event. Callers double-buffer the
  staging tensor so the next step's H2D cannot overwrite a staging buffer
  whose D2D has not yet drained. A pool of prep streams is used round-robin
  so consecutive copies in one prepare_inputs are not queued behind each
  other either.

* ``AsyncD2HCopyWorker``: run the (still-blocking) result readback on a
  dedicated daemon thread so the scheduler keeps issuing work (mirrors
  TensorRT-LLM PR #8463). The copy stays synchronous under Confidential
  Computing; it is merely non-blocking *to the scheduler thread*, restoring
  overlap.
"""

import queue
import threading
from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager, nullcontext
from functools import cache
from itertools import cycle

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)

_STAGED_H2D_POOL_SIZE = 16


@cache
def confidential_compute_enabled() -> bool:
    """Whether NVIDIA Confidential Computing is active on this platform.

    The single gate for every copy path in this module.
    """
    from vllm.platforms import current_platform

    return current_platform.is_confidential_compute()


@cache
def _staged_h2d_streams(device_index: int) -> Iterator[torch.cuda.Stream]:
    logger.info_once(
        "Using staged H2D input copies under Confidential Computing "
        "(H2D on a dedicated prep stream + D2D on the compute stream) to "
        "avoid blocking the scheduler on the in-flight forward."
    )
    device = torch.device(f"cuda:{device_index}")
    return cycle(
        [torch.cuda.Stream(device=device) for _ in range(_STAGED_H2D_POOL_SIZE)]
    )


def _staged_h2d_stream(device: torch.device) -> torch.cuda.Stream:
    """Return the next idle prep stream (round-robin) for this device."""
    idx = device.index if device.index is not None else torch.cuda.current_device()
    return next(_staged_h2d_streams(idx))


def prep_stream_ctx(device: torch.device) -> AbstractContextManager:
    """Make an idle prep stream current under Confidential Computing; no-op
    otherwise.

    Use around an H2D whose result is consumed immediately on the issuing
    thread: under Confidential Computing the copy is host-synchronous, so it
    is complete on return regardless of stream, and the prep stream only
    waits for its own transfer instead of the forward queued on the compute
    stream.
    """
    if not confidential_compute_enabled():
        return nullcontext()
    return torch.cuda.stream(_staged_h2d_stream(device))


class StagedH2DCopier:
    """Double-buffered staged H2D into a persistent GPU tensor.

    One instance per destination tensor. Callers must only use this when
    ``confidential_compute_enabled()`` is true; outside Confidential
    Computing the cross-stream handoff would race (see module docstring).
    """

    def __init__(self, gpu_base: torch.Tensor):
        self._gpu = gpu_base
        # Staging is mutable runtime state, not inference data.
        with torch.inference_mode(False):
            self._stage = [torch.empty_like(gpu_base) for _ in range(2)]
        self._idx = 0

    def copy_(self, cpu_base: torch.Tensor, n: int | None = None) -> torch.Tensor:
        """Copy ``cpu_base[:n]`` into the GPU tensor via the staged path."""
        gpu_dst = self._gpu if n is None else self._gpu[:n]
        cpu_src = cpu_base if n is None else cpu_base[:n]
        stage = self._stage[self._idx]
        self._idx ^= 1
        stage_dst = stage if n is None else stage[:n]
        # The H2D on the prep stream is host-synchronous under Confidential
        # Computing, so stage_dst is populated on return; the D2D on the
        # current (compute) stream is asynchronous and ordered after the
        # forward's read of the reused buffer.
        with torch.cuda.stream(_staged_h2d_stream(self._gpu.device)):
            stage_dst.copy_(cpu_src, non_blocking=True)
        return gpu_dst.copy_(stage_dst, non_blocking=True)


class AsyncD2HCopyWorker:
    """Runs the per-step result D2H readback on a dedicated daemon thread.

    ``submit(copy_fn)`` records a CUDA event on the caller's current stream
    (after the producing forward+sample) and returns immediately with a
    ``threading.Event``. This worker ``cudaEventSynchronize``-s on the CUDA
    event (event-sync, NOT stream-wait, so the blocking copy does not stall
    the scheduler's CUDA API calls), runs the copy on the dedicated copy
    stream, blocks until it completes, and sets the returned event
    (worker-done => copy-done). See the module docstring for the Confidential
    Computing rationale.
    """

    def __init__(self, device_module, copy_stream, device=None):
        self.device_module = device_module
        self.copy_stream = copy_stream
        self._device = device
        self._queue: queue.Queue = queue.Queue()
        self._thread = threading.Thread(
            target=self._loop, name="vllm-d2h-copy-worker", daemon=True
        )
        self._thread.start()

    def submit(self, copy_fn: Callable[[], None]) -> threading.Event:
        """Enqueue a readback ordered after the work currently queued on the
        caller's current stream. ``copy_fn`` performs the actual
        ``.to("cpu", ...)`` copies; the returned event is set once they have
        completed."""
        src_ready = torch.Event()
        src_ready.record()
        done = threading.Event()
        self._queue.put((src_ready, copy_fn, done))
        return done

    def _loop(self):
        # A new thread does not inherit the main thread's CUDA context; set the
        # device so CUDA runtime calls do not implicitly create one on device 0.
        if self._device is not None:
            from vllm.platforms import current_platform

            current_platform.set_device(self._device)
        while True:
            item = self._queue.get()
            if item is None:
                return
            src_ready, copy_fn, done = item
            try:
                # Wait until the producing forward+sample has materialized the
                # source tensors, then issue the copies on a dedicated stream
                # owned by this thread (the current stream is thread-local, so
                # this does not affect the scheduler thread's stream).
                src_ready.synchronize()
                with self.device_module.stream(self.copy_stream):
                    copy_fn()
                self.copy_stream.synchronize()
            except Exception:
                logger.exception("AsyncD2HCopyWorker readback failed")
            finally:
                done.set()

    def shutdown(self, timeout: float = 2.0):
        """Signal the worker to stop and join it (best-effort, bounded wait)."""
        if not self._thread.is_alive():
            return
        self._queue.put(None)
        self._thread.join(timeout=timeout)
