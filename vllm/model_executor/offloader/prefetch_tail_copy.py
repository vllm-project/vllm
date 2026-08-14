# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Background tail-copy scheduler for wraparound prefetches.

When a model forward starts a prefetch that wraps around to the next forward
(``target_unit_idx <= source_unit_idx``), the H2D copy is low priority work
that runs on the shared copy stream concurrently with regular prefetches.  We
chunk such copies and pace them through a daemon thread so they cannot stall
the next forward's critical-path prefetches.
"""

import threading
from collections import deque
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import torch

PREFETCH_H2D_CHUNK_BYTES = 128 * 1024 * 1024
# Backward-compatible name for the original tail-prefetch-only use.
TAIL_PREFETCH_H2D_CHUNK_BYTES = PREFETCH_H2D_CHUNK_BYTES
_TAIL_COPY_READY_POLL_S = 0.0005
# Backoff used when the main thread is currently capturing a CUDA graph and
# any event query/synchronize on streams entangled with that capture would
# raise ``cudaErrorStreamCaptureUnsupported``. Pause the tail-copy worker
# briefly and retry once capture finishes.
_TAIL_COPY_CAPTURE_BACKOFF_S = 0.005


def _is_stream_capture_unsupported_error(exc: BaseException) -> bool:
    """Return True when ``exc`` is the CUDA stream-capture-unsupported error.

    During cuda graph capture, ``cudaEventQuery``/``cudaEventSynchronize`` on
    streams that are entangled with the capturing stream are rejected with
    ``cudaErrorStreamCaptureUnsupported`` ("operation not permitted when
    stream is capturing"). This helper recognises that case from the message
    string so the caller can back off instead of aborting the worker.
    """
    if not isinstance(exc, RuntimeError) and not isinstance(
        exc, getattr(torch, "AcceleratorError", RuntimeError)
    ):
        return False
    msg = str(exc)
    return (
        "cudaErrorStreamCaptureUnsupported" in msg
        or "operation not permitted when stream is capturing" in msg
    )


TensorCopyItem = tuple[torch.Tensor, torch.Tensor, int]


def is_wraparound_prefetch(
    source_unit_idx: int,
    target_unit_idx: int | None,
) -> bool:
    """Whether prefetching ``target_unit_idx`` after ``source_unit_idx`` wraps."""
    return target_unit_idx is not None and target_unit_idx <= source_unit_idx


def iter_chunked_tensor_views(
    dst: torch.Tensor,
    src: torch.Tensor,
    num_bytes: int,
    chunk_bytes: int,
) -> Generator[TensorCopyItem, None, None]:
    """Yield per-chunk ``(dst, src, num_bytes)`` views or fall back to whole copy."""
    if (
        chunk_bytes <= 0
        or num_bytes <= chunk_bytes
        or dst.numel() != src.numel()
        or not dst.is_contiguous()
        or not src.is_contiguous()
    ):
        yield dst, src, num_bytes
        return

    element_size = dst.element_size()
    chunk_elems = max(1, chunk_bytes // element_size)
    dst_flat = dst.view(-1)
    src_flat = src.view(-1)
    for start in range(0, dst_flat.numel(), chunk_elems):
        end = min(start + chunk_elems, dst_flat.numel())
        yield dst_flat[start:end], src_flat[start:end], (end - start) * element_size


@dataclass
class TailCopyJob:
    module_offloader: Any
    fork_event: Any
    copy_items: tuple[TensorCopyItem, ...]
    next_index: int = 0


@dataclass
class CollectiveWindow:
    """CUDA-observed lifetime of one collective on its execution stream."""

    start_event: Any
    done_event: Any
    started: bool = False


def pop_next_ready_tail_copy_job(jobs: deque[Any]) -> Any | None:
    """Pop the first job whose compute-stream fork event is ready.

    If a query raises ``cudaErrorStreamCaptureUnsupported`` because the main
    thread is currently capturing a CUDA graph, treat the job as "not ready
    yet" and leave it in the queue: the worker will retry after the capture
    completes. Any other error is left to the caller to handle.
    """
    for index, job in enumerate(jobs):
        try:
            ready = job.fork_event.query()
        except Exception as exc:  # noqa: BLE001 - re-raise non-capture errors
            if _is_stream_capture_unsupported_error(exc):
                return None
            raise
        if ready:
            del jobs[index]
            return job
    return None


def requeue_active_tail_copy_job(jobs: deque[Any], job: Any) -> None:
    """Keep copying the active tail job before later ready jobs."""
    jobs.appendleft(job)


class TailCopyScheduler:
    """Ready-aware paced tail H2D copy scheduler."""

    def __init__(
        self,
        *,
        device: int | torch.device,
        copy_stream: torch.cuda.Stream,
    ) -> None:
        self.device = device
        self.copy_stream = copy_stream
        self._condition = threading.Condition()
        self._jobs: deque[TailCopyJob] = deque()
        self._collective_windows_by_stream: dict[int, deque[CollectiveWindow]] = {}
        self._thread = threading.Thread(
            target=self._pump,
            name="vllm-prefetch-tail-copy",
            daemon=True,
        )
        self._thread.start()

    def submit(self, job: TailCopyJob) -> None:
        with self._condition:
            self._jobs.append(job)
            self._condition.notify()

    def register_collective_window(
        self,
        stream: torch.cuda.Stream,
        start_event: Any,
        done_event: Any,
    ) -> None:
        """Register a collective's GPU lifetime without blocking its launch."""
        stream_id = int(stream.cuda_stream)
        with self._condition:
            windows = self._collective_windows_by_stream.setdefault(stream_id, deque())
            windows.append(CollectiveWindow(start_event, done_event))
            self._condition.notify_all()

    @contextmanager
    def gate_for_collective(self) -> Generator[None, None, None]:
        """Gate new chunks only during the collective's actual GPU lifetime.

        The start and done events are ordered around the collective on its
        execution stream. The background worker ignores future host-enqueued
        collectives, pauses at a chunk boundary once the GPU reaches the start
        event, and resumes after the done event. A chunk already in flight is
        deliberately allowed to finish.
        """
        stream = torch.cuda.current_stream()
        start_event = torch.cuda.Event()
        start_event.record(stream)
        try:
            yield
        except Exception:
            # The collective was not successfully enqueued. Do not leave an
            # incomplete window that could stop the copy worker indefinitely.
            raise
        else:
            done_event = torch.cuda.Event()
            done_event.record(stream)
            self.register_collective_window(stream, start_event, done_event)

    def _collective_is_active_locked(self) -> bool:
        """Whether any registered collective is active on the GPU now."""
        active = False
        empty_streams: list[int] = []
        for stream_id, windows in self._collective_windows_by_stream.items():
            while windows:
                window = windows[0]
                try:
                    if not window.started:
                        if not window.start_event.query():
                            # Events on one stream are ordered, so no later
                            # window on this stream can have started yet.
                            break
                        window.started = True
                    if not window.done_event.query():
                        active = True
                        break
                except Exception as exc:
                    if _is_stream_capture_unsupported_error(exc):
                        # Avoid submitting more work while event state cannot
                        # safely be observed during an unrelated graph capture.
                        active = True
                        break
                    raise
                windows.popleft()
            if not windows:
                empty_streams.append(stream_id)
        for stream_id in empty_streams:
            del self._collective_windows_by_stream[stream_id]
        return active

    def _pump(self) -> None:
        torch.accelerator.set_device_index(self.device)
        while True:
            with self._condition:
                while True:
                    if self._collective_is_active_locked():
                        self._condition.wait(timeout=_TAIL_COPY_READY_POLL_S)
                        continue
                    if not self._jobs:
                        self._condition.wait()
                        continue
                    try:
                        job = pop_next_ready_tail_copy_job(self._jobs)
                    except Exception:
                        # ``pop_next_ready_tail_copy_job`` already handles
                        # capture-in-progress errors. Anything else is fatal.
                        raise
                    if job is None:
                        self._condition.wait(timeout=_TAIL_COPY_READY_POLL_S)
                        continue
                    break

            status = self._copy_one_chunk(job)

            with self._condition:
                if status == "capture_in_progress":
                    # Main thread is capturing a CUDA graph. Put the job
                    # back at the head and back off briefly; we will resume
                    # the chunked copy once capture finishes.
                    requeue_active_tail_copy_job(self._jobs, job)
                    self._condition.wait(timeout=_TAIL_COPY_CAPTURE_BACKOFF_S)
                    continue
                if status == "failed":
                    pass
                elif job.next_index < len(job.copy_items):
                    requeue_active_tail_copy_job(self._jobs, job)
                else:
                    self._complete_job(job)
                self._condition.notify_all()

    def _copy_one_chunk(self, job: TailCopyJob) -> str:
        """Copy one chunk for ``job``.

        Returns one of:
          * ``"copied"``               - chunk successfully copied;
          * ``"capture_in_progress"``  - blocked because the main thread is
            currently capturing a CUDA graph (transient, retry later);
          * ``"failed"``               - any other error; the job has been
            marked failed on its module offloader.
        """
        module_offloader = job.module_offloader
        try:
            dst_chunk, src_chunk, _ = job.copy_items[job.next_index]
            with torch.cuda.stream(self.copy_stream):
                dst_chunk.copy_(src_chunk, non_blocking=True)
                chunk_done_event = torch.cuda.Event()
                chunk_done_event.record(self.copy_stream)
            chunk_done_event.synchronize()
            job.next_index += 1
            return "copied"
        except Exception as exc:
            if _is_stream_capture_unsupported_error(exc):
                # ``next_index`` was not advanced; this same chunk will be
                # retried on the next iteration after capture completes.
                return "capture_in_progress"
            module_offloader._copy_thread_error = exc
            module_offloader._copy_done_event_recorded.set()
            job.next_index = len(job.copy_items)
            return "failed"

    def _complete_job(self, job: TailCopyJob) -> None:
        module_offloader = job.module_offloader
        try:
            for offloader in module_offloader._param_offloaders.values():
                offloader.mark_cpu_master_synced()
            with torch.cuda.stream(self.copy_stream):
                module_offloader._copy_done_event.record(self.copy_stream)
        except Exception as exc:
            module_offloader._copy_thread_error = exc
        finally:
            module_offloader._copy_done_event_recorded.set()
