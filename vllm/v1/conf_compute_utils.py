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
  reads valid data without a cross-stream event. For the same reason the
  prep stream is fully drained whenever a copy call returns, so a single
  prep stream per device is always idle at issue time. Callers double-buffer
  the staging tensor so the next step's H2D cannot overwrite a staging
  buffer whose D2D has not yet drained.

* Deferred D2H readback: the per-step result copy is not issued at step time
  (host-synchronous, it would stall the scheduler thread on the in-flight
  forward) but from ``AsyncGPUModelRunnerOutput.get_output()``, which runs
  on the executor's async-output thread and blocks until the copy completes
  anyway. Same ``async_output_copy_stream`` + ready-event paradigm as the
  non-Confidential-Computing path, just issued on the consuming thread
  (mirrors TensorRT-LLM PR #8463).
"""

from contextlib import AbstractContextManager, nullcontext
from functools import cache

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)


@cache
def confidential_compute_enabled() -> bool:
    """Whether NVIDIA Confidential Computing is active on this platform.

    The single gate for every copy path in this module.
    """
    from vllm.platforms import current_platform

    return current_platform.is_confidential_compute()


@cache
def _prep_stream_for_index(device_index: int) -> torch.cuda.Stream:
    logger.info_once(
        "Using staged H2D input copies under Confidential Computing "
        "(H2D on a dedicated prep stream + D2D on the compute stream) to "
        "avoid blocking the scheduler on the in-flight forward."
    )
    return torch.cuda.Stream(device=torch.device(f"cuda:{device_index}"))


def _prep_stream(device: torch.device) -> torch.cuda.Stream:
    """Return the per-device prep stream.

    A single stream suffices: every H2D on it is host-synchronous under
    Confidential Computing, so the stream is drained by the time each copy
    call returns.
    """
    idx = device.index if device.index is not None else torch.cuda.current_device()
    return _prep_stream_for_index(idx)


def prep_stream_ctx(device: torch.device) -> AbstractContextManager:
    """Make the idle prep stream current under Confidential Computing; no-op
    otherwise.

    Use around an H2D whose result is consumed immediately on the issuing
    thread: under Confidential Computing the copy is host-synchronous, so it
    is complete on return regardless of stream, and the prep stream only
    waits for its own transfer instead of the forward queued on the compute
    stream.
    """
    if not confidential_compute_enabled():
        return nullcontext()
    return torch.cuda.stream(_prep_stream(device))


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
        with torch.cuda.stream(_prep_stream(self._gpu.device)):
            stage_dst.copy_(cpu_src, non_blocking=True)
        return gpu_dst.copy_(stage_dst, non_blocking=True)
