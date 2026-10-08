# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Copy helpers for NVIDIA Confidential Computing.

Under bounce-buffer Confidential Computing a host<->device
``cudaMemcpyAsync`` is host-synchronous: the issuing thread blocks until the
copy, and everything already queued on its stream, completes. An H2D issued
on the compute stream therefore blocks the engine for the in-flight forward,
and the per-step D2H readback blocks the thread that issues it.

``staged_h2d`` issues uploads on an idle prep stream instead, so the host only
pays the transfer itself. ``vllm.utils.torch_utils.async_tensor_h2d`` and the
V2 model runner's buffer pools dispatch to it under Confidential Computing;
outside Confidential Computing nothing here is used.
"""

from functools import cache

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)


@cache
def confidential_compute_enabled() -> bool:
    """Whether NVIDIA Confidential Computing is active on this platform."""
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


def prep_stream(device: torch.device) -> torch.cuda.Stream:
    """Return the per-device prep stream.

    A single stream suffices: every pinned H2D on it is host-synchronous under
    Confidential Computing, so the stream is drained by the time each copy
    call returns.
    """
    idx = (
        device.index
        if device.index is not None
        else torch.accelerator.current_device_index()
    )
    return _prep_stream_for_index(idx)


def staged_h2d(
    src: torch.Tensor,
    *,
    device: torch.device | str | None = None,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Upload a host tensor without blocking on the compute stream.

    The H2D is issued on the idle prep stream, so the host only waits for the
    transfer itself. Without ``out`` the fresh device tensor is returned with
    its lifetime tied to the compute stream: it was allocated on the prep
    stream, and ``record_stream`` keeps the caching allocator from recycling it
    before the consuming kernel has run. With ``out`` the upload lands in a
    staging tensor and an asynchronous D2D on the compute stream moves it into
    ``out``, so an in-flight reader of ``out`` is never overwritten.

    ``src`` must already have the destination dtype: a converting copy would
    run its cast kernel on the prep stream, unordered with the compute stream.
    A pageable ``src`` is not host-synchronous under Confidential Computing, so
    the compute stream is made to wait for the prep stream in that case.
    """
    target = out.device if out is not None else torch.device(device)  # type: ignore[arg-type]
    compute_stream = torch.cuda.current_stream(target)
    prep = prep_stream(target)
    with torch.cuda.stream(prep):
        staged = src.to(device=target, non_blocking=True)
    # Under Confidential Computing ``is_pinned()`` reports False even for
    # pin_memory=True allocations, so it cannot be used to decide whether the
    # upload is host-synchronous. Always order the compute stream after the
    # prep stream; the event wait is negligible and correct in both cases.
    compute_stream.wait_stream(prep)
    staged.record_stream(compute_stream)
    if out is None:
        return staged
    return out.copy_(staged, non_blocking=True)
