# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Torch profiler helpers for CUDA/HIP graph capture tracing.

Graph capture is driven by three subsystems (encoder, decoder and speculator)
that reach the same handful of capture loops by different routes.
``graph_capture_profiler`` binds one profiler around a subsystem's capture,
and ``graph_capture_step`` picks that binding up inside whichever capture loop
ends up running. Capture loops therefore carry no profiler plumbing, and a new
speculator built on an existing capture loop is traced without further changes.
"""

from __future__ import annotations

import contextvars
from collections.abc import Iterator
from contextlib import AbstractContextManager, contextmanager, nullcontext
from dataclasses import dataclass
from typing import Any

import torch

from vllm.config import VllmConfig
from vllm.distributed.parallel_state import get_world_group
from vllm.logger import init_logger

logger = init_logger(__name__)


@dataclass(frozen=True)
class _CaptureBinding:
    profiler: AbstractContextManager[Any]
    label_prefix: str | None


_active_binding: contextvars.ContextVar[_CaptureBinding | None] = (
    contextvars.ContextVar("vllm_graph_capture_binding", default=None)
)


def _make_profiler(
    vllm_config: VllmConfig,
    subsystem: str | None,
) -> AbstractContextManager[Any]:
    profiler_config = vllm_config.profiler_config
    local_rank = get_world_group().local_rank
    if local_rank != 0 or not profiler_config.capture_torch_profiler:
        logger.info_once(
            "Rank %d: Torch profiler disabled for CUDA graph capture", local_rank
        )
        return nullcontext()

    trace_dir = profiler_config.torch_profiler_dir + "/capture_traces"
    worker_name = f"graph_capture_rank_{local_rank}"
    if subsystem:
        worker_name += f"_{subsystem}"
    logger.info_once(
        "Rank %d: Torch profiler enabled for %s CUDA graph capture, "
        "traces will be saved to: %s",
        local_rank,
        subsystem or "decoder",
        trace_dir,
    )
    return torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
        on_trace_ready=torch.profiler.tensorboard_trace_handler(
            trace_dir,
            worker_name=worker_name,
            use_gzip=True,
        ),
    )


@contextmanager
def graph_capture_profiler(
    vllm_config: VllmConfig,
    subsystem: str | None = None,
    label_prefix: str | None = None,
) -> Iterator[None]:
    """Bind a graph-capture profiler for one subsystem's capture.

    Args:
        subsystem: Suffix for the trace file name. ``"encoder"`` writes
            ``graph_capture_rank_0_encoder.<timestamp>.pt.trace.json.gz``;
            the main decoder passes ``None`` and keeps the unsuffixed name.
        label_prefix: Inserted into every annotation recorded inside the
            block, so that e.g. speculator graphs are labelled
            ``capture_32_draft_FULL`` rather than reusing the decoder's
            ``capture_32_FULL``.
    """
    binding = _CaptureBinding(
        profiler=_make_profiler(vllm_config, subsystem),
        label_prefix=label_prefix,
    )
    token = _active_binding.set(binding)
    try:
        yield
    finally:
        _active_binding.reset(token)


@contextmanager
def graph_capture_step(num_tokens: int, mode: str) -> Iterator[None]:
    """Profile the capture of a single graph shape.

    Records a ``capture_{num_tokens}_{mode}`` annotation, or
    ``capture_{num_tokens}_{label_prefix}_{mode}`` when the active binding
    sets a prefix. Does nothing outside a ``graph_capture_profiler`` block, so
    capture loops can call this unconditionally.
    """
    binding = _active_binding.get()
    if binding is None:
        yield
        return

    if binding.label_prefix:
        label = f"capture_{num_tokens}_{binding.label_prefix}_{mode}"
    else:
        label = f"capture_{num_tokens}_{mode}"
    with binding.profiler, torch.profiler.record_function(label):
        yield
