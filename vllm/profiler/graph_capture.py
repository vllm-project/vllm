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
from vllm.profiler.wrapper import (
    TorchProfilerWrapper,
    default_torch_profiler_activities,
    graph_capture_profiler_config,
)

logger = init_logger(__name__)


@dataclass(frozen=True)
class _CaptureBinding:
    profiler: AbstractContextManager[Any]
    label_prefix: str | None


_active_binding: contextvars.ContextVar[_CaptureBinding | None] = (
    contextvars.ContextVar("vllm_graph_capture_binding", default=None)
)
_skip_capture_tracing: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "vllm_skip_graph_capture_tracing", default=False
)


@contextmanager
def skip_graph_capture_tracing() -> Iterator[None]:
    """Disable capture tracing for the throwaway memory-estimate capture."""
    token = _skip_capture_tracing.set(True)
    try:
        yield
    finally:
        _skip_capture_tracing.reset(token)


def make_graph_capture_profiler(
    vllm_config: VllmConfig,
    subsystem: str | None = None,
) -> AbstractContextManager[Any]:
    """Build the one-shot capture profiler, or a nullcontext when disabled.

    The single construction site for capture profilers, so callers that still
    pass the profiler down their capture loops by hand inherit the same config
    overrides and device activities as ``graph_capture_profiler``.
    """
    profiler_config = vllm_config.profiler_config
    local_rank = get_world_group().local_rank
    if local_rank != 0 or not profiler_config.capture_torch_profiler:
        logger.info_once(
            "Rank %d: Torch profiler disabled for GPU graph capture", local_rank
        )
        return nullcontext()

    capture_config = graph_capture_profiler_config(profiler_config)
    worker_name = f"graph_capture_rank_{local_rank}"
    if subsystem:
        worker_name += f"_{subsystem}"
    wrapper = TorchProfilerWrapper(
        capture_config,
        worker_name=worker_name,
        local_rank=local_rank,
        activities=default_torch_profiler_activities(
            vllm_config.device_config.device_type
        ),
    )
    return wrapper.profiler


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
    if _skip_capture_tracing.get() or _active_binding.get() is not None:
        yield
        return

    binding = _CaptureBinding(
        profiler=make_graph_capture_profiler(vllm_config, subsystem),
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
