# ABOUTME: EPS forward runtime context management.
# ABOUTME: Shares request/layer metadata with KV write hook logic.

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Dict, List

import torch

from vllm.v1.eps.config import EpsRuntimeConfig
from vllm.v1.eps.state import EpsJLState
from vllm.v1.eps.telemetry import EpsStepCounters


@dataclass
class EpsRequestRuntime:
    request_id: str
    state: EpsJLState | None
    block_mapping: Dict[int, int]


@dataclass
class EpsGroupRuntime:
    block_size: int
    request_runtimes: List[EpsRequestRuntime]


@dataclass
class EpsLayerInfo:
    group_id: int
    layer_index: int
    layer_name: str


@dataclass
class EpsForwardContext:
    enabled: bool
    cfg: EpsRuntimeConfig
    layer_map: Dict[str, EpsLayerInfo]
    group_runtimes: List[EpsGroupRuntime]
    token_request_indices: torch.Tensor
    cudagraph_capture: bool = False
    device_counters: EpsStepCounters | None = None
    device_union_groups: set[int] | None = None


_CTX: ContextVar[EpsForwardContext | None] = ContextVar("eps_ctx", default=None)


def get_eps_context() -> EpsForwardContext | None:
    return _CTX.get()


@contextmanager
def eps_context(ctx: EpsForwardContext | None):
    token = _CTX.set(ctx)
    try:
        yield
    finally:
        _CTX.reset(token)


__all__ = [
    "EpsForwardContext",
    "EpsGroupRuntime",
    "EpsLayerInfo",
    "EpsRequestRuntime",
    "eps_context",
    "get_eps_context",
]
