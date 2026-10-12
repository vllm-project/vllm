# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""What a model declares to run MonoKernels: its ``MonoSpec``, and the types
the spec's callbacks see."""

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any, Protocol

import torch

from vllm.config import CUDAGraphMode

if TYPE_CHECKING:
    from vllm.config import VllmConfig


class Feature(str, Enum):
    """What a deployment can turn on. A feature in use that a spec's
    ``supports`` does not list is refused."""

    PIPELINE_PARALLEL = "pipeline_parallel"
    DATA_PARALLEL = "data_parallel"
    EXPERT_PARALLEL = "expert_parallel"
    EPLB = "eplb"
    DECODE_CONTEXT_PARALLEL = "decode_context_parallel"
    PREFILL_CONTEXT_PARALLEL = "prefill_context_parallel"
    LORA = "lora"
    KV_CONNECTOR = "kv_connector"
    ROUTED_EXPERT_CAPTURE = "routed_expert_capture"
    SPECULATIVE_DECODING = "speculative_decoding"
    SLEEP_MODE = "sleep_mode"
    CPU_WEIGHT_OFFLOAD = "cpu_weight_offload"
    WEIGHT_UPDATES = "weight_updates"


class MonoUnsupported(Exception):
    """The configuration or the loaded model is outside what a spec serves."""


@dataclass(frozen=True)
class CacheReq:
    """A cache the kernels read: its KV cache group, dtype and layout."""

    group: str
    dtype: str
    layout: str


@dataclass
class MonoContext:
    vllm_config: "VllmConfig"
    rank: int  # TP rank
    device: torch.device
    state: Any = None  # the spec's per-rank launch state (``pack``'s)


@dataclass
class MonoStep:
    num_tokens: int
    width: int  # the padded rows the launch serves
    positions: torch.Tensor
    attn_metadata: Any  # by layer name
    runtime_mode: CUDAGraphMode


class LayerKernel(Protocol):
    # attention layers whose caches ``MonoSpec.bind`` checks
    cache_layers: tuple[str, ...]

    def launch(self, step: MonoStep, *args, **kwargs) -> Any:
        """Takes and returns what the vLLM layer does."""
        ...

    def health(self) -> torch.Tensor:
        """The error words (``abi``)."""
        ...

    def warmup_keys(self) -> list[Any]:
        """Hashable build keys of this width's launches, compiled before CUDA
        graph capture by vLLM's JIT warm-up."""
        ...

    def compile(self, key: Any) -> None:
        """Compiles one of ``warmup_keys`` without launching it."""
        ...


@dataclass(frozen=True)
class MonoSpec:
    name: str  # log and metric name
    layers: Callable[[Any], list[int]]  # decoder layers that get a kernel
    # rows a launch serves, by config (a spec-verify kernel's are whole
    # requests of num_speculative_tokens + 1 rows)
    widths: Callable[["VllmConfig"], tuple[int, ...]]
    supports: frozenset[Feature]
    caches: Mapping[str, CacheReq]
    weight_source: str  # "views", or "convert" once after load
    options: Mapping[str, Any]  # declared opt-ins, off by default
    refuse_config: Callable[["VllmConfig"], list[str]]  # before load
    refuse_model: Callable[[Any], list[str]]  # after load
    pack: Callable[[Any, MonoContext], Any]  # weights / launch state, once
    bind: Callable[[MonoContext, Mapping[str, Any]], list[str]]  # every re-bind
    eligible: Callable[[MonoContext, MonoStep], str | None]  # per-step no-go
    build: Callable[[MonoContext, Any, Any, int], LayerKernel]  # layer, width
