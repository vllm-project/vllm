# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The contract between a model's MonoKernel and the shared runtime."""

from __future__ import annotations

import enum
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, Protocol

import torch

if TYPE_CHECKING:
    from torch import nn
    from transformers import PretrainedConfig

    from vllm.config import VllmConfig

    from .peer import PeerBuffers


class Feature(enum.Enum):
    """A deployment feature. One in use that the spec does not support is refused."""

    PIPELINE_PARALLEL = "pipeline parallel"
    DATA_PARALLEL = "data parallel"
    EXPERT_PARALLEL = "expert parallel"
    EPLB = "EPLB"
    DECODE_CONTEXT_PARALLEL = "decode context parallel"
    PREFILL_CONTEXT_PARALLEL = "prefill context parallel"
    LORA = "LoRA"
    KV_CONNECTOR = "KV transfer connector"
    ROUTED_EXPERT_CAPTURE = "routed-expert capture"
    SPECULATIVE_DECODING = "speculative decoding"
    SLEEP_MODE = "sleep mode"
    CPU_WEIGHT_OFFLOAD = "CPU weight offload"
    WEIGHT_UPDATES = "weight updates"
    PROMPT_EMBEDS = "prompt embeds"
    EAGLE3_AUX_LAYERS = "EAGLE3 aux hidden-state layers"


@dataclass(frozen=True)
class CacheReq:
    """One cache a kernel layer reads, found on the vLLM layer at ``path``
    (attributes, then an optional ``[i]`` into a list or tuple); ``when`` limits it
    to the layers it is true for."""

    path: str
    dtypes: tuple[torch.dtype, ...] = ()
    contiguous_last_dim: bool = True
    when: Callable[[Any], bool] | None = None


@dataclass
class MonoStep:
    """One routed decode step, shared by every kernel layer of the forward."""

    tokens: int
    width: int
    positions: torch.Tensor
    attn_metadata: Mapping[str, Any]
    slot_mapping: Mapping[str, torch.Tensor]
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class MonoContext:
    """What a spec's callbacks need beyond the model: config, ranks, peer memory."""

    vllm_config: VllmConfig
    tp_size: int
    tp_rank: int
    cpu_group: Any
    device: torch.device
    peer: PeerBuffers | None = None
    state: dict[str, Any] = field(default_factory=dict)


class LayerKernel(Protocol):
    def launch(self, step: MonoStep, *args: Any) -> Any:
        """Takes and returns exactly what the replaced vLLM layer does."""

    def health(self) -> torch.Tensor:
        """This launch's error words (layout in ``abi``)."""


@dataclass(frozen=True)
class MonoSpec:
    """A model's MonoKernel. Named by import path so config-time code reads it
    without importing the model."""

    name: str
    layers: Callable[[PretrainedConfig], list[int]]
    widths: tuple[int, ...]
    supports: frozenset[Feature]
    caches: Mapping[str, CacheReq]
    weight_source: Literal["views", "convert"]
    grid: int
    stages: tuple[str, ...]
    refuse_config: Callable[[VllmConfig], list[str]]
    refuse_model: Callable[[nn.Module], list[str]]
    pack: Callable[[nn.Module, MonoContext], Any]
    bind: Callable[[MonoContext, Mapping[str, torch.Tensor]], list[str]]
    eligible: Callable[[MonoContext, MonoStep], str | None]
    build: Callable[[MonoContext, Any, nn.Module, int, int], LayerKernel]
    metadata_names: Callable[[Mapping[int, nn.Module]], list[str]]
    options: Mapping[str, Any] = field(default_factory=dict)

    def width_for(self, tokens: int) -> int | None:
        return next((w for w in sorted(self.widths) if w >= tokens), None)
