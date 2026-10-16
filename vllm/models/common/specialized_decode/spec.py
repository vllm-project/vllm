# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""What a model declares to run specialized decode: ``DecodeSpec`` and its
callbacks."""

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any, Generic, Protocol, TypeVar

if TYPE_CHECKING:
    import torch

    from vllm.config import VllmConfig


class Feature(str, Enum):
    """What a deployment can turn on. A feature in use that a spec's
    ``supports`` does not list is refused."""

    PIPELINE_PARALLEL = "pipeline_parallel"
    DATA_PARALLEL = "data_parallel"
    EXPERT_PARALLEL = "expert_parallel"
    EPLB = "eplb"
    DECODE_CONTEXT_PARALLEL = "decode_context_parallel"
    UBATCHING = "ubatching"
    PREFILL_CONTEXT_PARALLEL = "prefill_context_parallel"
    LORA = "lora"
    KV_CONNECTOR = "kv_connector"
    ROUTED_EXPERT_CAPTURE = "routed_expert_capture"
    SPECULATIVE_DECODING = "speculative_decoding"
    SLEEP_MODE = "sleep_mode"
    CPU_WEIGHT_OFFLOAD = "cpu_weight_offload"
    WEIGHT_UPDATES = "weight_updates"


class DecodeUnsupported(Exception):
    """The configuration or the loaded model is outside what a spec serves."""


@dataclass(frozen=True)
class Step:
    """One decode step the kernel takes, as ``SpecializedDecode.plan`` decided it."""

    width: int
    """The step's rows, one of the widths the spec serves."""

    positions: "torch.Tensor"
    """The step's positions, as the model's forward was handed them."""


class DecodeImpl(Protocol):
    """A spec's half of the contract: one model's decode on this rank, for
    every width the spec serves.

    The implementation owns its own layer loop, so the library has no opinion
    about layers: it may replace one layer, two, or the whole backbone.
    """

    cache_layers: tuple[str, ...]
    """Attention layers whose caches ``DecodeSpec.refuse_caches`` checks."""

    def run(self, step: Step, input_ids, inputs_embeds, **kwargs) -> Any:
        """Takes and returns what the model's own forward does.

        A launch replaces a *whole* decoder layer: it returns what that layer's
        forward returns, or declines, and the kernel runs the layer on vLLM's
        path instead. Covering part of a layer is the model's own problem, and
        is discouraged: a new kernel should cover a whole layer or fall back.
        """
        ...

    def bind_caches(self, caches: Mapping[str, Any]) -> None:
        """Takes the ``cache_layers`` caches the runner has bound, once
        ``DecodeSpec.refuse_caches`` has passed them.

        For a kernel whose launches close over the cache layout (a compile
        against its strides, a cached view), this is where that happens. Called
        again whenever the runner rebinds, so it must replace what the last call
        built, not add to it: a normal startup binds the throwaway cache memory
        profiling allocates before the real one.
        """
        ...

    def warmup_keys(self) -> list[Any]:
        """Hashable build keys of this kernel's launches, compiled before CUDA
        graph capture by vLLM's JIT warm-up."""
        ...

    def compile(self, key: Any) -> None:
        """Compiles one of ``warmup_keys`` without launching it."""
        ...


KernelT = TypeVar("KernelT", bound=DecodeImpl)
"""What one spec's ``build`` returns, and what its ``refuse_caches`` and
``skip_step`` are handed: a spec declared ``DecodeSpec[MyKernel]`` ties the
three together."""


@dataclass(frozen=True)
class DecodeSpec(Generic[KernelT]):
    """Definition of when specialized decode can run, and how to build it.

    Attributes:
        name: Log and metric name.
        supports: Features this spec serves. Default deny: a feature in use
            that is absent from it refuses the deployment.
        widths: The rows a launch serves, from the engine's config. These
            become the CUDA graph capture sizes.
        refuse_config: Deployment refusals the engine's config answers, before
            the weights are loaded.
        refuse_model: Deployment refusals only the loaded model answers.
        refuse_caches: Deployment refusals the KV caches the kernel will read
            answer, re-run whenever the runner binds caches.
        skip_step: Why one step takes vLLM's path, None when it takes the
            kernel's. Takes the step's width; anything else it needs it reads
            from the forward context, which every rank sees alike.
        build: This rank's kernel, built once off the loaded model (whose
            parameters carry the device it launches on) and every width the
            spec serves. The only place a weight copy is ever made.

    """

    name: str
    supports: frozenset[Feature]
    widths: Callable[["VllmConfig"], tuple[int, ...]]
    refuse_config: Callable[["VllmConfig"], list[str]]
    refuse_model: Callable[[Any], list[str]]
    refuse_caches: Callable[["VllmConfig", KernelT, Mapping[str, Any]], list[str]]
    skip_step: Callable[["VllmConfig", KernelT, int], str | None]
    build: Callable[["VllmConfig", Any, tuple[int, ...]], KernelT]
