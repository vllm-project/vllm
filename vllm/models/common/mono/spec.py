# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""What a model declares to run MonoKernels: ``MonoSpec`` and its callbacks."""

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any, Generic, Protocol, TypeVar

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


LaunchStateT = TypeVar("LaunchStateT")
"""Whatever one spec's ``launch_state`` builds. The library never looks inside
it, so a spec chooses the type and the callbacks that receive it agree."""


class MonoUnsupported(Exception):
    """The configuration or the loaded model is outside what a spec serves."""


class LayerKernel(Protocol):
    """One decoder layer's launch, at one width, on this rank."""

    cache_layers: tuple[str, ...]
    """Attention layers whose caches ``MonoSpec.refuse_caches`` checks."""

    def launch(self, *args, **kwargs) -> Any:
        """Takes and returns what the vLLM layer it replaces does, or None to
        decline this call.

        The decline should only trigger on what ``skip_step`` cannot answer,
        such as the layout of the tensors the layer was handed."""
        ...

    def warmup_keys(self) -> list[Any]:
        """Hashable build keys of this width's launches, compiled before CUDA
        graph capture by vLLM's JIT warm-up."""
        ...

    def compile(self, key: Any) -> None:
        """Compiles one of ``warmup_keys`` without launching it."""
        ...


@dataclass(frozen=True)
class MonoSpec(Generic[LaunchStateT]):
    """Definition of when a monokernel can run, and how to build it.

    Attributes:
        name: Log and metric name.
        supports: Features this spec serves. Default deny: a feature in use
            that is absent from it refuses the deployment.
        widths: The rows a launch serves, from the engine's config. These
            become the CUDA graph capture sizes.
        refuse_config: Deployment refusals the engine's config answers, before
            the weights are loaded.
        refuse_model: Deployment refusals only the loaded model answers.
        refuse_caches: Deployment refusals the KV caches the kernels will read
            answer, re-run whenever those caches move. Takes the launch state,
            so the caches are checked against what the kernels were built for.
        skip_layer: Why one layer takes vLLM's path, None when it takes the
            kernel's. Takes only the layer, so what it reads is what every rank
            of the same model sees alike.
        skip_step: Why one step takes vLLM's path, None when it takes the
            kernel's. Takes the step's width; anything else it needs it reads
            from the forward context, which every rank sees alike.
        launch_state: The weights and launch state, built once at attach, off
            the loaded model (whose parameters carry the device the kernels
            launch on). The only place a weight copy is ever made.
        build: One layer's kernels, by width, called once a layer with every
            width the spec serves. What a layer prepares once (its weight
            views, anything the layer itself must be told) is prepared here,
            and the widths share it. Must return a kernel for every width.

    """

    name: str
    supports: frozenset[Feature]
    widths: Callable[["VllmConfig"], tuple[int, ...]]
    refuse_config: Callable[["VllmConfig"], list[str]]
    refuse_model: Callable[[Any], list[str]]
    refuse_caches: Callable[["VllmConfig", LaunchStateT, Mapping[str, Any]], list[str]]
    skip_layer: Callable[[Any], str | None]
    skip_step: Callable[["VllmConfig", LaunchStateT, int], str | None]
    launch_state: Callable[[Any, "VllmConfig", tuple[int, ...]], LaunchStateT]
    build: Callable[
        ["VllmConfig", LaunchStateT, Any, tuple[int, ...]], Mapping[int, LayerKernel]
    ]
