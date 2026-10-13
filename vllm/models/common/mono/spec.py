# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""What a model declares to run MonoKernels: ``MonoSpec`` and its callbacks."""

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any, Generic, Protocol, TypeVar

import torch

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


@dataclass(frozen=True)
class MonoContext:
    """One rank's view of the deployment, built once at attach.

    Attributes:
        vllm_config: The engine's config.
        device: The device the kernels launch on.

    """

    vllm_config: "VllmConfig"
    device: torch.device


class LayerKernel(Protocol):
    """One decoder layer's launch, at one width, on this rank."""

    cache_layers: tuple[str, ...]
    """Attention layers whose caches ``MonoSpec.refuse_caches`` checks."""

    def launch(self, *args, **kwargs) -> Any:
        """Takes and returns what the vLLM layer it replaces does, or None to
        decline this call, which sends the layer down vLLM's path. Decline only
        on what ``skip_step`` cannot answer, such as the layout of the tensors
        the layer was handed."""
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
    """One (model, vendor, ISA)'s declaration: values and callbacks, no
    behaviour of its own.

    A spec is a value, so a second ISA is ``dataclasses.replace`` of the first
    at module scope rather than a subclass, and a model exports its candidates
    as a ``MONO_SPECS`` tuple the library selects from.

    The parameter is the spec's own launch state: ``launch_state`` builds it,
    and ``skip_step`` and ``build`` are handed back the same object. Annotate
    the spec with it (``MonoSpec[MyLaunchState](...)``) and those three
    callbacks are checked against each other.

    Gating runs in tiers, and the prefix is the consequence. ``refuse_*`` says
    the deployment asked for something it cannot have, so it may take the
    engine down; ``skip_*`` says only that this layer, or this step, takes
    vLLM's path, and never raises. If the same model with the same flags could
    legitimately have another layer, or a later step, that passes the check, it
    is a ``skip_*``. Every tier reports all of its reasons, never the first.

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
            answer, re-run whenever those caches move.
        skip_layer: Why one layer takes vLLM's path, None when it takes the
            kernel's. Takes only the layer, so what it reads is what every rank
            of the same model sees alike.
        skip_step: Why one step takes vLLM's path, None when it takes the
            kernel's. Takes the step's width; anything else it needs it reads
            from the forward context, which every rank sees alike.
        launch_state: The weights and launch state, built once at attach. The
            only place a weight copy is ever made. Opaque to the library, which
            only hands it back to the spec's own callbacks.
        build: One ``LayerKernel`` for (context, state, layer, width).

    """

    name: str
    supports: frozenset[Feature]
    widths: Callable[["VllmConfig"], tuple[int, ...]]
    refuse_config: Callable[["VllmConfig"], list[str]]
    refuse_model: Callable[[Any], list[str]]
    refuse_caches: Callable[[MonoContext, Mapping[str, Any]], list[str]]
    skip_layer: Callable[[Any], str | None]
    skip_step: Callable[[MonoContext, LaunchStateT, int], str | None]
    launch_state: Callable[[Any, MonoContext], LaunchStateT]
    build: Callable[[MonoContext, LaunchStateT, Any, int], LayerKernel]
