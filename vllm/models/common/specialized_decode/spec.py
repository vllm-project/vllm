# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Interfaces for when and how to run a specialized decode kernel."""

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import Enum
from typing import (
    TYPE_CHECKING,
    Any,
    Generic,
    Protocol,
    TypeVar,
    runtime_checkable,
)

if TYPE_CHECKING:
    import torch

    from vllm.config import VllmConfig


class Feature(str, Enum):
    """Features a specialized decode kernel may require."""

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
    """The step's rows, as the model's forward was handed them."""

    kernel_width: int
    """The width the kernel launches at, one of the widths the spec serves.

    Equal to ``width``, unless the spec pads a short step up to the next width
    it was built for, in which case it is the larger of the two and the kernel
    owns the padding.
    """

    positions: "torch.Tensor"
    """The step's positions, as the model's forward was handed them."""


class DecodeImpl(Protocol):
    """A specialized decode kernel, substituting a model's forward.

    The implementation owns its own layer loop, so the library has no opinion
    about layers: it may replace one layer, two, or the whole backbone.
    """

    cache_layers: tuple[str, ...]
    """Attention layers whose caches ``DecodeSpec.refuse_caches`` checks."""

    def run(self, step: Step, input_ids, inputs_embeds, **kwargs) -> Any:
        """Takes and returns what the model's own forward does."""
        ...

    def bind_caches(self, caches: Mapping[str, Any]) -> None:
        """Bind the KV caches the kernel will read."""
        ...


@runtime_checkable
class DecodeWarmable(Protocol):
    """A kernel whose launches compile from keys, registered for vLLM's JIT
    warm-up.

    SpecializedDecodes created inside process_weights_after_loading can implement
    this for automatic warm-up of their kernels, so that the first decode step does not
    pay the compilation cost.
    """

    def warmup_keys(self) -> list[Any]:
        """Build keys of this kernel's launches."""
        ...

    def compile(self, key: Any) -> None:
        """Compiles one of ``warmup_keys`` without launching it."""
        ...


KernelT = TypeVar("KernelT", bound=DecodeImpl)


@dataclass(frozen=True)
class DecodeSpec(Generic[KernelT]):
    """Definition of when specialized decode can run, and how to build its
    implementation.

    Attributes:
        name: Log and metric name.
        supports: Features this spec serves. Default deny: a feature in use
            that is absent from it refuses the deployment.
        widths: The rows a launch serves, from the engine's config.
        refuse_config: Conditions under which the specialized decode kernel cannot run,
            based on the config.
        refuse_model: Conditions under which the specialized decode kernel cannot run,
            based on the model.
        refuse_caches: Conditions under which the specialized decode kernel cannot run,
            based on the cache.
        skip_step: Whether to run the specialized decode kernel for this forward call,
            or to fall back to the model's own forward. Takes the step's
            ``kernel_width``.
        build: This rank's kernel.
        pad_steps: Whether a step of fewer rows than a served width runs on the
            next width up, the kernel padding it. Default deny: a step whose
            rows are not themselves a served width falls back.
        set_capture_sizes: Whether to make the served widths the CUDA graph
            capture sizes, so that no step is refused for its width alone. Turn
            it off for a spec that pads, which serves the engine's own capture
            sizes as they are.

    """

    name: str
    supports: frozenset[Feature]
    widths: Callable[["VllmConfig"], tuple[int, ...]]
    refuse_config: Callable[["VllmConfig"], list[str]]
    refuse_model: Callable[[Any], list[str]]
    refuse_caches: Callable[["VllmConfig", KernelT, Mapping[str, Any]], list[str]]
    skip_step: Callable[["VllmConfig", KernelT, int], str | None]
    build: Callable[["VllmConfig", Any, tuple[int, ...]], KernelT]
    pad_steps: bool = False
    set_capture_sizes: bool = True
