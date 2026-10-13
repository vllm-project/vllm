# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The reference MonoKernel: a ``LayerKernel`` that calls the vLLM layer it
replaces, and the ``MONO_SPECS`` that declare it.

It computes nothing of its own and launches nothing on a device, so it runs on
CPU in the conformance kit (``kit``). What it is for is the contract: a model
porting to MonoKernels can read one complete spec, and the library's own
behaviour can be tested without a GPU or a kernel backend.
"""

from dataclasses import dataclass, replace
from types import SimpleNamespace
from typing import Any

import torch

from vllm.config.compilation import CUDAGraphMode
from vllm.models.common.mono.spec import CacheReq, MonoContext, MonoSpec, MonoStep

CACHE = CacheReq(group="kv", dtype="bfloat16", layout="2nhd")

_LAYOUT_DIMS = {"2nhd": 4}


class ReferenceLayer(torch.nn.Module):
    """A decoder layer the reference kernel can replace: one weight, one
    attention layer name, and ``dense`` for the layers a spec skips."""

    def __init__(self, index: int, hidden: int, dense: bool = False):
        super().__init__()
        self.index = index
        self.dense = dense
        self.attn_name = f"layers.{index}.attn"
        self.weight = torch.nn.Parameter(torch.ones(hidden) * index)

    def forward(self, positions: torch.Tensor, hidden: torch.Tensor):
        return hidden + self.weight


class ReferenceModel(torch.nn.Module):
    """A model shaped like the layer loop a MonoKernel hooks into."""

    def __init__(self, layers: int = 4, hidden: int = 8):
        super().__init__()
        self.config = SimpleNamespace(num_hidden_layers=layers, hidden_size=hidden)
        self.layers = torch.nn.ModuleList(
            [ReferenceLayer(i, hidden, dense=i == 0) for i in range(layers)]
        )
        self.start_layer = 0
        self.end_layer = layers
        self.mono = None


@dataclass
class LaunchState:
    """What ``launch_state`` builds once at attach: views of the model's
    weights, never a copy."""

    weights: dict[int, torch.Tensor]


class ReferenceKernel:
    """One layer's launch at one width: calls the layer it replaces."""

    def __init__(self, ctx: MonoContext, state: LaunchState, layer, width: int):
        self.cache_layers = (layer.attn_name,)
        self.ctx = ctx
        self.state = state
        self.layer = layer
        self.width = width
        self.launched: list[int] = []
        self.compiled: list[Any] = []

    def launch(self, step: MonoStep, positions: torch.Tensor, *args, **kwargs):
        self.launched.append(step.num_tokens)
        return self.layer(positions, *args, **kwargs)

    def warmup_keys(self) -> list[Any]:
        return [(self.layer.index, self.width)]

    def compile(self, key: Any) -> None:
        self.compiled.append(key)


def layers(config) -> list[int]:
    return list(range(config.num_hidden_layers))


def widths(vllm_config) -> tuple[int, ...]:
    most = vllm_config.scheduler_config.max_num_seqs
    return tuple(w for w in (1, 2, 4, 8) if w <= most)


def refuse_config(vllm_config) -> list[str]:
    reasons = []
    if not widths(vllm_config):
        reasons.append("max_num_seqs below the smallest width")
    if vllm_config.model_config.dtype is not torch.bfloat16:
        reasons.append(f"dtype {vllm_config.model_config.dtype}: needs bfloat16")
    return reasons


def refuse_model(model) -> list[str]:
    return [] if isinstance(model, ReferenceModel) else ["not a ReferenceModel"]


def skip_layer(ctx: MonoContext, layer) -> str | None:
    return "dense layer" if layer.dense else None


def skip_step(ctx: MonoContext, state: LaunchState, step: MonoStep) -> str | None:
    return "prefill" if step.attn_metadata.get("prefill") else None


def launch_state(model, ctx: MonoContext) -> LaunchState:
    return LaunchState(
        weights={i: layer.weight.data for i, layer in enumerate(model.layers)}
    )


def reload_weights(model, ctx: MonoContext, state: LaunchState) -> None:
    for i, layer in enumerate(model.layers):
        state.weights[i].copy_(layer.weight.data)


def bind(ctx: MonoContext, caches) -> list[str]:
    reasons = []
    for name, cache in caches.items():
        if str(cache.dtype).removeprefix("torch.") != CACHE.dtype:
            reasons.append(f"{name}: dtype {cache.dtype}, needs {CACHE.dtype}")
        if cache.ndim != _LAYOUT_DIMS[CACHE.layout]:
            reasons.append(f"{name}: {cache.ndim}d, needs layout {CACHE.layout}")
    return reasons


def build(ctx: MonoContext, state: LaunchState, layer, width: int) -> ReferenceKernel:
    return ReferenceKernel(ctx, state, layer, width)


REFERENCE = MonoSpec[LaunchState](
    name="reference",
    supports=frozenset(),
    caches={"attn": CACHE},
    weight_source="views",
    layers=layers,
    widths=widths,
    refuse_config=refuse_config,
    refuse_model=refuse_model,
    skip_layer=skip_layer,
    skip_step=skip_step,
    launch_state=launch_state,
    reload_weights=reload_weights,
    bind=bind,
    build=build,
)

REFERENCE_WIDE = replace(
    REFERENCE,
    name="reference-wide",
    refuse_config=lambda c: (
        refuse_config(c)
        + (["needs max_num_seqs >= 8"] if c.scheduler_config.max_num_seqs < 8 else [])
    ),
)

MONO_SPECS = (REFERENCE_WIDE, REFERENCE)


def make_config(**overrides) -> Any:
    """A config shaped like the parts of ``VllmConfig`` the library reads.

    Args:
        **overrides: Top-level attributes to replace. Nested ones are set on
            the returned object (``config.parallel_config.enable_eplb = True``).

    """
    config = SimpleNamespace(
        kernel_config=SimpleNamespace(monokernel=True),
        model_config=SimpleNamespace(dtype=torch.bfloat16, enable_sleep_mode=False),
        scheduler_config=SimpleNamespace(max_num_seqs=4),
        parallel_config=SimpleNamespace(
            pipeline_parallel_size=1,
            data_parallel_size=1,
            enable_expert_parallel=False,
            enable_eplb=False,
            decode_context_parallel_size=1,
            prefill_context_parallel_size=1,
        ),
        offload_config=SimpleNamespace(
            uva=SimpleNamespace(cpu_offload_gb=0),
            prefetch=SimpleNamespace(offload_group_size=0),
        ),
        aux_output_config=SimpleNamespace(enable_return_routed_experts=False),
        compilation_config=SimpleNamespace(
            cudagraph_mode=CUDAGraphMode.FULL,
            cudagraph_capture_sizes=[1, 2, 3, 4, 16],
            static_forward_context={},
        ),
        lora_config=None,
        kv_transfer_config=None,
        speculative_config=None,
        weight_transfer_config=None,
        use_v2_model_runner=True,
    )
    for key, value in overrides.items():
        setattr(config, key, value)
    return config


def make_model(config, layers: int = 4) -> ReferenceModel:
    """A loaded reference model, with its KV caches already in the config's
    static forward context."""
    model = ReferenceModel(layers=layers)
    set_caches(config, model)
    return model


def set_caches(config, model, **kwargs) -> dict[str, torch.Tensor]:
    """Puts one cache a layer in the static forward context, and returns them."""
    caches = {
        layer.attn_name: make_cache(**kwargs)
        for layer in model.layers
        if not layer.dense
    }
    config.compilation_config.static_forward_context = {
        name: SimpleNamespace(kv_cache=cache) for name, cache in caches.items()
    }
    return caches


def make_cache(dtype=torch.bfloat16, ndim: int = 4) -> torch.Tensor:
    return torch.zeros((2,) + (1,) * (ndim - 1), dtype=dtype)
