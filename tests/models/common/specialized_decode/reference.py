# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The reference kernel: a ``DecodeImpl`` over the layers it replaces."""

from dataclasses import replace
from types import SimpleNamespace
from typing import Any

import torch

from vllm.config.compilation import CUDAGraphMode
from vllm.forward_context import get_forward_context
from vllm.models.common.specialized_decode.spec import DecodeSpec, Step

CACHE_DTYPE = torch.bfloat16
CACHE_NDIM = 4


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
    """A model shaped like the layer loop a kernel hooks into."""

    def __init__(self, layers: int = 4, hidden: int = 8):
        super().__init__()
        self.config = SimpleNamespace(num_hidden_layers=layers, hidden_size=hidden)
        self.layers = torch.nn.ModuleList(
            [ReferenceLayer(i, hidden, dense=i == 0) for i in range(layers)]
        )
        self.start_layer = 0
        self.end_layer = layers

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors=None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        hidden = inputs_embeds
        for layer in self.layers:
            hidden = layer(positions, hidden)
        return hidden


class ReferenceKernel:
    """The model's decode in one launch: the work of every layer it replaces,
    off views of the model's weights, with the dense layers left to vLLM."""

    def __init__(self, vllm_config, model: ReferenceModel, widths: tuple[int, ...]):
        self.model = model
        self.widths = widths
        served = [
            model.layers[i]
            for i in range(model.start_layer, model.end_layer)
            if not model.layers[i].dense
        ]
        self.weights = {layer.index: layer.weight.data for layer in served}
        self.cache_layers = tuple(layer.attn_name for layer in served)
        self.launched: list[int] = []
        self.bound: dict[str, Any] = {}
        self.compiled: set[Any] = set()

    def run(self, step: Step, input_ids, inputs_embeds, **kwargs):
        self.launched.append(step.kernel_width)
        hidden = inputs_embeds
        for layer in self.model.layers:
            weight = self.weights.get(layer.index)
            if weight is None:
                hidden = layer(step.positions, hidden)
            else:
                hidden = hidden + weight
        return hidden

    def bind_caches(self, caches) -> None:
        self.bound = dict(caches)

    def warmup_keys(self) -> list[Any]:
        return [(i, w) for i in self.weights for w in self.widths]

    def compile(self, key: Any) -> None:
        self.compiled.add(key)


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


def skip_step(vllm_config, kernel: ReferenceKernel, width: int) -> str | None:
    md = get_forward_context().attn_metadata
    return "prefill" if md.get("prefill") else None


def refuse_caches(vllm_config, kernel: ReferenceKernel, caches) -> list[str]:
    reasons = []
    for name, cache in caches.items():
        if cache.dtype is not CACHE_DTYPE:
            reasons.append(f"{name}: dtype {cache.dtype}, needs {CACHE_DTYPE}")
        if cache.ndim != CACHE_NDIM:
            reasons.append(f"{name}: {cache.ndim}d, needs {CACHE_NDIM}d")
    return reasons


def build(vllm_config, model, widths: tuple[int, ...]) -> ReferenceKernel:
    return ReferenceKernel(vllm_config, model, widths)


REFERENCE = DecodeSpec[ReferenceKernel](
    name="reference",
    supports=frozenset(),
    widths=widths,
    refuse_config=refuse_config,
    refuse_model=refuse_model,
    refuse_caches=refuse_caches,
    skip_step=skip_step,
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

DECODE_SPECS = (REFERENCE_WIDE, REFERENCE)


def make_config(**overrides) -> Any:
    """A config shaped like the parts of ``VllmConfig`` the library reads.

    Args:
        **overrides: Top-level attributes to replace. Nested ones are set on
            the returned object (``config.parallel_config.enable_eplb = True``).

    """
    config = SimpleNamespace(
        kernel_config=SimpleNamespace(specialized_decode="monokernel"),
        model_config=SimpleNamespace(dtype=torch.bfloat16, enable_sleep_mode=False),
        scheduler_config=SimpleNamespace(max_num_seqs=4),
        parallel_config=SimpleNamespace(
            pipeline_parallel_size=1,
            data_parallel_size=1,
            enable_expert_parallel=False,
            enable_eplb=False,
            decode_context_parallel_size=1,
            use_ubatching=False,
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
