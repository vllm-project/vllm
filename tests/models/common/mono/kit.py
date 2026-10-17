# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The MonoKernel test kit (RFC #60904): fake configs, a fake spec and kernel,
a golden-layer checker and a fault harness, for any model's MonoSpec tests."""

from collections.abc import Callable
from types import SimpleNamespace
from typing import Any

import torch
import torch.nn.functional as F

from vllm.config import CUDAGraphMode
from vllm.models.common.mono.spec import Feature, MonoSpec, MonoStep


def fake_vllm_config(**over) -> SimpleNamespace:
    """A supported TP8 FULL-graph config with speculative decoding; ``over``
    sets dotted paths, e.g. ``{"parallel_config.data_parallel_size": 2}``."""
    c = SimpleNamespace(
        parallel_config=SimpleNamespace(
            tensor_parallel_size=8,
            pipeline_parallel_size=1,
            data_parallel_size=1,
            enable_expert_parallel=False,
            enable_eplb=False,
            decode_context_parallel_size=1,
            prefill_context_parallel_size=1,
        ),
        lora_config=None,
        kv_transfer_config=None,
        aux_output_config=SimpleNamespace(enable_return_routed_experts=False),
        model_config=SimpleNamespace(enable_sleep_mode=False),
        offload_config=SimpleNamespace(
            uva=SimpleNamespace(cpu_offload_gb=0),
            prefetch=SimpleNamespace(offload_group_size=0),
        ),
        weight_transfer_config=None,
        use_v2_model_runner=True,
        speculative_config=SimpleNamespace(num_speculative_tokens=3),
        kernel_config=SimpleNamespace(monokernel=True),
        compilation_config=SimpleNamespace(
            cudagraph_mode=CUDAGraphMode.FULL_AND_PIECEWISE,
            cudagraph_capture_sizes=list(range(1, 17)),
            static_forward_context={},
        ),
    )
    for path, v in over.items():
        obj = c
        *head, last = path.split(".")
        for h in head:
            obj = getattr(obj, h)
        setattr(obj, last, v)
    return c


class FakeKernel:
    """A layer kernel that adds 1 to the hidden states and reports ``words``."""

    def __init__(self, layer: int, width: int, words: torch.Tensor, caches: bool):
        self.layer, self.width, self.words = layer, width, words
        self.cache_layers = (f"layers.{layer}.attn",) if caches else ()
        self.compiled: list[Any] = []

    def launch(self, step: MonoStep, positions, hidden_states, residual):
        return hidden_states + 1, residual

    def health(self) -> torch.Tensor:
        return self.words

    def warmup_keys(self) -> list[Any]:
        return [("fake", self.width)]

    def compile(self, key: Any) -> None:
        self.compiled.append(key)


def fake_spec(
    *,
    layers: list[int] | None = None,
    widths: tuple[int, ...] = (4, 8),
    supports: frozenset[Feature] = frozenset({Feature.SPECULATIVE_DECODING}),
    refuse_config: Callable[[Any], list[str]] = lambda c: [],
    refuse_model: Callable[[Any], list[str]] = lambda m: [],
    bind: Callable[[Any, Any], list[str]] = lambda ctx, caches: [],
    eligible: Callable[[Any, MonoStep], str | None] = lambda ctx, step: None,
    words: torch.Tensor | None = None,
    caches: bool = False,
) -> MonoSpec:
    """``caches``: every kernel checks its layer's cache, ``layers.<i>.attn``."""
    health = torch.zeros(8, dtype=torch.int32) if words is None else words
    return MonoSpec(
        name="fake",
        layers=lambda config: [1, 2] if layers is None else layers,
        widths=lambda vllm_config: widths,
        supports=supports,
        caches={},
        weight_source="views",
        options={},
        refuse_config=refuse_config,
        refuse_model=refuse_model,
        pack=lambda model, ctx: None,
        bind=bind,
        eligible=eligible,
        build=lambda ctx, packed, layer, width: FakeKernel(
            layer.layer_idx, width, health, caches
        ),
    )


def assert_matches_layer(
    kernel_out: torch.Tensor, layer_out: torch.Tensor, cos_min: float = 0.9999
) -> None:
    """The golden-layer check: a layer kernel's output against the vLLM layer's
    on the same inputs (cosine, so it holds across reduction orders)."""
    a, b = kernel_out.float().reshape(-1), layer_out.float().reshape(-1)
    cos = F.cosine_similarity(a, b, dim=0).item()
    assert cos > cos_min, f"cosine {cos} <= {cos_min}"


def failed_words(tp: int, ranks: list[int]) -> torch.Tensor:
    """The fault harness: the output rank's flags after ``ranks`` gave up."""
    words = torch.zeros(tp, dtype=torch.int32)
    words[ranks] = 1
    return words
