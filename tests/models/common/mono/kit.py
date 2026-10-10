# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Test kit for MonoKernel specs: fake configs, a fake kernel, a fake forward
context. Model tests reuse it with their own spec."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

import torch
from torch import nn

from vllm.config.compilation import CUDAGraphMode
from vllm.models.common.mono import CacheReq, Feature, MonoSpec

STAGES = ("attn", "ffn")


def make_vllm_config(**overrides: Any) -> SimpleNamespace:
    """The VllmConfig fields the shared guards read, all at a supported value."""
    parallel = dict(
        pipeline_parallel_size=1,
        data_parallel_size=1,
        enable_expert_parallel=False,
        enable_eplb=False,
        decode_context_parallel_size=1,
        prefill_context_parallel_size=1,
        tensor_parallel_size=4,
    )
    parallel.update(overrides.pop("parallel", {}))
    compilation = dict(
        cudagraph_mode=CUDAGraphMode.FULL_AND_PIECEWISE,
        cudagraph_capture_sizes=[1, 2, 4, 8, 16, 32],
    )
    compilation.update(overrides.pop("compilation", {}))
    cfg = SimpleNamespace(
        parallel_config=SimpleNamespace(**parallel),
        model_config=SimpleNamespace(
            enable_sleep_mode=False,
            enable_prompt_embeds=False,
            hf_text_config=SimpleNamespace(num_hidden_layers=4),
        ),
        offload_config=SimpleNamespace(
            uva=SimpleNamespace(cpu_offload_gb=0),
            prefetch=SimpleNamespace(offload_group_size=0),
        ),
        compilation_config=SimpleNamespace(**compilation),
        aux_output_config=SimpleNamespace(enable_return_routed_experts=False),
        speculative_config=None,
        lora_config=None,
        kv_transfer_config=None,
        weight_transfer_config=None,
        use_v2_model_runner=True,
    )
    for key, value in overrides.items():
        setattr(cfg, key, value)
    return cfg


class FakeLayer(nn.Module):
    """A vLLM-style layer: (positions, hidden, residual) -> (hidden, residual)."""

    def __init__(self, idx: int) -> None:
        super().__init__()
        self.idx = idx
        self.cache = torch.zeros(8, 4)
        self.calls = 0
        self.attn_name = f"layers.{idx}.attn"

    def forward(self, positions, hidden, residual):
        self.calls += 1
        residual = hidden if residual is None else residual + hidden
        return hidden + 1, residual


class FakeModel(nn.Module):
    def __init__(self, n: int = 4) -> None:
        super().__init__()
        self.layers = nn.ModuleList(FakeLayer(i) for i in range(n))
        self.start_layer, self.end_layer = 0, n


@dataclass
class FakeKernel:
    idx: int
    width: int
    words: torch.Tensor
    launches: list = field(default_factory=list)

    def launch(self, step, hidden, residual):
        self.launches.append((self.idx, step.tokens, step.width))
        residual = hidden if residual is None else residual + hidden
        return hidden + 1, residual

    def health(self) -> torch.Tensor:
        return self.words


def make_spec(
    supports: frozenset[Feature] = frozenset(),
    config_reasons: tuple[str, ...] = (),
    model_reasons: tuple[str, ...] = (),
    bind_reasons: tuple[str, ...] = (),
    eligible_reason: str | None = None,
    layers: tuple[int, ...] = (1, 2, 3),
    log: list | None = None,
) -> MonoSpec:
    log = [] if log is None else log
    words = torch.zeros(len(STAGES), dtype=torch.int32)

    def bind(ctx, caches):
        log.append(("bind", tuple(sorted(caches))))
        return list(bind_reasons)

    def build(ctx, packed, layer, idx, width):
        log.append(("build", idx, width))
        return FakeKernel(idx, width, words)

    return MonoSpec(
        name="fake",
        layers=lambda hf: list(layers),
        widths=(2, 4, 8),
        supports=supports,
        caches={"kv": CacheReq("cache", (torch.float32,))},
        weight_source="views",
        grid=256,
        stages=STAGES,
        refuse_config=lambda cfg: list(config_reasons),
        refuse_model=lambda model: list(model_reasons),
        pack=lambda model, ctx: log.append(("pack",)) or "packed",
        bind=bind,
        eligible=lambda ctx, step: eligible_reason,
        build=build,
        metadata_names=lambda ls: [ls[i].attn_name for i in sorted(ls)],
        options={"tokens": (1, 2, 4, 8)},
    )


def decode_metadata(names: list[str], **fields: Any) -> dict[str, SimpleNamespace]:
    base = dict(num_prefills=0, max_query_len=1)
    base.update(fields)
    return {n: SimpleNamespace(**base) for n in names}


def forward_context(
    metadata: Any, mode: CUDAGraphMode = CUDAGraphMode.FULL
) -> SimpleNamespace:
    return SimpleNamespace(
        attn_metadata=metadata, slot_mapping={}, cudagraph_runtime_mode=mode
    )
