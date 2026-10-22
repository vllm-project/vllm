# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Test kit for the GLM-5.3-Flash specialized decode on CPU: a supported fake
config, a fake backbone and kernels, a fake forward context."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

import torch
from torch import nn

from vllm.config.compilation import CompilationMode, CUDAGraphMode
from vllm.models.glm5next.amd.specialized_decode.runtime import spec

# kernel layers of the fake backbone: MoE layers 1..3 (layer 4 is the last)
KERNEL_LAYERS = (1, 2, 3)


def make_vllm_config(**overrides: Any) -> SimpleNamespace:
    """The VllmConfig fields the config refusals read, all at a supported value
    (only the platform is refused off ROCm / off gfx950)."""
    parallel = dict(
        pipeline_parallel_size=1,
        data_parallel_size=1,
        enable_expert_parallel=False,
        enable_eplb=False,
        decode_context_parallel_size=1,
        prefill_context_parallel_size=1,
        tensor_parallel_size=4,
        use_sequence_parallel_moe=False,
    )
    parallel.update(overrides.pop("parallel", {}))
    compilation = dict(
        mode=CompilationMode.VLLM_COMPILE,
        cudagraph_mode=CUDAGraphMode.FULL_AND_PIECEWISE,
        cudagraph_capture_sizes=[1, 2, 4, 8, 16, 32],
    )
    compilation.update(overrides.pop("compilation", {}))
    cache = dict(cache_dtype="auto", mamba_ssm_cache_dtype="auto")
    cache.update(overrides.pop("cache", {}))
    hf = SimpleNamespace(model_type="glm5_next_text", **spec.GEOMETRY)
    for key, value in overrides.pop("hf", {}).items():
        setattr(hf, key, value)
    cfg = SimpleNamespace(
        parallel_config=SimpleNamespace(**parallel),
        model_config=SimpleNamespace(
            enable_sleep_mode=False, enable_prompt_embeds=False, hf_text_config=hf
        ),
        cache_config=SimpleNamespace(**cache),
        offload_config=SimpleNamespace(
            uva=SimpleNamespace(cpu_offload_gb=0),
            prefetch=SimpleNamespace(offload_group_size=0),
        ),
        compilation_config=SimpleNamespace(**compilation),
        aux_output_config=SimpleNamespace(enable_return_routed_experts=False),
        kernel_config=SimpleNamespace(monokernel=True),
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


class FakeBackbone(nn.Module):
    def __init__(self, vllm_config: Any = None, n: int = 5) -> None:
        super().__init__()
        self.vllm_config = vllm_config or make_vllm_config()
        self.config = SimpleNamespace(
            num_hidden_layers=n, mlp_layer_types=["dense"] + ["sparse"] * (n - 1)
        )
        self.layers = nn.ModuleList(FakeLayer(i) for i in range(n))
        self.start_layer, self.end_layer = 0, n

    def forward_layers(self, step, tokens: int):
        hidden, residual = torch.zeros(tokens, 4), None
        positions = torch.zeros(tokens, dtype=torch.long)
        for idx, layer in enumerate(self.layers):
            kernel = step.layer(idx) if step is not None else None
            hidden, residual = (kernel or layer)(positions, hidden, residual)
        return hidden, residual


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


def patch_layers(
    monkeypatch,
    log: list,
    model_reasons: tuple[str, ...] = (),
    bind_reasons: tuple[str, ...] = (),
    eligible_reason: str | None = None,
    rank: int = 0,
) -> None:
    """Replace the GLM layer builders with fakes, so the executor runs on CPU."""
    from vllm.models.glm5next.amd.specialized_decode.runtime import decode, layers

    words = torch.zeros(len(spec.STAGES), dtype=torch.int32)

    def bind(ctx, caches):
        log.append(("bind", tuple(sorted(caches))))
        return list(bind_reasons)

    def build(ctx, packed, layer, idx, width):
        log.append(("build", idx, width))
        return FakeKernel(idx, width, words)

    monkeypatch.setattr(decode, "model_refusals", lambda b: list(model_reasons))
    monkeypatch.setattr(
        layers, "pack", lambda model, ctx: log.append(("pack",)) or "packed"
    )
    monkeypatch.setattr(layers, "bind", bind)
    monkeypatch.setattr(layers, "build", build)
    monkeypatch.setattr(layers, "eligible", lambda ctx, step: eligible_reason)
    monkeypatch.setattr(
        layers, "metadata_names", lambda ls: [ls[i].attn_name for i in sorted(ls)]
    )
    monkeypatch.setattr(
        layers, "CACHES", {"kv": ("cache", (torch.float32,), lambda layer: True)}
    )
    monkeypatch.setattr("vllm.distributed.get_tensor_model_parallel_rank", lambda: rank)
    monkeypatch.setattr(
        "vllm.distributed.get_tensor_model_parallel_world_size", lambda: 4
    )
    monkeypatch.setattr(
        "vllm.distributed.get_tp_group",
        lambda: type("G", (), {"cpu_group": None})(),
    )


def decode_metadata(names: list[str], **fields: Any) -> dict[str, SimpleNamespace]:
    base = dict(num_prefills=0, max_query_len=1)
    base.update(fields)
    return {n: SimpleNamespace(**base) for n in names}


def forward_context(
    metadata: Any, mode: CUDAGraphMode = CUDAGraphMode.FULL
) -> SimpleNamespace:
    return SimpleNamespace(attn_metadata=metadata, cudagraph_runtime_mode=mode)
