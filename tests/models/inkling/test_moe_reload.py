# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Weight updates of the Inkling NVFP4 routed experts (input_amax)."""

import pytest
import torch

from vllm.config import VllmConfig, set_current_vllm_config
from vllm.config.model import ModelConfig
from vllm.distributed.weight_transfer.base import ParamMeta
from vllm.model_executor.layers.quantization.modelopt import ModelOptNvFp4Config
from vllm.model_executor.model_loader.reload import (
    finish_reload,
    load_planned,
    plan_loads,
    record_metadata_for_reloading,
    start_reload,
)
from vllm.model_executor.model_loader.utils import process_weights_after_loading
from vllm.model_executor.utils import kernel_parameters
from vllm.models.inkling.configs import InklingModelConfig
from vllm.models.inkling.nvidia.moe import InklingMoE
from vllm.platforms import current_platform

E, H, N = 8, 256, 256  # routed experts, hidden size, intermediate size
_SHAPES = {
    "experts.w13_weight": ((E, 2 * N, H // 2), torch.uint8),
    "experts.w13_weight_scale": ((E, 2 * N, H // 16), torch.float8_e4m3fn),
    "experts.w13_weight_scale_2": ((E,), torch.float32),
    "experts.w13_weight.input_amax": ((1,), torch.bfloat16),
    "experts.w2_weight": ((E, H, N // 2), torch.uint8),
    "experts.w2_weight_scale": ((E, H, N // 16), torch.float8_e4m3fn),
    "experts.w2_weight_scale_2": ((E,), torch.float32),
    "experts.w2_weight.input_amax": ((1,), torch.bfloat16),
}


class _Model(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        config = InklingModelConfig(
            hidden_size=H,
            intermediate_size=N,
            n_routed_experts=E,
            n_shared_experts=1,
            num_experts_per_tok=2,
            shared_expert_sink=True,
        )
        quant_config = ModelOptNvFp4Config(
            is_checkpoint_nvfp4_serialized=True,
            kv_cache_quant_algo=None,
            exclude_modules=[],
        )
        with set_current_vllm_config(VllmConfig()), torch.device("cuda"):
            self.mlp = InklingMoE(
                config, prefix="layers.0.mlp", quant_config=quant_config
            )

    def load_weights(self, weights) -> None:
        for name, weight in weights:
            self.mlp.load_expert_weight(name, weight)


def _checkpoint(seed: int) -> dict[str, torch.Tensor]:
    gen = torch.Generator().manual_seed(seed)
    weights = {}
    for name, (shape, dtype) in _SHAPES.items():
        if dtype == torch.uint8:
            weights[name] = torch.randint(0, 256, shape, dtype=dtype, generator=gen)
        else:
            weights[name] = (torch.rand(shape, generator=gen) + 0.5).to(dtype)
    return weights


def _cold_start(
    weights, model_config: ModelConfig, *, scales_from_loader: bool = False
) -> _Model:
    model = _Model()
    model.load_weights(weights.items())
    if scales_from_loader:
        # The reference: input scales set as the loader used to, no amax.
        experts = model.mlp.experts.routed_experts
        for projection in ("w13", "w2"):
            amax = float(weights[f"experts.{projection}_weight.input_amax"].max())
            getattr(experts, f"{projection}_input_scale").fill_(amax / (448.0 * 6.0))
            delattr(experts, f"{projection}_input_amax")
    process_weights_after_loading(model, model_config, torch.device("cuda"))
    return model


def _state(model: _Model) -> dict[str, torch.Tensor]:
    """The routed experts' state dict plus tensor attributes (such as the
    ``_a1_gscale`` the kernels read, derived from the input scale)."""
    experts = model.mlp.experts.routed_experts
    tensors = dict(experts.state_dict(keep_vars=True))
    for module_name, module in experts.named_modules():
        prefix = f"{module_name}." if module_name else ""
        for attr, value in vars(module).items():
            if isinstance(value, torch.Tensor):
                tensors[prefix + attr] = value
    return {n: t.detach().clone() for n, t in tensors.items()}


@pytest.mark.skipif(
    not current_platform.has_device_capability(100), reason="NVFP4 MoE needs SM100"
)
def test_planned_update_rederives_input_scales(dist_init, monkeypatch) -> None:
    """``input_amax`` is copied as is, so the plan loads it directly, and
    post-load processing rederives the input scales from it on every update,
    leaving the experts as the loader-derived scales did at cold start."""
    monkeypatch.setenv("VLLM_RELOAD_MODE", "direct")
    model_config = ModelConfig()
    model = _cold_start(_checkpoint(seed=0), model_config)
    experts = model.mlp.experts.routed_experts
    ptrs = {n: t.data_ptr() for n, t in experts.state_dict(keep_vars=True).items()}

    start_reload(model)
    metadata = [ParamMeta(n, dtype, shape) for n, (shape, dtype) in _SHAPES.items()]
    plan = plan_loads(model, metadata)
    for projection in ("w13", "w2"):
        assert plan.get_dest(f"experts.{projection}_weight.input_amax")

    for seed in (1, 2):
        if seed > 1:
            start_reload(model)
        new = _checkpoint(seed)
        load_planned(model, plan, list(new.items()))
        finish_reload(model, model_config, torch.device("cuda"))

        actual = _state(model)
        assert ptrs == {
            n: t.data_ptr() for n, t in experts.state_dict(keep_vars=True).items()
        }
        expected = _state(_cold_start(new, model_config, scales_from_loader=True))
        for projection in ("w13", "w2"):
            amax = actual.pop(f"{projection}_input_amax")
            assert torch.equal(
                amax.cpu(), new[f"experts.{projection}_weight.input_amax"]
            )
        assert actual.keys() == expected.keys()
        for name, tensor in actual.items():
            assert torch.equal(
                tensor.view(torch.uint8), expected[name].view(torch.uint8)
            ), name


@pytest.mark.skipif(
    not current_platform.has_device_capability(100), reason="NVFP4 MoE needs SM100"
)
def test_layerwise_reload_keeps_routed_experts(dist_init, monkeypatch) -> None:
    """Layerwise reload doesn't see the routed-expert loads (they write
    ``.data``); the amax load must not make it reprocess the layer from
    uninitialized weights."""
    monkeypatch.setenv("VLLM_RELOAD_MODE", "layerwise")
    model_config = ModelConfig()
    weights = _checkpoint(seed=0)
    model = _Model()
    with torch.device("cuda"):
        record_metadata_for_reloading(model)  # as model init does
    model.load_weights(weights.items())
    process_weights_after_loading(model, model_config, torch.device("cuda"))
    before = _state(model)

    start_reload(model)
    model.load_weights(weights.items())
    finish_reload(model, model_config, torch.device("cuda"))

    after = _state(model)
    assert after.keys() == before.keys()
    for name, tensor in after.items():
        assert torch.equal(tensor.view(torch.uint8), before[name].view(torch.uint8)), (
            name
        )


@pytest.mark.skipif(
    not current_platform.has_device_capability(100), reason="NVFP4 MoE needs SM100"
)
def test_eplb_skips_input_amax(dist_init) -> None:
    """The amax is retired after post-load processing, so EPLB (which views
    every expert weight as [E, -1]) and the kernels skip it."""
    model = _cold_start(_checkpoint(seed=0), ModelConfig())
    experts = model.mlp.experts.routed_experts
    live = kernel_parameters(experts)
    assert "w13_input_amax" not in live and "w2_input_amax" not in live
    experts.get_expert_weights()  # a live [1] amax would fail view(E, -1)
