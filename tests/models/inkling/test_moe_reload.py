# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Weight updates of the Inkling NVFP4 routed experts under direct reload."""

from types import SimpleNamespace

import pytest
import torch

from vllm.config import VllmConfig, set_current_vllm_config
from vllm.distributed.weight_transfer.base import ParamMeta
from vllm.model_executor.layers.quantization.modelopt import ModelOptNvFp4Config
from vllm.model_executor.model_loader.reload import (
    finish_reload,
    load_planned,
    plan_loads,
    start_reload,
)
from vllm.model_executor.model_loader.utils import process_weights_after_loading
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
_MODEL_CONFIG = SimpleNamespace(
    word_embeddings_untied_by_checkpoint=False, quantization="modelopt_fp4"
)


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


def _cold_start(weights, *, scales_from_loader: bool = False) -> _Model:
    model = _Model()
    model.load_weights(weights.items())
    if scales_from_loader:
        # The reference: input scales set as the loader used to, no amax.
        experts = model.mlp.experts.routed_experts
        for projection in ("w13", "w2"):
            amax = float(weights[f"experts.{projection}_weight.input_amax"].max())
            getattr(experts, f"{projection}_input_scale").fill_(amax / (448.0 * 6.0))
            del experts._parameters[f"{projection}_input_amax"]
    process_weights_after_loading(model, _MODEL_CONFIG, torch.device("cuda"))
    return model


def _state(model: _Model) -> dict[str, torch.Tensor]:
    experts = model.mlp.experts.routed_experts
    return {n: t.clone() for n, t in experts.state_dict(keep_vars=True).items()}


@pytest.mark.skipif(
    not current_platform.has_device_capability(100), reason="NVFP4 MoE needs SM100"
)
def test_planned_update_rederives_input_scales(dist_init, monkeypatch) -> None:
    """``input_amax`` is copied as is, so the plan loads it directly, and
    post-load processing rederives the input scales from it on every update,
    leaving the experts as the loader-derived scales did at cold start."""
    monkeypatch.setenv("VLLM_RELOAD_MODE", "direct")
    model = _cold_start(_checkpoint(seed=0))
    experts = model.mlp.experts.routed_experts
    # EPLB views every expert weight as [E, -1]; the [1] amax must not be one.
    assert all(w.shape[0] == E for w in experts.get_expert_weights())

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
        finish_reload(model, _MODEL_CONFIG, torch.device("cuda"))

        actual = _state(model)
        expected = _state(_cold_start(new, scales_from_loader=True))
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
