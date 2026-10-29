# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
from torch import nn

from vllm.models.muse_spark.amd.model import MuseSpark1xForCausalLM


def test_compute_logits_local_applies_output_transform() -> None:
    model = MuseSpark1xForCausalLM.__new__(MuseSpark1xForCausalLM)
    nn.Module.__init__(model)
    model.lm_head = nn.Linear(4, 8, bias=False, dtype=torch.bfloat16)
    model.logits_processor = lambda lm_head, h, skip_gather=False: lm_head(h)
    model.metap_output_multiplier = 0.5
    model.output_soft_cap_temp = 20.0
    model.mul_tanh_soft_cap = None
    hidden = torch.randn(3, 4, dtype=torch.bfloat16) * 50

    torch.testing.assert_close(
        model.compute_logits_local(hidden), model.compute_logits(hidden)
    )


def test_nvidia_router_multiplier_folds_on_every_load():
    """The router weight holds the folded multiplier after every load of the
    raw weight, a reload included."""
    from vllm.models.muse_spark.nvidia.model import MuseSpark1xMoE

    moe = MuseSpark1xMoE.__new__(MuseSpark1xMoE)
    nn.Module.__init__(moe)
    moe.router = nn.Linear(4, 3, bias=False)
    moe.router.weight_loader = lambda param, w: param.data.copy_(w)
    moe.router_output_multiplier = 2.0
    raw = torch.randn(3, 4)
    for _ in range(2):  # cold start, reload
        moe._load_router_weight(moe.router.weight, raw)
        torch.testing.assert_close(moe.router.weight.detach(), raw * 2.0)


def test_nvidia_nvfp4_moe_constants_keep_addresses(monkeypatch):
    """Every post-load rerun builds a new quant config and experts object; the
    TRT-LLM NVFP4 constants captured CUDA graphs read stay at their first
    addresses, refilled with the fresh values."""
    from types import SimpleNamespace

    from vllm.models.muse_spark.nvidia import muse_moe_runner as mr

    def convert(**kw):
        return tuple(
            kw.get(k)
            for k in ("w13", "w13_scale", "", "a13_scale")
            + ("w2", "w2_scale", "", "a2_scale")
        )

    def make_kernel(**kw):
        experts = object.__new__(mr.MuseSparkTrtLlmNvFp4Experts)
        experts.quant_config = kw["moe_quant_config"]
        experts.gemm1_clamp_limit = experts.gemm1_alpha = experts.gemm1_beta = None
        return SimpleNamespace(fused_experts=experts)

    monkeypatch.setattr(mr, "convert_to_nvfp4_moe_kernel_format", convert)
    monkeypatch.setattr(mr, "make_nvfp4_moe_kernel", make_kernel)
    monkeypatch.setattr(mr, "_attach_post_expert_finalize", lambda method: None)
    method = object.__new__(mr.MuseSparkModelOptNvFp4FusedMoE)
    method.use_route_major = False
    method.nvfp4_backend = method.moe = method.experts_cls = None
    layer = nn.Module()
    for name, shape in [
        ("w13_weight", (2, 4)),
        ("w13_weight_scale", (2, 4)),
        ("w13_weight_scale_2", (2, 2)),
        ("w13_input_scale", (2,)),
        ("w2_weight", (2, 4)),
        ("w2_weight_scale", (2, 4)),
        ("w2_weight_scale_2", (2,)),
        ("w2_input_scale", (2,)),
    ]:
        layer.register_parameter(
            name, nn.Parameter(torch.ones(shape), requires_grad=False)
        )
    layer._expert_routing_tables = lambda: None

    def load(scale: float) -> dict[str, torch.Tensor]:
        layer.w13_weight_scale_2.data.fill_(scale)
        method.process_weights_after_loading(layer)
        config = method.moe_quant_config
        experts = method.moe_kernel.fused_experts
        assert experts.g1_scale_c is layer.g1_scale_c
        return {
            "g1_scale_c": config.g1_scale_c,
            "experts_g1_scale_c": layer.g1_scale_c,
            "g1_alphas": config._w1.alpha_or_gscale,
            "g2_alphas": config._w2.alpha_or_gscale,
            "a1_gscale": config._a1.alpha_or_gscale,
            "a2_gscale": config._a2.alpha_or_gscale,
        }

    cold = load(1.0)
    ptrs = {name: t.data_ptr() for name, t in cold.items()}
    reloaded = load(2.0)
    assert {name: t.data_ptr() for name, t in reloaded.items()} == ptrs
    torch.testing.assert_close(reloaded["g1_alphas"], torch.full((2,), 2.0))
    torch.testing.assert_close(reloaded["g1_scale_c"], torch.full((2,), 2.0))
