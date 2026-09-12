# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from vllm.config import VllmConfig, set_current_vllm_config
from vllm.distributed import parallel_state
from vllm.model_executor.layers.quantization.auto_gptq import (
    AutoGPTQConfig,
    AutoGPTQLinearMethod,
)
from vllm.model_executor.models import k2_horizon
from vllm.platforms import current_platform
from vllm.utils.torch_utils import set_default_torch_dtype


pytestmark = pytest.mark.skipif(
    not current_platform.is_xpu(), reason="Requires the XPU GPTQ weight layout"
)


@pytest.fixture(params=[(1, 0), (2, 0), (2, 1)])
def mova_attention(monkeypatch: pytest.MonkeyPatch, request):
    """Build the MoVA value projection without initializing an attention kernel."""
    monkeypatch.setattr(
        parallel_state,
        "_TP",
        SimpleNamespace(world_size=request.param[0], rank_in_group=request.param[1]),
    )
    monkeypatch.setattr(
        k2_horizon, "Attention", lambda *args, **kwargs: nn.Identity()
    )

    quant_config = AutoGPTQConfig(
        weight_bits=4,
        group_size=64,
        desc_act=False,
        is_sym=True,
        lm_head_quantized=False,
        dynamic={},
        full_config={},
        modules_in_block_to_quantize=[
            f"self_attn.v_experts.{expert}" for expert in range(4)
        ],
    )
    runtime_config = SimpleNamespace(
        model_config=SimpleNamespace(dtype=torch.float16),
        scheduler_config=SimpleNamespace(max_num_batched_tokens=2048),
    )
    with (
        set_current_vllm_config(VllmConfig()),
        set_default_torch_dtype(torch.float16),
        torch.device("xpu"),
    ):
        return k2_horizon.K2HorizonMoVAAttention(
            vllm_config=runtime_config,
            hidden_size=128,
            num_heads=2,
            num_kv_heads=2,
            rope_parameters={"rope_type": "default"},
            rope_head_dim=64,
            query_key_norm=False,
            num_experts=4,
            num_experts_per_tok=2,
            moe_gate_bias=False,
            router_score_func="softmax",
            router_scaling_factor=1.0,
            quant_config=quant_config,
            prefix="model.layers.0.self_attn",
        )


def test_mova_value_experts_use_gptq_method(mova_attention):
    """The fused value projection inherits its experts' GPTQ allocation."""
    assert isinstance(
        mova_attention.v_experts_fused.quant_method, AutoGPTQLinearMethod
    )


@pytest.mark.parametrize("num_tokens", [1, 4])
@torch.inference_mode()
def test_mova_gptq_shards_match_dense_reference_and_graph(mova_attention, num_tokens):
    """Packed expert loading and graph replay preserve sparse MoVA outputs."""
    layer = mova_attention
    fused = layer.v_experts_fused
    torch.manual_seed(17)
    values = torch.randint(0, 16, (4, 128, 128), device="xpu", dtype=torch.int32)
    scales = torch.rand(4, 2, 128, device="xpu", dtype=torch.float16) * 0.01
    shifts = torch.arange(0, 32, 4, device="xpu", dtype=torch.int32)
    packed = (values.reshape(4, 16, 8, 128) << shifts[None, None, :, None]).sum(2)
    for expert in range(4):
        for name, weight in (
            ("qweight", packed[expert].to(torch.int32)),
            ("scales", scales[expert]),
            ("qzeros", torch.full((2, 16), 0x77777777, device="xpu", dtype=torch.int32)),
        ):
            param = getattr(fused, name)
            param.weight_loader(param, weight, expert)
    fused.quant_method.process_weights_after_loading(fused)
    layer.v_router.weight.normal_(std=0.05)

    dense = ((values.float() - 8) * scales.float().repeat_interleave(64, dim=1))
    dense = dense.transpose(1, 2).to(torch.float16)
    dense = dense[:, layer.tp_rank * layer.kv_size : (layer.tp_rank + 1) * layer.kv_size]
    x = torch.randn(num_tokens, 128, device="xpu", dtype=torch.float16)

    def reference():
        scores = F.linear(x, layer.v_router.weight).float().softmax(-1)
        weights, indices = scores.topk(2, dim=-1)
        weights = (weights / weights.sum(-1, keepdim=True)).to(x.dtype)
        all_values = torch.stack([F.silu(F.linear(x, w)) for w in dense], dim=1)
        selected = all_values.gather(1, indices[..., None].expand(-1, -1, layer.kv_size))
        return (selected * weights[..., None]).sum(1)

    torch.testing.assert_close(layer.compute_mova_v_sparse(x), reference(), atol=5e-3, rtol=2e-2)
    graph = torch.xpu.XPUGraph()
    with torch.xpu.graph(graph):
        output = layer.compute_mova_v_sparse(x)
    for _ in range(2):
        x.normal_()
        graph.replay()
        torch.testing.assert_close(output, reference(), atol=5e-3, rtol=2e-2)
