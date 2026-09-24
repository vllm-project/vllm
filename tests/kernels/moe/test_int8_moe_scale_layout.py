# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""INT8 MoE activation vs weight scale layout.

Static per-tensor activations can be paired with per-output-channel weight
scales. Those layouts must be represented and indexed independently.
"""

from types import SimpleNamespace

import pytest
import torch

import vllm.model_executor.layers.fused_moe.fused_moe as fused_moe_module
from tests.kernels.moe.utils import make_dummy_moe_config
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.fused_moe import fused_experts
from vllm.model_executor.layers.fused_moe.config import int8_w8a8_moe_quant_config
from vllm.model_executor.layers.quantization.quark.quark_moe import (
    QuarkW8A8Int8MoEMethod,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    kInt8StaticChannelSym,
    kInt8StaticTensorSym,
)
from vllm.platforms import current_platform
from vllm.triton_utils import tl


def test_int8_w8a8_quant_config_static_act_per_channel_weights() -> None:
    w1_scale = torch.ones(4, 8, 1)
    w2_scale = torch.ones(4, 4, 1)
    a1_scale = torch.tensor(0.5)
    a2_scale = torch.tensor(0.25)

    quant_config = int8_w8a8_moe_quant_config(
        w1_scale=w1_scale,
        w2_scale=w2_scale,
        a1_scale=a1_scale,
        a2_scale=a2_scale,
        per_act_token_quant=False,
        per_out_ch_quant=True,
    )

    assert quant_config.per_act_token_quant is False
    assert quant_config.per_out_ch_quant is True
    assert quant_config.use_int8_w8a8 is True


def test_int8_w8a8_quant_config_defaults_tensor_weight_scales() -> None:
    quant_config = int8_w8a8_moe_quant_config(
        w1_scale=torch.ones(2),
        w2_scale=torch.ones(2),
        a1_scale=torch.tensor(1.0),
        a2_scale=torch.tensor(1.0),
        per_act_token_quant=False,
    )

    assert quant_config.per_act_token_quant is False
    assert quant_config.per_out_ch_quant is False


def test_quark_static_int8_per_channel_sets_per_out_ch_quant() -> None:
    method = QuarkW8A8Int8MoEMethod(
        make_dummy_moe_config(num_experts=4, hidden_dim=8, intermediate_size=8),
        kInt8StaticChannelSym,
        kInt8StaticTensorSym,
    )
    assert method.int8_backend is None
    assert method.weight_qscheme == "per_channel"

    layer = SimpleNamespace(
        w13_weight_scale=torch.ones(4, 16, 1),
        w2_weight_scale=torch.ones(4, 8, 1),
        w13_input_scale=torch.tensor(0.5),
        w2_input_scale=torch.tensor(0.25),
        w13_bias=None,
        w2_bias=None,
    )
    quant_config = method.get_fused_moe_quant_config(layer)

    assert quant_config is not None
    assert quant_config.per_out_ch_quant is True
    assert quant_config.per_act_token_quant is False


def test_quark_static_int8_per_tensor_keeps_tensor_weight_scales() -> None:
    method = QuarkW8A8Int8MoEMethod(
        make_dummy_moe_config(num_experts=4, hidden_dim=8, intermediate_size=8),
        kInt8StaticTensorSym,
        kInt8StaticTensorSym,
    )
    layer = SimpleNamespace(
        w13_weight_scale=torch.ones(4),
        w2_weight_scale=torch.ones(4),
        w13_input_scale=torch.tensor(0.5),
        w2_input_scale=torch.tensor(0.25),
        w13_bias=None,
        w2_bias=None,
    )
    quant_config = method.get_fused_moe_quant_config(layer)

    assert quant_config is not None
    assert quant_config.per_out_ch_quant is False
    assert quant_config.per_act_token_quant is False


def test_triton_moe_launcher_forwards_independent_scale_flags(monkeypatch) -> None:
    captured: dict[str, bool] = {}

    class FakeKernel:
        def __getitem__(self, grid):
            def launch(*args, **kwargs) -> None:
                captured["per_channel_quant"] = kwargs["per_channel_quant"]
                captured["per_out_ch_quant"] = kwargs["per_out_ch_quant"]

            return launch

    monkeypatch.setattr(fused_moe_module, "fused_moe_kernel", FakeKernel())

    fused_moe_module.invoke_fused_moe_triton_kernel(
        A=torch.ones((1, 1)),
        B=torch.ones((1, 1, 1)),
        C=torch.empty((1, 1, 1)),
        A_scale=torch.tensor([0.5]),
        B_scale=torch.ones((1, 1, 1)),
        topk_weights=torch.ones((1, 1)),
        sorted_token_ids=None,
        expert_ids=torch.zeros(1, dtype=torch.int32),
        num_tokens_post_padded=torch.ones(1, dtype=torch.int32),
        mul_routed_weight=True,
        top_k=1,
        config={"BLOCK_SIZE_M": 1, "BLOCK_SIZE_N": 1, "BLOCK_SIZE_K": 1},
        compute_type=tl.float32,
        use_fp8_w8a8=False,
        use_int8_w8a8=True,
        use_int8_w8a16=False,
        use_int4_w4a16=False,
        per_channel_quant=False,
        per_out_ch_quant=True,
    )

    assert captured["per_channel_quant"] is False
    assert captured["per_out_ch_quant"] is True


def test_fused_experts_forwards_independent_scale_flags(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_impl(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return args[0]

    monkeypatch.setattr(fused_moe_module, "fused_experts_impl", fake_impl)

    quant_config = int8_w8a8_moe_quant_config(
        w1_scale=torch.ones(2, 4, 1),
        w2_scale=torch.ones(2, 4, 1),
        a1_scale=torch.tensor(0.5),
        a2_scale=torch.tensor(0.25),
        per_act_token_quant=False,
        per_out_ch_quant=True,
    )
    fused_experts(
        hidden_states=torch.ones((1, 4), dtype=torch.bfloat16),
        w1=torch.ones((2, 8, 4), dtype=torch.int8),
        w2=torch.ones((2, 4, 4), dtype=torch.int8),
        topk_weights=torch.ones((1, 1)),
        topk_ids=torch.zeros((1, 1), dtype=torch.int32),
        quant_config=quant_config,
    )

    assert captured["kwargs"].get("per_out_ch_quant", captured["args"][-1]) is True
    assert captured["args"][12] is False


@pytest.mark.skipif(
    not current_platform.is_cuda_alike() or not torch.cuda.is_available(),
    reason="Requires a GPU to launch fused_experts INT8 kernels",
)
def test_static_int8_per_channel_weight_scale_affects_output() -> None:
    """Perturbing a routed expert's channel scales must change the output."""
    torch.manual_seed(0)
    device = torch.device(current_platform.device_type)
    num_tokens = 8
    num_experts = 4
    hidden = 64
    intermediate = 64
    top_k = 2

    hidden_states = torch.randn(
        (num_tokens, hidden), device=device, dtype=torch.bfloat16
    )
    w1 = torch.randint(
        -8, 8, (num_experts, 2 * intermediate, hidden), device=device, dtype=torch.int8
    )
    w2 = torch.randint(
        -8, 8, (num_experts, hidden, intermediate), device=device, dtype=torch.int8
    )
    w1_scale = torch.linspace(
        0.02, 0.08, num_experts * 2 * intermediate, device=device, dtype=torch.float32
    ).view(num_experts, 2 * intermediate, 1)
    w2_scale = torch.linspace(
        0.02, 0.08, num_experts * hidden, device=device, dtype=torch.float32
    ).view(num_experts, hidden, 1)
    a1_scale = torch.tensor(0.1, device=device, dtype=torch.float32)
    a2_scale = torch.tensor(0.1, device=device, dtype=torch.float32)

    # Expert 3 is routed on every token; expert 0 is never routed.
    topk_ids = torch.tensor([[3, 1]] * num_tokens, device=device, dtype=torch.int32)
    topk_weights = torch.ones((num_tokens, top_k), device=device, dtype=torch.bfloat16)

    quant_config = int8_w8a8_moe_quant_config(
        w1_scale=w1_scale.clone(),
        w2_scale=w2_scale.clone(),
        a1_scale=a1_scale,
        a2_scale=a2_scale,
        per_act_token_quant=False,
        per_out_ch_quant=True,
    )
    vllm_config = VllmConfig()
    with set_current_vllm_config(vllm_config):
        baseline = fused_experts(
            hidden_states,
            w1,
            w2,
            topk_weights,
            topk_ids,
            quant_config=quant_config,
        )

        routed_scales = w1_scale.clone()
        routed_scales[3] *= 4
        routed_out = fused_experts(
            hidden_states,
            w1,
            w2,
            topk_weights,
            topk_ids,
            quant_config=int8_w8a8_moe_quant_config(
                w1_scale=routed_scales,
                w2_scale=w2_scale.clone(),
                a1_scale=a1_scale,
                a2_scale=a2_scale,
                per_act_token_quant=False,
                per_out_ch_quant=True,
            ),
        )

        unrouted_scales = w1_scale.clone()
        unrouted_scales[0] *= 4
        unrouted_out = fused_experts(
            hidden_states,
            w1,
            w2,
            topk_weights,
            topk_ids,
            quant_config=int8_w8a8_moe_quant_config(
                w1_scale=unrouted_scales,
                w2_scale=w2_scale.clone(),
                a1_scale=a1_scale,
                a2_scale=a2_scale,
                per_act_token_quant=False,
                per_out_ch_quant=True,
            ),
        )

    assert not torch.equal(baseline, routed_out)
    torch.testing.assert_close(baseline, unrouted_out, atol=0, rtol=0)
