# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.quantization.moe_wna16 import MoeWNA16Method
from vllm.platforms import current_platform


@pytest.mark.skipif(not current_platform.is_cuda(), reason="Only test on CUDA")
def test_moe_wna16_apply_passes_layer_activation(monkeypatch):
    captured_kwargs = {}

    def fake_fused_experts(*args, **kwargs):
        captured_kwargs.update(kwargs)
        return torch.empty(1, 2)

    monkeypatch.setattr(
        "vllm.model_executor.layers.fused_moe.fused_experts",
        fake_fused_experts,
    )

    method = object.__new__(MoeWNA16Method)
    method.moe = SimpleNamespace(disable_inplace=False)
    method.moe_quant_config = object()
    layer = SimpleNamespace(
        w13_qweight=torch.empty(1, 2),
        w2_qweight=torch.empty(1, 2),
        activation=MoEActivation.GELU_TANH,
        apply_router_weight_on_input=False,
        global_num_experts=1,
        expert_map=None,
    )

    output = method.apply(
        layer,
        x=torch.empty(1, 2),
        topk_weights=torch.empty(1, 1),
        topk_ids=torch.empty(1, 1, dtype=torch.int32),
        shared_experts=None,
        shared_experts_input=None,
    )

    assert output.shape == (1, 2)
    assert captured_kwargs["activation"] is MoEActivation.GELU_TANH


@pytest.mark.skipif(not current_platform.is_cuda(), reason="Only test on CUDA")
def test_moe_wna16_gemm_checks_sorted_token_ids_bounds():
    from vllm import _custom_ops as ops

    size_m = 17
    size_k = 32
    size_n = 16
    block_size_m = 16
    top_k = 1

    x = torch.zeros(size_m, size_k, dtype=torch.float16, device="cuda")
    output = torch.empty(size_m * top_k, size_n, dtype=torch.float16, device="cuda")
    qweight = torch.empty(1, size_n, size_k // 2, dtype=torch.uint8, device="cuda")
    scales = torch.ones(1, size_n, 1, dtype=torch.float16, device="cuda")

    # Regression for #45496: num_tokens_post_pad can cover 9 blocks (144
    # tokens) while sorted_token_ids has only 129 entries. The last block must
    # not read sorted_token_ids[129].
    sorted_token_ids = torch.zeros(129, dtype=torch.int32, device="cuda")
    expert_ids = torch.full((9,), -1, dtype=torch.int32, device="cuda")
    num_tokens_post_pad = torch.tensor([144], dtype=torch.int32, device="cuda")

    ops.moe_wna16_gemm(
        x,
        output,
        qweight,
        scales,
        None,
        None,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_pad,
        top_k,
        block_size_m,
        size_n,
        size_k,
        4,
    )
    torch.cuda.synchronize()
    torch.testing.assert_close(output, torch.zeros_like(output))
