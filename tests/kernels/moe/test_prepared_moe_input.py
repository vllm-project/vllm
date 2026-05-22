# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.model_executor.layers.fused_moe.config import FusedMoEQuantConfig
from vllm.model_executor.layers.fused_moe.minimax_m2_kernels import (
    minimax_moe_topk_sigmoid_quant,
)
from vllm.model_executor.layers.fused_moe.prepare_finalize import no_dp_ep
from vllm.model_executor.layers.fused_moe.runner.moe_runner import MoERunner
from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    per_token_group_quant_fp8,
)


def _block_quant_config() -> FusedMoEQuantConfig:
    return FusedMoEQuantConfig.make(
        quant_dtype=torch.float8_e4m3fn,
        block_shape=[128, 128],
    )


def test_no_dp_ep_prepare_uses_prepared_fp8_input(monkeypatch):
    prepare_finalize = no_dp_ep.MoEPrepareAndFinalizeNoDPEPModular()
    quant_config = _block_quant_config()
    a1 = torch.randn(4, 256, dtype=torch.bfloat16)
    prepared_a1q = torch.empty(a1.shape, dtype=torch.float8_e4m3fn)
    prepared_a1q_scale = torch.randn(4, 2, dtype=torch.float32)
    topk_weights = torch.ones(4, 1, dtype=torch.float32)
    topk_ids = torch.zeros(4, 1, dtype=torch.int32)

    def fail_quantize(*args, **kwargs):
        raise AssertionError("prepared MoE input should skip internal quantization")

    monkeypatch.setattr(no_dp_ep, "moe_kernel_quantize_input", fail_quantize)

    a1q, a1q_scale, expert_tokens_meta, expert_topk_ids, expert_topk_weights = (
        prepare_finalize.prepare_prepared_input(
            a1,
            prepared_a1q,
            prepared_a1q_scale,
            topk_weights,
            topk_ids,
            num_experts=8,
            expert_map=None,
            apply_router_weight_on_input=False,
            quant_config=quant_config,
            defer_input_quant=False,
        )
    )

    assert a1q is prepared_a1q
    assert a1q_scale is prepared_a1q_scale
    assert expert_tokens_meta is None
    assert expert_topk_ids is None
    assert expert_topk_weights is None


def test_no_dp_ep_prepared_input_matches_prepare_quant_result(monkeypatch):
    prepare_finalize = no_dp_ep.MoEPrepareAndFinalizeNoDPEPModular()
    quant_config = _block_quant_config()
    a1 = torch.randn(4, 256, dtype=torch.bfloat16)
    prepared_a1q = torch.empty(a1.shape, dtype=torch.float8_e4m3fn)
    prepared_a1q_scale = torch.randn(4, 2, dtype=torch.float32)
    topk_weights = torch.ones(4, 1, dtype=torch.float32)
    topk_ids = torch.zeros(4, 1, dtype=torch.int32)

    def fake_quantize(*args, **kwargs):
        return prepared_a1q, prepared_a1q_scale

    monkeypatch.setattr(no_dp_ep, "moe_kernel_quantize_input", fake_quantize)

    regular = prepare_finalize.prepare(
        a1,
        topk_weights,
        topk_ids,
        num_experts=8,
        expert_map=None,
        apply_router_weight_on_input=False,
        quant_config=quant_config,
        defer_input_quant=False,
    )
    prepared = prepare_finalize.prepare_prepared_input(
        a1,
        prepared_a1q,
        prepared_a1q_scale,
        topk_weights,
        topk_ids,
        num_experts=8,
        expert_map=None,
        apply_router_weight_on_input=False,
        quant_config=quant_config,
        defer_input_quant=False,
    )

    assert regular[0] is prepared[0] is prepared_a1q
    assert regular[1] is prepared[1] is prepared_a1q_scale
    assert regular[2:] == prepared[2:] == (None, None, None)


def test_runner_prepared_topk_skips_router_select_experts():
    class FakeRouter:
        prepare_called = False

        def select_experts(self, *args, **kwargs):
            raise AssertionError("prepared topk should skip router selection")

        def prepare_precomputed_experts(self, topk_weights, topk_ids):
            self.prepare_called = True
            return topk_weights, topk_ids

    class FakeQuantMethod:
        is_monolithic = False
        supports_prepared_inputs = True

        def __init__(self):
            self.seen_kwargs = None

        def apply(self, **kwargs):
            self.seen_kwargs = kwargs
            return torch.empty_like(kwargs["x"])

    router = FakeRouter()
    quant_method = FakeQuantMethod()
    runner = object.__new__(MoERunner)
    runner.router = router
    runner._quant_method = quant_method
    runner._shared_experts = None

    hidden_states = torch.randn(3, 16, dtype=torch.bfloat16)
    router_logits = torch.randn(3, 8, dtype=torch.float32)
    prepared_topk_weights = torch.randn(3, 2, dtype=torch.float32)
    prepared_topk_ids = torch.randint(0, 8, (3, 2), dtype=torch.int32)
    prepared_a1q = torch.empty_like(hidden_states, dtype=torch.float8_e4m3fn)
    prepared_a1q_scale = torch.randn(3, 1, dtype=torch.float32)

    shared_output, fused_output = runner._apply_quant_method(
        layer=torch.nn.Module(),
        hidden_states=hidden_states,
        router_logits=router_logits,
        shared_experts_input=None,
        prepared_topk_weights=prepared_topk_weights,
        prepared_topk_ids=prepared_topk_ids,
        prepared_a1q=prepared_a1q,
        prepared_a1q_scale=prepared_a1q_scale,
    )

    assert shared_output is None
    assert fused_output.shape == hidden_states.shape
    assert router.prepare_called
    assert quant_method.seen_kwargs["topk_weights"] is prepared_topk_weights
    assert quant_method.seen_kwargs["topk_ids"] is prepared_topk_ids
    assert quant_method.seen_kwargs["prepared_a1q"] is prepared_a1q
    assert quant_method.seen_kwargs["prepared_a1q_scale"] is prepared_a1q_scale


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA-only kernel test")
@pytest.mark.parametrize("m", [1, 17, 128])
def test_minimax_moe_topk_sigmoid_quant_matches_reference(m: int):
    from vllm import _custom_ops as ops

    hidden_states = torch.randn((m, 3072), device="cuda", dtype=torch.bfloat16)
    router_logits = torch.randn((m, 256), device="cuda", dtype=torch.float32)
    e_score_correction_bias = torch.randn((256,), device="cuda", dtype=torch.float32)

    topk_weights, topk_ids, a1q, a1q_scale = minimax_moe_topk_sigmoid_quant(
        hidden_states,
        router_logits,
        e_score_correction_bias,
        top_k=8,
        block_k=128,
    )

    ref_topk_weights = torch.empty((m, 8), device="cuda", dtype=torch.float32)
    ref_topk_ids = torch.empty((m, 8), device="cuda", dtype=torch.int32)
    token_expert_indices = torch.empty((m, 8), device="cuda", dtype=torch.int32)
    ops.topk_sigmoid(
        ref_topk_weights,
        ref_topk_ids,
        token_expert_indices,
        router_logits,
        renormalize=True,
        e_score_correction_bias=e_score_correction_bias,
    )
    ref_a1q, ref_a1q_scale = per_token_group_quant_fp8(
        hidden_states, 128, use_ue8m0=False
    )

    torch.testing.assert_close(topk_ids, ref_topk_ids, atol=0, rtol=0)
    torch.testing.assert_close(topk_weights, ref_topk_weights, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(a1q.float(), ref_a1q.float(), atol=0, rtol=0)
    torch.testing.assert_close(a1q_scale, ref_a1q_scale, atol=1e-6, rtol=1e-6)
