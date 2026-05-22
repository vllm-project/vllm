# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.model_executor.layers.fused_moe.config import FusedMoEQuantConfig
from vllm.model_executor.layers.fused_moe.prepare_finalize import no_dp_ep


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
