# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import torch

from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEQuantConfig,
    FusedMoEQuantDesc,
)


class FakeMoeAllToAll:
    def __init__(self):
        self.dispatch_kwargs = None
        self.dispatched_payloads = None

    def dispatch(self, token_selected_experts, input_payloads, **kwargs):
        self.dispatch_kwargs = kwargs
        self.dispatched_payloads = input_payloads
        return input_payloads


class FakeAll2AllManager:
    rank = 0
    world_size = 2

    def __init__(self):
        self.moe_alltoall = FakeMoeAllToAll()
        self.initialize_kwargs = None

    def initialize(self, **kwargs):
        self.initialize_kwargs = kwargs


def _bf16_quant_config() -> FusedMoEQuantConfig:
    desc = FusedMoEQuantDesc(dtype=None)
    return FusedMoEQuantConfig(desc, desc, desc, desc)


def _mxfp8_quant_config() -> FusedMoEQuantConfig:
    a_desc = FusedMoEQuantDesc(dtype="mxfp8")
    w_desc = FusedMoEQuantDesc(dtype="mxfp4")
    return FusedMoEQuantConfig(a_desc, a_desc, w_desc, w_desc)


def test_one_sided_prepare_uses_out_of_range_invalid_expert_id_without_expert_map(
    monkeypatch,
):
    from vllm.model_executor.layers.fused_moe.prepare_finalize import (
        flashinfer_nvlink_one_sided as one_sided,
    )

    manager = FakeAll2AllManager()
    monkeypatch.setattr(
        one_sided,
        "get_ep_group",
        lambda: SimpleNamespace(
            device_communicator=SimpleNamespace(all2all_manager=manager)
        ),
    )
    monkeypatch.setattr(one_sided, "get_local_sizes", lambda: [2])

    def fail_quantize_input(*args, **kwargs):
        raise AssertionError("deferred BF16 dispatch should not quantize activations")

    monkeypatch.setattr(one_sided, "moe_kernel_quantize_input", fail_quantize_input)

    prepare_finalize = one_sided.FlashInferNVLinkOneSidedPrepareAndFinalize(
        max_num_tokens=8,
        top_k=2,
        num_experts=4,
        hidden_size=8,
        dispatch_dtype_bytes_per_elem=2,
        dispatch_scale_bytes_per_token=0,
    )
    a1 = torch.randn((2, 8), dtype=torch.bfloat16)
    topk_ids = torch.tensor([[0, 1], [2, 3]], dtype=torch.int32)
    topk_weights = torch.randn((2, 2), dtype=torch.float32)

    a1_recv, a1_scale_recv, _, topk_ids_recv, topk_weights_recv = (
        prepare_finalize.prepare(
            a1=a1,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            num_experts=4,
            expert_map=None,
            apply_router_weight_on_input=False,
            quant_config=_bf16_quant_config(),
            defer_input_quant=True,
        )
    )

    assert manager.initialize_kwargs["dispatch_dtype_bytes_per_elem"] == 2
    assert manager.initialize_kwargs["dispatch_scale_bytes_per_token"] == 0
    assert manager.moe_alltoall.dispatch_kwargs["invalid_token_expert_id"] == 4
    assert manager.moe_alltoall.dispatch_kwargs["expert_id_payload_index"] == 1
    assert manager.moe_alltoall.dispatched_payloads[0] is a1
    assert manager.moe_alltoall.dispatched_payloads[1] is topk_ids
    assert manager.moe_alltoall.dispatched_payloads[2] is topk_weights
    torch.testing.assert_close(a1_recv, a1)
    assert a1_scale_recv is None
    torch.testing.assert_close(topk_ids_recv, topk_ids)
    torch.testing.assert_close(topk_weights_recv, topk_weights)


def test_one_sided_prepare_uses_nonlocal_padding_expert_with_expert_map(monkeypatch):
    from vllm.model_executor.layers.fused_moe.prepare_finalize import (
        flashinfer_nvlink_one_sided as one_sided,
    )

    manager = FakeAll2AllManager()
    monkeypatch.setattr(
        one_sided,
        "get_ep_group",
        lambda: SimpleNamespace(
            device_communicator=SimpleNamespace(all2all_manager=manager)
        ),
    )
    monkeypatch.setattr(one_sided, "get_local_sizes", lambda: [2])

    a1q = torch.empty((2, 8), dtype=torch.uint8)
    a1q_scale = torch.empty((2, 1), dtype=torch.uint8)

    def fake_quantize_input(*args, **kwargs):
        return a1q, a1q_scale

    monkeypatch.setattr(one_sided, "moe_kernel_quantize_input", fake_quantize_input)

    prepare_finalize = one_sided.FlashInferNVLinkOneSidedPrepareAndFinalize(
        max_num_tokens=8,
        top_k=2,
        num_experts=4,
        hidden_size=8,
        dispatch_dtype_bytes_per_elem=1,
        dispatch_scale_bytes_per_token=1,
    )
    a1 = torch.randn((2, 8), dtype=torch.bfloat16)
    topk_ids = torch.tensor([[0, 1], [2, 3]], dtype=torch.int32)
    topk_weights = torch.randn((2, 2), dtype=torch.float32)

    a1_recv, a1_scale_recv, _, topk_ids_recv, topk_weights_recv = (
        prepare_finalize.prepare(
            a1=a1,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            num_experts=4,
            expert_map=torch.tensor([0, 1, -1, -1], dtype=torch.int32),
            apply_router_weight_on_input=False,
            quant_config=_mxfp8_quant_config(),
        )
    )

    assert manager.moe_alltoall.dispatch_kwargs["invalid_token_expert_id"] == 2
    assert manager.moe_alltoall.dispatch_kwargs["expert_id_payload_index"] == 2
    assert manager.moe_alltoall.dispatched_payloads[0] is a1q
    assert manager.moe_alltoall.dispatched_payloads[1] is a1q_scale
    assert manager.moe_alltoall.dispatched_payloads[2] is topk_ids
    assert manager.moe_alltoall.dispatched_payloads[3] is topk_weights
    torch.testing.assert_close(a1_recv, a1q)
    torch.testing.assert_close(a1_scale_recv, a1q_scale)
    torch.testing.assert_close(topk_ids_recv, topk_ids)
    torch.testing.assert_close(topk_weights_recv, topk_weights)
