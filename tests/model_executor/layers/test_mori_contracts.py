# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import importlib
import sys
from types import SimpleNamespace

import pytest
import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.distributed.device_communicators.all2all import MoriAll2AllManager


class _PrepareFinalizeUsesOriginalIds:
    def __init__(self):
        self.finalize_topk_ids = None
        self.finalize_topk_weights = None

    def finalize_uses_original_topk_ids(self):
        return True

    def supports_async(self):
        return False

    def prepare(
        self,
        hidden_states,
        topk_weights,
        topk_ids,
        global_num_experts,
        expert_map,
        apply_router_weight_on_input,
        quant_config,
        defer_input_quant,
    ):
        topk_ids.add_(100)
        topk_weights.add_(100)
        return hidden_states, None, None, topk_ids + 1, topk_weights + 1

    def finalize(
        self,
        output,
        fused_expert_output,
        topk_weights,
        topk_ids,
        apply_router_weight_on_input,
        weight_and_reduce_impl,
    ):
        self.finalize_topk_ids = topk_ids.clone()
        self.finalize_topk_weights = topk_weights.clone()
        output.copy_(fused_expert_output)


class _FusedExpertsStub:
    def __init__(self):
        self.moe_config = SimpleNamespace(moe_parallel_config=None)
        self.quant_config = object()
        self.expects_unquantized_inputs = False

    def finalize_weight_and_reduce_impl(self):
        return object()


def test_mori_prepare_finalize_requests_original_router_topk_ids(monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        "mori",
        SimpleNamespace(ops=SimpleNamespace(EpDispatchCombineOp=object)),
    )
    module_name = "vllm.model_executor.layers.fused_moe.prepare_finalize.mori"
    previous_module = sys.modules.pop(module_name, None)
    try:
        mori_module = importlib.import_module(module_name)
        prepare_finalize = mori_module.MoriPrepareAndFinalize(
            mori_op=object(),
            max_tokens_per_rank=16,
            num_dispatchers=1,
        )

        assert prepare_finalize.finalize_uses_original_topk_ids()
    finally:
        sys.modules.pop(module_name, None)
        if previous_module is not None:
            sys.modules[module_name] = previous_module


def test_modular_finalize_uses_original_ids_but_prepared_weights(monkeypatch):
    prepare_finalize = _PrepareFinalizeUsesOriginalIds()
    kernel = mk.FusedMoEKernelModularImpl(prepare_finalize, _FusedExpertsStub())
    monkeypatch.setattr(
        kernel,
        "_fused_experts",
        lambda **kwargs: torch.full_like(kwargs["a1q"], 3),
    )

    hidden_states = torch.zeros((2, 2), dtype=torch.float32)
    w1 = torch.empty((1, 1, 1), dtype=torch.float32)
    w2 = torch.empty((1, 1, 1), dtype=torch.float32)
    topk_ids = torch.tensor([[5], [6]], dtype=torch.int64)
    topk_weights = torch.tensor([[0.25], [0.75]], dtype=torch.float32)

    output = kernel.apply(
        hidden_states=hidden_states,
        w1=w1,
        w2=w2,
        topk_ids=topk_ids,
        topk_weights=topk_weights,
        global_num_experts=1,
    )

    torch.testing.assert_close(output, torch.full_like(hidden_states, 3))
    torch.testing.assert_close(
        prepare_finalize.finalize_topk_ids,
        torch.tensor([[5], [6]], dtype=torch.int64),
    )
    torch.testing.assert_close(
        prepare_finalize.finalize_topk_weights,
        torch.tensor([[101.25], [101.75]], dtype=torch.float32),
    )


class _PrepareFinalizeUsesPreparedIds(_PrepareFinalizeUsesOriginalIds):
    def finalize_uses_original_topk_ids(self):
        return False


def test_modular_finalize_uses_prepared_ids_when_not_opted_in(monkeypatch):
    prepare_finalize = _PrepareFinalizeUsesPreparedIds()
    kernel = mk.FusedMoEKernelModularImpl(prepare_finalize, _FusedExpertsStub())
    monkeypatch.setattr(
        kernel,
        "_fused_experts",
        lambda **kwargs: torch.full_like(kwargs["a1q"], 3),
    )

    hidden_states = torch.zeros((2, 2), dtype=torch.float32)
    w1 = torch.empty((1, 1, 1), dtype=torch.float32)
    w2 = torch.empty((1, 1, 1), dtype=torch.float32)
    topk_ids = torch.tensor([[5], [6]], dtype=torch.int64)
    topk_weights = torch.tensor([[0.25], [0.75]], dtype=torch.float32)

    output = kernel.apply(
        hidden_states=hidden_states,
        w1=w1,
        w2=w2,
        topk_ids=topk_ids,
        topk_weights=topk_weights,
        global_num_experts=1,
    )

    torch.testing.assert_close(output, torch.full_like(hidden_states, 3))
    torch.testing.assert_close(
        prepare_finalize.finalize_topk_ids,
        torch.tensor([[106], [107]], dtype=torch.int64),
    )
    torch.testing.assert_close(
        prepare_finalize.finalize_topk_weights,
        torch.tensor([[101.25], [101.75]], dtype=torch.float32),
    )


@pytest.mark.parametrize(
    ("internode", "backend", "expected_kernel_type"),
    [
        (False, "mori_high_throughput", "intra"),
        (True, "mori_low_latency", "inter_ll"),
    ],
)
def test_mori_all2all_kwargs_pin_dispatch_quant_contract(
    monkeypatch, internode, backend, expected_kernel_type
):
    monkeypatch.setitem(
        sys.modules,
        "mori",
        SimpleNamespace(
            ops=SimpleNamespace(
                EpDispatchCombineKernelType=SimpleNamespace(
                    IntraNode="intra",
                    InterNodeV1="inter",
                    InterNodeV1LL="inter_ll",
                )
            )
        ),
    )
    monkeypatch.setattr("vllm.platforms.rocm.on_gfx942", lambda: True)
    monkeypatch.setattr("vllm.platforms.rocm.on_gfx950", lambda: False)

    manager = MoriAll2AllManager.__new__(MoriAll2AllManager)
    manager.internode = internode
    manager._all2all_backend = backend

    kwargs = manager._make_all2all_kwargs(
        rank=1,
        num_ep_ranks=16,
        input_dtype=torch.bfloat16,
        quant_dtype=torch.float8_e4m3fnuz,
        token_hidden_size=7168,
        scale_dim=1,
        scale_type_size=4,
        max_num_tokens_per_dp_rank=64,
        num_local_experts=4,
        num_experts_per_token=8,
    )

    assert kwargs["kernel_type"] == expected_kernel_type
    assert kwargs["num_qp_per_pe"] == 2
    assert kwargs["quant_type"] == "none"
    assert kwargs["data_type"] is torch.float8_e4m3fnuz
    assert kwargs["max_token_type_size"] == torch.bfloat16.itemsize
