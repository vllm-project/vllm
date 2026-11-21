# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import importlib
import sys
import types
from types import SimpleNamespace

import pytest
import torch

from tests.kernels.moe.utils import make_dummy_moe_config
from vllm.model_executor.layers.fused_moe.config import FUSED_MOE_UNQUANTIZED_CONFIG
from vllm.model_executor.layers.fused_moe.experts.triton_moe import TritonExperts
from vllm.model_executor.layers.fused_moe.modular_kernel import (
    FusedMoEKernelModularImpl,
)
from vllm.model_executor.layers.fused_moe.prepare_finalize.no_dp_ep import (
    MoEPrepareAndFinalizeNoDPEPModular,
)
from vllm.model_executor.layers.fused_moe.topk_weight_and_reduce import (
    TopKWeightAndReduceNoOP,
)


def _describe(tensor):
    if tensor is None:
        return "None (the op was never called)"
    return f"a different tensor of shape {tuple(tensor.shape)}"


def _import_mori_prepare_finalize(monkeypatch):
    # This is a contract test for vLLM's adapter; it does not need a real MoRI
    # installation or GPU kernels.
    fake_mori = types.ModuleType("mori")
    fake_mori.ops = SimpleNamespace(  # type: ignore[attr-defined]
        EpDispatchCombineOp=object
    )
    monkeypatch.setitem(sys.modules, "mori", fake_mori)
    module_name = "vllm.model_executor.layers.fused_moe.prepare_finalize.mori"
    sys.modules.pop(module_name, None)
    return importlib.import_module(module_name)


class _FakeMoriOp:
    def __init__(self, combine_output=None):
        self.dispatch_indices = None
        self.combine_indices = None
        self.combine_output = combine_output

    def dispatch(self, a1, weights, scale, indices):
        self.dispatch_indices = indices
        # Model the global dispatched layout seen with EP=16: the experts need
        # these IDs, but passing them to combine violates MoRI's contract.
        dispatch_indices = indices.repeat(16, 1)
        dispatch_a1 = a1.repeat(16, 1)
        dispatch_weights = weights.repeat(16, 1)
        recv_token_num = torch.tensor([dispatch_a1.shape[0]])
        return (
            dispatch_a1,
            dispatch_weights,
            scale,
            dispatch_indices,
            recv_token_num,
        )

    def combine(self, fused_expert_output, weights, indices):
        self.combine_indices = indices
        if self.combine_output is not None:
            return (self.combine_output,)
        return (fused_expert_output,)


def test_mori_combine_uses_original_per_rank_topk_ids(monkeypatch):
    mori_module = _import_mori_prepare_finalize(monkeypatch)
    mori_op = _FakeMoriOp()
    adapter = mori_module.MoriPrepareAndFinalize(
        mori_op=mori_op,
        max_tokens_per_rank=16,
        num_dispatchers=1,
    )

    hidden_states = torch.randn(2, 4)
    router_topk_ids = torch.tensor([[0, 1], [2, 3]], dtype=torch.int32)
    topk_weights = torch.ones(2, 2)
    quant_config = SimpleNamespace(
        is_block_quantized=False,
        is_per_act_token=False,
    )

    (
        dispatch_a1,
        _,
        _,
        dispatch_ids,
        dispatch_weights,
    ) = adapter.prepare(
        hidden_states,
        topk_weights,
        router_topk_ids,
        num_experts=4,
        expert_map=None,
        apply_router_weight_on_input=False,
        quant_config=quant_config,
    )

    # Read the shapes before the identity asserts below, which narrow these
    # names to the fake op's Optional attribute type.
    router_shape = tuple(router_topk_ids.shape)
    dispatch_shape = tuple(dispatch_ids.shape)
    expected_rows = router_shape[0] * 16

    assert dispatch_shape[0] == expected_rows, (
        "prepare() must return the ids dispatch() produced: EP=16 expands "
        f"{router_shape[0]} tokens to {expected_rows} rows, but prepare() "
        f"returned {dispatch_shape[0]} rows"
    )
    assert mori_op.dispatch_indices is router_topk_ids, (
        "dispatch() must be handed this rank's own pre-dispatch routing of "
        f"shape {router_shape}, but it received "
        f"{_describe(mori_op.dispatch_indices)}"
    )

    output = torch.empty_like(hidden_states)
    adapter.finalize(
        output,
        dispatch_a1,
        dispatch_weights,
        dispatch_ids,
        apply_router_weight_on_input=False,
        weight_and_reduce_impl=SimpleNamespace(),
    )

    assert mori_op.combine_indices is router_topk_ids, (
        "combine() reduces over this rank's own routing, so it must be handed "
        f"the pre-dispatch tensor of shape {router_shape}, but it received "
        f"{_describe(mori_op.combine_indices)}. finalize() most likely "
        "forwarded its own topk_ids, which the modular kernel has already "
        "rebound to the ids dispatch() returned (ROCm/mori#475)"
    )
    assert mori_op.combine_indices is not dispatch_ids, (
        f"combine() was handed the dispatched ids of shape {dispatch_shape} "
        "instead of this rank's pre-dispatch routing, which silently corrupts "
        "the reduction (ROCm/mori#475)"
    )


def _prepared_mori_adapter(monkeypatch, combine_output, num_tokens):
    """Return an adapter whose fake combine() yields ``combine_output``, after
    prepare() on ``num_tokens`` tokens, with the inputs finalize() takes."""
    mori_module = _import_mori_prepare_finalize(monkeypatch)
    adapter = mori_module.MoriPrepareAndFinalize(
        mori_op=_FakeMoriOp(combine_output),
        max_tokens_per_rank=combine_output.shape[0],
        num_dispatchers=1,
    )
    hidden_states = torch.randn(
        num_tokens, combine_output.shape[1], dtype=combine_output.dtype
    )
    dispatch_a1, _, _, dispatch_ids, dispatch_weights = adapter.prepare(
        hidden_states,
        torch.ones(num_tokens, 2),
        torch.arange(2, dtype=torch.int32).repeat(num_tokens, 1),
        num_experts=4,
        expert_map=None,
        apply_router_weight_on_input=False,
        quant_config=FUSED_MOE_UNQUANTIZED_CONFIG,
    )
    return adapter, hidden_states, dispatch_a1, dispatch_weights, dispatch_ids


@pytest.mark.parametrize("num_tokens", [1, 48, 128, 224])
def test_mori_finalize_returns_combine_rows_without_copy(monkeypatch, num_tokens):
    """finalize() returns MoRI's combined rows instead of copying them.

    MoRI combines into a persistent buffer with max_tokens_per_rank rows. The
    result must equal copying its first num_tokens rows into output, alias the
    buffer, and leave output unwritten.
    """
    combine_output = torch.randn(256, 16, dtype=torch.bfloat16)
    adapter, hidden_states, dispatch_a1, dispatch_weights, dispatch_ids = (
        _prepared_mori_adapter(monkeypatch, combine_output, num_tokens)
    )
    output = torch.zeros_like(hidden_states)
    copied = output.clone().copy_(combine_output[:num_tokens])

    result = adapter.finalize(
        output,
        dispatch_a1,
        dispatch_weights,
        dispatch_ids,
        apply_router_weight_on_input=False,
        weight_and_reduce_impl=TopKWeightAndReduceNoOP(),
    )

    torch.testing.assert_close(result, copied, rtol=0, atol=0)
    assert result.data_ptr() == combine_output.data_ptr()
    assert not output.any()


def test_modular_kernel_returns_tensor_from_finalize(monkeypatch):
    """The modular kernel returns the tensor finalize() returns, else output.

    MoRI's finalize() leaves output unwritten and returns its combine rows;
    a finalize() that writes output returns None.
    """
    experts = TritonExperts(make_dummy_moe_config(), FUSED_MOE_UNQUANTIZED_CONFIG)
    combine_output = torch.randn(256, 16, dtype=torch.bfloat16)
    adapter, hidden_states, dispatch_a1, dispatch_weights, dispatch_ids = (
        _prepared_mori_adapter(monkeypatch, combine_output, num_tokens=48)
    )
    output = torch.zeros_like(hidden_states)

    result = FusedMoEKernelModularImpl(adapter, experts)._finalize(
        output,
        dispatch_a1,
        hidden_states,
        dispatch_weights,
        dispatch_ids,
        apply_router_weight_on_input=False,
        shared_experts=None,
        shared_experts_input=None,
    )

    assert result.data_ptr() == combine_output.data_ptr()
    torch.testing.assert_close(result, combine_output[:48], rtol=0, atol=0)
    assert not output.any()

    result = FusedMoEKernelModularImpl(
        MoEPrepareAndFinalizeNoDPEPModular(), experts
    )._finalize(
        output,
        hidden_states,
        hidden_states,
        torch.ones(48, 2),
        torch.arange(2, dtype=torch.int32).repeat(48, 1),
        apply_router_weight_on_input=False,
        shared_experts=None,
        shared_experts_input=None,
    )

    assert result is output
    torch.testing.assert_close(output, hidden_states, rtol=0, atol=0)
