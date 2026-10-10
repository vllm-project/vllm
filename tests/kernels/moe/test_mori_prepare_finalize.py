# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import importlib
import sys
import types
from types import SimpleNamespace

import torch


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
    def __init__(self):
        self.dispatch_indices = None
        self.combine_indices = None

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
