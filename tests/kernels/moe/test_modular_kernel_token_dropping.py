# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Routing and output-layout contracts for eager MoE token dropping."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.model_executor.layers.fused_moe.config import FusedMoEQuantConfig
from vllm.model_executor.layers.fused_moe.prepare_finalize.no_dp_ep import (
    MoEPrepareAndFinalizeNoDPEPModular,
)
from vllm.model_executor.layers.fused_moe.topk_weight_and_reduce import (
    TopKWeightAndReduceNoOP,
)
from vllm.platforms import current_platform


def make_kernel(monkeypatch, capacity, prepare_finalize=None):
    experts = SimpleNamespace(
        moe_config=SimpleNamespace(
            expert_capacity=capacity,
            is_lora_enabled=False,
            moe_parallel_config=None,
        ),
        expects_unquantized_inputs=False,
        quant_config=FusedMoEQuantConfig.make(None),
        finalize_weight_and_reduce_impl=TopKWeightAndReduceNoOP,
    )
    kernel = mk.FusedMoEKernelModularImpl(
        prepare_finalize or MoEPrepareAndFinalizeNoDPEPModular(), experts
    )
    observed: dict[str, torch.Tensor] = {}

    def expert_output(**kwargs):
        states, ids, weights = (
            kwargs["a1q"],
            kwargs["topk_ids"],
            kwargs["topk_weights"],
        )
        observed.update(states=states, ids=ids, weights=weights)
        factors = (ids + 1).clamp_min(0).to(states.dtype)
        if not kwargs["apply_router_weight_on_input"]:
            factors = factors * weights
        return states * factors.sum(dim=1, keepdim=True)

    monkeypatch.setattr(kernel, "_fused_experts", expert_output)
    return kernel, observed


@pytest.mark.parametrize("capacity", [None, 0, 1, 2, 8])
@pytest.mark.parametrize("ids_dtype", [torch.int32, torch.int64])
def test_dropping_keeps_best_routes_without_compacting_states(
    monkeypatch, capacity, ids_dtype
):
    kernel, observed = make_kernel(monkeypatch, capacity)
    states = torch.arange(1, 13, dtype=torch.float32).reshape(4, 3)
    ids = torch.tensor([[0, 1], [0, 1], [0, 1], [0, -1]], dtype=ids_dtype)
    weights = torch.tensor([[0.1, 0.2], [0.9, 0.1], [0.9, 0.8], [0.05, 1.0]])
    originals = [t.clone() for t in (states, ids, weights)]
    factors = {
        None: [0.5, 1.1, 2.5, 0.05],
        0: [0.0, 0.0, 0.0, 0.0],
        1: [0.0, 0.9, 1.6, 0.0],
        2: [0.4, 0.9, 2.5, 0.0],
        8: [0.5, 1.1, 2.5, 0.05],
    }[capacity]

    output = kernel.apply(
        states, torch.empty(2, 1, 1), torch.empty(2, 1, 1), ids, weights
    )

    torch.testing.assert_close(output, states * torch.tensor(factors).unsqueeze(1))
    for tensor, original in zip((states, ids, weights), originals):
        torch.testing.assert_close(tensor, original)
    assert observed["states"] is states
    if capacity == 1:
        # The equal-weight route on row 1 wins over row 2; neither is renormalized.
        torch.testing.assert_close(
            observed["ids"], ids.new_tensor([[-1, -1], [0, -1], [-1, 1], [-1, -1]])
        )
        torch.testing.assert_close(
            observed["weights"],
            weights.new_tensor([[0.0, 0.0], [0.9, 0.0], [0.0, 0.8], [0.0, 0.0]]),
        )


@pytest.mark.parametrize("apply_on_input", [False, True])
@pytest.mark.parametrize("num_tokens", [0, 3])
def test_dropping_top1_preserves_weight_application(
    monkeypatch, apply_on_input, num_tokens
):
    kernel, _ = make_kernel(monkeypatch, 1)
    states = torch.ones(num_tokens, 2)
    ids = torch.zeros(num_tokens, 1, dtype=torch.int64)
    weights = torch.tensor([[0.2], [0.8], [0.1]])[:num_tokens]
    output = kernel.apply(
        states,
        torch.empty(1, 1, 1),
        torch.empty(1, 1, 1),
        ids,
        weights,
        apply_router_weight_on_input=apply_on_input,
    )
    expected = torch.zeros_like(states)
    if num_tokens:
        expected[1] = 0.8
    torch.testing.assert_close(output, expected)


def test_dropping_preserves_rows_after_async_finalize_and_shared_input(monkeypatch):
    class AsyncPrepareFinalize(MoEPrepareAndFinalizeNoDPEPModular):
        def supports_async(self):
            return True

        def prepare_async(self, *args, **kwargs):
            return lambda: self.prepare(*args, **kwargs)

        def finalize_async(self, *args, **kwargs):
            def receiver():
                self.finalize(*args, **kwargs)
                # A combine backend may leave fully dropped rows unwritten.
                args[0][[0, 2]] = float("nan")

            return receiver

    kernel, _ = make_kernel(monkeypatch, 1, AsyncPrepareFinalize())
    states = torch.ones(3, 2)
    shared_input = torch.ones(3, 4)
    shared = Mock()
    output = kernel.apply(
        states,
        torch.empty(1, 1, 1),
        torch.empty(1, 1, 1),
        torch.zeros(3, 1, dtype=torch.int64),
        torch.tensor([[0.2], [0.8], [0.1]]),
        shared_experts=shared,
        shared_experts_input=shared_input,
    )
    torch.testing.assert_close(
        output, torch.tensor([[0.0, 0.0], [0.8, 0.8], [0.0, 0.0]])
    )
    assert shared.call_args.args[0] is shared_input


def test_unsupported_dispatch_leaves_routing_unchanged(monkeypatch):
    prepare_finalize = MoEPrepareAndFinalizeNoDPEPModular()
    prepare_finalize.supports_token_dropping = False
    kernel, observed = make_kernel(monkeypatch, 1, prepare_finalize)
    states = torch.ones(3, 2)
    ids = torch.zeros(3, 1, dtype=torch.int64)
    weights = torch.tensor([[0.2], [0.8], [0.1]])
    output = kernel.apply(
        states, torch.empty(1, 1, 1), torch.empty(1, 1, 1), ids, weights
    )
    torch.testing.assert_close(output, states * weights)
    assert observed["ids"] is ids
    assert observed["weights"] is weights


def test_dropping_rejects_compilation(monkeypatch):
    kernel, _ = make_kernel(monkeypatch, 1)
    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: True)
    with pytest.raises(RuntimeError, match="eager execution"):
        kernel.apply(
            torch.ones(1, 2),
            torch.empty(1, 1, 1),
            torch.empty(1, 1, 1),
            torch.zeros(1, 1, dtype=torch.int64),
            torch.ones(1, 1),
        )


def test_batched_dispatch_uses_expert_capacity_without_changing_rank_limit(monkeypatch):
    from vllm.model_executor.layers.fused_moe.prepare_finalize.batched import (
        BatchedPrepareAndFinalize,
    )

    prepare_finalize = BatchedPrepareAndFinalize(8, 2, 1, 0)
    kernel, _ = make_kernel(monkeypatch, 1, prepare_finalize)
    prepare_finalize.post_init_setup(kernel.fused_experts)
    states, _, metadata, _, _ = prepare_finalize.prepare(
        torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
        torch.tensor([[0.9, 0.0], [0.0, 0.8]]),
        torch.tensor([[0, -1], [-1, 1]]),
        2,
        None,
        False,
        kernel.fused_experts.quant_config,
        False,
    )
    torch.testing.assert_close(states, torch.tensor([[[1.0, 2.0]], [[3.0, 4.0]]]))
    torch.testing.assert_close(
        metadata.expert_num_tokens, torch.ones(2, dtype=torch.int)
    )
    assert prepare_finalize.max_num_tokens_per_rank() == 8


@pytest.mark.parametrize("batched", [False, True])
@pytest.mark.parametrize("capacity", [0, 1, 2])
@pytest.mark.skipif(not current_platform.is_cuda_alike(), reason="Requires a GPU")
def test_dropping_triton_matches_retained_expert_contributions(
    batched, capacity, workspace_init
):
    from tests.kernels.moe.utils import make_dummy_moe_config
    from vllm.model_executor.layers.fused_moe.experts.fused_batched_moe import (
        BatchedTritonExperts,
    )
    from vllm.model_executor.layers.fused_moe.experts.triton_moe import TritonExperts
    from vllm.model_executor.layers.fused_moe.prepare_finalize.batched import (
        BatchedPrepareAndFinalize,
    )

    config = make_dummy_moe_config(
        num_experts=2,
        experts_per_token=2,
        hidden_dim=128,
        intermediate_size=128,
        in_dtype=torch.float32,
        max_num_tokens=4,
    )
    config.expert_capacity = capacity
    if batched:
        prepare_finalize = BatchedPrepareAndFinalize(4, 2, 1, 0)
        experts = BatchedTritonExperts(config, FusedMoEQuantConfig.make(None), 4, 1)
    else:
        prepare_finalize = MoEPrepareAndFinalizeNoDPEPModular()
        experts = TritonExperts(config, FusedMoEQuantConfig.make(None))
    kernel = mk.FusedMoEKernel(prepare_finalize, experts)
    torch.manual_seed(0)
    states = torch.randn(4, 128, device="cuda") / 10
    w1 = torch.randn(2, 256, 128, device="cuda") / 10
    w2 = torch.randn(2, 128, 128, device="cuda") / 10
    ids = torch.tensor([[0, 1]] * 4, device="cuda")
    weights = torch.tensor(
        [[0.1, 0.2], [0.9, 0.1], [0.9, 0.8], [0.05, 0.05]], device="cuda"
    )
    expected = torch.zeros_like(states)
    for expert, rows in enumerate(([1, 2], [2, 0])):
        for row in rows[:capacity]:
            gate, up = (states[row] @ w1[expert].T).chunk(2)
            expert_output = (torch.nn.functional.silu(gate) * up) @ w2[expert].T
            expected[row] += expert_output * weights[row, expert]
    output = kernel.apply(
        states, w1, w2, weights, ids, config.activation, 2, None, False
    )
    torch.testing.assert_close(output, expected, atol=1e-4, rtol=1e-3)
    if batched:
        assert prepare_finalize.max_num_tokens_per_rank() == 4
        shapes = experts.workspace_shapes(
            max(1, capacity),
            256,
            128,
            2,
            2,
            2,
            None,
            config.activation,
        )
        assert shapes[2][1] == 4


@pytest.mark.parametrize("expert_name", ["naive", "triton", "deep_gemm", "marlin"])
@pytest.mark.parametrize("capacity", [None, 7])
@pytest.mark.parametrize("dispatched_tokens", [1, 32, 64, 513])
def test_batched_workspaces_follow_dispatch_layout(
    expert_name, capacity, dispatched_tokens
):
    """Capacity may shrink dispatch, but padding must still fit every workspace."""
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation
    from vllm.model_executor.layers.fused_moe.experts.batched_deep_gemm_moe import (
        BatchedDeepGemmExperts,
    )
    from vllm.model_executor.layers.fused_moe.experts.fused_batched_moe import (
        BatchedTritonExperts,
        NaiveBatchedExperts,
    )
    from vllm.model_executor.layers.fused_moe.experts.marlin_moe import (
        BatchedMarlinExperts,
    )

    cls = {
        "naive": NaiveBatchedExperts,
        "triton": BatchedTritonExperts,
        "deep_gemm": BatchedDeepGemmExperts,
        "marlin": BatchedMarlinExperts,
    }[expert_name]
    experts = object.__new__(cls)
    experts.max_num_tokens = 128
    experts.num_dispatchers = 4
    experts.expert_capacity = capacity
    workspace13, workspace2, output = experts.workspace_shapes(
        dispatched_tokens,
        64,
        32,
        2,
        4,
        4,
        None,
        MoEActivation.SILU,
    )
    rows = max(512, dispatched_tokens)
    assert output == (4, rows, 32)
    if expert_name == "naive":
        assert workspace13 == (4, rows, 32)
        scratch_rows = rows if capacity is None else min(rows, capacity * 4)
        assert workspace2 == (scratch_rows, 64)
    elif expert_name == "marlin":
        assert workspace13 == (4 * rows, 128)
        assert workspace2 == (4 * rows, 64)
    else:
        assert workspace13 == (4, rows, 64)
        assert workspace2 == (4, rows, 32)


@pytest.mark.parametrize("capacity", [None, 7])
def test_workspace_allocation_preserves_full_output_shape(monkeypatch, capacity):
    kernel, _ = make_kernel(monkeypatch, capacity)
    shapes = Mock(side_effect=lambda M, *args, **kwargs: ((0,), (0,), (M, 4)))
    kernel.fused_experts.workspace_shapes = shapes
    kernel.fused_experts.workspace_dtype = lambda dtype: dtype
    monkeypatch.setattr(current_platform, "is_cpu", lambda: True)
    _, _, output = kernel._allocate_buffers(
        torch.float32,
        torch.device("cpu"),
        16,
        32,
        8,
        4,
        2,
        4,
        4,
        None,
        mk.MoEActivation.SILU,
    )
    assert output.shape == (32, 4)
    assert [call.args[0] for call in shapes.call_args_list] == [16, 32]
