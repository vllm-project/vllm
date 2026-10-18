# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""NIXL EP execution-capacity regression tests."""

from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from vllm.model_executor.layers.fused_moe.config import FusedMoEQuantConfig


def test_dbo_nixl_dispatch_and_combine_use_stream_handoffs():
    pytest.importorskip("nixl_ep")
    from vllm.model_executor.layers.fused_moe.prepare_finalize import nixl_ep
    from vllm.model_executor.layers.fused_moe.topk_weight_and_reduce import (
        TopKWeightAndReduceDelegate,
    )

    events = []

    class Buffer:
        def dispatch(self, x, ids, capacity, experts, **kwargs):
            events.append(("dispatch", kwargs["return_recv_hook"]))
            return x.unsqueeze(0), torch.tensor([len(x)]), (capacity,), None, None

        def combine(self, x, ids, weights, handle, **kwargs):
            events.append(("combine", kwargs["return_recv_hook"]))
            return kwargs["out"], None, None

    prepare = nixl_ep.NixlEPPrepareAndFinalize(Buffer(), 64, 1, 1)
    switches = {
        "dbo_yield_and_switch_from_compute_to_comm": "compute_to_comm",
        "dbo_switch_to_compute": "compute",
        "dbo_switch_to_comm": "comm",
        "dbo_yield_and_switch_from_comm_to_compute": "comm_to_compute",
    }
    with ExitStack() as stack:
        stack.enter_context(patch.object(nixl_ep, "dbo_enabled", return_value=True))
        stack.enter_context(
            patch.object(nixl_ep, "dbo_current_ubatch_id", return_value=0)
        )
        for name, key in switches.items():
            stack.enter_context(
                patch.object(
                    nixl_ep, name, side_effect=lambda event=key: events.append(event)
                )
            )
        x = torch.ones(2, 2048)
        ids = torch.zeros(2, 1, dtype=torch.int64)
        weights = torch.ones(2, 1)
        result = prepare.prepare_async(
            x, weights, ids, 1, None, False, FusedMoEQuantConfig.make()
        )
        assert callable(result)
        received, _, _, _, _ = result()
        result = prepare.finalize_async(
            torch.empty_like(x),
            received,
            weights,
            ids,
            False,
            TopKWeightAndReduceDelegate(),
        )
        assert callable(result)
        result()

    assert events == [
        "compute_to_comm",
        ("dispatch", False),
        "compute",
        "comm",
        "comm_to_compute",
        "compute_to_comm",
        ("combine", False),
        "compute",
        "comm",
        "comm_to_compute",
    ]


@pytest.mark.parametrize(
    "sizes,dispatchers,sp,sp_expected,direct_expected",
    [
        ([63, 97, 80], 1, 1, 100, 100),
        ([128, 128], 2, 2, 64, 128),
        ([1, 1], 1, 2, 4, 4),
    ],
)
def test_runtime_capacity_uses_synchronized_execution_extent(
    sizes, dispatchers, sp, sp_expected, direct_expected
):
    pytest.importorskip("nixl_ep")
    from vllm.forward_context import (
        DPMetadata,
        ForwardContext,
        override_forward_context,
    )
    from vllm.model_executor.layers.fused_moe.prepare_finalize.nixl_ep import (
        NixlEPPrepareAndFinalize,
    )

    metadata = DPMetadata(torch.tensor(sizes))
    prepare = NixlEPPrepareAndFinalize(None, 8192, dispatchers, 1)
    with override_forward_context(ForwardContext({}, {}, {}, metadata)):
        with metadata.sp_local_sizes(sp):
            assert prepare._get_runtime_dispatch_capacity() == sp_expected
        assert prepare._get_runtime_dispatch_capacity() == direct_expected
        prepare.max_tokens_per_rank = direct_expected - 1
        with pytest.raises(AssertionError):
            prepare._get_runtime_dispatch_capacity()


@pytest.mark.parametrize("runtime_m", [512, 1024, 2048, 8192 * 8])
def test_batched_output_matches_dispatch_without_shrinking_reservation(runtime_m):
    from vllm.model_executor.layers.fused_moe import modular_kernel as mk
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation
    from vllm.model_executor.layers.fused_moe.experts.fused_batched_moe import (
        BatchedTritonExperts,
    )
    from vllm.v1.worker.workspace import WorkspaceManager

    experts = SimpleNamespace(
        num_dispatchers=8,
        max_num_tokens=8192,
        adjust_N_for_activation=lambda n, activation: n // 2,
        workspace_dtype=lambda dtype: dtype,
        activation_format=BatchedTritonExperts.activation_format,
        finalize_weight_and_reduce_impl=lambda: None,
    )
    experts.workspace_shapes = lambda *args: BatchedTritonExperts.workspace_shapes(
        experts, *args
    )
    kernel = SimpleNamespace(
        fused_experts=experts,
        prepare_finalize=SimpleNamespace(
            allocate_fused_expert_output=lambda *args: None
        ),
    )
    manager = WorkspaceManager(torch.device("cpu"))
    with patch.object(mk, "current_workspace_manager", return_value=manager):
        workspace13, workspace2, output, aliased = (
            mk.FusedMoEKernelModularImpl._allocate_buffers(
                kernel,
                torch.float32,
                torch.device("cpu"),
                runtime_m,
                runtime_m,
                8,
                4,
                1,
                8,
                2,
                None,
                MoEActivation.SILU,
            )
        )
    assert workspace13.shape == (2, 8192 * 8, 8)
    assert workspace2.shape == (2, 8192 * 8, 4)
    assert output.shape == (2, runtime_m, 4)
    assert output.is_contiguous()
    assert output.data_ptr() == workspace13.data_ptr()
    assert not aliased
