# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""NIXL EP execution-capacity regression tests."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch


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
