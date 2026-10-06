# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Routing and output-layout contracts for eager MoE token dropping."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.model_executor.layers.fused_moe.config import FusedMoEQuantConfig
from vllm.model_executor.layers.fused_moe.prepare_finalize.batched_compaction import (
    BatchedExpertCompaction,
)
from vllm.model_executor.layers.fused_moe.prepare_finalize.no_dp_ep import (
    MoEPrepareAndFinalizeNoDPEPModular,
)
from vllm.model_executor.layers.fused_moe.prepare_finalize.standard_compaction import (
    StandardTokenRowCompaction,
)
from vllm.model_executor.layers.fused_moe.topk_weight_and_reduce import (
    TopKWeightAndReduceNoOP,
)
from vllm.platforms import current_platform
from vllm.utils.deep_gemm import is_deep_gemm_supported


def make_kernel(monkeypatch, capacity, prepare_finalize=None):
    if prepare_finalize is None:
        prepare_finalize = MoEPrepareAndFinalizeNoDPEPModular(expert_capacity=capacity)
    experts = SimpleNamespace(
        moe_config=SimpleNamespace(
            expert_capacity=capacity,
            is_lora_enabled=False,
            moe_parallel_config=None,
        ),
        expects_unquantized_inputs=False,
        quant_config=FusedMoEQuantConfig.make(None),
        finalize_weight_and_reduce_impl=TopKWeightAndReduceNoOP,
        activation_format=lambda: mk.FusedMoEActivationFormat.Standard,
    )
    kernel = mk.FusedMoEKernelModularImpl(
        prepare_finalize,
        experts,
    )
    if (
        kernel.expert_capacity is not None
        and prepare_finalize.activation_format == experts.activation_format()
    ):
        kernel._configure_compaction()
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
def test_dropping_compacts_fully_dropped_standard_rows(
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
    expected_states = {
        None: states,
        0: states[[], :],
        1: states[[1, 2]],
        2: states[[0, 1, 2]],
        8: states,
    }[capacity]
    torch.testing.assert_close(observed["states"], expected_states)
    if capacity == 1:
        # The equal-weight route on row 1 wins over row 2; neither is renormalized.
        torch.testing.assert_close(
            observed["ids"],
            ids.new_tensor([[0, -1], [-1, 1]]),
        )
        torch.testing.assert_close(
            observed["weights"],
            weights.new_tensor([[0.9, 0.0], [0.0, 0.8]]),
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


@pytest.mark.parametrize("capacity", [0, 1])
def test_dropping_preserves_rows_after_async_finalize_and_shared_input(
    monkeypatch, capacity
):
    class AsyncPrepareFinalize(MoEPrepareAndFinalizeNoDPEPModular):
        def supports_async(self):
            return True

        def prepare_async(self, *args, **kwargs):
            return lambda: self.prepare(*args, **kwargs)

        def finalize_async(self, *args, **kwargs):
            def receiver():
                self.finalize(*args, **kwargs)
                # A combine backend may leave fully dropped rows unwritten.
                if capacity == 0:
                    args[0].fill_(float("nan"))
                else:
                    args[0][[0, 2]] = float("nan")

            return receiver

    kernel, _ = make_kernel(
        monkeypatch, capacity, AsyncPrepareFinalize(expert_capacity=capacity)
    )
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
    expected = torch.zeros_like(states)
    if capacity:
        expected[1] = 0.8
    torch.testing.assert_close(output, expected)
    assert shared.call_args.args[0] is shared_input


def test_unsupported_dispatch_leaves_routing_unchanged(monkeypatch):
    prepare_finalize = MoEPrepareAndFinalizeNoDPEPModular(expert_capacity=1)
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


def test_standard_compaction_copies_output_when_not_compacted():
    compaction = StandardTokenRowCompaction()
    output = torch.zeros(3, 2)
    source = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])

    receiver = compaction.scatter_or_copy_output(output, source, 0)
    receiver()

    torch.testing.assert_close(output, source)


@pytest.mark.parametrize("supports_dropping", [False, True])
def test_dispatch_support_controls_batched_physical_capacity(supports_dropping):
    """Compaction updates the physical row limit used by workspaces."""
    from tests.kernels.moe.utils import make_dummy_moe_config
    from vllm.model_executor.layers.fused_moe.experts.fused_batched_moe import (
        BatchedTritonExperts,
    )
    from vllm.model_executor.layers.fused_moe.prepare_finalize.batched import (
        BatchedPrepareAndFinalize,
    )

    config = make_dummy_moe_config(
        num_experts=2,
        experts_per_token=2,
        hidden_dim=128,
        intermediate_size=128,
        in_dtype=torch.float32,
        max_num_tokens=128,
    )
    config.expert_capacity = 7
    dispatcher = BatchedPrepareAndFinalize(128, 2, 1, 0, expert_capacity=7)
    dispatcher.supports_token_dropping = supports_dropping
    experts = BatchedTritonExperts(config, FusedMoEQuantConfig.make(None), 128, 1)
    mk.FusedMoEKernel(dispatcher, experts)
    rows = experts.max_num_tokens
    assert rows is not None
    scratch13, scratch2, output = experts.workspace_shapes(
        rows, 256, 128, 2, 2, 2, None, config.activation
    )
    assert scratch13 == (2, rows, 256)
    assert scratch2 == (2, rows, 128)
    assert output == (2, rows, 128)


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

    prepare_finalize = BatchedPrepareAndFinalize(8, 2, 1, 0, expert_capacity=1)
    kernel, _ = make_kernel(monkeypatch, 1, prepare_finalize)
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
        prepare_finalize = BatchedPrepareAndFinalize(
            4, 2, 1, 0, expert_capacity=capacity
        )
        experts = BatchedTritonExperts(config, FusedMoEQuantConfig.make(None), 4, 1)
    else:
        prepare_finalize = MoEPrepareAndFinalizeNoDPEPModular(expert_capacity=capacity)
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
        assert shapes[2][1] == max(1, capacity)


@pytest.mark.parametrize("expert_name", ["naive", "triton", "deep_gemm", "marlin"])
@pytest.mark.parametrize("capacity", [None, 7])
@pytest.mark.parametrize("dispatched_tokens", [4, 32, 64, 512])
def test_batched_workspaces_follow_dispatch_layout(
    expert_name, capacity, dispatched_tokens
):
    """Workspace padding follows the configured physical dispatch layout."""
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
    experts.max_num_tokens = dispatched_tokens // 4
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
    rows = dispatched_tokens
    scratch_rows = rows
    assert output == (4, rows, 32)
    if expert_name == "naive":
        assert workspace13 == (4, rows, 32)
        assert workspace2 == (scratch_rows, 64)
    elif expert_name == "marlin":
        assert workspace13 == (4 * scratch_rows, 128)
        assert workspace2 == (4 * scratch_rows, 64)
    else:
        assert workspace13 == (4, scratch_rows, 64)
        assert workspace2 == (4, scratch_rows, 32)


@pytest.mark.parametrize("dispatched_tokens", [1, 513])
def test_batched_workspaces_reject_dispatch_layout_mismatch(dispatched_tokens):
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation
    from vllm.model_executor.layers.fused_moe.experts.fused_batched_moe import (
        BatchedTritonExperts,
    )

    experts = object.__new__(BatchedTritonExperts)
    experts.max_num_tokens = 128
    experts.num_dispatchers = 4
    with pytest.raises(AssertionError, match="dispatched layout"):
        experts.workspace_shapes(
            dispatched_tokens,
            64,
            32,
            2,
            4,
            4,
            None,
            MoEActivation.SILU,
        )


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


def test_workspace_tracking_counts_shared_storage_once(monkeypatch):
    """Views share backing bytes; smaller requests must not reset the peak."""
    kernel, _ = make_kernel(monkeypatch, 128)
    log = Mock()
    monkeypatch.setattr(mk.logger, "info", log)
    storage = torch.empty(128, dtype=torch.float32)
    kernel._record_workspace_usage(storage[:32], storage[64:96], storage[:16], 4, 8)
    assert kernel._workspace_peaks == (128, 128, 64, 256, 512)
    kernel._record_workspace_usage(storage[:16], storage[64:80], storage[:8], 2, 4)
    assert kernel._workspace_peaks == (128, 128, 64, 256, 512)
    assert log.call_count == 1
    separate = torch.empty(256, dtype=torch.float32)
    kernel._record_workspace_usage(storage[:32], separate, storage[:16], 4, 8)
    assert kernel._workspace_peaks == (128, 1024, 64, 1152, 1536)
    assert log.call_count == 2


@pytest.mark.parametrize("expert_name", ["triton", "deep_gemm", "humming"])
@pytest.mark.parametrize("packed_ue8m0", [False, True])
def test_batched_compaction_handles_fp8_dispatch_scale_layout(
    expert_name, packed_ue8m0
):
    """Quantized dispatch payloads compact values and scales together.

    Humming receives row-major scales; the other kernels retain the native
    dispatch stride, including packed UE8M0's TMA-friendly layout.
    """
    from vllm.model_executor.layers.fused_moe.experts.batched_deep_gemm_moe import (
        BatchedDeepGemmExperts,
    )
    from vllm.model_executor.layers.fused_moe.experts.fused_batched_moe import (
        BatchedTritonExperts,
    )
    from vllm.model_executor.layers.fused_moe.experts.fused_humming_moe import (
        BatchedHummingGroupedExperts,
    )

    experts = object.__new__(
        {
            "triton": BatchedTritonExperts,
            "deep_gemm": BatchedDeepGemmExperts,
            "humming": BatchedHummingGroupedExperts,
        }[expert_name]
    )
    experts.expert_capacity = 2
    experts.max_num_tokens = 8
    experts.num_dispatchers = 2
    experts.quant_config = SimpleNamespace(is_quantized=True, use_fp8_w8a8=True)

    compaction = BatchedExpertCompaction()
    use_row_major_dispatch_scales = experts.use_row_major_dispatch_scales
    tokens_per_expert = compaction.configure(2, 8, 2, True, experts, True)
    assert tokens_per_expert == 2
    assert experts.max_num_tokens == 8

    num_experts, full_rows = 2, 16
    hidden_dim = 512
    scale_columns = hidden_dim // (512 if packed_ue8m0 else 128)
    values = (
        torch.arange(num_experts * full_rows * hidden_dim, dtype=torch.float32)
        .reshape(num_experts, full_rows, hidden_dim)
        .to(torch.float8_e4m3fn)
    )
    scale_storage = torch.arange(
        num_experts * full_rows * scale_columns, dtype=torch.int32
    )
    scales = scale_storage.as_strided(
        (num_experts, full_rows, scale_columns),
        (full_rows * scale_columns, 1, full_rows),
    )

    compact_values, compact_scales = compaction.compact((values, scales), 0)

    assert compact_values.shape == (num_experts, 4, hidden_dim)
    assert compact_values.is_contiguous()
    assert compact_scales.shape == (num_experts, 4, scale_columns)
    torch.testing.assert_close(compact_scales, scales[:, :4])
    expected_stride = (
        (4 * scale_columns, scale_columns, 1)
        if use_row_major_dispatch_scales
        else (4 * scale_columns, 1, 4)
    )
    assert compact_scales.stride() == expected_stride

    restored, receiver = compaction.restore(compact_values, 0)
    assert restored.shape == values.shape
    assert torch.equal(restored[:, :4], compact_values)
    receiver()


@pytest.mark.skipif(
    not current_platform.is_cuda() or not is_deep_gemm_supported(),
    reason="Requires CUDA and DeepGEMM",
)
def test_batched_compaction_feeds_deep_gemm_fp8(workspace_init):
    """DeepGEMM accepts the compacted FP8 values and scale layout."""
    from tests.kernels.moe.utils import (
        make_dummy_moe_config,
        make_test_weights,
        per_token_cast_to_fp8,
    )
    from vllm.model_executor.layers.fused_moe.config import (
        fp8_w8a8_moe_quant_config,
    )
    from vllm.model_executor.layers.fused_moe.experts.batched_deep_gemm_moe import (
        BatchedDeepGemmExperts,
    )
    from vllm.utils.deep_gemm import get_mk_alignment_for_contiguous_layout

    num_experts, full_rows, rows, hidden_dim, intermediate_dim = 2, 16, 4, 512, 128
    block_shape = get_mk_alignment_for_contiguous_layout()
    config = make_dummy_moe_config(
        num_experts=num_experts,
        experts_per_token=2,
        hidden_dim=hidden_dim,
        intermediate_size=intermediate_dim,
        in_dtype=torch.bfloat16,
        max_num_tokens=8,
    )
    config.expert_capacity = 2
    (_, w1, w1_scale, _), (_, w2, w2_scale, _) = make_test_weights(
        num_experts,
        intermediate_dim,
        hidden_dim,
        torch.bfloat16,
        torch.float8_e4m3fn,
        block_shape=block_shape,
    )
    quant_config = fp8_w8a8_moe_quant_config(
        w1_scale=w1_scale,
        w2_scale=w2_scale,
        block_shape=block_shape,
    )
    experts = BatchedDeepGemmExperts(config, quant_config, 8, 2)
    experts.expert_capacity = 2

    compaction = BatchedExpertCompaction()
    experts.max_num_tokens = compaction.configure(2, 8, 2, True, experts, True)
    states = (
        torch.randn(
            num_experts, full_rows, hidden_dim, device="cuda", dtype=torch.bfloat16
        )
        / 10
    )
    values, scales = per_token_cast_to_fp8(states.view(-1, hidden_dim))
    values = values.view(num_experts, full_rows, hidden_dim)
    scales = scales.view(num_experts, full_rows, hidden_dim // 128)
    compact_values, compact_scales = compaction.compact((values, scales), 0)

    expert_num_tokens = torch.full(
        (num_experts,), rows, device="cuda", dtype=torch.int32
    )
    metadata = mk.ExpertTokensMetadata(
        expert_num_tokens=expert_num_tokens,
        expert_num_tokens_cpu=expert_num_tokens.cpu(),
    )
    workspace13_shape, workspace2_shape, output_shape = experts.workspace_shapes(
        rows,
        w1.size(1),
        hidden_dim,
        2,
        num_experts,
        num_experts,
        metadata,
        config.activation,
    )
    output = torch.empty(output_shape, device="cuda", dtype=torch.bfloat16)
    experts.apply(
        output=output,
        hidden_states=compact_values,
        w1=w1,
        w2=w2,
        topk_weights=torch.ones(rows, 2, device="cuda"),
        topk_ids=torch.zeros(rows, 2, device="cuda", dtype=torch.int64),
        activation=config.activation,
        global_num_experts=num_experts,
        expert_map=None,
        a1q_scale=compact_scales,
        a2_scale=None,
        workspace13=torch.empty(workspace13_shape, device="cuda", dtype=torch.bfloat16),
        workspace2=torch.empty(workspace2_shape, device="cuda", dtype=torch.bfloat16),
        expert_tokens_meta=metadata,
        apply_router_weight_on_input=False,
    )

    assert output.shape == (num_experts, rows, hidden_dim)
    assert torch.isfinite(output).all()


@pytest.mark.parametrize("backend", ["nixl", "deepep_ll"])
@pytest.mark.parametrize("do_async", [False, True])
@pytest.mark.parametrize("capacity", [None, 0, 1, 2, 8])
def test_batched_compaction_restores_combine_layout(
    monkeypatch, backend, do_async, capacity
):
    """Both dispatchers restore their layout before sync or async combine."""
    pytest.importorskip("nixl_ep" if backend == "nixl" else "deep_ep")
    from tests.kernels.moe.utils import make_dummy_moe_config
    from vllm.model_executor.layers.fused_moe.experts.fused_batched_moe import (
        BatchedTritonExperts,
    )

    if backend == "nixl":
        from vllm.model_executor.layers.fused_moe.prepare_finalize import nixl_ep as pf
    else:
        from vllm.model_executor.layers.fused_moe.prepare_finalize import (
            deepep_ll as pf,
        )
    from vllm.model_executor.layers.fused_moe.topk_weight_and_reduce import (
        TopKWeightAndReduceDelegate,
    )

    config = make_dummy_moe_config(
        num_experts=4,
        experts_per_token=2,
        hidden_dim=128,
        intermediate_size=128,
        in_dtype=torch.bfloat16,
        max_num_tokens=8,
    )
    config.expert_capacity = capacity
    quant = FusedMoEQuantConfig.make(None)
    experts = BatchedTritonExperts(config, quant, 8, 2)
    experts.expert_capacity = capacity
    buffer = Mock()
    if backend == "nixl":
        dispatcher = pf.NixlEPPrepareAndFinalize(
            buffer, 8, 2, 4, expert_capacity=capacity
        )
        combine = buffer.combine
    else:
        dispatcher = pf.DeepEPLLPrepareAndFinalize(
            buffer, 8, 2, expert_capacity=capacity
        )
        combine = buffer.low_latency_combine
    compact_tokens = dispatcher._configure_batched_compaction(
        capacity,
        experts,
    )
    if compact_tokens is not None:
        experts.max_num_tokens = compact_tokens
    dispatcher.post_init_setup(experts)
    received = torch.arange(2 * 16 * 128).reshape(2, 16, 128).to(torch.bfloat16)
    counts = torch.tensor([4, 1] if capacity is None else [min(4, capacity * 2), 0])
    rows = 16 if capacity is None else max(1, min(8, capacity)) * 2
    assert dispatcher.max_num_tokens_per_rank() == 8
    assert experts.max_num_tokens == (8 if capacity is None else max(1, capacity))

    # Prepare two outstanding microbatches before finalizing either one.
    for slot in (0, 1):
        compact, _, metadata, _, _ = dispatcher._receiver(
            received + slot, counts, None, torch.bfloat16, quant, slot
        )
        torch.testing.assert_close(compact, (received + slot)[:, :rows])
        assert compact.is_contiguous()
        assert metadata.expert_num_tokens is counts
        dispatcher.handles[slot] = (slot,)
    shapes = experts.workspace_shapes(
        rows, 256, 128, 2, 4, 2, metadata, config.activation
    )
    assert shapes[2] == (2, rows, 128)

    for slot in (1, 0):
        monkeypatch.setattr(pf, "dbo_current_ubatch_id", lambda slot=slot: slot)
        monkeypatch.setattr(pf, "dbo_enabled", lambda: False)
        monkeypatch.setattr(pf, "dbo_maybe_run_recv_hook", lambda: None)
        hook = Mock()
        combine.return_value = (None, None, hook)
        compact_output = (received + slot)[:, :rows].contiguous()
        finalize = dispatcher.finalize_async if do_async else dispatcher.finalize
        finalize_result = finalize(
            torch.empty(3, 128),
            compact_output,
            torch.ones(3, 2),
            torch.zeros(3, 2, dtype=torch.int64),
            False,
            TopKWeightAndReduceDelegate(),
        )
        restored = combine.call_args.args[0]
        assert restored.shape == received.shape
        torch.testing.assert_close(restored[:, :rows], compact_output)
        assert combine.call_args.args[3] == (slot,)
        if do_async:
            recv_hook, receiver = finalize_result
            recv_hook()
            receiver()
            hook.assert_called_once()
        else:
            hook.assert_not_called()
        restored_again, receiver = dispatcher._compaction.restore(restored, slot)
        assert restored_again is restored
        receiver()


@pytest.mark.parametrize("ids_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("weights_dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("shape", [(7, 3), (256, 8)])
@pytest.mark.skipif(not current_platform.is_cuda(), reason="Requires CUDA")
def test_drop_tokens_triton_matches_reference(
    monkeypatch, ids_dtype, weights_dtype, shape
):
    """The fused CUDA path preserves the reference capacity ordering."""
    kernel, _ = make_kernel(monkeypatch, 2)
    torch.manual_seed(1234)
    ids = torch.randint(-1, 12, shape, device="cuda", dtype=ids_dtype)
    weights = torch.rand(shape, device="cuda", dtype=weights_dtype)
    weights[weights < 0.1] = 0.25
    expected = kernel._drop_tokens_reference(ids, weights)

    def fail_reference(*args, **kwargs):
        raise AssertionError("Expected the Triton token-dropping path")

    monkeypatch.setattr(kernel, "_drop_tokens_reference", fail_reference)
    actual = kernel._drop_tokens(ids, weights)

    assert torch.equal(actual[0], expected[0])
    torch.testing.assert_close(actual[1], expected[1])
    assert torch.equal(actual[2], expected[2])


@pytest.mark.parametrize("capacity", [0, 2])
@pytest.mark.skipif(not current_platform.is_cuda(), reason="Requires CUDA graphs")
def test_dropping_cuda_graph_replay_uses_current_routes(monkeypatch, capacity):
    """Capture must not freeze the selected assignments or all-dropped mask."""
    kernel, _ = make_kernel(monkeypatch, capacity)
    ids = torch.zeros((8, 2), device="cuda", dtype=torch.int64)
    weights = torch.zeros((8, 2), device="cuda")
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            kernel._drop_tokens(ids, weights)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        captured = kernel._drop_tokens(ids, weights)
    for reverse in (False, True):
        routes = torch.arange(16, device="cuda").reshape(8, 2) % 3
        scores = torch.arange(16, device="cuda", dtype=torch.float32).reshape(8, 2)
        if reverse:
            routes = routes.flip(0)
            scores = scores.flip(0)
        ids.copy_(routes)
        weights.copy_(scores)
        expected = kernel._drop_tokens(ids, weights)
        graph.replay()
        for actual, reference in zip(captured, expected):
            torch.testing.assert_close(actual, reference)
