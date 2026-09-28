# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""NIXL receive-buffer ownership across overlapping microbatches."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from vllm.model_executor.layers.fused_moe.config import FusedMoEQuantConfig


class _TwoBufferDispatch:
    """Overwrite aliased inputs and handle metadata only when receive completes."""

    def __init__(self):
        self.buffers = [
            (
                torch.empty(1, 2, 2048),
                torch.empty(1, 2, 16),
                torch.empty(1, dtype=torch.int32),
                torch.empty(2, dtype=torch.int32),
                torch.empty(2, dtype=torch.int64),
            )
            for _ in range(2)
        ]
        self.calls = 0

    def dispatch(self, x, topk_ids, max_tokens, experts, **kwargs):
        values, scales, counts, src, layout = self.buffers[self.calls % 2]
        self.calls += 1

        def receive():
            values.copy_(x.unsqueeze(0))
            scales.fill_(1)
            counts.fill_(self.calls)
            src.fill_(self.calls + 10)
            layout.fill_(self.calls + 20)

        return (
            (values, scales) if kwargs["use_fp8"] else values,
            counts,
            (src, layout, max_tokens, 2048),
            None,
            receive,
        )


@pytest.mark.parametrize("k", [1, 2, 3, 4, 8])
@pytest.mark.parametrize("fp8", [False, True])
def test_dispatch_owns_data_until_its_microbatch_consumes_it(k, fp8):
    pytest.importorskip("nixl_ep")
    from vllm.model_executor.layers.fused_moe.prepare_finalize import nixl_ep

    buffer = _TwoBufferDispatch()
    prepare = nixl_ep.NixlEPPrepareAndFinalize(
        buffer, 2, 1, 1, use_fp8_dispatch=fp8, num_ubatches=k
    )
    for step in range(3):
        receivers = []
        for mb in range(k):
            x = torch.full((2, 2048), step * k + mb + 1.0)
            with (
                patch.object(nixl_ep, "dbo_current_ubatch_id", return_value=mb),
                patch.object(nixl_ep, "dbo_num_ubatches", return_value=k),
            ):
                hook, receiver = prepare.prepare_async(
                    x,
                    torch.ones(2, 1),
                    torch.zeros(2, 1, dtype=torch.int64),
                    1,
                    None,
                    False,
                    FusedMoEQuantConfig.make(),
                )
            hook()
            receivers.append(receiver)

        for mb, receiver in enumerate(receivers):
            value = step * k + mb + 1
            x, _, metadata, _, _ = receiver()
            torch.testing.assert_close(x, torch.full_like(x, value))
            assert metadata.expert_num_tokens.item() == value
            src, layout, max_tokens, hidden = prepare.handles[mb]
            assert (src == value + 10).all()
            assert (layout == value + 20).all()
            assert (max_tokens, hidden) == (2, 2048)
            if k <= 2:
                # The original one/two-microbatch path remains allocation-free.
                original = buffer.buffers[(step * k + mb) % 2]
                assert src.data_ptr() == original[3].data_ptr()
                assert layout.data_ptr() == original[4].data_ptr()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="NIXL needs CUDA")
@pytest.mark.parametrize("k", [2, 3, 4, 8, (2, 3, 4, 8, 2)])
def test_nixl_microbatches_capture_and_replay(
    k, monkeypatch, world_size=1, rank=0, port=0, shared_aux=False
):
    """Real NIXL with production staging, threaded handoff and graph replay.

    Identity experts isolate dispatch/combine ownership from model numerics.
    Defaults to one EP rank; the standalone driver also runs two EP ranks.
    This does not exercise the DP scheduler or real Attention/model layers.
    """
    nixl_ep = pytest.importorskip("nixl_ep")
    from tests.v1.worker.test_gpu_ubatch_slicing import (
        _make_dbo_config,
        _make_execution_runner,
        _make_input_batch,
        _make_model_inputs,
        _make_ubatch_state,
    )
    from vllm.forward_context import DPMetadata
    from vllm.model_executor.layers.fused_moe.prepare_finalize.nixl_ep import (
        NixlEPPrepareAndFinalize,
    )
    from vllm.model_executor.layers.fused_moe.runner.shared_experts import (
        SharedExperts,
        SharedExpertsOrder,
    )
    from vllm.model_executor.layers.fused_moe.topk_weight_and_reduce import (
        TopKWeightAndReduceDelegate,
    )
    from vllm.v1.worker.gpu.input_batch import InputBatch, InputBuffers
    from vllm.v1.worker.gpu.ubatch_utils import (
        create_ubatch_slices,
        restore_staged_inputs,
        stage_decode_tokens,
    )
    from vllm.v1.worker.ubatching import (
        dbo_maybe_run_recv_hook,
        dbo_register_recv_hook,
        dbo_yield,
    )

    monkeypatch.setenv("VLLM_DBO_COMM_SMS", "0")
    monkeypatch.setenv("VLLM_DISABLE_SHARED_EXPERTS_STREAM", "0")
    counts = (k,) if isinstance(k, int) else k
    capacity = max(counts)
    config = _make_dbo_config()
    config.parallel_config.ubatch_size = capacity
    config.parallel_config.all2all_backend = "nixl_ep"
    runner = _make_execution_runner(config)
    store = torch.distributed.TCPStore("127.0.0.1", port, world_size, rank == 0)
    buffer = nixl_ep.Buffer(rank=rank, tcp_store_group=store, explicitly_destroy=True)
    buffer.update_memory_buffers(
        num_ranks=world_size,
        num_experts_per_rank=8,
        num_rdma_bytes=nixl_ep.Buffer.get_rdma_size_hint(
            64, 2048, world_size, 8 * world_size
        ),
    )
    buffer.connect_ranks(list(range(world_size)))
    prepare = NixlEPPrepareAndFinalize(
        buffer, 64, world_size, 8 * world_size, num_ubatches=capacity
    )

    class SharedIdentity(torch.nn.Module):
        def forward(self, x):
            return x + 0

    class IdentityExperts(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.shared = SharedExperts(
                SharedIdentity(),
                SimpleNamespace(
                    moe_parallel_config=SimpleNamespace(
                        enable_eplb=False,
                        use_fi_nvl_two_sided_kernels=False,
                    )
                ),
                lambda: not shared_aux,
                num_ubatches=capacity,
            )

        def forward(self, input_ids, **kwargs):
            x = input_ids[:, None].expand(-1, 2048).to(torch.bfloat16).contiguous()
            topk = (
                (
                    torch.arange(4, device=x.device, dtype=prepare.topk_indices_dtype())
                    * (world_size * 2)
                )
                .expand(x.shape[0], -1)
                .contiguous()
            )
            weights = torch.full(topk.shape, 0.25, device=x.device)
            for _ in range(2):
                if shared_aux:
                    assert self.shared.maybe_forward_async(x)
                else:
                    self.shared(x, SharedExpertsOrder.MK_INTERNAL_OVERLAPPED)
                dbo_maybe_run_recv_hook()
                result = prepare.prepare_async(
                    x,
                    weights,
                    topk,
                    8 * world_size,
                    None,
                    False,
                    FusedMoEQuantConfig.make(),
                )
                if isinstance(result, tuple):
                    hook, receiver = result
                    dbo_register_recv_hook(hook)
                    dbo_yield()
                else:
                    receiver = result
                expert_x, _, _, _, _ = receiver()
                assert expert_x.shape[1] == x.shape[0] * world_size
                output = torch.empty_like(x)
                result = prepare.finalize_async(
                    output,
                    expert_x * 2 + 1,
                    weights,
                    topk,
                    False,
                    TopKWeightAndReduceDelegate(),
                )
                if isinstance(result, tuple):
                    hook, receiver = result
                    dbo_register_recv_hook(hook)
                    dbo_yield()
                else:
                    receiver = result
                receiver()
                if shared_aux:
                    self.shared.wait()
                x = output + self.shared.output
            return x

    try:
        n = 127
        inputs = _make_model_inputs(n, torch.device("cuda:0"))
        buffers = InputBuffers(n, n, torch.device("cuda:0"))
        inputs["input_ids"] = buffers.input_ids[:n]
        inputs["input_ids"].copy_(torch.arange(n, device="cuda"))
        inputs["input_ids"].add_(rank).remainder_(8)
        batch = InputBatch.make_dummy(n, n, InputBuffers(n, n, torch.device("cpu")))
        model = IdentityExperts()
        graphs = {}
        for active in dict.fromkeys(counts):
            state = _make_ubatch_state(config, create_ubatch_slices(batch, active))
            for ctx, execution_slice in zip(state.forward_contexts, state.slices):
                ctx.dp_metadata = DPMetadata(
                    torch.full((world_size,), execution_slice.num_tokens)
                )
            for _ in range(2):
                output = runner.run(model, inputs, state)
                torch.testing.assert_close(
                    output,
                    (inputs["input_ids"][:, None] * 9 + 4)
                    .expand_as(output)
                    .to(output.dtype),
                    rtol=0,
                    atol=0,
                )
            torch.accelerator.synchronize()
            graph = torch.cuda.CUDAGraph()
            finish = runner.begin_capturable_run(model, inputs, state, for_capture=True)
            with torch.cuda.graph(graph, stream=runner.capture_stream):
                output = finish()
            graphs[active] = graph, output, state
        slots = torch.arange(n, device="cuda").unsqueeze(0)
        for step in range(4 * len(counts)):
            active = counts[step % len(counts)]
            graph, output, state = graphs[active]
            last_start = state.slices[-1].token_slice.start
            real = [active, 97, last_start, last_start + 1][(step + rank) % 4]
            batch = _make_input_batch([1] * real, [100] * real, buffers, n, n)
            slots.copy_(torch.arange(n, device="cuda").unsqueeze(0))
            inputs["input_ids"].copy_(
                (torch.arange(n, device="cuda") + step + rank) % 8
            )
            expected = (inputs["input_ids"][:real, None] * 9 + 4).to(output.dtype)
            if real <= last_start:
                rows = stage_decode_tokens(batch, (), slots, state.slices)
            else:
                rows = None
            graph.replay()
            if rows is not None:
                output[:real] = output[rows]
                restore_staged_inputs(batch, (), slots, rows)
            torch.accelerator.synchronize()
            torch.testing.assert_close(
                output[:real],
                expected.expand_as(output[:real]),
                rtol=0,
                atol=0,
            )
    finally:
        torch.accelerator.synchronize()
        buffer.destroy()


@pytest.mark.parametrize("configured_ubatches", [3, 4, 8])
@pytest.mark.parametrize("active_ubatches", [1, 2])
def test_small_active_count_does_not_snapshot(configured_ubatches, active_ubatches):
    """Spare handle capacity must not add copies to a one/two-batch step."""
    pytest.importorskip("nixl_ep")
    from vllm.model_executor.layers.fused_moe.prepare_finalize import nixl_ep

    buffer = _TwoBufferDispatch()
    prepare = nixl_ep.NixlEPPrepareAndFinalize(
        buffer, 2, 1, 1, num_ubatches=configured_ubatches
    )
    with patch.object(nixl_ep, "dbo_num_ubatches", return_value=active_ubatches):
        hook, receiver = prepare.prepare_async(
            torch.ones(2, 2048),
            torch.ones(2, 1),
            torch.zeros(2, 1, dtype=torch.int64),
            1,
            None,
            False,
            FusedMoEQuantConfig.make(),
        )
    hook()
    x, _, metadata, _, _ = receiver()
    original_x, _, original_counts, src, layout = buffer.buffers[0]
    assert x.data_ptr() == original_x.data_ptr()
    assert metadata.expert_num_tokens.data_ptr() == original_counts.data_ptr()
    assert prepare.handles[0][0].data_ptr() == src.data_ptr()
    assert prepare.handles[0][1].data_ptr() == layout.data_ptr()


@pytest.mark.parametrize(
    "bucket,k,slice_sizes",
    [
        (128, 1, [128]),
        (128, 2, [64, 64]),
        (256, 1, [256]),
        (256, 2, [128, 128]),
        (128, 3, [42, 42, 44]),
        (256, 4, [64, 64, 64, 64]),
    ],
)
def test_runtime_dispatch_uses_synchronized_execution_slices(bucket, k, slice_sizes):
    """Imbalanced ranks must dispatch identical capacities and preserve routing."""
    pytest.importorskip("nixl_ep")
    import numpy as np

    from vllm.config import CUDAGraphMode
    from vllm.forward_context import (
        BatchDescriptor,
        DPMetadata,
        ForwardContext,
        override_forward_context,
    )
    from vllm.model_executor.layers.fused_moe.prepare_finalize import nixl_ep
    from vllm.model_executor.layers.fused_moe.topk_weight_and_reduce import (
        TopKWeightAndReduceDelegate,
    )
    from vllm.v1.worker.gpu.ubatch_utils import UBatchRunner, create_ubatch_slices

    class IdentityDispatch:
        def __init__(self):
            self.capacities: list[int] = []

        def dispatch(self, x, ids, capacity, experts, **kwargs):
            self.capacities.append(capacity)
            self.ids = ids.clone()
            self.x = x.clone()
            recv = torch.zeros(1, capacity, x.shape[1])
            recv[0, : len(x)] = x
            return recv, torch.tensor([len(x)]), (capacity,), None, lambda: None

        def combine(self, x, ids, weights, handle, **kwargs):
            assert x.shape == (1, handle[0], self.x.shape[1])
            assert x.is_contiguous()
            torch.testing.assert_close(ids, self.ids)
            kwargs["out"].copy_(x[0, : len(ids)] * weights)
            return kwargs["out"], None, lambda: None

    loads = [63, 97, 80, 80, 80, 80, 80, 80]
    for rank, real_tokens in enumerate(loads):
        if k == 1:
            contexts = [
                ForwardContext(
                    {},
                    {},
                    {},
                    DPMetadata(torch.full((len(loads),), bucket)),
                    cudagraph_runtime_mode=CUDAGraphMode.FULL,
                    batch_descriptor=BatchDescriptor(bucket),
                )
            ]
        else:
            batch = SimpleNamespace(
                num_scheduled_tokens=np.ones(real_tokens, dtype=np.int32),
                num_tokens_after_padding=bucket,
                num_reqs_after_padding=bucket,
                num_reqs=real_tokens,
            )
            slices = create_ubatch_slices(batch, k)
            config = SimpleNamespace(
                compilation_config=SimpleNamespace(
                    fast_moe_cold_start=False, static_forward_context={}
                ),
            )
            runner = SimpleNamespace(
                vllm_config=config,
                parallel_config=SimpleNamespace(
                    data_parallel_size=len(loads),
                    data_parallel_rank=rank,
                    is_moe_model=True,
                ),
            )
            contexts = UBatchRunner._make_forward_contexts(
                runner, slices, [{}] * k, [{}] * k, [None] * k
            )
        buffer = IdentityDispatch()
        physical_ids = torch.tensor(
            [2, 0, 3, 1], dtype=nixl_ep.NIXL_EP_TOPK_INDICES_DTYPE
        )
        prepare = nixl_ep.NixlEPPrepareAndFinalize(
            buffer,
            8192,
            1,
            4,
            num_ubatches=k,
            global_to_physical=physical_ids,
        )
        for mb, (ctx, num_tokens) in enumerate(zip(contexts, slice_sizes)):
            ids = torch.arange(num_tokens).remainder(4).view(-1, 1)
            original_ids = ids.clone()
            x = torch.arange(num_tokens, dtype=torch.float32).view(-1, 1)
            x = x.expand(-1, 2048).contiguous()
            weights = torch.full((num_tokens, 1), 0.5)
            with (
                override_forward_context(ctx),
                patch.object(nixl_ep, "dbo_current_ubatch_id", return_value=mb),
            ):
                recv, _, _, _, _ = prepare.prepare(
                    x, weights, ids, 1, None, False, FusedMoEQuantConfig.make()
                )
                out = torch.empty_like(x)
                prepare.finalize(
                    out, recv, weights, ids, False, TopKWeightAndReduceDelegate()
                )
            torch.testing.assert_close(out, x * weights)
            torch.testing.assert_close(ids, original_ids)
            torch.testing.assert_close(buffer.ids, physical_ids[original_ids])
        # A single NIXL dispatcher rounds its receive extent up to 4 rows.
        assert buffer.capacities == [(size + 3) // 4 * 4 for size in slice_sizes]
        assert prepare.max_num_tokens_per_rank() == 8192


@pytest.mark.parametrize(
    "sizes,sp,sp_expected,direct_expected",
    [
        ([63, 97, 80], 1, 100, 100),
        ([128, 128], 2, 64, 128),
        ([1, 1], 1, 2, 2),
    ],
)
def test_runtime_dispatch_uses_global_max_and_existing_sp_sizes(
    sizes, sp, sp_expected, direct_expected
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

    meta = DPMetadata(torch.tensor(sizes))
    prepare = NixlEPPrepareAndFinalize(None, 8192, len(sizes), 1)
    with override_forward_context(ForwardContext({}, {}, {}, meta)):
        with meta.sp_local_sizes(sp):
            assert prepare._get_runtime_dispatch_capacity() == sp_expected
        assert prepare._get_runtime_dispatch_capacity() == direct_expected
        prepare.max_tokens_per_rank = direct_expected - 1
        with pytest.raises(AssertionError):
            prepare._get_runtime_dispatch_capacity()


@pytest.mark.parametrize("runtime_m", [512, 1024, 2048, 8192 * 8])
def test_batched_output_matches_dispatch_without_shrinking_reservation(runtime_m):
    """Combine needs a contiguous runtime view; workspaces retain static bounds."""
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
    )
    experts.workspace_shapes = lambda *args: BatchedTritonExperts.workspace_shapes(
        experts, *args
    )
    kernel = SimpleNamespace(fused_experts=experts)
    manager = WorkspaceManager(torch.device("cpu"))
    with patch.object(mk, "current_workspace_manager", return_value=manager):
        workspace13, workspace2, output = (
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
