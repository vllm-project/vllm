# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""NIXL receive-buffer ownership across overlapping microbatches."""

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


@pytest.mark.parametrize("k", [1, 2, 3, 4])
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
                patch.object(nixl_ep, "dbo_enabled", return_value=True),
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
@pytest.mark.parametrize("k", [2, 3, 4])
def test_nixl_microbatches_capture_and_replay(
    k, monkeypatch, world_size=1, rank=0, port=0
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
    from vllm.model_executor.layers.fused_moe.prepare_finalize.nixl_ep import (
        NixlEPPrepareAndFinalize,
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
    config = _make_dbo_config()
    config.parallel_config.enable_dbo = False
    config.parallel_config.ubatch_size = k
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
        buffer, 64, world_size, 8 * world_size, num_ubatches=k
    )

    class IdentityExperts(torch.nn.Module):
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
                dbo_maybe_run_recv_hook()
                hook, receiver = prepare.prepare_async(
                    x,
                    weights,
                    topk,
                    8 * world_size,
                    None,
                    False,
                    FusedMoEQuantConfig.make(),
                )
                dbo_register_recv_hook(hook)
                dbo_yield()
                expert_x, _, _, _, _ = receiver()
                output = torch.empty_like(x)
                hook, receiver = prepare.finalize_async(
                    output,
                    expert_x * 2 + 1,
                    weights,
                    topk,
                    False,
                    TopKWeightAndReduceDelegate(),
                )
                dbo_register_recv_hook(hook)
                dbo_yield()
                receiver()
                x = output
            return x

    try:
        n = 127
        inputs = _make_model_inputs(n, torch.device("cuda:0"))
        buffers = InputBuffers(n, n, torch.device("cuda:0"))
        inputs["input_ids"] = buffers.input_ids[:n]
        inputs["input_ids"].copy_(torch.arange(n, device="cuda"))
        inputs["input_ids"].add_(rank).remainder_(8)
        batch = InputBatch.make_dummy(n, n, InputBuffers(n, n, torch.device("cpu")))
        state = _make_ubatch_state(config, create_ubatch_slices(batch, k))
        model = IdentityExperts()
        for _ in range(2):
            output = runner.run(model, inputs, state)
            torch.testing.assert_close(
                output,
                (inputs["input_ids"][:, None] * 4 + 3)
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
        slots = torch.arange(n, device="cuda").unsqueeze(0)
        last_start = state.slices[-1].token_slice.start
        for step in range(4):
            real = [k, 97, last_start, last_start + 1][(step + rank) % 4]
            batch = _make_input_batch([1] * real, [100] * real, buffers, n, n)
            slots.copy_(torch.arange(n, device="cuda").unsqueeze(0))
            inputs["input_ids"].copy_(
                (torch.arange(n, device="cuda") + step + rank) % 8
            )
            expected = (inputs["input_ids"][:real, None] * 4 + 3).to(output.dtype)
            rows = (
                stage_decode_tokens(batch, (), slots, state.slices)
                if real <= last_start
                else None
            )
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


@pytest.mark.parametrize("configured_ubatches", [3, 4])
def test_non_microbatched_dispatch_does_not_snapshot(configured_ubatches):
    """A configured maximum must not add copies to a normal single-batch step."""
    pytest.importorskip("nixl_ep")
    from vllm.model_executor.layers.fused_moe.prepare_finalize import nixl_ep

    buffer = _TwoBufferDispatch()
    prepare = nixl_ep.NixlEPPrepareAndFinalize(
        buffer, 2, 1, 1, num_ubatches=configured_ubatches
    )
    with patch.object(nixl_ep, "dbo_enabled", return_value=False):
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
