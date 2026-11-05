# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SubGPU pipelining in the V2 model runner.

Covers what differs from dual batch overlap: the microbatching decision on a
single DP rank (no handshake, the thresholds decide) and the green-context
pools the microbatches run on. The slicing and threading are DBO's and are
tested in test_gpu_ubatch_slicing.py.
"""

from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import patch

import pytest
import torch

from vllm.config import (
    CompilationConfig,
    CUDAGraphMode,
    ModelConfig,
    ParallelConfig,
    VllmConfig,
)
from vllm.forward_context import create_forward_context
from vllm.utils.torch_utils import current_stream
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.worker.gpu import cudagraph_utils, dp_utils
from vllm.v1.worker.gpu.cudagraph_utils import CudaGraphManager
from vllm.v1.worker.gpu.model_states.interface import ModelState
from vllm.v1.worker.gpu.ubatch_utils import UBatchRunner, UBatchState
from vllm.v1.worker.ubatch_utils import UBatchSlice
from vllm.v1.worker.ubatching import (
    dbo_current_ubatch_id,
    dbo_maybe_run_recv_hook,
    dbo_register_recv_hook,
    dbo_subgpu_ensure_attn,
    dbo_subgpu_ensure_rest,
    dbo_yield,
)

DECODE_THRESHOLD = 32
PREFILL_THRESHOLD = 128


def _make_config(**parallel_kwargs: Any) -> VllmConfig:
    return VllmConfig(
        model_config=ModelConfig(model="facebook/opt-125m", dtype="float16", seed=0),
        parallel_config=ParallelConfig(
            subgpu_pipelining=True,
            dbo_decode_token_threshold=DECODE_THRESHOLD,
            dbo_prefill_token_threshold=PREFILL_THRESHOLD,
            **parallel_kwargs,
        ),
        compilation_config=CompilationConfig(
            cudagraph_mode=CUDAGraphMode.FULL, cudagraph_capture_sizes=[64, 128]
        ),
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a graph pool")
def test_one_rank_microbatches_by_the_thresholds():
    """Without other ranks the thresholds decide, and only a captured graph is taken."""
    vllm_config = _make_config()
    with patch.object(
        cudagraph_utils,
        "get_pp_group",
        lambda: SimpleNamespace(is_first_rank=True, is_last_rank=True),
    ):
        manager = CudaGraphManager(
            vllm_config,
            torch.device("cuda:0"),
            CUDAGraphMode.FULL,
            decode_query_len=1,
            ubatch_runner=cast(UBatchRunner, object()),
        )
    manager._graphs_captured = True

    def dispatch(num_tokens: int, uniform: bool = True, **kwargs: Any):
        desc, dp_sync_state = dp_utils.dispatch_cg_and_sync_dp(
            manager,
            num_reqs=num_tokens if uniform else 8,
            num_tokens=num_tokens,
            uniform_token_count=1 if uniform else None,
            dp_size=1,
            dp_rank=0,
            parallel_config=vllm_config.parallel_config,
            allow_ubatching=kwargs.pop("allow_ubatching", True),
            uniform_decode=uniform,
            **kwargs,
        )
        assert dp_sync_state is None
        return desc

    desc = dispatch(64)
    assert desc.cg_mode == CUDAGraphMode.FULL
    assert desc.num_ubatches == 2

    # Below the decode threshold the batch runs whole.
    desc = dispatch(16)
    assert desc.cg_mode == CUDAGraphMode.FULL
    assert desc.num_ubatches == 1

    # A mixed batch below the prefill threshold too.
    assert dispatch(64, uniform=False).num_ubatches == 1

    assert dispatch(64, allow_ubatching=False).num_ubatches == 1

    desc = dispatch(64, need_eager=True)
    assert desc.cg_mode == CUDAGraphMode.NONE
    assert desc.num_ubatches == 1


class _TwoStageModel(torch.nn.Module):
    """Toy layer: an attention stage on the attention pool, the rest elsewhere.

    With `hook_streams` the rest stage also plays the MoE's all-to-all protocol:
    run the peer's pending receive hook, register its own and yield. Each hook
    records the stream it ran on under its microbatch's id.
    """

    def __init__(
        self,
        streams: list[tuple[int, torch.cuda.Stream, torch.cuda.Stream]],
        hook_streams: dict[int, torch.cuda.Stream] | None = None,
    ):
        super().__init__()
        self.streams = streams
        self.hook_streams = hook_streams

    def forward(self, input_ids, positions, intermediate_tensors=None, **kwargs):
        dbo_subgpu_ensure_attn()
        attn_stream = current_stream()
        out = input_ids.float().unsqueeze(-1) * 2
        dbo_subgpu_ensure_rest()
        rest_stream = current_stream()
        out = out + positions.float().unsqueeze(-1)
        if self.hook_streams is not None:
            dbo_maybe_run_recv_hook()
            ubatch_id = dbo_current_ubatch_id()
            hook_streams = self.hook_streams

            def hook() -> None:
                hook_streams[ubatch_id] = current_stream()

            dbo_register_recv_hook(hook)
            dbo_yield()
        # Microbatches must end on the stream they started on.
        dbo_subgpu_ensure_attn()
        self.streams.append((dbo_current_ubatch_id(), attn_stream, rest_stream))
        return out


def _make_runner(vllm_config: VllmConfig) -> UBatchRunner:
    try:
        return UBatchRunner(
            vllm_config,
            torch.device("cuda:0"),
            model_state=cast(ModelState, None),
            attn_groups=[],
            kv_cache_config=cast(KVCacheConfig, None),
            max_num_reqs=32,
        )
    except RuntimeError as e:
        if "CUDA_ERROR_NOT_SUPPORTED" in str(e):
            pytest.skip(f"green contexts are not available on this device: {e}")
        raise


def _run_two_stage_model(
    runner: UBatchRunner,
    vllm_config: VllmConfig,
    hook_streams: dict[int, torch.cuda.Stream] | None = None,
) -> dict[int, tuple[torch.cuda.Stream, torch.cuda.Stream]]:
    device = torch.device("cuda:0")
    num_tokens = 16
    model_inputs = {
        "input_ids": torch.arange(num_tokens, device=device),
        "positions": torch.arange(num_tokens, device=device),
        "inputs_embeds": None,
        "intermediate_tensors": None,
    }
    ubatch_state = UBatchState(
        slices=[
            UBatchSlice(slice(0, 4), slice(0, 8)),
            UBatchSlice(slice(4, 8), slice(8, 16)),
        ],
        forward_contexts=[create_forward_context(None, vllm_config) for _ in range(2)],
    )
    streams: list[tuple[int, torch.cuda.Stream, torch.cuda.Stream]] = []
    output = runner.run(
        _TwoStageModel(streams, hook_streams), model_inputs, ubatch_state
    )
    torch.cuda.synchronize()

    expected = model_inputs["input_ids"].float().unsqueeze(-1) * 2 + model_inputs[
        "positions"
    ].float().unsqueeze(-1)
    torch.testing.assert_close(output, expected)
    by_ubatch = {ubatch_id: (attn, rest) for ubatch_id, attn, rest in streams}
    assert set(by_ubatch) == {0, 1}
    return by_ubatch


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_microbatches_run_on_the_pools():
    """Attention stages share the attention pool's stream; each microbatch's
    rest stage has its own stream in the rest pool."""
    pytest.importorskip("cuda.bindings")
    total_sms = torch.cuda.get_device_properties(0).multi_processor_count
    vllm_config = _make_config(subgpu_attn_sms=8, subgpu_rest_sms=total_sms - 8)
    runner = _make_runner(vllm_config)

    pools = runner.subgpu_pools
    assert pools is not None
    assert (pools.attn_sms, pools.rest_sms) == (8, total_sms - 8)
    assert runner.capture_stream == pools.attn_stream
    assert runner.comm_streams == pools.rest_streams
    assert len(pools.rest_streams) == 2
    assert pools.rest_streams[0] != pools.rest_streams[1]

    by_ubatch = _run_two_stage_model(runner, vllm_config)
    assert all(attn == pools.attn_stream for attn, _ in by_ubatch.values())
    assert by_ubatch[0][1] == pools.rest_streams[0]
    assert by_ubatch[1][1] == pools.rest_streams[1]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_receive_hooks_run_on_the_registering_microbatch_stream():
    """A receive hook is run by the peer microbatch (at its next all-to-all or
    on exit, when it is back on the attention pool) but completes the
    registering microbatch's receive: it must run on that microbatch's rest
    stream, ahead of the work that consumes the received data."""
    pytest.importorskip("cuda.bindings")
    total_sms = torch.cuda.get_device_properties(0).multi_processor_count
    vllm_config = _make_config(subgpu_attn_sms=8, subgpu_rest_sms=total_sms - 8)
    runner = _make_runner(vllm_config)
    pools = runner.subgpu_pools
    assert pools is not None

    hook_streams: dict[int, torch.cuda.Stream] = {}
    by_ubatch = _run_two_stage_model(runner, vllm_config, hook_streams)
    assert set(hook_streams) == {0, 1}
    for ubatch_id, (_, rest) in by_ubatch.items():
        assert hook_streams[ubatch_id] == rest
        assert hook_streams[ubatch_id] != pools.attn_stream


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_one_rank_forward_contexts_carry_no_dp_metadata():
    """DP metadata describes the ranks' token counts; a single rank has none."""
    vllm_config = _make_config(subgpu_schedule="dbo")
    runner = _make_runner(vllm_config)
    slices = [
        UBatchSlice(slice(0, 4), slice(0, 8)),
        UBatchSlice(slice(4, 8), slice(8, 16)),
    ]
    contexts = runner._make_forward_contexts(
        slices, [None, None], [{}, {}], [None, None]
    )
    assert [ctx.dp_metadata for ctx in contexts] == [None, None]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_dbo_schedule_keeps_the_whole_gpu():
    """The baseline schedule microbatches without pools or stage switches."""
    vllm_config = _make_config(subgpu_schedule="dbo")
    runner = _make_runner(vllm_config)
    assert runner.subgpu_pools is None
    assert runner.comm_streams is None

    hook_streams: dict[int, torch.cuda.Stream] = {}
    by_ubatch = _run_two_stage_model(runner, vllm_config, hook_streams)
    for ubatch_id, (attn, rest) in by_ubatch.items():
        assert attn == rest
        assert hook_streams[ubatch_id] == attn


def test_subgpu_deepep_v2_buffer_per_microbatch(monkeypatch):
    """With micro-batching, DeepEP v2 dispatches and combines on the buffer of
    the current micro-batch; with a single buffer the index is ignored."""
    pytest.importorskip("deep_ep")
    from vllm.model_executor.layers.fused_moe.prepare_finalize import (
        deepep_v2 as mod,
    )

    b0, b1 = object(), object()
    pf = mod.DeepEPV2PrepareAndFinalize(
        buffer=b0,
        num_dispatchers=1,
        dp_size=1,
        rank_expert_offset=0,
        num_experts=4,
        num_topk=2,
        buffers=[b0, b1],
    )
    assert len(pf.handles) == 2
    monkeypatch.setattr(mod, "dbo_current_ubatch_id", lambda: 0)
    assert pf._ubatch_buffer() is b0
    monkeypatch.setattr(mod, "dbo_current_ubatch_id", lambda: 1)
    assert pf._ubatch_buffer() is b1

    single = mod.DeepEPV2PrepareAndFinalize(
        buffer=b0,
        num_dispatchers=1,
        dp_size=1,
        rank_expert_offset=0,
        num_experts=4,
        num_topk=2,
    )
    assert single.buffers == [b0]
    assert single._ubatch_buffer() is b0


def test_subgpu_deepep_v2_manager_caches_one_buffer_per_microbatch(monkeypatch):
    """`get_handle` keys the ElasticBuffer cache by the micro-batch index and
    does not pass the index to the buffer constructor."""
    deep_ep = pytest.importorskip("deep_ep")
    from vllm.distributed.device_communicators import all2all as mod

    made = []

    class FakeBuffer:
        def __init__(self, **kw):
            self.kw = kw
            made.append(self)

        def get_theoretical_num_sms(self, **kw):
            return 1

    monkeypatch.setattr(deep_ep, "ElasticBuffer", FakeBuffer)
    mgr = mod.DeepEPV2All2AllManager.__new__(mod.DeepEPV2All2AllManager)
    mgr.handle_cache = mod.Cache()
    mgr._num_sms = None
    mgr._gin_checked = True
    monkeypatch.setattr(
        mgr,
        "_make_all2all_kwargs",
        lambda **kw: {"hidden": kw["hidden"], "num_topk": kw["num_topk"]},
    )
    args = {
        "num_max_tokens_per_rank": 8,
        "hidden": 16,
        "num_topk": 2,
        "use_fp8_dispatch": False,
        "num_experts": 4,
    }
    h0 = mgr.get_handle(args)
    h1 = mgr.get_handle(dict(args, ubatch_idx=1))
    assert h0 is not h1
    assert mgr.get_handle(dict(args, ubatch_idx=0)) is h0
    assert mgr.get_handle(dict(args, ubatch_idx=1)) is h1
    assert len(made) == 2
    assert all("ubatch_idx" not in b.kw for b in made)
