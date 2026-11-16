# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The attention-feed-forward-overlap DBO schedule in the V2 model runner.

Covers what differs from dispatch-combine-overlap: the microbatching decision
on a single DP rank (no handshake, the thresholds decide) and the green-context
pools the microbatches run on. The slicing and threading are shared and are
tested in test_gpu_ubatch_slicing.py.
"""

from types import SimpleNamespace
from typing import Any, NamedTuple, cast
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
    dbo_yield,
    dbo_yield_and_switch_from_comm_to_compute,
    dbo_yield_and_switch_from_compute_to_comm,
)

DECODE_THRESHOLD = 32
PREFILL_THRESHOLD = 128


class _Streams(NamedTuple):
    """The streams a microbatch ran on: on entry, during attention, after it."""

    entry: torch.cuda.Stream
    attn: torch.cuda.Stream
    rest: torch.cuda.Stream


def _make_config(**parallel_kwargs: Any) -> VllmConfig:
    parallel_kwargs.setdefault("dbo_schedule", "attention-feed-forward-overlap")
    return VllmConfig(
        model_config=ModelConfig(model="facebook/opt-125m", dtype="float16", seed=0),
        parallel_config=ParallelConfig(
            enable_dbo=True,
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
            decode_query_lens={1: vllm_config.scheduler_config.max_num_seqs},
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
    """Toy layer: a pre-attention stage, attention (bracketed by the
    attention-feed-forward-overlap handoffs, as in gpt-oss), then the rest.

    With `hook_streams` the rest stage also plays the MoE's all-to-all protocol:
    run the peer's pending receive hook, register its own and yield. Each hook
    records the stream it ran on under its microbatch's id.
    """

    def __init__(
        self,
        streams: dict[int, _Streams],
        hook_streams: dict[int, torch.cuda.Stream] | None = None,
    ):
        super().__init__()
        self.streams = streams
        self.hook_streams = hook_streams

    def forward(self, input_ids, positions, intermediate_tensors=None, **kwargs):
        entry_stream = current_stream()
        out = input_ids.float().unsqueeze(-1)
        dbo_yield_and_switch_from_comm_to_compute("attention-feed-forward-overlap")
        attn_stream = current_stream()
        out = out * 2
        dbo_yield_and_switch_from_compute_to_comm("attention-feed-forward-overlap")
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
        self.streams[dbo_current_ubatch_id()] = _Streams(
            entry_stream, attn_stream, rest_stream
        )
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
) -> dict[int, _Streams]:
    """Run the toy model; return the streams each microbatch ran on."""
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
    streams: dict[int, _Streams] = {}
    model = _TwoStageModel(streams, hook_streams)
    output = runner.run(model, model_inputs, ubatch_state)
    torch.accelerator.synchronize()

    expected = model_inputs["input_ids"].float().unsqueeze(-1) * 2 + model_inputs[
        "positions"
    ].float().unsqueeze(-1)
    torch.testing.assert_close(output, expected)
    assert set(streams) == {0, 1}
    return streams


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_microbatches_run_on_the_pools():
    """Attention shares the attention pool's stream; everything else of a
    microbatch, from its first kernel on, runs on its own feed-forward pool
    stream."""
    pytest.importorskip("cuda.bindings")
    total_sms = torch.cuda.get_device_properties(0).multi_processor_count
    vllm_config = _make_config(dbo_attention_sms=8)
    runner = _make_runner(vllm_config)

    pools = runner.sm_pools
    assert pools is not None
    assert (pools.attn_sms, pools.rest_sms) == (8, total_sms - 8)
    assert runner.capture_stream == pools.attn_stream
    assert runner.comm_streams == pools.rest_streams
    assert len(pools.rest_streams) == 2
    assert pools.rest_streams[0] != pools.rest_streams[1]

    by_ubatch = _run_two_stage_model(runner, vllm_config)
    for ubatch_id, (entry, attn, rest) in by_ubatch.items():
        assert attn == pools.attn_stream
        assert entry == rest == pools.rest_streams[ubatch_id]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_receive_hooks_run_on_the_registering_microbatch_stream():
    """A receive hook is run by the peer microbatch (at its next all-to-all or
    on exit) but completes the registering microbatch's receive: it must run on
    that microbatch's feed-forward stream, ahead of the work that consumes the
    received data."""
    pytest.importorskip("cuda.bindings")
    vllm_config = _make_config(dbo_attention_sms=8)
    runner = _make_runner(vllm_config)
    pools = runner.sm_pools
    assert pools is not None

    hook_streams: dict[int, torch.cuda.Stream] = {}
    by_ubatch = _run_two_stage_model(runner, vllm_config, hook_streams)
    assert set(hook_streams) == {0, 1}
    for ubatch_id, (_, _, rest) in by_ubatch.items():
        assert hook_streams[ubatch_id] == rest
        assert hook_streams[ubatch_id] != pools.attn_stream


# dispatch-combine-overlap needs a supported all2all backend even without
# expert parallelism; these tests build the runner directly.
_DCO = {
    "dbo_schedule": "dispatch-combine-overlap",
    "all2all_backend": "deepep_low_latency",
}


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_one_rank_forward_contexts_carry_no_dp_metadata():
    """DP metadata describes the ranks' token counts; a single rank has none."""
    vllm_config = _make_config(**_DCO)
    runner = _make_runner(vllm_config)
    slices = [
        UBatchSlice(slice(0, 4), slice(0, 8)),
        UBatchSlice(slice(4, 8), slice(8, 16)),
    ]
    contexts = runner._make_forward_contexts(slices, [{}, {}], [{}, {}], [None, None])
    assert [ctx.dp_metadata for ctx in contexts] == [None, None]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_dispatch_combine_overlap_ignores_attention_handoffs():
    """The default schedule keeps the whole GPU: no pools, and handoffs tagged
    with attention-feed-forward-overlap neither switch streams nor yield."""
    vllm_config = _make_config(**_DCO)
    runner = _make_runner(vllm_config)
    assert runner.sm_pools is None
    assert runner.comm_streams is None

    hook_streams: dict[int, torch.cuda.Stream] = {}
    by_ubatch = _run_two_stage_model(runner, vllm_config, hook_streams)
    for ubatch_id, (entry, attn, rest) in by_ubatch.items():
        assert entry == attn == rest
        assert hook_streams[ubatch_id] == attn
