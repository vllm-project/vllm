# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Dual batch overlap (DBO) in the V2 model runner.

The V2 runner slices the batch *before* building attention metadata, where V1
builds metadata for the whole batch and slices it afterwards. Both must describe
the same microbatches, so the main test here pins V2's slicing against V1's
`split_attn_metadata`. The rest covers the decision to microbatch and the
threaded execution of the microbatches.
"""

import threading
from contextlib import nullcontext
from dataclasses import replace
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import patch

import numpy as np
import pytest
import torch

from tests.v1.attention.utils import BatchSpec, create_common_attn_metadata
from vllm.config import (
    CompilationConfig,
    CUDAGraphMode,
    ModelConfig,
    ParallelConfig,
    VllmConfig,
)
from vllm.forward_context import create_forward_context
from vllm.model_executor.layers.fused_moe.routed_experts_capturer import (
    RoutedExpertsCapturer,
)
from vllm.v1.attention.backends.utils import get_dcp_local_seq_lens
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.worker import gpu_ubatch_wrapper as legacy_gpu_ubatch_wrapper
from vllm.v1.worker.gpu import cp_utils as gpu_cp_utils
from vllm.v1.worker.gpu import cudagraph_utils, dp_utils
from vllm.v1.worker.gpu.cudagraph_utils import (
    BatchExecutionDescriptor,
    CudaGraphManager,
)
from vllm.v1.worker.gpu.input_batch import InputBatch, InputBuffers
from vllm.v1.worker.gpu.model_states.interface import ModelState
from vllm.v1.worker.gpu.ubatch_utils import (
    UBatchRunner,
    UBatchState,
    _slice_input_batch,
    create_ubatch_slices,
    restore_staged_inputs,
    slice_model_inputs,
    stage_decode_tokens,
)
from vllm.v1.worker.ubatch_utils import (
    UBatchSlice,
    maybe_create_ubatch_slices,
    split_attn_metadata,
)
from vllm.v1.worker.ubatching import (
    dbo_current_ubatch_id,
    dbo_register_recv_hook,
    dbo_yield,
)

MAX_NUM_REQS = 32
MAX_NUM_TOKENS = 128


def _make_buffers() -> InputBuffers:
    return InputBuffers(
        max_num_reqs=MAX_NUM_REQS,
        max_num_tokens=MAX_NUM_TOKENS,
        device=torch.device("cpu"),
    )


def _make_ubatch_buffers(num_ubatches: int = 2) -> list[tuple[torch.Tensor, ...]]:
    """The per-microbatch (query_start_loc, seq_lens) buffers UBatchRunner owns."""
    return [
        (
            torch.zeros(MAX_NUM_REQS + 1, dtype=torch.int32),
            torch.zeros(MAX_NUM_REQS, dtype=torch.int32),
        )
        for _ in range(num_ubatches)
    ]


def _make_input_batch(
    query_lens: list[int],
    seq_lens: list[int],
    buffers: InputBuffers,
    num_reqs_padded: int | None = None,
    num_tokens_padded: int | None = None,
) -> InputBatch:
    """Build an InputBatch shaped like the one `prepare_inputs` produces."""
    num_reqs = len(query_lens)
    num_tokens = int(sum(query_lens))
    num_reqs_padded = num_reqs_padded or num_reqs
    num_tokens_padded = num_tokens_padded or num_tokens

    # make_dummy writes the shared buffers, so fill them afterwards.
    base = InputBatch.make_dummy(num_reqs, num_tokens, buffers)

    query_start_loc_np = np.zeros(num_reqs_padded + 1, dtype=np.int32)
    np.cumsum(query_lens, out=query_start_loc_np[1 : num_reqs + 1])
    # Padded entries repeat the token count, as `prepare_inputs` does.
    query_start_loc_np[num_reqs + 1 :] = num_tokens
    buffers.query_start_loc[: num_reqs_padded + 1] = torch.from_numpy(
        query_start_loc_np
    )

    buffers.seq_lens[:num_reqs] = torch.tensor(seq_lens, dtype=torch.int32)
    buffers.seq_lens[num_reqs:num_reqs_padded] = 0

    seq_lens_upper_bound = np.zeros(num_reqs_padded, dtype=np.int32)
    seq_lens_upper_bound[:num_reqs] = seq_lens

    return replace(
        base,
        req_ids=[f"req_{i}" for i in range(num_reqs)],
        num_reqs=num_reqs,
        num_reqs_after_padding=num_reqs_padded,
        idx_mapping=torch.arange(num_reqs, dtype=torch.int32),
        idx_mapping_np=np.arange(num_reqs, dtype=np.int32),
        num_scheduled_tokens=np.array(query_lens, dtype=np.int32),
        num_tokens=num_tokens,
        num_tokens_after_padding=num_tokens_padded,
        query_start_loc=buffers.query_start_loc[: num_reqs_padded + 1],
        query_start_loc_np=query_start_loc_np,
        seq_lens=buffers.seq_lens[:num_reqs_padded],
        seq_lens_cpu_upper_bound=torch.from_numpy(seq_lens_upper_bound),
        num_computed_tokens_np=np.array(seq_lens, dtype=np.int32)
        - np.array(query_lens, dtype=np.int32),
        prefill_len_np=np.zeros(num_reqs, dtype=np.int32),
        num_computed_prefill_tokens_np=np.zeros(num_reqs, dtype=np.int32),
        is_prefilling_np=np.zeros(num_reqs, dtype=np.bool_),
        input_ids=buffers.input_ids[:num_tokens_padded],
        positions=buffers.positions[:num_tokens_padded],
        is_padding=buffers.is_padding[:num_tokens_padded],
    )


# (query_lens, seq_lens): uniform decode, a boundary that lands inside a
# request, and a boundary that lands exactly on a request start.
BATCHES = {
    "uniform_decode": ([1] * 8, [128 + 4 * i for i in range(8)]),
    "split_request": ([1, 1, 10, 1, 1], [64, 96, 512, 32, 48]),
    "aligned_prefills": ([6, 2, 4, 4], [6, 130, 260, 4]),
    "split_first_and_last": ([3, 9, 3], [70, 300, 41]),
    # A long prefill trailing a decode: it straddles the boundary and is also
    # the last request, so the second microbatch holds nothing else.
    "trailing_prefill_spans_boundary": ([2, 20], [64, 300]),
}


@pytest.mark.parametrize("batch_name", list(BATCHES))
def test_slicing_matches_v1_split_attn_metadata(batch_name: str):
    """V2's pre-sliced inputs describe the same microbatches as V1's."""
    query_lens, seq_lens = BATCHES[batch_name]
    num_scheduled_tokens = np.array(query_lens, dtype=np.int32)
    num_tokens = int(num_scheduled_tokens.sum())

    ubatch_slices, _ = maybe_create_ubatch_slices(
        True, num_scheduled_tokens, num_tokens, len(query_lens), num_ubatches=2
    )
    assert ubatch_slices is not None

    v1_metadata = create_common_attn_metadata(
        BatchSpec(seq_lens=seq_lens, query_lens=query_lens),
        block_size=16,
        device=torch.device("cpu"),
    )
    v1_ubatches = split_attn_metadata(ubatch_slices, v1_metadata)

    buffers = _make_buffers()
    ubatch_buffers = _make_ubatch_buffers()
    input_batch = _make_input_batch(query_lens, seq_lens, buffers)

    for i, (ubatch_slice, v1_ubatch) in enumerate(zip(ubatch_slices, v1_ubatches)):
        v2_ubatch = _slice_input_batch(input_batch, ubatch_slice, *ubatch_buffers[i])

        assert v2_ubatch.num_reqs == v1_ubatch.num_reqs
        assert v2_ubatch.num_tokens == v1_ubatch.num_actual_tokens
        assert int(v2_ubatch.num_scheduled_tokens.max()) == v1_ubatch.max_query_len
        torch.testing.assert_close(v2_ubatch.query_start_loc, v1_ubatch.query_start_loc)
        torch.testing.assert_close(v2_ubatch.seq_lens, v1_ubatch.seq_lens)
        torch.testing.assert_close(
            torch.from_numpy(v2_ubatch.query_start_loc_np),
            v1_ubatch.query_start_loc_cpu,
        )
        # `prepare_attn` derives max_seq_len from the upper bound. Compare the
        # bound itself rather than V1's max_seq_len: V1 floors it with the full
        # batch's max_seq_len to keep the value CUDA-graph capture installed,
        # which V2 gets from `prepare_attn(for_capture=True)` instead.
        torch.testing.assert_close(
            v2_ubatch.seq_lens_cpu_upper_bound, v1_ubatch.seq_lens_cpu_upper_bound
        )


def test_microbatches_do_not_share_buffers():
    """Both microbatches are live at once, so their buffers must be distinct."""
    query_lens, seq_lens = BATCHES["split_request"]
    num_scheduled_tokens = np.array(query_lens, dtype=np.int32)
    num_tokens = int(num_scheduled_tokens.sum())

    ubatch_slices, _ = maybe_create_ubatch_slices(
        True, num_scheduled_tokens, num_tokens, len(query_lens), num_ubatches=2
    )
    assert ubatch_slices is not None

    buffers = _make_buffers()
    ubatch_buffers = _make_ubatch_buffers()
    input_batch = _make_input_batch(query_lens, seq_lens, buffers)

    first = _slice_input_batch(input_batch, ubatch_slices[0], *ubatch_buffers[0])
    first_query_start_loc = first.query_start_loc.clone()
    first_seq_lens = first.seq_lens.clone()

    # Slicing the second microbatch must not disturb the first.
    _slice_input_batch(input_batch, ubatch_slices[1], *ubatch_buffers[1])

    torch.testing.assert_close(first.query_start_loc, first_query_start_loc)
    torch.testing.assert_close(first.seq_lens, first_seq_lens)


def test_trailing_microbatch_absorbs_cudagraph_padding():
    """The padded microbatch keeps padded rows empty and query_start_loc flat."""
    query_lens, seq_lens = [1] * 6, [64] * 6
    num_tokens_padded = 8

    buffers = _make_buffers()
    input_batch = _make_input_batch(
        query_lens,
        seq_lens,
        buffers,
        num_reqs_padded=num_tokens_padded,
        num_tokens_padded=num_tokens_padded,
    )
    ubatch_slices_padded = create_ubatch_slices(input_batch, num_ubatches=2)

    last = _slice_input_batch(
        input_batch, ubatch_slices_padded[-1], *_make_ubatch_buffers()[1]
    )

    # Two real decodes plus two padded rows.
    assert last.num_reqs == 2
    assert last.num_tokens == 2
    assert last.num_reqs_after_padding == 4
    assert last.num_tokens_after_padding == 4
    torch.testing.assert_close(
        last.query_start_loc, torch.tensor([0, 1, 2, 2, 2], dtype=torch.int32)
    )
    torch.testing.assert_close(
        last.seq_lens, torch.tensor([64, 64, 0, 0], dtype=torch.int32)
    )


DECODE_THRESHOLD = 32
PREFILL_THRESHOLD = 128
DECODE_QUERY_LEN = 1


def test_dummy_batch_can_be_microbatched():
    """A DP rank with no work still has to split, so its dummy batch must slice.

    Microbatching is all-or-nothing across the group, so a rank whose step is a
    dummy batch runs it microbatched like everyone else.
    """
    num_reqs, num_tokens = 1, 32
    buffers = _make_buffers()
    input_batch = InputBatch.make_dummy(num_reqs, num_tokens, buffers)

    ubatch_slices_padded = create_ubatch_slices(input_batch, num_ubatches=2)

    ubatch_buffers = _make_ubatch_buffers()
    ubatches = [
        _slice_input_batch(input_batch, ubatch_slice, *ubatch_buffers[i])
        for i, ubatch_slice in enumerate(ubatch_slices_padded)
    ]

    assert [u.num_tokens for u in ubatches] == [16, 16]
    # The one dummy request spans both microbatches, 16 of its tokens in each.
    assert all(u.num_reqs == 1 for u in ubatches)
    for ubatch in ubatches:
        torch.testing.assert_close(
            ubatch.query_start_loc, torch.tensor([0, 16], dtype=torch.int32)
        )


def _sync_dp(
    num_tokens_per_rank: list[int],
    uniform_token_count_per_rank: list[int] | None = None,
    allow_ubatching: bool = True,
    cudagraph_manager: Any = None,
    staging_per_rank: list[bool] | None = None,
    decode_threshold: int = DECODE_THRESHOLD,
) -> tuple[BatchExecutionDescriptor, dp_utils.DPSyncState | None]:
    """Run the DP handshake with the all-reduce stubbed out.

    The microbatching decision lives inside `sync_cudagraph_and_dp_padding`, so
    the cases below stand in for the collective rather than for the decision,
    and need no process group. Every rank asks for eager, which keeps the
    non-microbatched path off the cudagraph manager. This rank is rank 0.
    """
    dp_size = len(num_tokens_per_rank)
    uniform_token_counts = uniform_token_count_per_rank or [0] * dp_size
    reduced = torch.zeros(8, dp_size, dtype=torch.int32)
    reduced[0] = torch.tensor(num_tokens_per_rank, dtype=torch.int32)
    reduced[1] = CUDAGraphMode.NONE.value
    reduced[2] = torch.tensor(uniform_token_counts, dtype=torch.int32)
    reduced[3] = -1  # max_query_len, -1 means None
    reduced[4] = int(allow_ubatching)
    reduced[5] = 8  # num_reqs
    reduced[6] = 0  # allow_skip_drafts
    reduced[7] = torch.tensor(staging_per_rank or [False] * dp_size)

    dp_group = SimpleNamespace(cpu_all_reduce_=lambda t: t.copy_(reduced))
    with patch.object(dp_utils, "get_dp_group", lambda: dp_group):
        return dp_utils.sync_cudagraph_and_dp_padding(
            cudagraph_manager=cudagraph_manager,
            desired_batch_desc=BatchExecutionDescriptor(
                cg_mode=CUDAGraphMode.NONE,
                num_tokens=num_tokens_per_rank[0],
                num_reqs=8,
            ),
            num_tokens=num_tokens_per_rank[0],
            num_reqs=8,
            uniform_token_count=uniform_token_counts[0] or None,
            dp_size=dp_size,
            dp_rank=0,
            parallel_config=ParallelConfig(
                enable_dbo=True,
                dbo_decode_token_threshold=decode_threshold,
                dbo_prefill_token_threshold=PREFILL_THRESHOLD,
            ),
            allow_ubatching=allow_ubatching,
            uniform_decode=uniform_token_counts[0] == DECODE_QUERY_LEN,
        )


def test_every_dp_rank_must_agree_to_microbatch():
    """One rank below the threshold is enough to keep the group on one batch.

    The thresholds are not communicated -- they are checked against the token
    counts the all-reduce already carries -- so this also pins that the whole
    vector is consulted, not just this rank's own entry.
    """
    assert _sync_dp([256, 256])[0].num_ubatches == 2
    assert _sync_dp([256, 100])[0].num_ubatches == 1
    assert _sync_dp([100, 256])[0].num_ubatches == 1


def test_decode_threshold_only_applies_when_every_rank_is_a_uniform_decode():
    """Otherwise the group is held to the prefill threshold."""
    # 64 tokens clears the decode threshold but not the prefill one, so it only
    # microbatches when both ranks are uniform decodes.
    assert _sync_dp([64, 64], uniform_token_count_per_rank=[1, 1])[0].num_ubatches == 2
    assert _sync_dp([64, 64], uniform_token_count_per_rank=[1, 0])[0].num_ubatches == 1


def test_microbatching_pads_all_ranks_to_the_largest():
    """Ranks must run the same token count so microbatch sizes line up."""
    desc, dp_sync_state = _sync_dp([200, 256])
    assert desc.num_tokens == 256
    assert dp_sync_state is not None
    assert dp_sync_state.num_tokens_across_dp.tolist() == [256, 256]


def test_microbatching_survives_a_rank_that_cannot_fill_it():
    """A rank whose real tokens all land in the first microbatch still splits.

    Rank 0 has 128 real tokens but the split point is at 512 // 2 = 256, so its
    second microbatch is pure padding. That microbatch does no work, which is
    fine -- it still has to run so the expert all-to-all stays collective.
    """
    desc, _ = _sync_dp([128, 512])
    assert desc.num_tokens == 512
    assert desc.num_ubatches == 2


def test_all_padding_microbatch_has_no_work_to_do():
    """The microbatch past the last real token is well-formed and empty.

    It carries the last request with none of its query tokens, so attention and
    sampling see nothing to do while the microbatch still runs.
    """
    query_lens, seq_lens = [1] * 6, [64] * 6
    num_tokens_padded = 16  # split at 8, past all 6 real tokens

    buffers = _make_buffers()
    input_batch = _make_input_batch(
        query_lens, seq_lens, buffers, num_tokens_padded=num_tokens_padded
    )
    ubatch_slices_padded = create_ubatch_slices(input_batch, num_ubatches=2)
    ubatch_buffers = _make_ubatch_buffers()

    first = _slice_input_batch(input_batch, ubatch_slices_padded[0], *ubatch_buffers[0])
    last = _slice_input_batch(input_batch, ubatch_slices_padded[1], *ubatch_buffers[1])

    # The first microbatch keeps every real request; the second gets none.
    assert first.num_reqs == 6
    assert first.num_tokens == 6
    assert last.num_tokens == 0
    assert last.num_reqs == 1
    assert last.num_reqs_after_padding == 1
    assert last.num_tokens_after_padding == 8

    # Zero query tokens for the one request it holds, so no attention work.
    torch.testing.assert_close(
        last.query_start_loc, torch.tensor([0, 0], dtype=torch.int32)
    )
    assert last.num_scheduled_tokens.tolist() == [0]
    # ...and its sequence still ends where its tokens do: they all live in the
    # first microbatch, which computes them before this one runs.
    torch.testing.assert_close(last.seq_lens, torch.tensor([64], dtype=torch.int32))
    torch.testing.assert_close(
        last.seq_lens_cpu_upper_bound, torch.tensor([64], dtype=torch.int32)
    )


def test_microbatching_off_when_a_rank_does_not_allow_it():
    assert _sync_dp([256, 256], allow_ubatching=False)[0].num_ubatches == 1


def test_slice_model_inputs_handles_mrope_positions():
    model_inputs = {
        "input_ids": torch.arange(8),
        # M-RoPE positions carry a leading section dim.
        "positions": torch.arange(24).view(3, 8),
        "inputs_embeds": None,
        "intermediate_tensors": None,
    }
    sliced = slice_model_inputs(model_inputs, slice(4, 8))

    torch.testing.assert_close(sliced["input_ids"], torch.arange(4, 8))
    torch.testing.assert_close(sliced["positions"], torch.arange(24).view(3, 8)[:, 4:8])
    assert sliced["inputs_embeds"] is None
    assert sliced["intermediate_tensors"] is None


@pytest.mark.parametrize(
    ("cudagraph_runtime_mode", "execution_method"),
    [
        (CUDAGraphMode.FULL, "_capture_ubatches"),
        (CUDAGraphMode.NONE, "_run_ubatches"),
    ],
)
def test_legacy_ubatch_contexts_slice_padding_mask(
    cudagraph_runtime_mode: CUDAGraphMode, execution_method: str
):
    """Replacement contexts preserve each microbatch's padding rows."""
    ubatch_slices = [
        UBatchSlice(slice(0, 2), slice(0, 3)),
        UBatchSlice(slice(2, 4), slice(3, 6)),
    ]
    is_padding = torch.tensor([False, True, False, True, False, True])
    forward_context = SimpleNamespace(
        batch_descriptor=object(),
        ubatch_slices=ubatch_slices,
        cudagraph_runtime_mode=cudagraph_runtime_mode,
        attn_metadata=[{}, {}],
        slot_mapping={},
        dp_metadata=object(),
        is_padding=is_padding,
    )

    wrapper = object.__new__(legacy_gpu_ubatch_wrapper.UBatchWrapper)
    wrapper.vllm_config = SimpleNamespace(
        parallel_config=SimpleNamespace(data_parallel_size=2)
    )
    wrapper.comm_stream = object()
    wrapper.ready_barrier = object()
    wrapper.cudagraphs = {}
    wrapper.sm_control = nullcontext()
    wrapper.runnable = object()

    def make_contexts(*, forward_contexts, **kwargs):
        return [
            SimpleNamespace(id=i, forward_context=context)
            for i, context in enumerate(forward_contexts)
        ]

    def create_context(attn_metadata, vllm_config, **kwargs):
        return SimpleNamespace(attn_metadata=attn_metadata, **kwargs)

    model_inputs = {
        "input_ids": torch.arange(6),
        "positions": torch.arange(6),
        "inputs_embeds": None,
        "intermediate_tensors": None,
    }
    expected = object()
    with (
        patch.object(
            legacy_gpu_ubatch_wrapper,
            "get_forward_context",
            return_value=forward_context,
        ),
        patch.object(
            legacy_gpu_ubatch_wrapper.torch.cuda,
            "current_stream",
            return_value=object(),
        ),
        patch.object(
            legacy_gpu_ubatch_wrapper.DPMetadata, "make", return_value=object()
        ),
        patch.object(
            legacy_gpu_ubatch_wrapper,
            "create_forward_context",
            side_effect=create_context,
        ),
        patch.object(
            legacy_gpu_ubatch_wrapper,
            "make_ubatch_contexts",
            side_effect=make_contexts,
        ),
        patch.object(wrapper, execution_method, return_value=expected) as execute,
    ):
        result = wrapper(**model_inputs)

    assert result is expected
    contexts = [
        metadata.context.forward_context for metadata in execute.call_args.args[0]
    ]
    torch.testing.assert_close(contexts[0].is_padding, is_padding[:3])
    torch.testing.assert_close(contexts[1].is_padding, is_padding[3:])


def _make_dbo_config() -> VllmConfig:
    return VllmConfig(
        model_config=ModelConfig(model="facebook/opt-125m", dtype="float16", seed=0),
        parallel_config=ParallelConfig(
            enable_dbo=True, all2all_backend="deepep_low_latency"
        ),
    )


def _make_model_inputs(num_tokens: int, device: torch.device) -> dict[str, Any]:
    return {
        "input_ids": torch.arange(num_tokens, device=device),
        "positions": torch.arange(num_tokens, device=device),
        "inputs_embeds": None,
        "intermediate_tensors": None,
    }


@pytest.fixture(autouse=True)
def _no_comm_sm_reservation(monkeypatch):
    """CI runs on MIG slices with fewer SMs than the default comm reservation."""
    monkeypatch.setenv("VLLM_DBO_COMM_SMS", "0")


def _make_execution_runner(vllm_config: VllmConfig) -> UBatchRunner:
    """A runner for the execution tests below, which never call `prepare()`.

    The attention-side dependencies are only read while building per-microbatch
    metadata, so they are left out here.
    """
    return UBatchRunner(
        vllm_config,
        torch.device("cuda:0"),
        model_state=cast(ModelState, None),
        attn_groups=[],
        kv_cache_config=cast(KVCacheConfig, None),
        max_num_reqs=MAX_NUM_REQS,
    )


def _make_ubatch_state(
    vllm_config: VllmConfig, ubatch_slices: list[UBatchSlice]
) -> UBatchState:
    return UBatchState(
        slices=ubatch_slices,
        forward_contexts=[
            create_forward_context(None, vllm_config) for _ in ubatch_slices
        ],
    )


def test_thresholds_below_the_microbatch_count_are_rejected():
    """A batch with fewer tokens than microbatches cannot be split at all.

    Nothing downstream copes with an unsplittable batch, so the thresholds are
    what keep one out: a one-token decode split two ways leaves an empty
    microbatch.
    """
    with pytest.raises(ValueError, match="number of microbatches"):
        ParallelConfig(enable_dbo=True, dbo_decode_token_threshold=1)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="DBO needs a GPU")
def test_runner_allocates_one_buffer_pair_per_microbatch():
    """All microbatches are live at once, so none may share rebase buffers."""
    vllm_config = _make_dbo_config()
    runner = _make_execution_runner(vllm_config)

    for buffers in (runner.ubatch_query_start_loc, runner.ubatch_seq_lens):
        assert len(buffers) == runner.num_ubatches
        storages = {b.untyped_storage().data_ptr() for b in buffers}
        assert len(storages) == runner.num_ubatches


class _YieldingModel(torch.nn.Module):
    """Toy model that hands off to the other microbatch mid-forward."""

    def __init__(
        self,
        trace: list[tuple[str, int]],
        capturer: RoutedExpertsCapturer | None = None,
    ):
        super().__init__()
        self.trace = trace
        self.capturer = capturer

    def forward(self, input_ids, positions, intermediate_tensors=None, **kwargs):
        self.trace.append(("enter", dbo_current_ubatch_id()))
        out = input_ids.float().unsqueeze(-1) * 2 + positions.float().unsqueeze(-1)
        # Stands in for the expert all-to-all handoff point.
        dbo_yield()
        if self.capturer is not None:
            self.capturer.capture(0, input_ids.view(-1, 1).int())
        self.trace.append(("exit", dbo_current_ubatch_id()))
        return out


def _make_capturer(num_tokens: int, device: torch.device) -> RoutedExpertsCapturer:
    capturer = RoutedExpertsCapturer.__new__(RoutedExpertsCapturer)
    capturer.dp_rank = 0
    capturer.tp_size = 1
    capturer.device_buffer = torch.full(
        (num_tokens, 1, 1), -1, dtype=torch.int32, device=device
    )
    return capturer


@pytest.mark.skipif(not torch.cuda.is_available(), reason="DBO needs a GPU")
def test_ubatch_runner_overlaps_and_matches_single_batch():
    """Microbatches interleave at the yield point and produce the same output."""
    vllm_config = _make_dbo_config()
    device = torch.device("cuda:0")
    runner = _make_execution_runner(vllm_config)

    model_inputs = _make_model_inputs(16, device)
    ubatch_state = _make_ubatch_state(
        vllm_config,
        [
            UBatchSlice(slice(0, 4), slice(0, 8)),
            UBatchSlice(slice(4, 8), slice(8, 16)),
        ],
    )

    trace: list[tuple[str, int]] = []
    capturer = _make_capturer(16, device)
    model = _YieldingModel(trace, capturer)
    output = runner.run(model, model_inputs, ubatch_state)

    expected = model_inputs["input_ids"].float().unsqueeze(-1) * 2 + model_inputs[
        "positions"
    ].float().unsqueeze(-1)
    torch.testing.assert_close(output, expected)

    # Both microbatches reach the handoff before either finishes.
    assert trace == [("enter", 0), ("enter", 1), ("exit", 0), ("exit", 1)]
    torch.testing.assert_close(
        capturer.device_buffer[:, 0, 0], model_inputs["input_ids"].int()
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="DBO needs a GPU")
@pytest.mark.parametrize("failed_ubatch", [0, 1])
def test_ubatch_runner_names_the_microbatch_that_failed(failed_ubatch):
    """A failing microbatch surfaces as a named error, not a downstream KeyError.

    The first recorded failure is reported, not microbatch 0's later error from
    registering a recv hook on the exited microbatch 1. A microbatch that fails
    *while its sibling is parked at a yield* hangs the step instead -- the
    shared handoff protocol in `ubatching.py` has no way to unwind a parked
    microbatch, and the V1 runner has the same gap. Fixing that means changing
    `ubatching.py`, which is out of scope for the V2 runner.
    """
    vllm_config = _make_dbo_config()
    device = torch.device("cuda:0")
    runner = _make_execution_runner(vllm_config)
    failed_threads: list[threading.Thread] = []

    class _FailingModel(torch.nn.Module):
        def forward(self, input_ids, positions, **kwargs):
            if dbo_current_ubatch_id() == failed_ubatch:
                failed_threads.append(threading.current_thread())
                raise ValueError("boom")
            if failed_ubatch == 0:
                return input_ids.float().unsqueeze(-1)
            dbo_yield()
            failed_threads[0].join(timeout=5.0)
            assert not failed_threads[0].is_alive()
            dbo_register_recv_hook(lambda: None)

    model_inputs = _make_model_inputs(8, device)
    ubatch_state = _make_ubatch_state(
        vllm_config,
        [
            UBatchSlice(slice(0, 2), slice(0, 4)),
            UBatchSlice(slice(2, 4), slice(4, 8)),
        ],
    )

    result: dict[str, BaseException] = {}

    def _call():
        try:
            runner.run(_FailingModel(), model_inputs, ubatch_state)
        except BaseException as e:  # noqa: BLE001
            result["error"] = e

    # Run behind a watchdog: `UBatchRunner.run` joins without a timeout, so a
    # regression here would wedge the suite rather than fail it.
    caller = threading.Thread(target=_call, daemon=True)
    caller.start()
    caller.join(timeout=60.0)
    assert not caller.is_alive(), "UBatchRunner.run hung on a failing microbatch"

    assert isinstance(result["error"], RuntimeError)
    assert f"Microbatch {failed_ubatch}" in str(result["error"])
    assert isinstance(result["error"].__cause__, ValueError)


# DCP x DBO: each microbatch recomputes DCP-local seq_lens from its own
# truncated seq_lens into its own persistent buffer (the parent's values for
# a straddling request are wrong for the leading microbatch).

DCP_SIZE = 2
CP_INTERLEAVE = 1


def _make_cuda_input_batch(
    query_lens: list[int], seq_lens: list[int]
) -> tuple[InputBatch, InputBuffers]:
    buffers = InputBuffers(
        max_num_reqs=MAX_NUM_REQS,
        max_num_tokens=MAX_NUM_TOKENS,
        device=torch.device("cuda:0"),
    )
    return _make_input_batch(query_lens, seq_lens, buffers), buffers


def _make_dcp_ubatch_buffers(
    num_ubatches: int = 2,
) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Per-microbatch (query_start_loc, seq_lens, dcp_local_seq_lens)."""
    return [
        (
            torch.zeros(MAX_NUM_REQS + 1, dtype=torch.int32, device="cuda:0"),
            torch.zeros(MAX_NUM_REQS, dtype=torch.int32, device="cuda:0"),
            torch.zeros(MAX_NUM_REQS, dtype=torch.int32, device="cuda:0"),
        )
        for _ in range(num_ubatches)
    ]


def _slice_with_dcp(
    input_batch: InputBatch, dcp_rank: int, num_ubatches: int = 2
) -> tuple[list[InputBatch], list[torch.Tensor]]:
    dcp_buffers = [
        torch.zeros(MAX_NUM_REQS, dtype=torch.int32, device="cuda:0")
        for _ in range(num_ubatches)
    ]
    ubatch_buffers = _make_dcp_ubatch_buffers(num_ubatches)
    ubatches = [
        _slice_input_batch(
            input_batch,
            ubatch_slice,
            ubatch_buffers[i][0],
            ubatch_buffers[i][1],
            dcp_buffers[i],
            dcp_size=DCP_SIZE,
            dcp_rank=dcp_rank,
            cp_interleave=CP_INTERLEAVE,
        )
        for i, ubatch_slice in enumerate(
            create_ubatch_slices(input_batch, num_ubatches)
        )
    ]
    return ubatches, dcp_buffers


@pytest.mark.skipif(not torch.cuda.is_available(), reason="triton kernel needs CUDA")
@pytest.mark.parametrize("dcp_rank", [0, 1])
def test_microbatches_recompute_dcp_lens_from_truncated_seq_lens(dcp_rank: int):
    """A boundary-split request keeps a truncated seq_len on the leading
    microbatch, whose DCP-local length may differ (in parity) from the full
    batch's. Microbatches must therefore derive DCP lengths from their own
    seq_lens, not reuse the parent's row.
    """
    input_batch, buffers = _make_cuda_input_batch(
        [1, 1, 10, 1, 1], [64, 96, 512, 32, 48]
    )
    # execute_model has populated the merged batch's DCP metadata already.
    input_batch.dcp_local_seq_lens = gpu_cp_utils.prepare_dcp_local_seq_lens(
        buffers.dcp_local_seq_lens,
        input_batch.seq_lens,
        input_batch.num_reqs,
        DCP_SIZE,
        dcp_rank,
        CP_INTERLEAVE,
        num_reqs_padded=input_batch.num_reqs_after_padding,
    )
    assert input_batch.dcp_local_seq_lens is not None
    parent_lens = input_batch.dcp_local_seq_lens.clone()

    ubatches, _ = _slice_with_dcp(input_batch, dcp_rank)

    for ubatch in ubatches:
        assert ubatch.dcp_local_seq_lens is not None
        expected = get_dcp_local_seq_lens(
            ubatch.seq_lens.cpu(), DCP_SIZE, dcp_rank, CP_INTERLEAVE
        )
        assert torch.equal(
            ubatch.dcp_local_seq_lens[: ubatch.num_reqs].cpu(),
            expected.to(torch.int32),
        )

    # The straddling request (512 tokens, 5 of its 10 query tokens truncated)
    # lands on this rank with a different local length than the full batch
    # reports (507 vs 512 -> 254 vs 256 for dcp_size=2, interleave=1); copying
    # the parent's row would ship the stale value.
    assert ubatches[0].num_reqs == 3
    assert ubatches[0].seq_lens[2].item() == 507
    assert ubatches[0].dcp_local_seq_lens is not None
    assert ubatches[0].dcp_local_seq_lens[2].item() != parent_lens[2].item()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="triton kernel needs CUDA")
def test_each_microbatch_owns_its_dcp_buffer():
    """Microbatch DCP lengths live in per-microbatch persistent buffers so
    CPU-visible shapes stay valid under CUDA graph capture and replay."""
    input_batch, _ = _make_cuda_input_batch([4, 4, 4, 4], [8, 9, 10, 11])

    ubatches, dcp_buffers = _slice_with_dcp(input_batch, dcp_rank=0)

    for ubatch, dcp_buffer in zip(ubatches, dcp_buffers):
        assert ubatch.dcp_local_seq_lens is not None
        assert ubatch.dcp_local_seq_lens.data_ptr() == dcp_buffer.data_ptr()
    assert dcp_buffers[0].data_ptr() != dcp_buffers[1].data_ptr()


def test_slicing_drops_stale_dcp_metadata_when_dcp_is_off():
    """dcp_size == 1 yields None even when the parent carries a value, so a
    microbatch can never consume DCP metadata this deployment did not
    enable.
    """
    buffers = _make_buffers()
    input_batch = _make_input_batch([4, 4, 4, 4], [8, 9, 10, 11], buffers)
    input_batch.dcp_local_seq_lens = buffers.dcp_local_seq_lens[:4]

    ubatch_buffers = _make_ubatch_buffers()
    ubatches = [
        _slice_input_batch(input_batch, ubatch_slice, *ubatch_buffers[i])
        for i, ubatch_slice in enumerate(create_ubatch_slices(input_batch, 2))
    ]

    assert all(ubatch.dcp_local_seq_lens is None for ubatch in ubatches)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="DBO needs a GPU")
def test_capturable_run_replays_as_a_cudagraph():
    """Microbatched graph replay reads updated persistent input buffers."""
    vllm_config = VllmConfig(
        model_config=ModelConfig(model="facebook/opt-125m", dtype="float16", seed=0),
        parallel_config=ParallelConfig(
            enable_dbo=True, all2all_backend="deepep_low_latency"
        ),
    )
    device = torch.device("cuda:0")
    runner = _make_execution_runner(vllm_config)

    num_tokens = 16
    input_ids = torch.arange(num_tokens, device=device)
    positions = torch.arange(num_tokens, device=device)
    model_inputs = {
        "input_ids": input_ids,
        "positions": positions,
        "inputs_embeds": None,
        "intermediate_tensors": None,
    }
    ubatch_state = _make_ubatch_state(
        vllm_config,
        [
            UBatchSlice(slice(0, 4), slice(0, 8)),
            UBatchSlice(slice(4, 8), slice(8, 16)),
        ],
    )

    capturer = _make_capturer(num_tokens, device)
    model = _YieldingModel([], capturer)
    graph = torch.cuda.CUDAGraph()
    finish = runner.begin_capturable_run(
        model, model_inputs, ubatch_state, for_capture=True
    )
    with torch.cuda.graph(graph, stream=runner.capture_stream):
        captured = finish()

    input_ids.fill_(3)
    positions.fill_(5)
    graph.replay()
    torch.accelerator.synchronize()

    expected = torch.full(
        (num_tokens, 1), 3 * 2 + 5, dtype=torch.float32, device=device
    )
    torch.testing.assert_close(captured, expected)
    assert (capturer.device_buffer == 3).all()


def _request_slices(input_batch: InputBatch) -> list[slice]:
    return [s.request_slice for s in create_ubatch_slices(input_batch, 2)]


def test_captured_split_survives_a_replay_with_enough_requests():
    """Replay preserves the captured split only if real requests reach it."""
    buffers = _make_buffers()
    captured = _request_slices(InputBatch.make_dummy(16, 16, buffers))

    def replay(num_reqs: int) -> list[slice]:
        return _request_slices(
            _make_input_batch(
                [1] * num_reqs,
                [128] * num_reqs,
                buffers,
                num_reqs_padded=16,
                num_tokens_padded=16,
            )
        )

    assert replay(16) == captured
    assert replay(9) == captured
    assert replay(4) != captured


def _make_cudagraph_manager(capture_sizes: list[int]) -> CudaGraphManager:
    vllm_config = VllmConfig(
        model_config=ModelConfig(model="facebook/opt-125m", dtype="float16", seed=0),
        parallel_config=ParallelConfig(
            enable_dbo=True,
            all2all_backend="deepep_low_latency",
            dbo_decode_token_threshold=DECODE_THRESHOLD,
            dbo_prefill_token_threshold=PREFILL_THRESHOLD,
        ),
        compilation_config=CompilationConfig(
            cudagraph_mode=CUDAGraphMode.FULL,
            cudagraph_capture_sizes=capture_sizes,
        ),
    )
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
            ubatch_runner=_make_execution_runner(vllm_config),
        )
    manager._graphs_captured = True
    return manager


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a graph pool")
def test_microbatched_graphs_are_only_offered_to_uniform_batches():
    """Mixed batches cannot reuse graphs captured with uniform query lengths."""
    manager = _make_cudagraph_manager([64, 128])

    desc = manager.dispatch(
        num_reqs=64,
        num_tokens=64,
        uniform_token_count=1,
        num_active_loras=0,
        num_ubatches=2,
    )
    assert desc.cg_mode == CUDAGraphMode.FULL
    assert desc.num_ubatches == 2

    desc = manager.dispatch(
        num_reqs=8,
        num_tokens=64,
        uniform_token_count=None,
        num_active_loras=0,
        num_ubatches=2,
    )
    assert desc.cg_mode == CUDAGraphMode.NONE


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a graph pool")
@pytest.mark.parametrize("tokens", [[33, 128], [64, 128]])
def test_microbatched_graph_needs_every_rank_to_reach_the_split(tokens):
    """Fall back to eager on all ranks if any rank cannot reach the captured split."""
    manager = _make_cudagraph_manager([64, 128])
    uniform = [1, 1]

    desc, _ = _sync_dp([64, 64], uniform, cudagraph_manager=manager)
    assert desc.cg_mode == CUDAGraphMode.FULL
    assert desc.num_ubatches == 2

    # Rank 1 pads the group up to a 128-token graph, but rank 0 only reaches 33.
    desc, dp_sync_state = _sync_dp(tokens, uniform, cudagraph_manager=manager)
    assert desc.cg_mode == CUDAGraphMode.NONE
    assert desc.num_ubatches == 2
    assert dp_sync_state is not None and dp_sync_state.eager


@pytest.mark.parametrize("padded,k", [(127, 2), (128, 3), (129, 4), (127, 8)])
@pytest.mark.parametrize("overflow", [False, True])
def test_staged_rows_roundtrip_preserves_inputs_and_kv_slots(padded, k, overflow):
    n = padded if overflow else 32
    buffers = InputBuffers(padded, padded, torch.device("cpu"))
    batch = _make_input_batch([1] * n, [100] * n, buffers, padded, padded)
    batch.input_ids.copy_(torch.arange(padded))
    batch.positions.copy_(torch.arange(padded) + 100)
    tables = (torch.arange(padded * 2).reshape(padded, 2),)
    slots = torch.arange(padded).unsqueeze(0)
    values = (batch.input_ids, batch.positions, *tables)
    expected = [value[:n].clone() for value in values]
    slices = create_ubatch_slices(batch, k)
    if overflow:
        with pytest.raises(AssertionError):
            stage_decode_tokens(batch, tables, slots, slices)
        for value, original in zip(values, expected):
            torch.testing.assert_close(value[:n], original)
        return
    rows = stage_decode_tokens(batch, tables, slots, slices)
    for value, original in zip(values, expected):
        torch.testing.assert_close(value[rows], original)
    torch.testing.assert_close(slots[:, rows], torch.arange(n).unsqueeze(0))
    assert (~batch.is_padding).sum() == n
    for region in slices:
        assert (~batch.is_padding[region.token_slice]).sum() in (n // k, n // k + 1)
    restore_staged_inputs(batch, tables, slots, rows)
    for value, original in zip(values, expected):
        torch.testing.assert_close(value[:n], original)
        assert not value[n:].any()
    torch.testing.assert_close(slots[:, :n], torch.arange(n).unsqueeze(0))
    assert (slots[:, n:] == -1).all()
    assert not batch.is_padding[:n].any() and batch.is_padding[n:].all()


@pytest.fixture
def staging_runner(tmp_path, dist_init, request):
    from transformers import Qwen3MoeConfig

    from vllm.config import AttentionConfig, SchedulerConfig, set_current_vllm_config
    from vllm.v1.attention.backends.flash_attn import FlashAttentionBackend
    from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheGroupSpec
    from vllm.v1.worker.gpu.model_states.default import DefaultModelState
    from vllm.v1.worker.utils import AttentionGroup

    padded, k = getattr(request, "param", (128, 2))
    Qwen3MoeConfig(
        architectures=["Qwen3MoeForCausalLM"],
        hidden_size=256,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=64,
        num_hidden_layers=1,
        moe_intermediate_size=256,
        num_experts=8,
        num_experts_per_tok=2,
        vocab_size=256,
        max_position_embeddings=256,
    ).save_pretrained(tmp_path)
    config = VllmConfig(
        model_config=ModelConfig(
            model=str(tmp_path), dtype="bfloat16", skip_tokenizer_init=True
        ),
        parallel_config=ParallelConfig(
            ubatch_size=k,
            all2all_backend="nixl_ep",
            data_parallel_size=2,
            distributed_executor_backend="mp",
        ),
        attention_config=AttentionConfig(flash_attn_version=3),
        scheduler_config=SchedulerConfig(
            is_encoder_decoder=False,
            max_model_len=256,
            max_num_seqs=padded,
            max_num_batched_tokens=256,
        ),
        compilation_config=CompilationConfig(
            cudagraph_mode=CUDAGraphMode.FULL,
            cudagraph_capture_sizes=[padded],
            max_cudagraph_capture_size=padded,
        ),
    )
    device = torch.device("cuda", 0)
    spec = FullAttentionSpec(
        block_size=16, num_kv_heads=2, head_size=64, dtype=torch.bfloat16
    )
    cache = KVCacheConfig(
        num_blocks=1024,
        kv_cache_tensors=[],
        kv_cache_groups=[KVCacheGroupSpec(["attn"], spec)],
    )
    with set_current_vllm_config(config):
        group = AttentionGroup(FlashAttentionBackend, ["attn"], spec, 0)
        group.create_metadata_builders(config, device, num_metadata_builders=k)
        state = DefaultModelState(config, torch.nn.Identity(), None, device)
        return UBatchRunner(config, device, state, [[group]], cache, padded)


@pytest.mark.cuda_only
@pytest.mark.parametrize(
    "staging_runner", [(127, 2), (128, 3), (129, 4), (127, 8)], indirect=True
)
def test_prepare_stages_decode_with_each_captured_regions_attention(staging_runner):
    runner = staging_runner
    padded = runner.vllm_config.scheduler_config.max_num_seqs
    n = 32
    buffers = InputBuffers(padded, padded, runner.device)
    seq_lens = [8 + i for i in range(n)]
    batch = _make_input_batch([1] * n, seq_lens, buffers, padded, padded)
    batch.input_ids.copy_(torch.arange(padded, device=runner.device))
    tables = (torch.arange(padded * 8, device=runner.device).reshape(padded, 8),)
    slots = torch.arange(padded, device=runner.device).unsqueeze(0)
    original = [value[:n].clone() for value in (batch.input_ids, *tables)]
    state = runner.prepare(batch, tables, slots, CUDAGraphMode.FULL)
    assert state.staged_rows is not None
    start = 0
    for i, (region, ctx) in enumerate(zip(state.slices, state.forward_contexts)):
        count = n // runner.num_ubatches + (i < n % runner.num_ubatches)
        metadata = ctx.attn_metadata["attn"]
        expected_query = torch.arange(region.num_tokens + 1, device=runner.device)
        torch.testing.assert_close(
            metadata.query_start_loc, expected_query.clamp_max(count).int()
        )
        torch.testing.assert_close(
            metadata.seq_lens[:count],
            torch.tensor(seq_lens[start : start + count], device=runner.device).int(),
        )
        assert not metadata.seq_lens[count:].any()
        torch.testing.assert_close(
            metadata.block_table[:count], original[1][start : start + count]
        )
        assert (
            ctx.dp_metadata.num_tokens_across_dp_cpu.tolist() == [region.num_tokens] * 2
        )
        start += count
    restore_staged_inputs(batch, tables, slots, state.staged_rows)
    for value, expected in zip((batch.input_ids, *tables), original):
        torch.testing.assert_close(value[:n], expected)


@pytest.mark.cuda_only
@pytest.mark.parametrize(
    ("loads", "staging_votes", "expected_mode"),
    [
        ([63, 97], [True, True], CUDAGraphMode.FULL),
        ([63, 97], [True, False], CUDAGraphMode.NONE),
        ([1, 97], [True, True], CUDAGraphMode.NONE),
    ],
)
def test_dp_keeps_imbalanced_graph_only_when_every_rank_can_stage(
    staging_runner, loads, staging_votes, expected_mode
):
    runner = staging_runner
    from vllm.config import set_current_vllm_config

    with set_current_vllm_config(runner.vllm_config):
        manager = CudaGraphManager(
            runner.vllm_config,
            runner.device,
            CUDAGraphMode.FULL,
            1,
            ubatch_runner=runner,
        )
        buf = torch.ones(128, device=runner.device)
        manager.capture(lambda desc, warmup: lambda mode: buf.add_(1))
    desc, sync = _sync_dp(
        loads,
        [1, 1],
        cudagraph_manager=manager,
        staging_per_rank=staging_votes,
        decode_threshold=1,
    )
    assert desc.cg_mode == expected_mode
    assert sync is not None and sync.eager == (expected_mode == CUDAGraphMode.NONE)
