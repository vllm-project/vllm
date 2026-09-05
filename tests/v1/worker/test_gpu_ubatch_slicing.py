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
from dataclasses import replace
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import patch

import numpy as np
import pytest
import torch

from tests.v1.attention.utils import BatchSpec, create_common_attn_metadata
from vllm.config import CUDAGraphMode, ModelConfig, ParallelConfig, VllmConfig
from vllm.forward_context import create_forward_context
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.worker.gpu import dp_utils
from vllm.v1.worker.gpu.cudagraph_utils import BatchExecutionDescriptor
from vllm.v1.worker.gpu.input_batch import InputBatch, InputBuffers
from vllm.v1.worker.gpu.model_states.interface import ModelState
from vllm.v1.worker.gpu.ubatch_utils import (
    UBatchRunner,
    UBatchState,
    _slice_input_batch,
    create_ubatch_slices,
    slice_model_inputs,
)
from vllm.v1.worker.ubatch_utils import (
    UBatchSlice,
    maybe_create_ubatch_slices,
    split_attn_metadata,
)
from vllm.v1.worker.ubatching import dbo_current_ubatch_id, dbo_yield

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
) -> tuple[BatchExecutionDescriptor, dp_utils.DPSyncState | None]:
    """Run the DP handshake with the all-reduce stubbed out.

    The microbatching decision lives inside `sync_cudagraph_and_dp_padding`, so
    the cases below stand in for the collective rather than for the decision,
    and need no process group. Every rank asks for eager, which keeps the
    non-microbatched path off the cudagraph manager. This rank is rank 0.
    """
    dp_size = len(num_tokens_per_rank)
    uniform_token_counts = uniform_token_count_per_rank or [0] * dp_size
    reduced = torch.zeros(6, dp_size, dtype=torch.int32)
    reduced[0] = torch.tensor(num_tokens_per_rank, dtype=torch.int32)
    reduced[1] = CUDAGraphMode.NONE.value
    reduced[2] = torch.tensor(uniform_token_counts, dtype=torch.int32)
    reduced[3] = -1  # max_query_len, -1 means None
    reduced[4] = int(allow_ubatching)
    reduced[5] = 8  # num_reqs

    with (
        patch.object(dp_utils.dist, "all_reduce", lambda t, group: t.copy_(reduced)),
        patch.object(dp_utils, "get_dp_group", lambda: SimpleNamespace(cpu_group=None)),
    ):
        return dp_utils.sync_cudagraph_and_dp_padding(
            cudagraph_manager=None,
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
                dbo_decode_token_threshold=DECODE_THRESHOLD,
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
    desc, dp_sync = _sync_dp([200, 256])
    assert desc.num_tokens == 256
    assert dp_sync is not None
    assert dp_sync.num_tokens_across_dp.tolist() == [256, 256]


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

    def __init__(self, trace: list[tuple[str, int]]):
        super().__init__()
        self.trace = trace

    def forward(self, input_ids, positions, intermediate_tensors=None, **kwargs):
        self.trace.append(("enter", dbo_current_ubatch_id()))
        out = input_ids.float().unsqueeze(-1) * 2 + positions.float().unsqueeze(-1)
        # Stands in for the expert all-to-all handoff point.
        dbo_yield()
        self.trace.append(("exit", dbo_current_ubatch_id()))
        return out


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
    model = _YieldingModel(trace)
    output = runner.run(model, model_inputs, ubatch_state)

    expected = model_inputs["input_ids"].float().unsqueeze(-1) * 2 + model_inputs[
        "positions"
    ].float().unsqueeze(-1)
    torch.testing.assert_close(output, expected)

    # Both microbatches reach the handoff before either finishes.
    assert trace == [("enter", 0), ("enter", 1), ("exit", 0), ("exit", 1)]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="DBO needs a GPU")
def test_ubatch_runner_names_the_microbatch_that_failed():
    """A failing microbatch surfaces as a named error, not a downstream KeyError.

    The model here never yields, so the microbatches run back to back and no
    handoff is left outstanding when one of them dies. A microbatch that fails
    *while its sibling is parked at a yield* hangs the step instead -- the
    shared handoff protocol in `ubatching.py` has no way to unwind a parked
    microbatch, and the V1 runner has the same gap. Fixing that means changing
    `ubatching.py`, which is out of scope for the V2 runner.
    """
    vllm_config = _make_dbo_config()
    device = torch.device("cuda:0")
    runner = _make_execution_runner(vllm_config)

    class _FailingModel(torch.nn.Module):
        def forward(self, input_ids, positions, **kwargs):
            if dbo_current_ubatch_id() == 0:
                raise ValueError("boom")
            return input_ids.float().unsqueeze(-1)

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
    assert "Microbatch 0" in str(result["error"])
    assert isinstance(result["error"].__cause__, ValueError)
