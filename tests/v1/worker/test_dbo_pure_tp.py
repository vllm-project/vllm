# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DBO under pure tensor parallelism (one DP rank, no EP), without GPUs."""

import threading
from types import SimpleNamespace

import pytest
import torch

import vllm.distributed.parallel_state as parallel_state
import vllm.v1.worker.ubatching as ubatching
import vllm.v1.worker.workspace as workspace
from vllm.config import CUDAGraphMode, ParallelConfig, VllmConfig
from vllm.sequence import IntermediateTensors
from vllm.v1.worker.gpu.dp_utils import dispatch_cg_and_sync_dp
from vllm.v1.worker.gpu.ubatch_utils import merge_ubatch_outputs


def test_merge_tensors_concatenates_along_tokens():
    a, b = torch.arange(6.0).view(3, 2), torch.arange(6.0, 10.0).view(2, 2)
    assert torch.equal(merge_ubatch_outputs([a, b]), torch.cat([a, b]))


def test_merge_recurses_into_tuples_and_lists():
    # EAGLE3 returns (hidden, aux); DSpark returns (hidden, [aux, ...]).
    h = [torch.full((2, 4), float(i)) for i in range(2)]
    aux = [
        [torch.full((2, 3), 10.0 + i), torch.full((2, 5), 20.0 + i)] for i in range(2)
    ]
    merged = merge_ubatch_outputs([(h[0], aux[0]), (h[1], aux[1])])
    assert isinstance(merged, tuple) and isinstance(merged[1], list)
    assert torch.equal(merged[0], torch.cat(h))
    assert torch.equal(merged[1][0], torch.cat([aux[0][0], aux[1][0]]))
    assert torch.equal(merged[1][1], torch.cat([aux[0][1], aux[1][1]]))


def test_merge_keeps_none_and_intermediate_tensors():
    assert merge_ubatch_outputs([None, None]) is None
    parts = [
        IntermediateTensors({"hidden_states": torch.ones(2, 4) * i}) for i in range(2)
    ]
    merged = merge_ubatch_outputs(parts)
    assert merged["hidden_states"].shape == (4, 4)


def _parallel_config(tp=4, dp=1, ep=False, dbo=True, schedule=None):
    # An explicit backend skips the GPU count check: CI may run this on one GPU.
    return ParallelConfig(
        tensor_parallel_size=tp,
        data_parallel_size=dp,
        enable_expert_parallel=ep,
        enable_dbo=dbo,
        dbo_schedule=schedule,
        distributed_executor_backend="mp",
    )


@pytest.mark.parametrize(
    ("tp", "dp", "ep", "schedule"),
    [
        (4, 1, False, "all-reduce-overlap"),
        (4, 2, False, "dispatch-combine-overlap"),
        (4, 1, True, "dispatch-combine-overlap"),
        (1, 1, False, "dispatch-combine-overlap"),
    ],
)
def test_default_dbo_schedule(tp, dp, ep, schedule):
    assert _parallel_config(tp, dp, ep).dbo_schedule == schedule


@pytest.mark.parametrize("kwargs", [dict(dp=2), dict(ep=True), dict(tp=1)])
def test_all_reduce_overlap_rejected_outside_pure_tp(kwargs):
    with pytest.raises(ValueError, match="all-reduce-overlap"):
        _parallel_config(schedule="all-reduce-overlap", **kwargs)


def test_explicit_dbo_schedule_is_kept():
    config = _parallel_config(schedule="dispatch-combine-overlap")
    assert config.dbo_schedule == "dispatch-combine-overlap"


def _config(tp=4, dp=1, ep=False, spec=False, mm=False, enc_dec=False):
    return SimpleNamespace(
        model_config=SimpleNamespace(
            is_multimodal_model=mm, is_encoder_decoder=enc_dec, is_hybrid=False
        ),
        parallel_config=_parallel_config(tp, dp, ep),
        lora_config=None,
        speculative_config=object() if spec else None,
        compilation_config=SimpleNamespace(cudagraph_mode=CUDAGraphMode.FULL),
        is_mm_encoder_only=False,
    )


@pytest.mark.parametrize(
    ("kwargs", "rejected"),
    [
        # pure TP: the vision encoder runs outside the split
        (dict(mm=True), None),
        (dict(spec=True, mm=True), None),
        # anything that is not pure TP keeps the old restrictions
        (dict(mm=True, dp=2), "multimodal"),
        (dict(mm=True, ep=True), "multimodal"),
        (dict(mm=True, tp=1), "multimodal"),
        # encoder-decoder models stay out under pure TP too
        (dict(enc_dec=True), "multimodal"),
    ],
)
def test_dbo_unsupported_features(kwargs, rejected):
    unsupported = VllmConfig._get_dbo_unsupported_features(_config(**kwargs))
    if rejected is None:
        assert unsupported == []
    else:
        assert any(rejected in u for u in unsupported), unsupported


def _parallel(threshold_prefill=8192, threshold_decode=32768):
    return SimpleNamespace(
        use_ubatching=True,
        dbo_schedule="all-reduce-overlap",
        num_ubatches=2,
        dbo_prefill_token_threshold=threshold_prefill,
        dbo_decode_token_threshold=threshold_decode,
    )


@pytest.mark.parametrize(
    ("num_tokens", "uniform_decode", "allow", "expected_ubatches"),
    [
        (16384, False, True, 2),  # prefill above its threshold: split
        (4096, False, True, 1),  # below: whole
        (16384, True, True, 1),  # a decode step uses the decode threshold
        (16384, False, False, 1),  # the caller did not allow a split
    ],
)
def test_single_dp_rank_splits_on_its_own(
    num_tokens, uniform_decode, allow, expected_ubatches
):
    desc, sync = dispatch_cg_and_sync_dp(
        None,
        num_reqs=2,
        num_tokens=num_tokens,
        uniform_token_count=None,
        dp_size=1,
        dp_rank=0,
        need_eager=True,
        parallel_config=_parallel(),
        allow_ubatching=allow,
        uniform_decode=uniform_decode,
    )
    assert sync is None
    assert desc.num_ubatches == expected_ubatches
    if expected_ubatches > 1:
        # a split step runs eager
        assert desc.cg_mode == CUDAGraphMode.NONE


@pytest.mark.parametrize(("num_ubatches", "expected"), [(1, 0), (2, 1)])
def test_workspace_id_shared_across_microbatches(monkeypatch, num_ubatches, expected):
    # TP-only DBO on the V2 runner keeps one workspace set.
    monkeypatch.setattr(workspace, "dbo_current_ubatch_id", lambda: 1)
    manager = workspace.WorkspaceManager(
        torch.device("cpu"), num_ubatches=num_ubatches, num_lanes=1
    )
    assert manager._get_workspace_id() == expected


def _ubatch_contexts(schedule):
    return ubatching.make_ubatch_contexts(
        num_micro_batches=2,
        compute_stream=object(),
        comm_stream=object(),
        forward_contexts=[object(), object()],  # type: ignore[list-item]  # stand-ins
        ready_barrier=threading.Barrier(3),
        schedule=schedule,
    )


def test_dbo_enabled_for_a_schedule(monkeypatch):
    # Checks that belong to one schedule (the MoE all-to-all hand-off) must
    # not fire under another.
    monkeypatch.setattr(ubatching, "_SCHEDULE", None)
    _ubatch_contexts("all-reduce-overlap")
    assert not ubatching.dbo_enabled("all-reduce-overlap")
    monkeypatch.setattr(ubatching, "_THREAD_ID_TO_CONTEXT", {0: 0})
    assert ubatching.dbo_enabled()
    assert ubatching.dbo_enabled("all-reduce-overlap")
    assert not ubatching.dbo_enabled("dispatch-combine-overlap")


def test_all_reduce_hand_off_is_a_no_op_under_another_schedule(monkeypatch):
    monkeypatch.setattr(ubatching, "_SCHEDULE", None)
    ctx = _ubatch_contexts("dispatch-combine-overlap")[0]
    # The streams are stand-ins: a hand-off that ran would raise.
    ctx.switch_to_comm_sync("all-reduce-overlap")
    ctx.yield_and_switch_from_comm_to_compute("all-reduce-overlap")
    assert ctx.current_stream is ctx.compute_stream


@pytest.mark.parametrize("is_tp_group", [True, False])
def test_all_reduce_uses_device_communicator_outside_a_microbatch(
    monkeypatch, is_tp_group
):
    # A step that is not split takes the normal all-reduce.
    monkeypatch.setattr(ubatching, "_THREAD_ID_TO_CONTEXT", {})
    group = parallel_state.GroupCoordinator.__new__(parallel_state.GroupCoordinator)
    group.device_communicator = SimpleNamespace(all_reduce=lambda x: x + 1)
    monkeypatch.setattr(parallel_state, "_TP", group if is_tp_group else None)
    x = torch.ones(2)
    assert torch.equal(group._all_reduce_out_place(x), x + 1)
