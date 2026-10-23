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


def _parallel_config(tp=4, dp=1, ep=False):
    return ParallelConfig(
        tensor_parallel_size=tp,
        data_parallel_size=dp,
        enable_expert_parallel=ep,
        enable_dbo=True,
    )


@pytest.mark.parametrize(
    ("tp", "dp", "ep", "tp_only"),
    [
        (4, 1, False, True),
        (4, 2, False, False),
        (4, 1, True, False),
        (1, 1, False, False),
    ],
)
def test_dbo_tp_only(tp, dp, ep, tp_only):
    assert _parallel_config(tp, dp, ep).dbo_tp_only is tp_only
    assert not ParallelConfig(tensor_parallel_size=tp).dbo_tp_only  # DBO off


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
        # pure TP: the drafter and the vision encoder run outside the split
        (dict(spec=True), None),
        (dict(mm=True), None),
        (dict(spec=True, mm=True), None),
        # anything that is not pure TP keeps the old restrictions
        (dict(spec=True, dp=2), "speculative decoding"),
        (dict(spec=True, ep=True), "speculative decoding"),
        (dict(spec=True, tp=1), "speculative decoding"),
        (dict(mm=True, dp=2), "multimodal"),
        # encoder-decoder models stay out under pure TP too
        (dict(enc_dec=True), "multimodal"),
    ],
)
def test_dbo_unsupported_features(monkeypatch, kwargs, rejected):
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "1")
    unsupported = VllmConfig._get_dbo_unsupported_features(_config(**kwargs))
    if rejected is None:
        assert unsupported == []
    else:
        assert any(rejected in u for u in unsupported), unsupported


def _parallel(threshold_prefill=8192, threshold_decode=32768):
    return SimpleNamespace(
        use_ubatching=True,
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


@pytest.mark.parametrize(
    "kwargs",
    [dict(dp=2), dict(ep=True), dict(dbo=False)],
)
def test_hook_not_installed_outside_pure_tp_dbo(monkeypatch, kwargs):
    monkeypatch.setattr(parallel_state, "_dbo_tp_all_reduce_hook", None)
    monkeypatch.setattr(ubatching, "_TP_UBATCH_COMMS", None)
    dbo = kwargs.pop("dbo", True)
    pc = _parallel_config(**kwargs) if dbo else ParallelConfig(tensor_parallel_size=4)
    ubatching.init_dbo_tp_comms(pc)
    assert parallel_state._dbo_tp_all_reduce_hook is None
    assert ubatching._TP_UBATCH_COMMS is None


def test_hook_is_inert_outside_a_microbatch(monkeypatch):
    # Outside a microbatch thread or for a non-TP group the hook returns None.
    monkeypatch.setattr(ubatching, "_THREAD_ID_TO_CONTEXT", {})
    assert ubatching._dbo_tp_all_reduce(object(), torch.ones(2)) is None
    monkeypatch.setattr(ubatching, "_THREAD_ID_TO_CONTEXT", {threading.get_ident(): 0})
    monkeypatch.setattr(parallel_state, "_TP", object())
    assert ubatching._dbo_tp_all_reduce(object(), torch.ones(2)) is None


@pytest.mark.parametrize(("rocm", "priority"), [(True, -1), (False, 0)])
def test_comm_stream_priority(monkeypatch, rocm, priority):
    from vllm.platforms import current_platform

    monkeypatch.setattr(current_platform, "is_rocm", lambda: rocm)
    assert ubatching.dbo_comm_stream_priority() == priority
