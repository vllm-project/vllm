# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import vllm.distributed.elastic_ep.elastic_execute as elastic_execute
from vllm.config import CUDAGraphMode
from vllm.distributed.elastic_ep.elastic_execute import ElasticEPScalingExecutor


def _make_executor(cudagraph_mode: CUDAGraphMode) -> ElasticEPScalingExecutor:
    worker = SimpleNamespace(
        vllm_config=SimpleNamespace(
            compilation_config=SimpleNamespace(cudagraph_mode=cudagraph_mode)
        ),
    )
    executor = object.__new__(ElasticEPScalingExecutor)
    executor.worker_ref = lambda: worker
    return executor


@pytest.mark.parametrize(
    ("is_rocm", "cudagraph_mode", "expected"),
    [
        (True, CUDAGraphMode.NONE, False),
        (True, CUDAGraphMode.PIECEWISE, True),
        (True, CUDAGraphMode.FULL, True),
        (True, CUDAGraphMode.FULL_DECODE_ONLY, True),
        (True, CUDAGraphMode.FULL_AND_PIECEWISE, True),
        (False, CUDAGraphMode.NONE, False),
        (False, CUDAGraphMode.FULL_AND_PIECEWISE, False),
    ],
)
def test_should_defer_target_group_warmup(
    monkeypatch: pytest.MonkeyPatch,
    is_rocm: bool,
    cudagraph_mode: CUDAGraphMode,
    expected: bool,
) -> None:
    platform = SimpleNamespace(is_rocm=lambda: is_rocm)
    monkeypatch.setattr("vllm.platforms.current_platform", platform)
    executor = _make_executor(cudagraph_mode)

    assert executor._should_defer_target_group_warmup() is expected


@pytest.mark.parametrize("defer", [False, True])
def test_maybe_warm_target_groups_during_prepare(defer: bool) -> None:
    executor = _make_executor(CUDAGraphMode.FULL_AND_PIECEWISE)
    executor._should_defer_target_group_warmup = Mock(return_value=defer)
    executor._warm_dp_ep_device_groups = Mock()
    dp_group = object()
    ep_group = object()

    executor._maybe_warm_target_groups_during_prepare(dp_group, ep_group)

    if defer:
        executor._warm_dp_ep_device_groups.assert_not_called()
    else:
        executor._warm_dp_ep_device_groups.assert_called_once_with(dp_group, ep_group)


@pytest.mark.parametrize("defer", [False, True])
def test_deferred_target_group_warmup_runs_between_graph_release_and_recapture(
    monkeypatch: pytest.MonkeyPatch, defer: bool
) -> None:
    events: list[object] = []

    class FakeMultiBlockTable:
        block_tables: list[object] = []

        def clear(self) -> None:
            events.append("clear")

    runner = SimpleNamespace(
        input_batch=SimpleNamespace(block_table=FakeMultiBlockTable()),
        max_num_tokens=32,
    )
    runner._dummy_run = lambda *args, **kwargs: events.append("dummy")
    worker = SimpleNamespace(model_runner=runner)
    worker.compile_or_warm_up_model = lambda: events.append("compile")

    executor = object.__new__(ElasticEPScalingExecutor)
    executor.worker_ref = lambda: worker
    executor._release_cuda_graphs = lambda: events.append("release")
    executor._should_defer_target_group_warmup = lambda: defer
    executor._warm_dp_ep_device_groups = lambda dp, ep: events.append(("warm", dp, ep))

    dp_group = object()
    ep_group = object()
    monkeypatch.setattr(elastic_execute, "get_dp_group", lambda: dp_group)
    monkeypatch.setattr(elastic_execute, "get_ep_group", lambda: ep_group)
    monkeypatch.setattr(
        elastic_execute, "unlock_workspace", lambda: events.append("unlock")
    )
    monkeypatch.setattr(
        elastic_execute, "lock_workspace", lambda: events.append("lock")
    )

    executor.warm_and_capture()

    expected: list[object] = ["clear", "release"]
    if defer:
        expected.append(("warm", dp_group, ep_group))
    expected.extend(["unlock", "dummy", "compile", "lock"])
    assert events == expected


@pytest.mark.parametrize("is_existing_worker", [False, True])
def test_commit_scale_up_warms_every_target_member(
    is_existing_worker: bool,
) -> None:
    executor = _make_executor(CUDAGraphMode.FULL_AND_PIECEWISE)
    executor.worker.model_runner = SimpleNamespace(setup_eplb_from_mapping=Mock())
    retired_groups = object()
    mapping = object()
    executor.broadcast_expert_mapping = Mock()
    executor.switch_and_prepare = Mock(return_value=retired_groups)
    executor.receive_expert_mapping = Mock(return_value=mapping)
    executor.warm_and_capture = Mock()
    executor._perform_eplb_reshuffle = Mock()
    executor._start_group_cleanup = Mock()

    executor.commit_scale_up(is_existing_worker)

    executor.warm_and_capture.assert_called_once_with()
    executor._perform_eplb_reshuffle.assert_called_once_with(async_op=True)
    if is_existing_worker:
        executor.broadcast_expert_mapping.assert_called_once_with()
        executor.switch_and_prepare.assert_called_once_with()
        executor._start_group_cleanup.assert_called_once_with(retired_groups)
        executor.receive_expert_mapping.assert_not_called()
        executor.worker.model_runner.setup_eplb_from_mapping.assert_not_called()
    else:
        executor.broadcast_expert_mapping.assert_not_called()
        executor.switch_and_prepare.assert_not_called()
        executor.receive_expert_mapping.assert_called_once_with()
        executor.worker.model_runner.setup_eplb_from_mapping.assert_called_once_with(
            mapping
        )
        executor._start_group_cleanup.assert_not_called()


@pytest.mark.parametrize(("removing", "expected_warmups"), [(False, 1), (True, 0)])
def test_commit_scale_down_warms_only_surviving_target_members(
    removing: bool, expected_warmups: int
) -> None:
    executor = _make_executor(CUDAGraphMode.FULL_AND_PIECEWISE)
    retired_groups = object()
    executor.perform_scale_down_eplb_reshuffle = Mock()
    executor.switch_and_remove = Mock()
    executor.switch_and_prepare = Mock(return_value=retired_groups)
    executor.warm_and_capture = Mock()
    executor._start_group_cleanup = Mock()

    executor.commit_scale_down(new_dp_size=2, removing=removing)

    executor.perform_scale_down_eplb_reshuffle.assert_called_once_with(2)
    assert executor.warm_and_capture.call_count == expected_warmups
    if removing:
        executor.switch_and_remove.assert_called_once_with()
        executor.switch_and_prepare.assert_not_called()
        executor._start_group_cleanup.assert_not_called()
    else:
        executor.switch_and_remove.assert_not_called()
        executor.switch_and_prepare.assert_called_once_with()
        executor._start_group_cleanup.assert_called_once_with(retired_groups)
