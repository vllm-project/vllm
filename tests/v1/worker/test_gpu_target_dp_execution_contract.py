# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch

from vllm.config.compilation import CUDAGraphMode
from vllm.v1.worker.gpu import model_runner as model_runner_module
from vllm.v1.worker.gpu.cudagraph_utils import BatchExecutionDescriptor
from vllm.v1.worker.gpu.dp_utils import DPSyncState
from vllm.v1.worker.gpu.input_batch import InputBatch, InputBuffers
from vllm.v1.worker.gpu.model_runner import GPUModelRunner
from vllm.v1.worker.gpu_worker import Worker

pytestmark = [pytest.mark.cpu_test, pytest.mark.skip_global_cleanup]


def _runner(monkeypatch, order):
    runner = object.__new__(GPUModelRunner)
    runner._pending_target_dp_sync = None
    runner.execute_model_state = None
    runner.dp_execution_contract_enabled = True
    runner.input_buffers = InputBuffers(4, 16, torch.device("cpu"))
    runner.device = torch.device("cpu")
    runner.dp_size = 2
    runner.dp_rank = 0
    runner.lora_config = None
    runner.pcp_manager = None
    runner.is_encoder_decoder = False
    runner.is_first_pp_rank = True
    runner.is_last_pp_rank = True
    runner.uses_inputs_embeds = False
    runner.use_aux_hidden_state_outputs = False
    runner.routed_experts_capturer = None
    runner.attn_groups = []
    runner.model_config = Mock()
    runner.kv_cache_config = Mock()
    runner.req_states = SimpleNamespace(num_computed_tokens=SimpleNamespace(gpu=Mock()))
    runner.model_state = Mock()
    runner.model_state.prepare_attn.return_value = {}
    runner.model_state.prepare_inputs.return_value = {}
    runner.eplb = Mock()
    runner.step_timing = Mock()
    runner.kv_connector = Mock()
    runner.kv_connector.no_forward.return_value = Mock()
    runner.ec_connector = Mock()
    runner.speculator = None

    for name in (
        "update_pp_decode_requests",
        "finish_requests",
        "free_states",
        "add_requests",
        "update_requests",
    ):
        setattr(runner, name, Mock())
    runner.block_tables = Mock()
    runner.block_tables.apply_staged_writes = Mock()
    runner.prepare_attn = Mock(return_value=((torch.zeros(1),), torch.zeros(1)))
    runner.prepare_dummy_attn = Mock(return_value=((torch.zeros(1),), torch.zeros(1)))
    runner._merge_ec_connector_no_forward = Mock(return_value="empty")

    manager = Mock()
    manager.run_fullgraph.side_effect = lambda desc: (
        order.append("forward") or torch.zeros(desc.num_tokens, 4)
    )
    runner.cudagraph_manager = manager
    monkeypatch.setattr(
        model_runner_module,
        "build_slot_mappings_by_layer",
        Mock(return_value={}),
    )
    return runner


def _scheduler_output(num_tokens, num_reqs):
    return SimpleNamespace(
        total_num_scheduled_tokens=num_tokens,
        num_scheduled_tokens={f"req-{i}": num_tokens for i in range(num_reqs)},
        scheduled_encoder_inputs={},
        finished_req_ids=set(),
    )


def _future(order, batch_desc, sync):
    future = Mock()
    resolved = False

    def result(manager):
        nonlocal resolved
        if not resolved:
            order.append("finish")
            resolved = True
        return batch_desc, sync

    future.result.side_effect = result
    future.release.side_effect = lambda: order.append("release")
    return future


def test_target_dp_sync_overlaps_local_input_preparation(monkeypatch):
    order: list[str] = []
    runner = _runner(monkeypatch, order)
    batch_desc = BatchExecutionDescriptor(
        cg_mode=CUDAGraphMode.FULL,
        num_tokens=2,
        num_reqs=1,
        uniform_token_count=2,
    )
    sync = DPSyncState(
        torch.tensor([2, 2]),
        2,
        False,
        generation=3,
        live_num_tokens_across_dp=(2, 2),
        live_num_reqs_across_dp=(1, 1),
        execution_num_reqs=1,
    )
    future = _future(order, batch_desc, sync)
    coordinator = Mock()

    def start(*args, **kwargs):
        order.append("start")
        return future

    coordinator.start.side_effect = start
    runner._target_dp_sync = coordinator
    batch_state = SimpleNamespace(
        num_tokens=2,
        num_scheduled_tokens=np.array([2], dtype=np.int32),
        is_prefilling_np=np.array([False]),
    )
    runner.gather_batch_req_state = Mock(return_value=(batch_state, 2))
    input_batch = InputBatch.make_dummy(1, 2, runner.input_buffers)

    def prepare_inputs(*args):
        order.append("prepare_local")
        args[-1].result(runner.cudagraph_manager)
        return input_batch

    runner.prepare_inputs = Mock(side_effect=prepare_inputs)

    output = runner.execute_model(_scheduler_output(2, 1))

    assert output is None
    assert order == ["start", "prepare_local", "finish", "forward"]
    assert runner.execute_model_state is not None
    assert runner.execute_model_state.target_dp_future is future
    assert runner.execute_model_state.dp_sync is sync
    assert runner._pending_target_dp_sync is None
    future.release.assert_not_called()


def test_zero_token_rank_executes_one_sentinel_and_releases(monkeypatch):
    order: list[str] = []
    runner = _runner(monkeypatch, order)
    batch_desc = BatchExecutionDescriptor(
        cg_mode=CUDAGraphMode.FULL,
        num_tokens=4,
        num_reqs=2,
        uniform_token_count=2,
    )
    sync = DPSyncState(
        torch.tensor([4, 4]),
        2,
        False,
        generation=5,
        live_num_tokens_across_dp=(0, 4),
        live_num_reqs_across_dp=(0, 2),
        execution_num_reqs=2,
    )
    future = _future(order, batch_desc, sync)
    coordinator = Mock()

    def start(*args, **kwargs):
        order.append("start")
        return future

    coordinator.start.side_effect = start
    runner._target_dp_sync = coordinator
    runner.speculator = Mock()

    def dummy_speculator(**kwargs):
        order.append("speculator")
        assert kwargs["input_batch"].is_padding.tolist() == [False, True, True, True]
        assert kwargs["dp_sync"] is sync

    runner._execute_dummy_speculator_stage = Mock(side_effect=dummy_speculator)

    output = runner.execute_model(_scheduler_output(0, 0))

    assert output == "empty"
    assert order == ["start", "finish", "forward", "speculator", "release"]
    runner.kv_connector.pre_forward.assert_not_called()
    assert runner.execute_model_state is None
    assert runner._pending_target_dp_sync is None


def test_engine_core_dummy_contributes_zero_live_work(monkeypatch):
    order: list[str] = []
    runner = _runner(monkeypatch, order)
    batch_desc = BatchExecutionDescriptor(
        cg_mode=CUDAGraphMode.FULL,
        num_tokens=4,
        num_reqs=2,
        uniform_token_count=2,
    )
    sync = DPSyncState(
        torch.tensor([4, 4]),
        2,
        False,
        generation=6,
        live_num_tokens_across_dp=(0, 4),
        live_num_reqs_across_dp=(0, 2),
        execution_num_reqs=2,
    )
    future = _future(order, batch_desc, sync)
    coordinator = Mock()

    def start(*args, **kwargs):
        order.append("start")
        return future

    coordinator.start.side_effect = start
    runner._target_dp_sync = coordinator

    output = runner.execute_model(
        _scheduler_output(1, 1),
        dummy_run=True,
        dp_idle=True,
    )

    assert output == "empty"
    assert coordinator.start.call_args.args[1:3] == (0, 0)
    assert order == ["start", "finish", "forward", "release"]
    assert runner._pending_target_dp_sync is None


@pytest.mark.parametrize("enabled", [False, True])
def test_worker_marks_execution_contract_dummy_as_idle(enabled):
    worker = object.__new__(Worker)
    worker.model_runner = SimpleNamespace(
        uniform_decode_query_len=4,
        dp_execution_contract_enabled=enabled,
        _dummy_run=Mock(),
    )

    worker.execute_dummy_batch()

    expected_kwargs = {"uniform_decode": True}
    if enabled:
        expected_kwargs["dp_idle"] = True
    worker.model_runner._dummy_run.assert_called_once_with(4, **expected_kwargs)


def test_target_dp_sync_releases_when_forward_fails(monkeypatch):
    order: list[str] = []
    runner = _runner(monkeypatch, order)
    batch_desc = BatchExecutionDescriptor(
        cg_mode=CUDAGraphMode.FULL,
        num_tokens=2,
        num_reqs=1,
        uniform_token_count=2,
    )
    sync = DPSyncState(
        torch.tensor([2, 2]),
        2,
        False,
        generation=7,
        live_num_tokens_across_dp=(2, 2),
        live_num_reqs_across_dp=(1, 1),
        execution_num_reqs=1,
    )
    future = _future(order, batch_desc, sync)
    coordinator = Mock()

    def start(*args, **kwargs):
        order.append("start")
        return future

    coordinator.start.side_effect = start
    runner._target_dp_sync = coordinator
    batch_state = SimpleNamespace(
        num_tokens=2,
        num_scheduled_tokens=np.array([2], dtype=np.int32),
        is_prefilling_np=np.array([False]),
    )
    runner.gather_batch_req_state = Mock(return_value=(batch_state, 2))
    input_batch = InputBatch.make_dummy(1, 2, runner.input_buffers)

    def prepare_inputs(*args):
        args[-1].result(runner.cudagraph_manager)
        return input_batch

    runner.prepare_inputs = Mock(side_effect=prepare_inputs)

    def fail_forward(desc):
        order.append("forward")
        raise RuntimeError("forward failed")

    runner.cudagraph_manager.run_fullgraph.side_effect = fail_forward

    with pytest.raises(RuntimeError, match="forward failed"):
        runner.execute_model(_scheduler_output(2, 1))

    assert order == ["start", "finish", "forward", "release"]
    assert runner._pending_target_dp_sync is None


def test_target_dp_sync_releases_when_all_ranks_are_idle(monkeypatch):
    order: list[str] = []
    runner = _runner(monkeypatch, order)
    batch_desc = BatchExecutionDescriptor(
        cg_mode=CUDAGraphMode.NONE,
        num_tokens=0,
        num_reqs=0,
    )
    future = _future(order, batch_desc, None)
    coordinator = Mock()

    def start(*args, **kwargs):
        order.append("start")
        return future

    coordinator.start.side_effect = start
    runner._target_dp_sync = coordinator

    output = runner.execute_model(_scheduler_output(0, 0))

    assert output == "empty"
    assert order == ["start", "finish", "release"]
    runner.cudagraph_manager.run_fullgraph.assert_not_called()
    assert runner._pending_target_dp_sync is None


def test_target_dp_sync_releases_when_local_preparation_fails(monkeypatch):
    order: list[str] = []
    runner = _runner(monkeypatch, order)
    future = Mock()
    future.release.side_effect = lambda: order.append("release")
    coordinator = Mock()

    def start(*args, **kwargs):
        order.append("start")
        return future

    coordinator.start.side_effect = start
    runner._target_dp_sync = coordinator
    batch_state = SimpleNamespace(
        num_tokens=2,
        num_scheduled_tokens=np.array([2], dtype=np.int32),
        is_prefilling_np=np.array([False]),
    )
    runner.gather_batch_req_state = Mock(return_value=(batch_state, 2))
    runner.prepare_inputs = Mock(side_effect=RuntimeError("prepare failed"))

    with pytest.raises(RuntimeError, match="prepare failed"):
        runner.execute_model(_scheduler_output(2, 1))

    assert order == ["start", "release"]
    assert runner._pending_target_dp_sync is None
