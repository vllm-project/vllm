# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import Mock

import torch

from vllm.sequence import IntermediateTensors
from vllm.v1.worker import gpu_worker
from vllm.v1.worker.gpu_worker import Worker


class _MockPPGroup:
    def __init__(self):
        self.is_first_rank = False
        self.is_last_rank = False
        self.recv_all_gather_tensors: dict[str, bool] | None = None
        self.send_all_gather_tensors: dict[str, bool] | None = None

    def irecv_tensor_dict(self, all_gather_group, all_gather_tensors):
        self.recv_all_gather_tensors = all_gather_tensors
        return {"residual": torch.zeros(8)}, [], []

    def isend_tensor_dict(self, tensors, all_gather_group, all_gather_tensors):
        self.send_all_gather_tensors = all_gather_tensors
        return []


def _build_worker(enable_sp: bool, enable_sp_moe: bool):
    worker = Worker.__new__(Worker)
    worker._pp_send_work = []
    worker.profiler = None
    worker.use_v2_model_runner = False
    worker.vllm_config = SimpleNamespace(
        compilation_config=SimpleNamespace(
            pass_config=SimpleNamespace(
                enable_sp=enable_sp,
                enable_sp_moe=enable_sp_moe,
            )
        ),
        parallel_config=SimpleNamespace(
            pipeline_parallel_size=2,
            distributed_executor_backend="mp",
        ),
    )

    determine_padding = Mock(
        return_value=(None, SimpleNamespace(num_tokens=8), None, None, None)
    )
    execute_model = Mock(
        return_value=IntermediateTensors(
            {
                "residual": torch.zeros(8),
            }
        )
    )

    worker.model_runner = SimpleNamespace(
        _determine_batch_execution_and_padding=determine_padding,
        execute_model=execute_model,
    )
    return worker, determine_padding


def test_execute_model_sets_residual_all_gather_override_for_sp_moe(monkeypatch):
    worker, determine_padding = _build_worker(enable_sp=False, enable_sp_moe=True)
    pp_group = _MockPPGroup()

    monkeypatch.setattr(gpu_worker, "get_pp_group", lambda: pp_group)
    monkeypatch.setattr(gpu_worker, "get_tp_group", lambda: object())
    monkeypatch.setattr(
        gpu_worker, "is_residual_scattered_for_sp", lambda *_args, **_kwargs: True
    )

    scheduler_output = SimpleNamespace(
        total_num_scheduled_tokens=8,
        num_scheduled_tokens={"req_0": 8},
    )

    result = Worker.execute_model(worker, scheduler_output)

    assert result is None
    determine_padding.assert_called_once()
    assert pp_group.recv_all_gather_tensors == {"residual": False}
    assert pp_group.send_all_gather_tensors == {"residual": False}


def test_execute_model_skips_override_without_sp_or_sp_moe(monkeypatch):
    worker, determine_padding = _build_worker(enable_sp=False, enable_sp_moe=False)
    pp_group = _MockPPGroup()

    monkeypatch.setattr(gpu_worker, "get_pp_group", lambda: pp_group)
    monkeypatch.setattr(gpu_worker, "get_tp_group", lambda: object())

    scheduler_output = SimpleNamespace(
        total_num_scheduled_tokens=8,
        num_scheduled_tokens={"req_0": 8},
    )

    result = Worker.execute_model(worker, scheduler_output)

    assert result is None
    determine_padding.assert_not_called()
    assert pp_group.recv_all_gather_tensors == {}
    assert pp_group.send_all_gather_tensors == {}
