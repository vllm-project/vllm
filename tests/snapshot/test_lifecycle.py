# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock, call, patch

from vllm.v1.engine.core import EngineCoreProc
from vllm.v1.executor.abstract import Executor


def _engine(*, transport_reconnected: bool, dp_group=None):
    parallel_config = SimpleNamespace(
        data_parallel_master_ip="10.0.0.1",
        _data_parallel_master_port_list=[1234],
        stateless_init_dp_group=Mock(return_value="new-dp-group"),
    )
    engine = SimpleNamespace(
        _transport_lock=nullcontext(),
        _transport_reconnected=transport_reconnected,
        _reconnect_transport=Mock(),
        dp_group=dp_group,
        model_executor=Mock(),
        vllm_config=SimpleNamespace(
            parallel_config=parallel_config,
            kv_transfer_config=None,
        ),
    )
    return engine


def test_suspend_and_unlock_delegate_to_model_executor():
    engine = _engine(transport_reconnected=True)

    EngineCoreProc.suspend(engine, "/snapshot/model")
    EngineCoreProc.device_unlock(engine)

    engine.model_executor.suspend.assert_called_once_with("/snapshot/model")
    engine.model_executor.device_unlock.assert_called_once_with()


def test_model_executor_delegates_lifecycle_to_workers():
    executor = SimpleNamespace(collective_rpc=Mock())

    Executor.suspend(executor, "/snapshot/model")
    Executor.device_unlock(executor)
    Executor.resume(executor, "10.0.0.2", "10.0.0.3", "/snapshot/model", None)

    assert executor.collective_rpc.call_args_list == [
        call("suspend", args=("/snapshot/model",)),
        call("device_unlock"),
        call(
            "resume",
            args=("10.0.0.2", "10.0.0.3", "/snapshot/model", None),
        ),
    ]


def test_resume_reconnects_transport_before_worker_restore():
    engine = _engine(transport_reconnected=False)

    with (
        patch("vllm.v1.engine.core.get_local_ip", return_value="10.0.0.2"),
        patch("vllm.v1.engine.core.refresh_scheduler_after_resume") as refresh,
        patch(
            "vllm.v1.engine.core.refresh_scheduler_handshake_metadata_after_resume"
        ) as refresh_metadata,
    ):
        EngineCoreProc.resume(engine, "10.0.0.3", "/snapshot/model")

        assert engine.vllm_config.parallel_config.data_parallel_master_ip == "10.0.0.3"
        engine.model_executor.resume.assert_called_once_with(
            "10.0.0.2",
            "10.0.0.3",
            "/snapshot/model",
            None,
        )
    engine._reconnect_transport.assert_called_once_with("10.0.0.3")
    assert engine._transport_reconnected
    refresh.assert_called_once_with(engine, "10.0.0.2")
    refresh_metadata.assert_called_once_with(engine)


def test_resume_rebuilds_engine_core_dp_group():
    engine = _engine(transport_reconnected=True, dp_group="old-dp-group")

    with (
        patch(
            "vllm.v1.engine.core.stateless_destroy_torch_distributed_process_group"
        ) as destroy_dp_group,
        patch("vllm.v1.engine.core.get_local_ip", return_value="10.0.0.2"),
        patch("vllm.v1.engine.core.refresh_scheduler_after_resume"),
        patch("vllm.v1.engine.core.refresh_scheduler_handshake_metadata_after_resume"),
    ):
        EngineCoreProc.resume(engine, "10.0.0.3", None)

    destroy_dp_group.assert_called_once_with("old-dp-group")
    engine._reconnect_transport.assert_not_called()
    assert engine.vllm_config.parallel_config._data_parallel_master_port_list == []
    assert engine.dp_group == "new-dp-group"
