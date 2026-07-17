from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from vllm.v1.core.sched.interface import PauseState
from vllm.v1.engine.core import EngineCoreProc, _resolve_abort_kv_offload
from vllm.v1.request import RequestStatus

pytestmark = pytest.mark.cpu_test


def _scheduler(capability: bool):
    scheduler = Mock()
    scheduler.supports_abort_kv_offload.return_value = capability
    return scheduler


@pytest.mark.parametrize("requested", [True, False])
def test_explicit_abort_offload_value_wins(monkeypatch, requested):
    monkeypatch.setenv("VLLM_ABORT_KV_OFFLOAD", "0" if requested else "1")
    scheduler = _scheduler(capability=not requested)

    assert _resolve_abort_kv_offload(scheduler, requested) is requested
    scheduler.supports_abort_kv_offload.assert_not_called()


@pytest.mark.parametrize(
    ("value", "expected"),
    [("1", True), ("true", True), ("0", False), ("off", False)],
)
def test_environment_override_used_when_argument_is_none(monkeypatch, value, expected):
    monkeypatch.setenv("VLLM_ABORT_KV_OFFLOAD", value)
    scheduler = _scheduler(capability=not expected)

    assert _resolve_abort_kv_offload(scheduler, None) is expected
    scheduler.supports_abort_kv_offload.assert_not_called()


def test_connector_capability_is_default(monkeypatch):
    monkeypatch.delenv("VLLM_ABORT_KV_OFFLOAD", raising=False)
    scheduler = _scheduler(capability=True)

    assert _resolve_abort_kv_offload(scheduler, None) is True


def test_invalid_environment_value_is_rejected(monkeypatch):
    monkeypatch.setenv("VLLM_ABORT_KV_OFFLOAD", "sometimes")

    with pytest.raises(ValueError, match="VLLM_ABORT_KV_OFFLOAD"):
        _resolve_abort_kv_offload(_scheduler(False), None)


def test_engine_core_proc_pause_marks_abort_and_waits_for_idle(monkeypatch):
    monkeypatch.delenv("VLLM_ABORT_KV_OFFLOAD", raising=False)
    scheduler = _scheduler(capability=True)
    scheduler.finish_requests.return_value = [("request-0", 0)]
    engine = SimpleNamespace(
        scheduler=scheduler,
        _send_abort_outputs=Mock(),
        _pause_complete=Mock(return_value=False),
        _idle_state_callbacks=[],
    )

    future = EngineCoreProc.pause_scheduler(
        engine, mode="abort", clear_cache=False
    )

    scheduler.finish_requests.assert_called_once_with(
        None,
        RequestStatus.FINISHED_ABORTED,
        offload_aborted_kv=True,
    )
    scheduler.set_pause_state.assert_called_once_with(PauseState.PAUSED_NEW)
    engine._send_abort_outputs.assert_called_once_with([("request-0", 0)])
    assert future is not None and not future.done()
    assert len(engine._idle_state_callbacks) == 1

    engine._idle_state_callbacks.pop()(engine)
    assert future.done()
