# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from vllm.models.minimax_m3.amd.specialized_decode.runtime import decode as runtime


@dataclass
class _Health:
    failed: bool = False
    rank: int = -1
    epoch: int = -1


class _Executor:
    def __init__(self):
        self.current_health = _Health()
        self.after_calls = 0
        self.close_calls = 0
        self.after_failure: Exception | None = None
        self.abort_calls = 0

    def health(self):
        return self.current_health

    def after_step(self):
        self.after_calls += 1
        if self.after_failure is not None:
            raise self.after_failure

    def abort(self):
        self.abort_calls += 1

    def close(self):
        self.close_calls += 1


def _decode(executor: _Executor) -> runtime.SpecializedDecode:
    decode = object.__new__(runtime.SpecializedDecode)
    decode.executor = executor
    decode._closed = False
    return decode


@pytest.fixture(autouse=True)
def _single_rank(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        runtime, "get_tp_group", lambda: SimpleNamespace(cpu_group=None)
    )


def test_sticky_health_stops_before_a_new_step():
    executor = _Executor()
    executor.current_health = _Health(True, rank=2, epoch=7)
    decode = _decode(executor)
    with pytest.raises(RuntimeError, match="before step.*rank=2.*epoch=7"):
        decode._check_health("before step")


def test_after_step_checks_only_a_launched_plan():
    executor = _Executor()
    decode = _decode(executor)
    decode.after_step(None)
    decode.after_step(runtime.Step(3, (None,)))
    assert executor.after_calls == 0

    kernel = lambda *args: (args[1], args[2])
    decode.after_step(runtime.Step(3, (kernel,)))
    assert executor.after_calls == 1


def test_after_step_exception_is_fail_stop():
    executor = _Executor()
    executor.after_failure = ValueError("injected")
    decode = _decode(executor)
    kernel = lambda *args: (args[1], args[2])
    with pytest.raises(RuntimeError, match="failed on a TP rank") as error:
        decode.after_step(runtime.Step(3, (kernel,)))
    assert isinstance(error.value.__cause__, ValueError)


def test_layer_exception_aborts_peers_then_fails_on_every_rank():
    executor = _Executor()
    decode = _decode(executor)
    kernel = lambda *args: (args[1], args[2])
    with pytest.raises(RuntimeError, match="failed on a TP rank") as error:
        decode.after_step(runtime.Step(3, (kernel,)), ValueError("launch"))
    assert isinstance(error.value.__cause__, ValueError)
    assert executor.abort_calls == 1
    assert executor.after_calls == 0


def test_stock_step_exception_is_reraised_unchanged():
    decode = _decode(_Executor())
    with pytest.raises(ValueError, match="stock"):
        decode.after_step(None, ValueError("stock"))


def test_close_is_idempotent():
    executor = _Executor()
    decode = _decode(executor)
    decode.close()
    decode.close()
    assert executor.close_calls == 1
