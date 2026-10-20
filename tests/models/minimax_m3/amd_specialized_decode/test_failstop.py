# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm.models.minimax_m3.amd.model_state import MiniMaxM3ModelState
from vllm.models.minimax_m3.amd.specialized_decode.runtime import decode as runtime
from vllm.models.minimax_m3.amd.specialized_decode.runtime.executor import (
    FfnExecutor,
)


class _Executor:
    def __init__(self):
        self.after_calls = 0
        self.abort_calls = 0
        self.close_calls = 0

    def after_step(self):
        self.after_calls += 1

    def abort(self):
        self.abort_calls += 1

    def close(self):
        self.close_calls += 1


def _decode(executor: _Executor) -> runtime.SpecializedDecode:
    decode = object.__new__(runtime.SpecializedDecode)
    decode.executor = executor
    decode._closed = False
    return decode


def _kernel(*args):
    return args[1], args[2]


def test_after_step_counts_only_a_launched_plan():
    executor = _Executor()
    decode = _decode(executor)
    decode.after_step(None)
    decode.after_step(runtime.Step(3, (None,)))
    assert executor.after_calls == 0
    decode.after_step(runtime.Step(3, (_kernel,)))
    assert executor.after_calls == 1


def test_layer_exception_marks_peers_and_propagates():
    executor = _Executor()
    decode = _decode(executor)
    with pytest.raises(ValueError, match="launch"):
        decode.after_step(runtime.Step(3, (_kernel,)), ValueError("launch"))
    assert executor.abort_calls == 1
    assert executor.after_calls == 0


def test_stock_step_exception_is_reraised_unchanged():
    executor = _Executor()
    decode = _decode(executor)
    with pytest.raises(ValueError, match="stock"):
        decode.after_step(None, ValueError("stock"))
    assert executor.abort_calls == 0


def test_abort_never_raises():
    executor = object.__new__(FfnExecutor)
    executor.rank = 1

    def broken():
        raise RuntimeError("peer view")

    executor.peer = SimpleNamespace(
        bytes=torch.zeros(16, dtype=torch.uint8),
        addresses=SimpleNamespace(tolist=broken),
    )
    executor.abort()


def test_fault_description_names_the_failed_rank():
    state = object.__new__(MiniMaxM3ModelState)
    message = state.describe_fault([1, 2, 7, 3])
    assert "rank 2" in message and "discarded" in message


def test_close_is_idempotent():
    executor = _Executor()
    decode = _decode(executor)
    decode.close()
    decode.close()
    assert executor.close_calls == 1
