# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm.models.minimax_m3.amd.specialized_decode.runtime import decode as runtime
from vllm.models.minimax_m3.amd.specialized_decode.runtime.executor import (
    FfnExecutor,
)
from vllm.v1.worker.gpu.async_utils import AsyncOutput


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


@pytest.mark.parametrize("word", [[0, 0, 0, 0], [1, 2, 7, 3]])
def test_step_output_raises_on_a_set_fault_word(word):
    output = object.__new__(AsyncOutput)
    output.copy_event = SimpleNamespace(synchronize=lambda: None)
    output._model_fault = torch.tensor(word, dtype=torch.int32)
    if any(word):
        with pytest.raises(RuntimeError, match="discarded"):
            output.get_output()
    else:
        with pytest.raises(AttributeError):
            # past the fault check: the rest of the output is not set up here
            output.get_output()


def test_close_is_idempotent():
    executor = _Executor()
    decode = _decode(executor)
    decode.close()
    decode.close()
    assert executor.close_calls == 1
