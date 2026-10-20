# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import resource
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch

from vllm.models.minimax_m3.amd.specialized_decode.runtime import decode as runtime
from vllm.models.minimax_m3.amd.specialized_decode.runtime.executor import (
    FfnExecutor,
)


class _Executor:
    def __init__(self):
        self.after_calls = 0
        self.abort_calls = 0
        self.close_calls = 0
        self.fault_word = torch.zeros(4, dtype=torch.int32)

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


@pytest.mark.parametrize("step", [None, runtime.Step(3, (_kernel,))])
def test_every_forward_asserts_the_error_word_is_clear(step):
    """Stock and mono steps alike, so a word set between steps (a host
    abort) stops the next step whichever path it takes."""
    executor = _Executor()
    decode = _decode(executor)
    decode.after_step(step)
    executor.fault_word[0] = 1
    with pytest.raises(RuntimeError, match="error word set"):
        decode.after_step(step)


def test_closed_decode_does_not_read_the_freed_word():
    executor = _Executor()
    decode = _decode(executor)
    decode.close()
    executor.fault_word[0] = 1
    decode.after_step(None)


_GATE_IN_FULL_GRAPH = """
import torch
from vllm.models.minimax_m3.amd.specialized_decode.runtime import decode as runtime

decode = object.__new__(runtime.SpecializedDecode)
decode.executor = type("E", (), {})()
decode.executor.fault_word = torch.zeros(4, dtype=torch.int32, device="cuda")
decode._closed = False
token = torch.zeros(1, dtype=torch.int64, device="cuda")

def forward():
    decode.after_step(None)
    token.fill_(42)

forward()
graph = torch.cuda.CUDAGraph()
with torch.cuda.graph(graph):
    forward()
token.zero_()
decode.executor.fault_word[0] = FAIL
graph.replay()
copy_stream = torch.cuda.Stream()
copy_stream.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(copy_stream):
    host = token.to("cpu", non_blocking=True)
    done = torch.cuda.Event(blocking=True)
    done.record(copy_stream)
done.synchronize()
print("token", host.item(), flush=True)
"""


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
@pytest.mark.parametrize("fail", [0, 1])
def test_set_word_stops_a_replayed_step_before_its_output_copy(fail):
    """The gate replayed from a FULL graph: with the word set, work queued
    after it (here, the token write and its copy-stream copy) never reaches
    the host and the process dies; this also fails on a torch build whose
    device asserts compile to nothing."""
    run = subprocess.run(
        [sys.executable, "-c", _GATE_IN_FULL_GRAPH.replace("FAIL", str(fail))],
        capture_output=True,
        text=True,
        timeout=300,
        preexec_fn=lambda: resource.setrlimit(resource.RLIMIT_CORE, (0, 0)),
    )
    if fail:
        assert run.returncode != 0 and "token" not in run.stdout
    else:
        assert run.returncode == 0 and "token 42" in run.stdout, run.stderr


def test_close_is_idempotent():
    executor = _Executor()
    decode = _decode(executor)
    decode.close()
    decode.close()
    assert executor.close_calls == 1
