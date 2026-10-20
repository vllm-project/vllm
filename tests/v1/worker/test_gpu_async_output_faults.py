# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Device fault sources of the V2 runner's ``AsyncOutput``: a set source stops a
step's results before any is returned, ordered after that step's main-stream
work and independent of later steps."""

import pytest
import torch

from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.worker.gpu.async_utils import AsyncOutput, DeviceFault
from vllm.v1.worker.gpu.sample.output import SamplerOutput

if not torch.cuda.is_available():
    pytest.skip("needs a GPU", allow_module_level=True)


def _output(token, faults, copy_stream):
    sampler = SamplerOutput(
        torch.tensor([[token]], device="cuda"),
        None,
        None,
        None,
        num_rejected=torch.tensor([0], device="cuda"),
    )
    return AsyncOutput(
        ModelRunnerOutput(["request"], {"request": 0}),
        sampler,
        torch.tensor([1], device="cuda"),
        torch.cuda.current_stream(),
        copy_stream,
        pending_aux_output=None,
        faults=faults,
    )


def _fault(tensor):
    return DeviceFault(tensor, lambda values: f"device fault {values}")


@pytest.fixture
def copy_stream():
    return torch.cuda.Stream()


def test_no_fault_source_returns_tokens(copy_stream):
    assert _output(7, (), copy_stream).get_output().sampled_token_ids == [[7]]


def test_clear_source_returns_tokens(copy_stream):
    word = torch.zeros(4, dtype=torch.int32, device="cuda")
    assert _output(7, (_fault(word),), copy_stream).get_output().sampled_token_ids == [
        [7]
    ]


def test_set_source_raises_before_results_are_converted(copy_stream):
    word = torch.tensor([1, 2, 3, 4], dtype=torch.int32, device="cuda")
    output = _output(7, (_fault(word),), copy_stream)
    with pytest.raises(RuntimeError, match=r"device fault \[1, 2, 3, 4\]"):
        output.get_output()
    assert output.model_runner_output.sampled_token_ids == []


def test_fault_set_by_earlier_main_stream_work_is_seen(copy_stream):
    """The copy waits for the step's main-stream work, however long it runs."""
    word = torch.zeros(4, dtype=torch.int32, device="cuda")
    torch.cuda._sleep(50_000_000)
    word.fill_(1)
    with pytest.raises(RuntimeError, match="device fault"):
        _output(7, (_fault(word),), copy_stream).get_output()


def test_in_flight_outputs_snapshot_their_own_step(copy_stream):
    """With several outputs in flight (async scheduling), each holds the
    source as of its own step: a later fault fails only later steps."""
    word = torch.zeros(4, dtype=torch.int32, device="cuda")
    first = _output(1, (_fault(word),), copy_stream)
    torch.cuda._sleep(10_000_000)
    word.fill_(1)
    second = _output(2, (_fault(word),), copy_stream)
    third = _output(3, (_fault(word),), copy_stream)
    assert first.get_output().sampled_token_ids == [[1]]
    for later in (second, third):
        with pytest.raises(RuntimeError, match="device fault"):
            later.get_output()


def test_output_keeps_its_source_alive(copy_stream):
    word = torch.ones(4, dtype=torch.int32, device="cuda")
    output = _output(7, (_fault(word),), copy_stream)
    del word
    torch.accelerator.empty_cache()
    with pytest.raises(RuntimeError, match="device fault"):
        output.get_output()
