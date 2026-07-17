# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.utils.import_utils import has_helion

if not has_helion():
    pytest.skip("Helion is not installed", allow_module_level=True)

from scripts.benchmark_helion_kernels import check_correctness


class _FakeKernel:
    helion_settings = None

    def __init__(self, offset: float):
        self.calls = 0
        self.offset = offset

    def __call__(self, output: torch.Tensor, input: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        output.copy_(input + self.offset)
        return input * 2


def test_check_correctness_runs_once_without_mutating_benchmark_inputs():
    kernel = _FakeKernel(offset=1)
    baseline_calls = 0

    def baseline(output: torch.Tensor, input: torch.Tensor) -> torch.Tensor:
        nonlocal baseline_calls
        baseline_calls += 1
        output.copy_(input + 1)
        return input * 2

    output = torch.zeros(4)
    inputs = (output, torch.arange(4))

    check_correctness(kernel, baseline, inputs, "matching")

    assert kernel.calls == 1
    assert baseline_calls == 1
    torch.testing.assert_close(output, torch.zeros(4))


def test_check_correctness_reports_mutated_input_mismatch():
    kernel = _FakeKernel(offset=2)

    def baseline(output: torch.Tensor, input: torch.Tensor) -> torch.Tensor:
        output.copy_(input + 1)
        return input * 2

    inputs = (torch.zeros(4), torch.arange(4))

    with pytest.raises(AssertionError, match="Numerics check failed for case bad"):
        check_correctness(kernel, baseline, inputs, "bad")


def test_check_correctness_reports_return_value_mismatch():
    kernel = _FakeKernel(offset=1)

    def baseline(output: torch.Tensor, input: torch.Tensor) -> torch.Tensor:
        output.copy_(input + 1)
        return input * 3

    inputs = (torch.zeros(4), torch.arange(4))

    with pytest.raises(AssertionError, match="Numerics check failed for case bad"):
        check_correctness(kernel, baseline, inputs, "bad")
