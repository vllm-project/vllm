# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.multimodal.inputs import MultiModalKwargs, NestedTensors

pytestmark = pytest.mark.cpu_test


def assert_nested_tensors_equal(expected: NestedTensors, actual: NestedTensors):
    assert type(expected) == type(actual)  # noqa: E721
    if isinstance(expected, torch.Tensor):
        assert torch.equal(expected, actual)
    else:
        for expected_item, actual_item in zip(expected, actual):
            assert_nested_tensors_equal(expected_item, actual_item)


def assert_multimodal_inputs_equal(
    expected: MultiModalKwargs, actual: MultiModalKwargs
):
    assert set(expected.keys()) == set(actual.keys())
    for key in expected:
        assert_nested_tensors_equal(expected[key], actual[key])


def test_multimodal_input_batch_single_tensor():
    t = torch.rand([1, 2])
    result = MultiModalKwargs.batch([{"image": t}])
    assert_multimodal_inputs_equal(result, {"image": t.unsqueeze(0)})


def test_multimodal_input_batch_multiple_tensors():
    a = torch.rand([1, 1, 2])
    b = torch.rand([1, 1, 2])
    c = torch.rand([1, 1, 2])
    result = MultiModalKwargs.batch([{"image": a}, {"image": b}, {"image": c}])
    assert_multimodal_inputs_equal(result, {"image": torch.stack([a, b, c])})


def test_multimodal_input_batch_multiple_heterogeneous_tensors():
    a = torch.rand([1, 2, 2])
    b = torch.rand([1, 3, 2])
    c = torch.rand([1, 4, 2])
    result = MultiModalKwargs.batch([{"image": a}, {"image": b}, {"image": c}])
    assert_multimodal_inputs_equal(result, {"image": [a, b, c]})


def test_multimodal_input_batch_nested_tensors():
    a = torch.rand([2, 3])
    b = torch.rand([2, 3])
    c = torch.rand([2, 3])
    result = MultiModalKwargs.batch([{"image": [a]}, {"image": [b]}, {"image": [c]}])
    assert_multimodal_inputs_equal(
        result, {"image": torch.stack([a.unsqueeze(0), b.unsqueeze(0), c.unsqueeze(0)])}
    )


def test_multimodal_input_batch_heterogeneous_lists():
    a = torch.rand([1, 2, 3])
    b = torch.rand([1, 2, 3])
    c = torch.rand([1, 2, 3])
    result = MultiModalKwargs.batch([{"image": [a, b]}, {"image": [c]}])
    assert_multimodal_inputs_equal(
        result, {"image": [torch.stack([a, b]), c.unsqueeze(0)]}
    )


def test_multimodal_input_batch_multiple_batchable_lists():
    a = torch.rand([1, 2, 3])
    b = torch.rand([1, 2, 3])
    c = torch.rand([1, 2, 3])
    d = torch.rand([1, 2, 3])
    result = MultiModalKwargs.batch([{"image": [a, b]}, {"image": [c, d]}])
    assert_multimodal_inputs_equal(
        result, {"image": torch.stack([torch.stack([a, b]), torch.stack([c, d])])}
    )


def test_multimodal_input_batch_mixed_stacking_depths():
    a = torch.rand([1, 2, 3])
    b = torch.rand([1, 3, 3])
    c = torch.rand([1, 4, 3])

    result = MultiModalKwargs.batch([{"image": [a, b]}, {"image": [c]}])
    assert_multimodal_inputs_equal(result, {"image": [[a, b], c.unsqueeze(0)]})

    result = MultiModalKwargs.batch([{"image": [a]}, {"image": [b, c]}])
    assert_multimodal_inputs_equal(result, {"image": [a.unsqueeze(0), [b, c]]})
