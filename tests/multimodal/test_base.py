import torch

from vllm.multimodal.base import MultiModalInputs, NestedTensors


def assert_nested_tensors_equal(expected: NestedTensors,
                                actual: NestedTensors):
    assert type(expected) == type(actual)
    if isinstance(expected, torch.Tensor):
        assert torch.equal(expected, actual)
    else:
        for expected_item, actual_item in zip(expected, actual):
            assert_nested_tensors_equal(expected_item, actual_item)


def assert_multimodal_inputs_equal(expected: MultiModalInputs,
                                   actual: MultiModalInputs):
    assert set(expected.keys()) == set(actual.keys())
    for key in expected:
        assert_nested_tensors_equal(expected[key], actual[key])


def test_multimodal_input_batch_single_tensor():
    t = torch.rand([1, 2])
    result = MultiModalInputs.batch([{"image": t}])
    assert_multimodal_inputs_equal(result, {"image": t.unsqueeze(0)})


def test_multimodal_input_batch_multiple_tensors():
    a = torch.rand([1, 1, 2])
    b = torch.rand([1, 1, 2])
    c = torch.rand([1, 1, 2])
    result = MultiModalInputs.batch([{"image": a}, {"image": b}, {"image": c}])
    assert_multimodal_inputs_equal(result, {"image": torch.stack((a, b, c))})


def test_multimodal_input_batch_nested_tensors():
    a = torch.rand([1, 2, 3])
    b = torch.rand([1, 2, 3])
    c = torch.rand([1, 2, 3])
    result = MultiModalInputs.batch([{
        "image": [a]
    }, {
        "image": [b]
    }, {
        "image": [c]
    }])
    assert_multimodal_inputs_equal(
        result, {
            "image": torch.stack(
                (a.unsqueeze(0), b.unsqueeze(0), c.unsqueeze(0)))
        })


def test_lists_of_tensors():
    a = torch.rand([1, 2, 3])
    b = torch.rand([1, 2, 3])
    c = torch.rand([1, 2, 3])
    result = MultiModalInputs.batch([{"image": [a, b]}, {"image": [c]}])
    assert_multimodal_inputs_equal(
        result, {"image": [torch.stack(
            (a, b)), c.unsqueeze(0)]})


def test_batched_lists_of_tensors():
    a = torch.rand([1, 2, 3])
    b = torch.rand([1, 2, 3])
    c = torch.rand([1, 2, 3])
    d = torch.rand([1, 2, 3])
    result = MultiModalInputs.batch([{"image": [a, b]}, {"image": [c, d]}])
    assert_multimodal_inputs_equal(
        result,
        {"image": torch.stack((torch.stack((a, b)), torch.stack((c, d))))})


def test_heterogenous_tensors():
    a = torch.rand([1, 2, 3])
    b = torch.rand([1, 4, 5])
    c = torch.rand([1, 2, 3])
    result = MultiModalInputs.batch([{"image": [a, b]}, {"image": [c]}])
    assert_multimodal_inputs_equal(result, {"image": [[a, b], c.unsqueeze(0)]})
