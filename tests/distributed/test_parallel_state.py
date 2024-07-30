import pytest
import torch

from vllm.distributed.parallel_state import _split_tensor_dict


def test_split_tensor_dict():
    test_dict = {
        "key_a": "a",
        "key_b": torch.arange(8, dtype=torch.float32),
        "key_c": {
            "key_1": torch.arange(5, dtype=torch.float32),
            "key_2": torch.tensor([], dtype=torch.float32),
            "key_3": 123,
        },
        "key_d": {},
    }
    metadata_list, tensor_list = _split_tensor_dict(test_dict)
    assert len(metadata_list) == 6
    assert torch.allclose(tensor_list[0], test_dict["key_b"])
    assert torch.allclose(tensor_list[1], test_dict["key_c"]["key_1"])
    assert torch.allclose(tensor_list[2], test_dict["key_c"]["key_2"])


def test_split_tensor_dict_invalid_key():
    test_dict = {
        "a%b": "a",
    }
    with pytest.raises(AssertionError):
        _split_tensor_dict(test_dict)
