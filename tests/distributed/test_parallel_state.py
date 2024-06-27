from typing import Any, Dict

import torch

from vllm.distributed.parallel_state import (_split_tensor_dict,
                                             _update_nested_dict)


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


def test_update_nested_dict():
    flattened_keys_values = [("key1%key2%key3", "value1"),
                             ("key1%key2%key4", "value2"),
                             ("key1%key5", "value3"), ("key6%key7", "value4"),
                             ("key8", "value5")]
    res: Dict[str, Any] = {}

    # Update the nested dictionary with each flattened key-value pair
    for flat_key, value in flattened_keys_values:
        _update_nested_dict(res, flat_key, value)
    assert res == {
        "key1": {
            "key2": {
                "key3": "value1",
                "key4": "value2"
            },
            "key5": "value3"
        },
        "key6": {
            "key7": "value4"
        },
        "key8": "value5"
    }
