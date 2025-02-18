# SPDX-License-Identifier: Apache-2.0
"""Tests for the KVTransferParams class.
"""
import json

import pytest

from vllm.distributed.kv_transfer.kv_transfer_params import KVTransferParams


# Note(shangming): KVCache transfer may have different granularity, such
# as prompt-level or block-level, so the length of kvcache_load_keys and
# kvcache_store_keys has no strong correspondence with the length of
# prefix_prompt_ids.
@pytest.mark.parametrize(
    "prefix_prompt_ids, kvcache_load_keys, kvcache_store_keys",
    [
        (None, None, None),
        ([None], [None], [None]),
        # prefill node cases
        ([1, 2, 3], None, ["key1", "key2", "key3"]),
        ([None, [1, 2, 3]], [None, None], [None, ["key1"]]),
        # decode node cases
        ([1, 2, 3], ["key1", "key2", "key3"], None),
        ([None, [1, 2, 3]], [None, ["key1"]], [None, None]),
        # prefix cache sharing cases
        ([[1, 2, 3], [1, 2]], [["key1", "key2", "key3"], ["key1", "key2"]
                               ], [None, None]),
        ([[1, 2, 3], [4, 5, 6]], [["key1", "key2", "key3"], None
                                  ], [None, ["key4", "key5", "key6"]]),
    ])
def test_kv_transfer_params_none_and_various(prefix_prompt_ids,
                                             kvcache_load_keys,
                                             kvcache_store_keys):
    KVTransferParams(prefix_prompt_ids=prefix_prompt_ids,
                     kvcache_load_keys=kvcache_load_keys,
                     kvcache_store_keys=kvcache_store_keys)


@pytest.mark.parametrize(
    "prefix_prompt_ids, kvcache_load_keys, kvcache_store_keys, exception_msg",
    [
        (["a", "b", "c"], None, None, "prefix_prompt_ids"),
        (None, [1, 2, 3], None, "kvcache_load_keys"),
        (None, None, [1, 2, 3], "kvcache_store_keys"),
        ([[1, "a"], [3, 4]], None, None, "prefix_prompt_ids"),
        (None, [["key1", 2], ["key3"]], None, "kvcache_load_keys"),
    ])
def test_kv_transfer_params_invalid_initialization_with_messages(
        prefix_prompt_ids, kvcache_load_keys, kvcache_store_keys,
        exception_msg):
    with pytest.raises(ValueError, match=exception_msg):
        KVTransferParams(prefix_prompt_ids=prefix_prompt_ids,
                         kvcache_load_keys=kvcache_load_keys,
                         kvcache_store_keys=kvcache_store_keys)


def test_kv_transfer_params_from_optional():
    # Valid JSON string input
    json_input = """{
        "prefix_prompt_ids": [1, 2, 3],
        "kvcache_load_keys": ["key1"],
        "kvcache_store_keys": ["key2"]
    }"""
    params = KVTransferParams.from_optional(json_input)
    assert params.prefix_prompt_ids == [1, 2, 3]
    assert params.kvcache_load_keys == ["key1"]
    assert params.kvcache_store_keys == ["key2"]

    # Valid dictionary input
    dict_input = {
        "prefix_prompt_ids": [1, 2],
        "kvcache_load_keys": ["key3"],
        "kvcache_store_keys": ["key4"]
    }
    params = KVTransferParams.from_optional(dict_input)
    assert params.prefix_prompt_ids == [1, 2]
    assert params.kvcache_load_keys == ["key3"]
    assert params.kvcache_store_keys == ["key4"]

    # None input
    params = KVTransferParams.from_optional(None)
    assert params is None

    # Invalid JSON string format
    with pytest.raises(json.JSONDecodeError):
        KVTransferParams.from_optional("invalid json")

    # Invalid input type
    with pytest.raises(ValueError):
        KVTransferParams.from_optional(12345)


def test_kv_transfer_params_empty_lists():
    params = KVTransferParams(prefix_prompt_ids=[],
                              kvcache_load_keys=[],
                              kvcache_store_keys=[])
    assert params.prefix_prompt_ids == []
    assert params.kvcache_load_keys == []
    assert params.kvcache_store_keys == []


if __name__ == "__main__":
    pytest.main([__file__])
