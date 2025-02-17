# SPDX-License-Identifier: Apache-2.0
"""Tests for the KVTransferParams class.
"""
from vllm.distributed.kv_transfer.kv_transfer_params import KVTransferParams


def test_all_none():
    # None should be allowed
    KVTransferParams(prefix_prompt_ids=None,
                     kvcache_load_keys=None,
                     kvcache_store_keys=None)
    KVTransferParams(prefix_prompt_ids=[None],
                     kvcache_load_keys=[None],
                     kvcache_store_keys=[None])

    # Note(shangming): KVCache transfer may have different granularity,
    # such as block-level, so the length of kvcache_load_keys and
    # kvcache_store_keys has no strong correspondence with the length of
    # prefix_prompt_ids.

    # prefill node cases
    KVTransferParams(prefix_prompt_ids=[1, 2, 3],
                     kvcache_load_keys=None,
                     kvcache_store_keys=["key1", "key2", "key3"])

    KVTransferParams(prefix_prompt_ids=[None, [1, 2, 3]],
                     kvcache_load_keys=[None, None],
                     kvcache_store_keys=[None, ["key1"]])

    # decode node cases
    KVTransferParams(prefix_prompt_ids=[1, 2, 3],
                     kvcache_load_keys=["key1", "key2", "key3"],
                     kvcache_store_keys=None)
    KVTransferParams(prefix_prompt_ids=[None, [1, 2, 3]],
                     kvcache_load_keys=[None, ["key1"]],
                     kvcache_store_keys=[None, None])

    # prefix cache sharing cases
    KVTransferParams(prefix_prompt_ids=[[1, 2, 3], [1, 2]],
                     kvcache_load_keys=[["key1", "key2", "key3"],
                                        ["key1", "key2"]],
                     kvcache_store_keys=[None, None])
    KVTransferParams(prefix_prompt_ids=[[1, 2, 3], [4, 5, 6]],
                     kvcache_load_keys=[["key1", "key2", "key3"], None],
                     kvcache_store_keys=[None, ["key4", "key5", "key6"]])


if __name__ == "__main__":
    import pytest
    pytest.main([__file__])
