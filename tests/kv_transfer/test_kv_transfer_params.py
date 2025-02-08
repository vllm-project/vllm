# SPDX-License-Identifier: Apache-2.0
"""Tests for the KVTransferParams class.
"""
from vllm import KVTransferParams


def test_all_none():
    """None should be allowed"""
    KVTransferParams(
        prefix_prompt_ids=None,
        kvcache_load_keys=None,
        kvcache_store_keys=None)
    KVTransferParams(
        prefix_prompt_ids=[None],
        kvcache_load_keys=[None],
        kvcache_store_keys=[None])


if __name__ == "__main__":
    import pytest
    pytest.main([__file__])
