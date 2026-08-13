# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.config.cache import CacheConfig


def test_requested_block_size_survives_group_min_reduction():
    """Engine init reduces block_size to the min KV-cache-group granularity;
    on hybrid models (DeepSeek-V4: groups of 256/64/8/4) that erases the value
    the user passed from cache_config_info while the attention group still
    honors it (vllm-project/vllm#51163). The pre-reduction value must survive
    into the metric labels."""
    config = CacheConfig(block_size=256)
    assert config.user_specified_block_size

    # The ordering EngineCore._initialize_kv_caches / _apply_ready_response use.
    config.requested_block_size = config.block_size
    config.block_size = 4

    info = config.metrics_info()
    assert info["block_size"] == "4"
    assert info["requested_block_size"] == "256"
    assert info["user_specified_block_size"] == "True"


def test_requested_block_size_is_none_until_kv_cache_init():
    config = CacheConfig(block_size=256)
    assert config.requested_block_size is None
    assert config.metrics_info()["requested_block_size"] == "None"
