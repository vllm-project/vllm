# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.config.scheduler import SchedulerConfig


def test_disable_hybrid_kv_cache_manager_changes_compilation_hash():
    """`disable_hybrid_kv_cache_manager` makes `get_kv_cache_config` call
    `unify_hybrid_kv_cache_specs`, which rewrites every layer's KVCacheSpec
    (e.g. SlidingWindowSpec -> FullAttentionSpec). That changes KV group
    splitting, page size and the selected attention backend/kernel, so it must
    change the config hash that keys the compiled-artifact cache."""
    default = SchedulerConfig.default_factory(max_model_len=2048)
    unified = SchedulerConfig.default_factory(
        max_model_len=2048, disable_hybrid_kv_cache_manager=True
    )

    assert default.compute_hash() != unified.compute_hash()


def test_unset_and_false_disable_hybrid_hash_identically():
    """The field defaults to None, which means the same thing as an explicit
    False; the two must not be treated as different compiled artifacts."""
    unset = SchedulerConfig.default_factory(max_model_len=2048)
    explicit_false = SchedulerConfig.default_factory(
        max_model_len=2048, disable_hybrid_kv_cache_manager=False
    )

    assert unset.disable_hybrid_kv_cache_manager is None
    assert unset.compute_hash() == explicit_false.compute_hash()
