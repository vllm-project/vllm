# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.v1.worker.sleep_tags import expand_weight_sleep_tags


def test_expand_weight_sleep_tags_preserves_weight_alias():
    assert expand_weight_sleep_tags(["weights"]) == [
        "weights",
        "shared_weights",
        "expert_weights",
    ]


def test_expand_weight_sleep_tags_deduplicates_explicit_buckets():
    assert expand_weight_sleep_tags(["weights", "expert_weights", "kv_cache"]) == [
        "weights",
        "shared_weights",
        "expert_weights",
        "kv_cache",
    ]
