# SPDX-License-Identifier: Apache-2.0
"""Unit tests for QuestConfig dataclass."""
from __future__ import annotations

import pytest


def test_quest_config_default_disabled():
    from vllm.config.quest import QuestConfig

    cfg = QuestConfig()
    assert cfg.enabled is False
    assert cfg.backend_name == "QUEST_SPARSE_OFFLOAD"
    assert cfg.top_k == 64
    assert cfg.block_size == 32
    assert cfg.full_kv_layers == [0, 1]
    assert cfg.gpu_cache_blocks_per_seq == 256
    assert cfg.cpu_cache_blocks == 65536
    assert cfg.eviction_policy == "lru"
    assert cfg.enable_async_prefetch is False
    assert cfg.enable_double_buffering is False
    assert cfg.selection_impl == "torch"
    assert cfg.unsupported_model_policy == "error"


def test_quest_config_validates_top_k_positive():
    from vllm.config.quest import QuestConfig

    with pytest.raises(ValueError, match="top_k must be positive"):
        QuestConfig(top_k=0).validate()
    with pytest.raises(ValueError, match="top_k must be positive"):
        QuestConfig(top_k=-1).validate()


def test_quest_config_validates_top_k_against_gpu_budget():
    from vllm.config.quest import QuestConfig

    cfg = QuestConfig(top_k=300, gpu_cache_blocks_per_seq=256)
    with pytest.raises(ValueError, match="gpu_cache_blocks_per_seq"):
        cfg.validate()


def test_quest_config_validates_eviction_policy():
    from vllm.config.quest import QuestConfig

    with pytest.raises(ValueError, match="eviction_policy"):
        QuestConfig(eviction_policy="random").validate()  # type: ignore[arg-type]


def test_quest_config_validates_selection_impl():
    from vllm.config.quest import QuestConfig

    with pytest.raises(ValueError, match="selection_impl"):
        QuestConfig(selection_impl="cublas").validate()  # type: ignore[arg-type]


def test_quest_config_validates_unsupported_model_policy():
    from vllm.config.quest import QuestConfig

    with pytest.raises(ValueError, match="unsupported_model_policy"):
        QuestConfig(unsupported_model_policy="ignore").validate()  # type: ignore[arg-type]


def test_quest_config_full_kv_layers_must_be_list_of_int():
    from vllm.config.quest import QuestConfig

    with pytest.raises(ValueError, match="full_kv_layers"):
        QuestConfig(full_kv_layers=[0, "1"]).validate()  # type: ignore[list-item]


def test_quest_config_to_dict_round_trip():
    from vllm.config.quest import QuestConfig

    original = QuestConfig(enabled=True, top_k=128, full_kv_layers=[0, 1, 2])
    d = original.to_dict()
    assert d["enabled"] is True
    assert d["top_k"] == 128
    restored = QuestConfig.from_dict(d)
    assert restored == original
