import pytest

from vllm.weave.kv_offload.spec import WeaveOffloadingConfig

pytestmark = pytest.mark.skip_global_cleanup


def test_from_dict_minimal_defaults() -> None:
    cfg = WeaveOffloadingConfig.from_dict({})
    defaults = WeaveOffloadingConfig()

    assert cfg.seed_pool_size_GB == defaults.seed_pool_size_GB
    assert cfg.cxl_kvcache_size_GB == defaults.cxl_kvcache_size_GB
    assert cfg.loom_recompute_ratio == defaults.loom_recompute_ratio
    assert cfg.loom_disable_store_for_recompute == defaults.loom_disable_store_for_recompute
    assert cfg.cxl_numa_node == defaults.cxl_numa_node
    assert cfg.eviction_policy == defaults.eviction_policy


def test_from_dict_parses_extended_fields() -> None:
    cfg = WeaveOffloadingConfig.from_dict(
        {
            "seed_pool_size_GB": 4,
            "cxl_kvcache_size_GB": 5,
            "loom_recompute_ratio": "auto",
            "loom_disable_store_for_recompute": "true",
            "cxl_numa_node": 2,
            "eviction_policy": "arc",
        }
    )

    assert cfg.seed_pool_size_GB == 4
    assert cfg.cxl_kvcache_size_GB == 5
    assert cfg.loom_recompute_ratio == "auto"
    assert cfg.loom_disable_store_for_recompute is True
    assert cfg.cxl_numa_node == 2
    assert cfg.eviction_policy == "arc"


def test_from_dict_rejects_ratio_out_of_bounds() -> None:
    with pytest.raises(ValueError):
        WeaveOffloadingConfig.from_dict({"loom_recompute_ratio": 1.5})


def test_from_dict_rejects_invalid_bool() -> None:
    with pytest.raises(TypeError):
        WeaveOffloadingConfig.from_dict(
            {"loom_disable_store_for_recompute": "maybe"}
        )


def test_from_dict_rejects_negative_bytes() -> None:
    with pytest.raises(ValueError):
        WeaveOffloadingConfig.from_dict({"seed_pool_size_GB": -1})
