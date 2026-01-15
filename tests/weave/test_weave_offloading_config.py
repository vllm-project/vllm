import pytest

from vllm.weave.offloading_spec import WeaveOffloadingConfig, WeaveOffloadingMode

pytestmark = pytest.mark.skip_global_cleanup


def test_from_dict_minimal_dram_bytes_to_use() -> None:
    cfg = WeaveOffloadingConfig.from_dict({"dram_bytes_to_use": 1024})
    defaults = WeaveOffloadingConfig()

    assert cfg.dram_bytes_to_use == 1024
    assert cfg.cxl_bytes_to_use == 0
    assert cfg.mode == WeaveOffloadingMode.DEFAULT

    assert cfg.dram_high_watermark == defaults.dram_high_watermark
    assert cfg.dram_low_watermark == defaults.dram_low_watermark
    assert cfg.kv_prefill_dram_ratio == defaults.kv_prefill_dram_ratio
    assert cfg.decode_allow_sync_cxl_read == defaults.decode_allow_sync_cxl_read


def test_from_dict_accepts_cpu_bytes_to_use_alias() -> None:
    cfg = WeaveOffloadingConfig.from_dict({"cpu_bytes_to_use": 2048})
    assert cfg.dram_bytes_to_use == 2048


def test_from_dict_accepts_pool_size_gb_keys() -> None:
    cfg = WeaveOffloadingConfig.from_dict({"dram_pool_size_gb": 2, "cxl_pool_size_gb": 3})

    assert cfg.dram_bytes_to_use == 2 * (1024**3)
    assert cfg.cxl_bytes_to_use == 3 * (1024**3)


def test_from_dict_parses_extended_fields() -> None:
    cfg = WeaveOffloadingConfig.from_dict(
        {
            "dram_bytes_to_use": 1,
            "cxl_bytes_to_use": 2,
            "mode": "dram",
            "dram_high_watermark": 0.9,
            "dram_low_watermark": 0.7,
            "kv_prefill_dram_ratio": "auto",
            "flush_batch_size_MB": 32,
            "flush_budget_MBps": 123,
            "kv_hot_window_tokens": 111,
            "kv_prefetch_blocks": 4,
            "promotion_budget_MBps": 55,
            "decode_allow_sync_cxl_read": "false",
        }
    )

    assert cfg.dram_bytes_to_use == 1
    assert cfg.cxl_bytes_to_use == 2
    assert cfg.mode == WeaveOffloadingMode.DRAM_ONLY

    assert cfg.dram_high_watermark == 0.9
    assert cfg.dram_low_watermark == 0.7

    assert cfg.kv_prefill_dram_ratio == "auto"
    assert cfg.flush_batch_size_MB == 32
    assert cfg.flush_budget_MBps == 123

    assert cfg.kv_hot_window_tokens == 111
    assert cfg.kv_prefetch_blocks == 4
    assert cfg.promotion_budget_MBps == 55
    assert cfg.decode_allow_sync_cxl_read is False


def test_weave_offloading_config_from_dict_success_minimal() -> None:
    cfg = WeaveOffloadingConfig.from_dict(
        {
            "dram_pool_size_gb": 1,
            "cxl_pool_size_gb": 2,
        }
    )
    assert cfg.dram_bytes_to_use == 1 * (1024**3)
    assert cfg.cxl_bytes_to_use == 2 * (1024**3)
    assert cfg.cxl_numa_node is None


def test_weave_offloading_config_from_dict_success_full() -> None:
    cfg = WeaveOffloadingConfig.from_dict(
        {
            "dram_pool_size_gb": 1,
            "cxl_pool_size_gb": 2,
            "mode": "default",
            "dram_high_watermark": 0.9,
            "dram_low_watermark": 0.5,
            "kv_prefill_dram_ratio": 0.5,
            "flush_batch_size_MB": 32,
            "flush_budget_MBps": 128,
            "kv_hot_window_tokens": 1024,
            "kv_prefetch_blocks": 4,
            "promotion_budget_MBps": 512,
            "decode_allow_sync_cxl_read": "true",
            "cxl_numa_node": 1,
        }
    )

    assert cfg.dram_bytes_to_use == 1 * (1024**3)
    assert cfg.cxl_bytes_to_use == 2 * (1024**3)
    assert cfg.mode == WeaveOffloadingMode.DEFAULT
    assert cfg.dram_high_watermark == 0.9
    assert cfg.dram_low_watermark == 0.5
    assert cfg.kv_prefill_dram_ratio == 0.5
    assert cfg.flush_batch_size_MB == 32
    assert cfg.flush_budget_MBps == 128
    assert cfg.kv_hot_window_tokens == 1024
    assert cfg.kv_prefetch_blocks == 4
    assert cfg.promotion_budget_MBps == 512
    assert cfg.decode_allow_sync_cxl_read is True
    assert cfg.cxl_numa_node == 1


def test_from_dict_rejects_ratio_out_of_bounds() -> None:
    with pytest.raises(ValueError):
        WeaveOffloadingConfig.from_dict({"dram_bytes_to_use": 1, "kv_prefill_dram_ratio": 1.5})


def test_from_dict_rejects_inverted_watermarks() -> None:
    with pytest.raises(ValueError):
        WeaveOffloadingConfig.from_dict(
            {
                "dram_bytes_to_use": 1,
                "dram_high_watermark": 0.6,
                "dram_low_watermark": 0.7,
            }
        )


def test_from_dict_rejects_invalid_bool() -> None:
    with pytest.raises(TypeError):
        WeaveOffloadingConfig.from_dict(
            {"dram_bytes_to_use": 1, "decode_allow_sync_cxl_read": "maybe"}
        )


def test_from_dict_rejects_negative_bytes() -> None:
    with pytest.raises(ValueError):
        WeaveOffloadingConfig.from_dict({"dram_bytes_to_use": -1})
