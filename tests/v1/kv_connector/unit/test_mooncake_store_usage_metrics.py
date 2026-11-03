# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Store usage, allocation-size and dedup metrics of MooncakeStoreConnector."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from prometheus_client import REGISTRY, Counter, Gauge, Histogram

from vllm.config import set_current_vllm_config
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store import (
    connector as mooncake_store_connector,
)
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store.metrics import (
    MASTER_USAGE_KEY,
    MooncakeStoreConnectorStats,
    master_usage_fractions,
    parse_master_metrics,
)
from vllm.v1.metrics.prometheus import unregister_vllm_metrics

from .test_mooncake_store_connector import _make_kv_cache_config, _make_vllm_config

TEST_MODEL = "test-model"
GiB = 1 << 30

MASTER_METRICS_TEXT = """\
# HELP master_total_capacity_bytes Total capacity
# TYPE master_total_capacity_bytes gauge
master_total_capacity_bytes 4294967296
master_allocated_bytes 2147483648
master_allocated_bytes_per_segment{segment="a"} 1073741824
master_mem_processing_bytes 536870912
master_mem_leased_bytes{tenant="x y"} 268435456
"""


@pytest.fixture(autouse=True)
def _clean_registry():
    unregister_vllm_metrics()
    yield
    unregister_vllm_metrics()


def _labels() -> dict[str, str]:
    return {"model_name": TEST_MODEL, "engine": "0"}


def _sample(series: str, extra: dict[str, str] | None = None) -> float | None:
    return REGISTRY.get_sample_value(series, {**_labels(), **(extra or {})})


def _observe(stats: MooncakeStoreConnectorStats) -> None:
    prom = mooncake_store_connector.MooncakeStoreConnector.build_prom_metrics(
        vllm_config=SimpleNamespace(kv_transfer_config=SimpleNamespace()),  # type: ignore[arg-type]
        metric_types={Gauge: Gauge, Counter: Counter, Histogram: Histogram},
        labelnames=["model_name", "engine"],
        per_engine_labelvalues={0: [TEST_MODEL, "0"]},
    )
    prom.observe(stats.data, engine_idx=0)


def test_parse_master_metrics_picks_memory_gauges():
    assert parse_master_metrics(MASTER_METRICS_TEXT) == {
        "capacity_bytes": 4 * GiB,
        "allocated_bytes": 2 * GiB,
        "processing_bytes": GiB // 2,
        "leased_bytes": GiB // 4,
    }


def test_usage_fractions_from_full_snapshot():
    fractions = master_usage_fractions(parse_master_metrics(MASTER_METRICS_TEXT))

    assert fractions == {
        "occupancy": 0.5,
        "write_usage": 0.125,
        "read_usage": 0.0625,
        "usage": 0.1875,
    }


def test_unpatched_master_reports_occupancy_only():
    snapshot = {"capacity_bytes": 4 * GiB, "allocated_bytes": GiB}

    assert master_usage_fractions(snapshot) == {"occupancy": 0.25}
    assert master_usage_fractions({"capacity_bytes": 0}) == {}


def test_reduce_reports_latest_master_snapshot():
    stats = MooncakeStoreConnectorStats()
    stats.record_master_usage({"capacity_bytes": 4 * GiB, "allocated_bytes": GiB})
    stats.record_master_usage(parse_master_metrics(MASTER_METRICS_TEXT))
    stats.record_operation("save_exists", 0.01, 4, num_existing_keys=3)

    reduced = stats.reduce()

    assert reduced["store_usage_perc"] == 18.75
    assert reduced["store_write_usage_perc"] == 12.5
    assert reduced["store_read_usage_perc"] == 6.25
    assert reduced["store_occupancy_perc"] == 50.0
    assert reduced["save_exists_existing_keys"] == 3
    assert not any(key.startswith(MASTER_USAGE_KEY) for key in reduced)


def test_prom_exports_usage_allocation_and_existing_keys():
    stats = MooncakeStoreConnectorStats()
    stats.record_master_usage(parse_master_metrics(MASTER_METRICS_TEXT))
    stats.record_operation("save_exists", 0.01, 8, num_existing_keys=5)
    stats.record_operation("save_put", 0.01, 3, num_bytes=96)
    stats.record_operation(
        "save_put", 0.01, 20, num_bytes=640, status="error", num_failed_keys=20
    )

    _observe(stats)

    assert _sample("vllm:mooncake_store_cache_usage_perc") == 0.1875
    assert _sample("vllm:mooncake_store_cache_write_usage_perc") == 0.125
    assert _sample("vllm:mooncake_store_cache_read_usage_perc") == 0.0625
    assert _sample("vllm:mooncake_store_cache_occupancy_perc") == 0.5
    assert _sample("vllm:mooncake_store_save_existing_keys_total") == 5
    # Failed attempts are observed too, as for vllm:kv_offload_cpu_allocation_size.
    assert _sample("vllm:mooncake_store_allocation_size_count") == 2
    assert _sample("vllm:mooncake_store_allocation_size_sum") == 23
    assert _sample("vllm:mooncake_store_allocation_size_bucket", {"le": "4.0"}) == 1


def test_prom_skips_usage_gauges_without_master_inputs():
    stats = MooncakeStoreConnectorStats()
    stats.record_master_usage({"capacity_bytes": 4 * GiB, "allocated_bytes": GiB})

    _observe(stats)

    assert _sample("vllm:mooncake_store_cache_occupancy_perc") == 0.25
    assert _sample("vllm:mooncake_store_cache_usage_perc") is None


def _make_scheduler_connector():
    with (
        set_current_vllm_config(_make_vllm_config()),
        patch(
            "vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store."
            "connector.MooncakeStoreScheduler"
        ),
    ):
        return mooncake_store_connector.MooncakeStoreConnector(
            _make_vllm_config(), KVConnectorRole.SCHEDULER, _make_kv_cache_config()
        )


def test_scheduler_stats_carry_master_snapshot_once():
    connector = _make_scheduler_connector()
    assert connector._master_usage_poller is None
    assert connector.get_kv_connector_stats() is None

    poller = MagicMock()
    snapshot = parse_master_metrics(MASTER_METRICS_TEXT)
    poller.take.side_effect = [snapshot, None]
    connector._master_usage_poller = poller

    stats = connector.get_kv_connector_stats()
    assert isinstance(stats, MooncakeStoreConnectorStats)
    assert stats.data[MASTER_USAGE_KEY] == [snapshot]
    assert connector.get_kv_connector_stats() is None

    connector.shutdown()
    poller.close.assert_called_once_with()
