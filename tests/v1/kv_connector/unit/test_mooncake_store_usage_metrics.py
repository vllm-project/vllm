# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Store usage, allocation-size and dedup metrics of MooncakeStoreConnector."""

import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
from prometheus_client import REGISTRY, Counter, Gauge, Histogram

from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store import (
    connector as mooncake_store_connector,
)
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store.metrics import (
    LeaseTracker,
    MooncakeStoreConnectorStats,
    master_usage_fractions,
    parse_master_metrics,
    poll_master_metrics,
)
from vllm.v1.metrics.prometheus import unregister_vllm_metrics

from .utils import create_vllm_config

TEST_MODEL = "test-model"
GiB = 1 << 30

MASTER_METRICS_TEXT = """\
# HELP master_total_capacity_bytes Total capacity
# TYPE master_total_capacity_bytes gauge
master_total_capacity_bytes 4294967296
master_allocated_bytes 2147483648
master_allocated_bytes_per_segment{segment="a"} 1073741824
master_mem_processing_bytes{tenant="x y"} 536870912
master_put_start_discarded_staging_size 268435456
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


def _build_prom(extra_config: dict | None = None, num_engines: int = 1):
    vllm_config = create_vllm_config(
        kv_connector="MooncakeStoreConnector",
        kv_role="kv_both",
        kv_connector_extra_config=extra_config,
    )
    return mooncake_store_connector.MooncakeStoreConnector.build_prom_metrics(
        vllm_config=vllm_config,
        metric_types={Gauge: Gauge, Counter: Counter, Histogram: Histogram},
        labelnames=["model_name", "engine"],
        per_engine_labelvalues={i: [TEST_MODEL, str(i)] for i in range(num_engines)},
    )


def _observe(stats: MooncakeStoreConnectorStats) -> None:
    _build_prom().observe(stats.data, engine_idx=0)


@pytest.fixture
def serve_master():
    """Starts fake Mooncake masters serving the given /metrics text."""
    servers = []

    def serve(text: str) -> str:
        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                body = text.encode()
                self.send_response(200)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args):
                pass

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        servers.append(server)
        return f"http://127.0.0.1:{server.server_port}/metrics"

    yield serve
    for server in servers:
        server.shutdown()


@pytest.fixture
def master_url(serve_master):
    return serve_master(MASTER_METRICS_TEXT)


def _wait_for_sample(series: str, extra: dict[str, str] | None = None) -> None:
    deadline = time.monotonic() + 5
    while _sample(series, extra) is None:
        assert time.monotonic() < deadline, f"{series} was never set"
        time.sleep(0.05)


def _wait_for_value(series: str, value: float) -> None:
    deadline = time.monotonic() + 8
    while _sample(series) != value:
        assert time.monotonic() < deadline, f"{series} never reached {value}"
        time.sleep(0.05)


def test_parse_master_metrics_picks_memory_gauges():
    # master_mem_processing_bytes is not used: vLLM counts its own puts.
    assert parse_master_metrics(MASTER_METRICS_TEXT) == {
        "capacity_bytes": 4 * GiB,
        "allocated_bytes": 2 * GiB,
        "discarded_bytes": GiB // 4,
    }


def test_usage_fractions_from_full_snapshot():
    snapshot = parse_master_metrics(MASTER_METRICS_TEXT)

    assert master_usage_fractions(
        snapshot, leased_bytes=GiB // 4, writing_bytes=GiB // 2
    ) == {
        "occupancy": 0.5,
        "write_usage": 0.125,
        "read_usage": 0.0625,
        "usage": 0.25,
    }


def test_usage_without_discarded_staging_gauge():
    snapshot = {"capacity_bytes": 4 * GiB}

    fractions = master_usage_fractions(snapshot, leased_bytes=GiB, writing_bytes=GiB)
    assert fractions["usage"] == 0.5


def test_stock_master_reports_every_gauge():
    snapshot = {"capacity_bytes": 4 * GiB, "allocated_bytes": GiB}

    assert master_usage_fractions(snapshot, leased_bytes=0, writing_bytes=0) == {
        "occupancy": 0.25,
        "write_usage": 0.0,
        "read_usage": 0.0,
        "usage": 0.0,
    }
    assert (
        master_usage_fractions({"capacity_bytes": 0}, leased_bytes=0, writing_bytes=0)
        == {}
    )


def test_lease_tracker_moves_extended_leases():
    tracker = LeaseTracker(ttl_s=10)

    assert tracker.grant([("a", 256)], now=100.0) == {110: 256}
    # Leased again before expiry: the bytes move, they are not counted twice.
    assert tracker.grant([("a", 256)], now=105.0) == {110: -256, 115: 256}
    # Leased again after expiry: a fresh lease.
    assert tracker.grant([("a", 256)], now=120.0) == {130: 256}


def test_reduce_skips_leases_and_writes():
    stats = MooncakeStoreConnectorStats()
    stats.record_leases({110: 256})
    stats.record_writes(256)

    assert stats.reduce() == {}


def test_prom_exports_allocation_and_existing_keys():
    stats = MooncakeStoreConnectorStats()
    stats.record_operation("save_exists", 0.01, 8, num_existing_keys=5)
    stats.record_operation("save_put", 0.01, 3, num_bytes=96)
    stats.record_operation(
        "save_put", 0.01, 20, num_bytes=640, status="error", num_failed_keys=20
    )

    _observe(stats)

    assert _sample("vllm:mooncake_store_save_existing_keys_total") == 5
    # Failed attempts are observed too, as for vllm:kv_offload_cpu_allocation_size.
    assert _sample("vllm:mooncake_store_allocation_size_count") == 2
    assert _sample("vllm:mooncake_store_allocation_size_sum") == 23
    assert _sample("vllm:mooncake_store_allocation_size_bucket", {"le": "4.0"}) == 1


def test_poll_master_metrics_reads_endpoint(master_url):
    assert poll_master_metrics(master_url) == parse_master_metrics(MASTER_METRICS_TEXT)


def test_poll_master_metrics_skips_unreachable_master():
    assert poll_master_metrics("http://127.0.0.1:1/metrics") is None


def test_prom_usage_gauges_track_master_without_engine_steps(master_url):
    # Nothing is observed: the gauges must not depend on engine steps, which
    # stop while the engine is idle.
    _build_prom({"mooncake_master_metrics_url": master_url}, num_engines=2)
    _wait_for_sample("vllm:mooncake_store_cache_usage_perc", {"engine": "1"})

    for engine in ("0", "1"):
        labels = {"engine": engine}
        assert _sample("vllm:mooncake_store_cache_usage_perc", labels) == 0.0625
        assert _sample("vllm:mooncake_store_cache_write_usage_perc", labels) == 0.0
        assert _sample("vllm:mooncake_store_cache_read_usage_perc", labels) == 0.0
        assert _sample("vllm:mooncake_store_cache_occupancy_perc", labels) == 0.5


def test_prom_read_usage_expires_without_engine_steps(master_url):
    prom = _build_prom({"mooncake_master_metrics_url": master_url})
    stats = MooncakeStoreConnectorStats()
    stats.record_leases({int(time.time()) + 3: GiB})
    prom.observe(stats.data, engine_idx=0)

    _wait_for_value("vllm:mooncake_store_cache_read_usage_perc", 0.25)
    # No further steps: the lease expires on the frontend's clock alone.
    _wait_for_value("vllm:mooncake_store_cache_read_usage_perc", 0.0)


def test_prom_write_usage_tracks_in_flight_puts(master_url):
    prom = _build_prom({"mooncake_master_metrics_url": master_url})
    started = MooncakeStoreConnectorStats()
    started.record_writes(GiB)
    prom.observe(started.data, engine_idx=0)
    _wait_for_value("vllm:mooncake_store_cache_write_usage_perc", 0.25)

    finished = MooncakeStoreConnectorStats()
    finished.record_writes(-GiB)
    prom.observe(finished.data, engine_idx=0)
    _wait_for_value("vllm:mooncake_store_cache_write_usage_perc", 0.0)
