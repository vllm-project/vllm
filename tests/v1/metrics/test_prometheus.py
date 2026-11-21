# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
import subprocess
import sys

import pytest
from prometheus_client import CollectorRegistry, multiprocess

from vllm.v1.metrics import prometheus

pytestmark = pytest.mark.cpu_test


@pytest.mark.parametrize("managed_directory", [False, True])
def test_worker_shutdown_removes_only_its_live_gauges(
    monkeypatch, tmp_path, managed_directory
):
    """Spawned API workers inherit the metrics path, not its temporary owner."""
    monkeypatch.setattr(prometheus, "_prometheus_multiproc_dir", None)
    if managed_directory:
        monkeypatch.delenv("PROMETHEUS_MULTIPROC_DIR", raising=False)
        prometheus.setup_multiprocess_prometheus()
    else:
        monkeypatch.setenv("PROMETHEUS_MULTIPROC_DIR", str(tmp_path))

    worker = """
import sys

from prometheus_client import Counter, Gauge
from vllm.v1.metrics import prometheus

assert prometheus._prometheus_multiproc_dir is None
Gauge("vllm:test_active", "Active work", multiprocess_mode="livesum").set(5)
Counter("vllm:test_completed", "Completed work").inc(3)
if sys.argv[1] == "shutdown":
    prometheus.shutdown_prometheus()
    prometheus.shutdown_prometheus()
"""
    try:
        for action in ("leave", "shutdown"):
            subprocess.run(
                [sys.executable, "-c", worker, action],
                check=True,
                capture_output=True,
                text=True,
                timeout=30,
            )

        registry = CollectorRegistry()
        multiprocess.MultiProcessCollector(
            registry, path=os.environ["PROMETHEUS_MULTIPROC_DIR"]
        )
        assert registry.get_sample_value("vllm:test_active") == 5
        assert registry.get_sample_value("vllm:test_completed_total") == 6
    finally:
        if managed_directory:
            prometheus._prometheus_multiproc_dir.cleanup()
