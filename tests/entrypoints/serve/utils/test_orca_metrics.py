# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
import logging

import pytest

from vllm.entrypoints.serve.utils import orca_metrics

pytestmark = pytest.mark.skip_global_cleanup

_NAMED_METRICS = [("kv_cache_usage_perc", 0.4), ("num_requests_waiting", 2.0)]


def test_create_orca_header_json_roundtrip():
    result = orca_metrics.create_orca_header("JSON", _NAMED_METRICS)
    assert result is not None
    raw = result["endpoint-load-metrics"]
    assert raw.startswith("JSON ")
    parsed = json.loads(raw.removeprefix("JSON "))
    assert parsed == {
        "named_metrics": {
            "kv_cache_usage_perc": 0.4,
            "num_requests_waiting": 2.0,
        }
    }


def test_create_orca_header_text_unchanged():
    result = orca_metrics.create_orca_header("TEXT", _NAMED_METRICS)
    assert result == {
        "endpoint-load-metrics": (
            "TEXT named_metrics.kv_cache_usage_perc=0.4, "
            "named_metrics.num_requests_waiting=2.0"
        )
    }


def test_create_orca_header_logs_at_debug_not_info(caplog):
    caplog.set_level(logging.INFO)
    orca_metrics.create_orca_header("JSON", _NAMED_METRICS)
    assert "Created ORCA header" not in caplog.text

    caplog.clear()
    caplog.set_level(logging.DEBUG, logger="vllm.entrypoints.serve.utils.orca_metrics")
    orca_metrics.create_orca_header("JSON", _NAMED_METRICS)
    assert "Created ORCA header" in caplog.text
