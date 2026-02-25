# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from argparse import Namespace

from vllm.entrypoints.openai.server_utils import get_uvicorn_log_config


def test_get_uvicorn_log_config_with_metrics_flag():
    args = Namespace(
        log_config_file=None,
        disable_access_log_for_endpoints=None,
        disable_uvicorn_metrics_access_log=True,
        uvicorn_log_level="info",
    )

    log_config = get_uvicorn_log_config(args)
    assert log_config is not None
    assert log_config["filters"]["access_log_filter"]["excluded_paths"] == ["/metrics"]


def test_get_uvicorn_log_config_combines_and_deduplicates_paths():
    args = Namespace(
        log_config_file=None,
        disable_access_log_for_endpoints="/health,/metrics,/ping",
        disable_uvicorn_metrics_access_log=True,
        uvicorn_log_level="info",
    )

    log_config = get_uvicorn_log_config(args)
    assert log_config is not None
    assert log_config["filters"]["access_log_filter"]["excluded_paths"] == [
        "/health",
        "/metrics",
        "/ping",
    ]


def test_get_uvicorn_log_config_prefers_log_config_file(tmp_path):
    expected = {"version": 1, "disable_existing_loggers": False}
    log_config_file = tmp_path / "uvicorn.json"
    log_config_file.write_text(json.dumps(expected))

    args = Namespace(
        log_config_file=str(log_config_file),
        disable_access_log_for_endpoints="/health",
        disable_uvicorn_metrics_access_log=True,
        uvicorn_log_level="info",
    )

    assert get_uvicorn_log_config(args) == expected
