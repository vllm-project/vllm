# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Regression tests for vLLM usage telemetry (vllm/usage/usage_lib.py).

CPU info collection is best-effort; a failure inside a forked worker (e.g.
py-cpuinfo re-executing itself) must degrade gracefully instead of killing
engine startup (#51825).
"""

import json

from vllm.usage import usage_lib
from vllm.usage.usage_lib import UsageContext, UsageMessage


def test_report_usage_survives_cpuinfo_failure(monkeypatch):
    """The unguarded ``cpuinfo.get_cpu_info()`` raises inside forked workers;
    telemetry must fall back to an empty ``info`` dict."""

    def fail_cpu_info():
        raise json.JSONDecodeError("invalid cpuinfo output", "not-json", 0)

    monkeypatch.setattr(usage_lib.cpuinfo, "get_cpu_info", fail_cpu_info)
    monkeypatch.setattr(UsageMessage, "_write_to_file", lambda *_: None)
    monkeypatch.setattr(UsageMessage, "_send_to_server", lambda *_: None)

    message = UsageMessage()
    message._report_usage_once("TestModel", UsageContext.ENGINE_CONTEXT, {})

    assert message.num_cpu is None
    assert message.cpu_type == ""
    assert message.cpu_family_model_stepping == ",,"
    assert message.model_architecture == "TestModel"


def test_report_usage_normal_path_unchanged(monkeypatch):
    """The guard must not change the normal-path behavior."""
    monkeypatch.setattr(
        usage_lib.cpuinfo,
        "get_cpu_info",
        lambda: {
            "count": 8,
            "brand_raw": "Intel Core i7",
            "family": 6,
            "model": 158,
            "stepping": 10,
        },
    )
    monkeypatch.setattr(UsageMessage, "_write_to_file", lambda *_: None)
    monkeypatch.setattr(UsageMessage, "_send_to_server", lambda *_: None)

    message = UsageMessage()
    message._report_usage_once("TestModel", UsageContext.ENGINE_CONTEXT, {})

    assert message.num_cpu == 8
    assert message.cpu_type == "Intel Core i7"
    assert message.cpu_family_model_stepping == "6,158,10"
