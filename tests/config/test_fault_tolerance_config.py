# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the fault-tolerance timeout warning."""

import logging

import pytest

from vllm.config.fault_tolerance import FaultToleranceConfig, warn_ft_config_conflicts
from vllm.config.parallel import EPLBConfig, ParallelConfig
from vllm.platforms import current_platform

RECOVERY = "engine_recovery_timeout_sec"
RPC = "VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS"


def _warnings(caplog_vllm, **overrides) -> list[str]:
    params = {
        "enable_fault_tolerance": True,
        "all2all_backend": "nixl_ep",
        "data_parallel_size": 4,
        "distributed_executor_backend": "mp",
        "cpu_distributed_timeout_seconds": 30,
        "fault_tolerance_config": FaultToleranceConfig(),
    }
    with caplog_vllm.at_level(logging.WARNING, logger="vllm"):
        warn_ft_config_conflicts(ParallelConfig(**{**params, **overrides}))
    return [
        r.getMessage()
        for r in caplog_vllm.records
        if r.getMessage().startswith("Fault tolerance")
    ]


@pytest.mark.parametrize(
    ("overrides", "rpc", "expected"),
    [
        ({}, 60, []),
        ({"enable_fault_tolerance": False}, 300, []),
        ({"cpu_distributed_timeout_seconds": None}, 60, [RECOVERY, RPC]),
        ({"all2all_backend": "deepep_low_latency"}, 60, [RECOVERY, RPC]),
        ({}, 300, [RECOVERY]),
        ({"distributed_executor_backend": "uni", "data_parallel_size": 1}, 300, []),
    ],
)
def test_ft_timeout_warnings(monkeypatch, caplog_vllm, overrides, rpc, expected):
    monkeypatch.setenv(RPC, str(rpc))
    messages = _warnings(caplog_vllm, **overrides)
    reported = [m.split(" not shorter than ")[1].split("=")[0] for m in messages]
    assert reported == expected


@pytest.mark.parametrize(
    ("communicator", "warned"), [("pynccl", True), ("nixl", False)]
)
def test_nccl_eplb_communicator_warns(monkeypatch, caplog_vllm, communicator, warned):
    monkeypatch.setenv(RPC, "60")
    monkeypatch.setattr(current_platform, "is_cuda_alike", lambda: True)
    messages = _warnings(
        caplog_vllm,
        enable_eplb=True,
        enable_expert_parallel=True,
        eplb_config=EPLBConfig(communicator=communicator, use_async=False),
    )
    assert any("NCCL watchdog" in m for m in messages) is warned
