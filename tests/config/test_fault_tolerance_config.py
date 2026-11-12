# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the fault-tolerance timeout check."""

import logging

import pytest

from vllm.config.fault_tolerance import (
    FaultToleranceConfig,
    ft_execute_model_timeout_conflicts,
    ft_timeout_conflicts,
    warn_ft_config_conflicts,
)
from vllm.config.parallel import EPLBConfig, ParallelConfig
from vllm.platforms import current_platform

DEEPEP_LL = "DeepEP low-latency kernel timeout"
NIXL_EP = "NIXL EP kernel timeout"
CPU_GROUP = "CPU distributed group timeout"
EXECUTE_MODEL = "execute_model RPC timeout"
WORKER_STORE = "Worker recovery store timeout"
STARTUP_DP_STORE = "Startup DP store timeout"
EPLB_BARRIER = "NIXL EPLB post-read barrier timeout"


def _parallel_config(**overrides) -> ParallelConfig:
    params = {
        "enable_fault_tolerance": True,
        "all2all_backend": "nixl_ep",
        "cpu_distributed_timeout_seconds": 30,
    }
    params.update(overrides)
    return ParallelConfig(**params)


def _recovery(seconds: int) -> FaultToleranceConfig:
    return FaultToleranceConfig(engine_recovery_timeout_sec=seconds)


def _names(parallel_config: ParallelConfig) -> list[str]:
    return [c.name for c in ft_timeout_conflicts(parallel_config)]


def test_fault_tolerance_disabled_reports_nothing(caplog_vllm):
    parallel_config = ParallelConfig(
        enable_fault_tolerance=False,
        tensor_parallel_size=2,
        cpu_distributed_timeout_seconds=1800,
    )
    assert ft_timeout_conflicts(parallel_config) == []

    with caplog_vllm.at_level(logging.WARNING, logger="vllm"):
        warn_ft_config_conflicts(parallel_config)
    assert not any(
        "Fault tolerance is enabled" in r.getMessage() for r in caplog_vllm.records
    )


def test_timeouts_shorter_than_recovery_report_nothing():
    parallel_config = _parallel_config(
        fault_tolerance_config=_recovery(1000),
    )
    assert ft_timeout_conflicts(parallel_config) == []


def test_timeout_equal_to_recovery_is_reported():
    parallel_config = _parallel_config(
        tensor_parallel_size=2,
        fault_tolerance_config=_recovery(300),
    )
    assert _names(parallel_config) == [EXECUTE_MODEL, WORKER_STORE]


def test_timeout_above_recovery_is_reported():
    parallel_config = _parallel_config(
        tensor_parallel_size=2,
        fault_tolerance_config=_recovery(250),
    )
    conflicts = ft_timeout_conflicts(parallel_config)
    assert [c.name for c in conflicts] == [EXECUTE_MODEL, WORKER_STORE]
    assert {c.seconds for c in conflicts} == {300}


def test_unset_cpu_timeout_uses_gloo_default():
    parallel_config = _parallel_config(
        tensor_parallel_size=2,
        cpu_distributed_timeout_seconds=None,
        fault_tolerance_config=_recovery(120),
    )
    conflicts = {c.name: c.seconds for c in ft_timeout_conflicts(parallel_config)}
    assert conflicts[CPU_GROUP] == 1800


def test_startup_dp_store_uses_gloo_default():
    parallel_config = _parallel_config(
        data_parallel_size=2,
        fault_tolerance_config=_recovery(400),
    )
    conflicts = {c.name: c.seconds for c in ft_timeout_conflicts(parallel_config)}
    assert conflicts == {STARTUP_DP_STORE: 1800}


def test_configured_cpu_timeout_is_used():
    parallel_config = _parallel_config(
        tensor_parallel_size=2,
        cpu_distributed_timeout_seconds=200,
        fault_tolerance_config=_recovery(120),
    )
    conflicts = {c.name: c.seconds for c in ft_timeout_conflicts(parallel_config)}
    assert conflicts[CPU_GROUP] == 200


def test_execute_model_timeout_reads_env(monkeypatch):
    monkeypatch.setenv("VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS", "30")
    parallel_config = _parallel_config(
        tensor_parallel_size=2,
        fault_tolerance_config=_recovery(50),
    )
    assert _names(parallel_config) == [WORKER_STORE]


@pytest.mark.parametrize(
    ("backend", "expected"),
    [
        ("nixl_ep", {NIXL_EP}),
        ("deepep_low_latency", {DEEPEP_LL}),
        ("allgather_reducescatter", set()),
    ],
)
def test_all2all_backend_selects_kernel_timeout(backend, expected):
    parallel_config = _parallel_config(
        all2all_backend=backend,
        fault_tolerance_config=_recovery(1),
    )
    assert set(_names(parallel_config)) == expected | {WORKER_STORE}


@pytest.mark.parametrize(
    ("communicator", "reported"),
    [("nixl", True), ("torch_gloo", False)],
)
def test_nixl_eplb_post_read_barrier_timeout(monkeypatch, communicator, reported):
    monkeypatch.setattr(current_platform, "is_cuda_alike", lambda: True)
    parallel_config = _parallel_config(
        tensor_parallel_size=2,
        enable_eplb=True,
        enable_expert_parallel=True,
        eplb_config=EPLBConfig(communicator=communicator),
        fault_tolerance_config=_recovery(1),
    )
    assert (EPLB_BARRIER in _names(parallel_config)) is reported


def test_single_rank_reports_only_worker_recovery_store():
    parallel_config = _parallel_config(
        cpu_distributed_timeout_seconds=None,
        fault_tolerance_config=_recovery(200),
    )
    assert _names(parallel_config) == [WORKER_STORE]


def test_execute_model_shorter_than_cpu_timeout_is_reported():
    parallel_config = _parallel_config(
        tensor_parallel_size=2,
        cpu_distributed_timeout_seconds=600,
        fault_tolerance_config=_recovery(1000),
    )
    assert ft_timeout_conflicts(parallel_config) == []
    assert [c.name for c in ft_execute_model_timeout_conflicts(parallel_config)] == [
        CPU_GROUP
    ]


def test_execute_model_longer_than_kernel_and_cpu_timeouts_reports_nothing():
    parallel_config = _parallel_config(
        tensor_parallel_size=2,
        fault_tolerance_config=_recovery(1000),
    )
    assert ft_execute_model_timeout_conflicts(parallel_config) == []


def test_execute_model_shorter_than_kernel_timeout_is_reported(monkeypatch):
    monkeypatch.setenv("VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS", "20")
    parallel_config = _parallel_config(
        tensor_parallel_size=2,
        cpu_distributed_timeout_seconds=10,
        fault_tolerance_config=_recovery(1000),
    )
    assert [c.name for c in ft_execute_model_timeout_conflicts(parallel_config)] == [
        NIXL_EP
    ]


def test_execute_model_order_ignored_without_tensor_parallelism():
    parallel_config = _parallel_config(
        cpu_distributed_timeout_seconds=600,
        fault_tolerance_config=_recovery(1000),
    )
    assert ft_execute_model_timeout_conflicts(parallel_config) == []


@pytest.mark.parametrize(
    ("communicator", "warned"),
    [("torch_nccl", True), ("pynccl", True), ("nixl", False)],
)
def test_nccl_eplb_communicator_warns(monkeypatch, caplog_vllm, communicator, warned):
    monkeypatch.setattr(current_platform, "is_cuda_alike", lambda: True)
    parallel_config = _parallel_config(
        tensor_parallel_size=2,
        enable_eplb=True,
        enable_expert_parallel=True,
        eplb_config=EPLBConfig(communicator=communicator, use_async=False),
        fault_tolerance_config=_recovery(1000),
    )
    with caplog_vllm.at_level(logging.WARNING, logger="vllm"):
        warn_ft_config_conflicts(parallel_config)

    messages = [
        r.getMessage() for r in caplog_vllm.records if "communicator" in r.getMessage()
    ]
    assert bool(messages) is warned


def test_warning_lists_conflicts(caplog_vllm):
    parallel_config = _parallel_config(
        tensor_parallel_size=2,
        fault_tolerance_config=_recovery(100),
    )
    with caplog_vllm.at_level(logging.WARNING, logger="vllm"):
        warn_ft_config_conflicts(parallel_config)

    records = [
        r for r in caplog_vllm.records if "Fault tolerance is enabled" in r.getMessage()
    ]
    assert len(records) == 1
    message = records[0].getMessage()
    assert "engine_recovery_timeout_sec=100" in message
    for name in (EXECUTE_MODEL, WORKER_STORE):
        assert name in message
    assert message.isascii()
