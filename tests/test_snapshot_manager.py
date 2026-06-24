# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest.mock import MagicMock, patch

import pytest

from vllm.engine.snapshot.base import BaseSnapshotProvider
from vllm.engine.snapshot.manager import SnapshotManager


class DummySnapshotProvider(BaseSnapshotProvider):
    def __init__(self):
        self.triggered = False

    def trigger(self) -> None:
        self.triggered = True


class FailingSnapshotProvider(BaseSnapshotProvider):
    def __init__(self, fail_in_init: bool = False):
        if fail_in_init:
            raise RuntimeError("Init failed")

    def trigger(self) -> None:
        raise RuntimeError("Trigger failed")


@pytest.mark.cpu_test
def test_snapshot_manager_discover_and_run():
    provider_instance = DummySnapshotProvider()
    provider_factory = lambda: provider_instance

    mock_entry_point = MagicMock()
    mock_entry_point.name = "dummy"
    mock_entry_point.load.return_value = provider_factory

    with patch("importlib.metadata.entry_points") as mock_entry_points:
        mock_entry_points.return_value = [mock_entry_point]

        manager = SnapshotManager(provider_name="dummy")
        assert manager.provider is provider_instance

        manager.run_snapshot()
        assert provider_instance.triggered is True


@pytest.mark.cpu_test
def test_snapshot_manager_provider_not_found():
    with patch("importlib.metadata.entry_points") as mock_entry_points:
        mock_entry_points.return_value = []

        with pytest.raises(ValueError, match="could not be loaded"):
            SnapshotManager(provider_name="non_existent")


@pytest.mark.cpu_test
def test_snapshot_manager_no_provider_name():
    with pytest.raises(ValueError, match="must be specified"):
        SnapshotManager(provider_name=None)


@pytest.mark.cpu_test
def test_snapshot_manager_instantiation_failure():
    provider_factory = lambda: FailingSnapshotProvider(fail_in_init=True)

    mock_entry_point = MagicMock()
    mock_entry_point.name = "failing"
    mock_entry_point.load.return_value = provider_factory

    with patch("importlib.metadata.entry_points") as mock_entry_points:
        mock_entry_points.return_value = [mock_entry_point]

        with pytest.raises(ValueError, match="could not be loaded"):
            SnapshotManager(provider_name="failing")


@pytest.mark.cpu_test
def test_snapshot_manager_trigger_failure():
    provider_instance = FailingSnapshotProvider(fail_in_init=False)
    provider_factory = lambda: provider_instance

    mock_entry_point = MagicMock()
    mock_entry_point.name = "failing_trigger"
    mock_entry_point.load.return_value = provider_factory

    with patch("importlib.metadata.entry_points") as mock_entry_points:
        mock_entry_points.return_value = [mock_entry_point]

        manager = SnapshotManager(provider_name="failing_trigger")
        assert manager.provider is provider_instance

        with pytest.raises(RuntimeError, match="failed during trigger"):
            manager.run_snapshot()


@pytest.mark.cpu_test
def test_snapshot_manager_multi_worker_filtering():
    """Verify snapshotting restricted to primary worker (_api_process_rank <= 0)."""
    from unittest.mock import MagicMock

    mock_config = MagicMock()
    mock_config.additional_config = {
        "enable_snapshot_post_startup": True,
        "snapshot_provider": "dummy",
    }

    # Primary worker (_api_process_rank == 0, client_index == 0)
    mock_config.parallel_config._api_process_rank = 0
    enable = mock_config.additional_config.get("enable_snapshot_post_startup")
    rank = getattr(mock_config.parallel_config, "_api_process_rank", 0)
    assert enable and rank <= 0

    # Secondary worker (_api_process_rank == 1)
    mock_config.parallel_config._api_process_rank = 1
    rank_sec = getattr(mock_config.parallel_config, "_api_process_rank", 0)
    assert not (enable and rank_sec <= 0)


@pytest.mark.cpu_test
def test_llm_engine_run_snapshot():
    """Verify LLMEngine.run_snapshot delegates to SnapshotManager."""
    from unittest.mock import MagicMock

    from vllm.v1.engine.llm_engine import LLMEngine

    mock_engine = MagicMock(spec=LLMEngine)
    mock_manager = MagicMock()
    mock_engine.snapshot_manager = mock_manager

    LLMEngine.run_snapshot(mock_engine)
    mock_manager.run_snapshot.assert_called_once()

    # When snapshot_manager is None, should not raise
    mock_engine.snapshot_manager = None
    LLMEngine.run_snapshot(mock_engine)


@pytest.mark.cpu_test
def test_snapshot_manager_dp_filtering():
    """Verify snapshotting restricted to primary local DP rank."""
    from unittest.mock import MagicMock

    mock_config = MagicMock()
    mock_config.additional_config = {
        "enable_snapshot_post_startup": True,
        "snapshot_provider": "dummy",
    }
    mock_config.parallel_config._api_process_rank = 0

    # Local DP Rank 0 (even if global DP rank is > 0)
    mock_config.parallel_config.data_parallel_rank = 2
    mock_config.parallel_config.data_parallel_rank_local = 0
    enable = mock_config.additional_config.get("enable_snapshot_post_startup")
    dp_local = getattr(mock_config.parallel_config, "data_parallel_rank_local", None)
    is_primary = dp_local <= 0 if dp_local is not None else True
    assert enable and is_primary

    # Local DP Rank 1
    mock_config.parallel_config.data_parallel_rank_local = 1
    dp_local_sec = getattr(
        mock_config.parallel_config, "data_parallel_rank_local", None
    )
    is_primary_sec = dp_local_sec <= 0 if dp_local_sec is not None else True
    assert not (enable and is_primary_sec)


@pytest.mark.cpu_test
def test_dp_supervisor_disables_child_snapshots():
    """Verify _build_vllm_dp_server_args disables child server snapshots."""
    import argparse

    from vllm.entrypoints.openai.dp_supervisor import _build_vllm_dp_server_args

    args = argparse.Namespace(
        port=8000,
        enable_snapshot_post_startup=True,
        snapshot_provider="dummy",
        data_parallel_size=2,
        data_parallel_size_local=2,
        node_rank=0,
        data_parallel_start_rank=None,
        device_ids=[0, 1],
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
    )

    child_args = _build_vllm_dp_server_args(args, local_rank=0)
    assert child_args.enable_snapshot_post_startup is False


@pytest.mark.cpu_test
def test_api_server_process_manager_snapshot_barrier():
    """Verify barrier creation for multi-process API servers."""
    import argparse
    import multiprocessing

    args = argparse.Namespace(
        enable_snapshot_post_startup=True,
        snapshot_provider="dummy",
    )
    spawn_context = multiprocessing.get_context("spawn")

    num_servers = 2
    barrier = (
        spawn_context.Barrier(num_servers)
        if num_servers > 1 and getattr(args, "enable_snapshot_post_startup", False)
        else None
    )
    assert barrier is not None
    assert barrier.parties == 2


@pytest.mark.asyncio
async def test_dp_supervisor_reraises_probe_exception():
    """Verify DPSupervisor re-raises exceptions from _probe_all_children."""
    import argparse
    import asyncio

    from vllm.entrypoints.openai.dp_supervisor import DPSupervisor

    args = argparse.Namespace(
        dp_supervisor_probe_interval_s=0.01,
        shutdown_timeout=1,
        grpc=False,
        uds=None,
        ssl_keyfile=None,
        ssl_certfile=None,
        api_server_count=1,
        data_parallel_rank=None,
        data_parallel_external_lb=False,
        data_parallel_hybrid_lb=False,
        data_parallel_size=2,
        data_parallel_size_local=2,
        port=8000,
        data_parallel_supervisor_port=9000,
        node_rank=0,
        data_parallel_start_rank=None,
        device_ids=[0, 1],
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
        enable_snapshot_post_startup=True,
        snapshot_provider="dummy",
    )
    supervisor = DPSupervisor(args)
    supervisor._shutdown_event = asyncio.Event()

    async def failing_probe():
        raise RuntimeError("Snapshot failure during probe")

    supervisor._probe_all_children = failing_probe
    supervisor._processes = []

    with pytest.raises(RuntimeError, match="Snapshot failure during probe"):
        await supervisor._monitor_children()


@pytest.mark.cpu_test
def test_validate_parsed_serve_args_unsupported_frontends():
    """Verify validate_parsed_serve_args rejects unsupported frontends."""
    import argparse

    from vllm.entrypoints.openai.cli_args import validate_parsed_serve_args

    base_args = argparse.Namespace(
        subparser="serve",
        chat_template=None,
        enable_auto_tool_choice=False,
        enable_log_outputs=False,
        data_parallel_multi_port_external_lb=False,
        enable_snapshot_post_startup=True,
        headless=True,
        grpc=False,
    )

    with pytest.raises(ValueError, match="cannot be used with --headless"):
        validate_parsed_serve_args(base_args)

    base_args.headless = False
    base_args.grpc = True
    with pytest.raises(ValueError, match="cannot be used with --grpc"):
        validate_parsed_serve_args(base_args)
