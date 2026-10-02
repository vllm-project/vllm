# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the ``vllm serve`` CLI subcommand."""

import argparse
import os
import socket
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from vllm.entrypoints.cli.serve import run_headless, run_multi_api_server


class _StopAfterEngineConfig(Exception):
    pass


def test_headless_imports_reasoning_parser_plugin_before_engine_config():
    plugin_path = "/tmp/custom_reasoning_parser.py"
    args = argparse.Namespace(
        api_server_count=0,
        reasoning_parser_plugin=plugin_path,
    )
    engine_args = MagicMock()

    with (
        patch(
            "vllm.entrypoints.cli.serve.vllm.AsyncEngineArgs.from_cli_args",
            return_value=engine_args,
        ),
        patch(
            "vllm.reasoning.ReasoningParserManager.import_reasoning_parser"
        ) as import_plugin,
    ):

        def create_engine_config(**kwargs):
            import_plugin.assert_called_once_with(plugin_path)
            raise _StopAfterEngineConfig

        engine_args.create_engine_config.side_effect = create_engine_config

        with pytest.raises(_StopAfterEngineConfig):
            run_headless(args)


@pytest.mark.skipif(sys.platform != "linux", reason="Linux REUSEPORT semantics")
@pytest.mark.parametrize("startup_failure", [False, True])
@pytest.mark.parametrize(
    "platform,count,uds,independent",
    [
        ("linux", 3, None, True),
        ("linux", 1, None, False),
        ("linux", 3, "/tmp/vllm.sock", False),
        ("darwin", 3, None, False),
    ],
)
def test_multi_server_selects_listener_policy(
    platform, count, uds, independent, startup_failure
):
    """HTTP startup selects listeners; the process manager receives that choice."""
    from vllm.entrypoints.launchers.launcher import create_server_socket

    args = argparse.Namespace(headless=False, api_server_count=count, uds=uds)
    config = SimpleNamespace(
        parallel_config=SimpleNamespace(
            data_parallel_rank=0,
            local_engines_only=True,
            data_parallel_backend="mp",
        ),
    )
    addresses = SimpleNamespace(inputs=["in"] * count, outputs=["out"] * count)
    engine_launch = SimpleNamespace(
        engine_manager=MagicMock(),
        coordinator=MagicMock(),
        addresses=addresses,
        tensor_queue=None,
    )
    with (
        create_server_socket(("127.0.0.1", 0), reuse_port=True) as sock,
        patch("vllm.entrypoints.cli.serve.sys.platform", platform),
        patch("vllm.entrypoints.cli.serve.signal.signal"),
        patch("vllm.entrypoints.cli.serve.setup_multiprocess_prometheus"),
        patch("vllm.entrypoints.cli.serve.envs.VLLM_USE_RUST_FRONTEND", False),
        patch(
            "vllm.entrypoints.cli.serve.setup_server", return_value=("http://", sock)
        ),
        patch(
            "vllm.entrypoints.cli.serve.vllm.AsyncEngineArgs.from_cli_args"
        ) as engine,
        patch("vllm.entrypoints.cli.serve.Executor.get_class"),
        patch("vllm.v1.engine.utils.get_engine_zmq_addresses", return_value=addresses),
        patch("vllm.entrypoints.cli.serve.launch_core_engines") as launch,
        patch("vllm.entrypoints.cli.serve.APIServerProcessManager") as manager,
        patch("vllm.entrypoints.cli.serve.wait_for_completion_or_failure"),
    ):
        engine.return_value.create_engine_config.return_value = config
        launch.return_value.__enter__.return_value = engine_launch
        manager.return_value.gather_actual_addresses.return_value = (
            addresses.inputs,
            addresses.outputs,
        )
        instance = manager.return_value

        def make_manager(**kwargs):
            factory = kwargs["socket_factory"]
            if independent:
                with factory() as listener:
                    assert listener.getsockname() == sock.getsockname()
                    assert (
                        os.fstat(listener.fileno()).st_ino
                        != os.fstat(sock.fileno()).st_ino
                    )
            else:
                assert factory is None
            return instance

        manager.side_effect = make_manager
        if startup_failure:
            instance.gather_actual_addresses.side_effect = RuntimeError(
                "handshake failed"
            )
            with pytest.raises(RuntimeError, match="handshake failed"):
                run_multi_api_server(args)
        else:
            run_multi_api_server(args)
        instance.shutdown.assert_called_once()
        engine_launch.engine_manager.shutdown.assert_called_once()
        engine_launch.coordinator.shutdown.assert_called_once()
        assert sock.fileno() == -1


def test_multi_server_closes_reserved_socket_on_config_failure():
    """Startup owns the initial port reservation even before workers exist."""
    args = argparse.Namespace(headless=False, api_server_count=1)
    with (
        socket.socket() as sock,
        patch("vllm.entrypoints.cli.serve.signal.signal"),
        patch("vllm.entrypoints.cli.serve.envs.VLLM_USE_RUST_FRONTEND", False),
        patch(
            "vllm.entrypoints.cli.serve.setup_server", return_value=("http://", sock)
        ),
        patch(
            "vllm.entrypoints.cli.serve.vllm.AsyncEngineArgs.from_cli_args",
            side_effect=ValueError("invalid config"),
        ),
    ):
        with pytest.raises(ValueError, match="invalid config"):
            run_multi_api_server(args)
        assert sock.fileno() == -1
