# SPDX-License-Identifier: Apache-2.0
"""Tests for the ``lmcache server`` CLI command."""

# Standard
from unittest.mock import patch
import argparse

# Third Party
import pytest

# First Party
from lmcache.cli.commands.server import ServerCommand


@pytest.fixture
def cmd():
    return ServerCommand()


@pytest.fixture
def parser(cmd):
    """An ArgumentParser with ServerCommand's arguments registered."""
    p = argparse.ArgumentParser()
    sub = p.add_subparsers()
    cmd.register(sub)
    return p


class TestServerCommandMetadata:
    def test_name(self, cmd):
        assert cmd.name() == "server"

    def test_help(self, cmd):
        assert "server" in cmd.help().lower()


class TestServerCommandArguments:
    def test_registers_subcommand(self, parser):
        """The 'server' subcommand should be parseable."""
        args = parser.parse_args(
            [
                "server",
                "--l1-size-gb",
                "4",
                "--eviction-policy",
                "LRU",
            ]
        )
        assert hasattr(args, "func")

    def test_mp_server_args_registered(self, parser):
        args = parser.parse_args(
            [
                "server",
                "--host",
                "0.0.0.0",
                "--port",
                "6666",
                "--l1-size-gb",
                "4",
                "--eviction-policy",
                "LRU",
            ]
        )
        assert args.host == "0.0.0.0"
        assert args.port == 6666

    def test_http_frontend_args_registered(self, parser):
        args = parser.parse_args(
            [
                "server",
                "--http-host",
                "127.0.0.1",
                "--http-port",
                "9000",
                "--l1-size-gb",
                "4",
                "--eviction-policy",
                "LRU",
            ]
        )
        assert args.http_host == "127.0.0.1"
        assert args.http_port == 9000

    def test_prometheus_args_registered(self, parser):
        args = parser.parse_args(
            [
                "server",
                "--prometheus-port",
                "9999",
                "--l1-size-gb",
                "4",
                "--eviction-policy",
                "LRU",
            ]
        )
        assert args.prometheus_port == 9999

    def test_default_values(self, parser):
        """Required args only — everything else should get defaults."""
        args = parser.parse_args(
            [
                "server",
                "--l1-size-gb",
                "4",
                "--eviction-policy",
                "LRU",
            ]
        )
        assert args.host == "localhost"
        assert args.port == 5555
        assert args.http_host == "0.0.0.0"
        assert args.http_port == 8080


class TestServerCommandExecute:
    def test_func_bound_to_execute(self, cmd, parser):
        """parser.parse_args should bind func to ServerCommand.execute."""
        args = parser.parse_args(
            [
                "server",
                "--l1-size-gb",
                "4",
                "--eviction-policy",
                "LRU",
            ]
        )
        assert args.func == cmd.execute

    def test_execute_calls_run_http_server(self, parser):
        """execute() should call run_http_server with parsed configs."""
        http_server = pytest.importorskip("lmcache.v1.multiprocess.http_server")
        with patch.object(http_server, "run_http_server") as mock_run:
            args = parser.parse_args(
                [
                    "server",
                    "--l1-size-gb",
                    "4",
                    "--eviction-policy",
                    "LRU",
                ]
            )
            cmd = ServerCommand()
            cmd.execute(args)

            mock_run.assert_called_once()
            kwargs = mock_run.call_args.kwargs
            assert "http_config" in kwargs
            assert "mp_config" in kwargs
            assert "storage_manager_config" in kwargs
            assert "obs_config" in kwargs
