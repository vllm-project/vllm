# SPDX-License-Identifier: Apache-2.0
"""Tests for the ``lmcache coordinator`` CLI command."""

# Standard
from unittest.mock import MagicMock, patch
import argparse

# Third Party
import pytest

# First Party
from lmcache.cli.commands.coordinator import CoordinatorCommand


@pytest.fixture
def cmd():
    return CoordinatorCommand()


@pytest.fixture
def parser(cmd):
    """An ArgumentParser with CoordinatorCommand's arguments registered."""
    p = argparse.ArgumentParser()
    sub = p.add_subparsers()
    cmd.register(sub)
    return p


class TestCoordinatorCommandMetadata:
    def test_name(self, cmd):
        assert cmd.name() == "coordinator"

    def test_help(self, cmd):
        assert "coordinator" in cmd.help().lower()


class TestCoordinatorCommandArguments:
    def test_all_flags_registered(self, parser):
        """Every MPCoordinatorConfig field is settable via a CLI flag."""
        args = parser.parse_args(
            [
                "coordinator",
                "--host",
                "127.0.0.1",
                "--port",
                "9999",
                "--instance-timeout",
                "15",
                "--health-check-interval",
                "7",
                "--eviction-check-interval",
                "3",
                "--eviction-ratio",
                "0.5",
                "--trigger-watermark",
                "0.9",
                "--chunk-size",
                "512",
                "--hash-algorithm",
                "sha256",
                "--blend-probe-stride",
                "2",
                "--timeout-keep-alive",
                "15",
            ]
        )
        assert args.host == "127.0.0.1"
        assert args.port == 9999
        assert args.chunk_size == 512
        assert args.hash_algorithm == "sha256"
        assert args.blend_probe_stride == 2
        assert args.timeout_keep_alive == 15

    def test_flags_default_to_none(self, parser):
        """Unset flags default to None so env/config defaults win."""
        args = parser.parse_args(["coordinator"])
        assert args.chunk_size is None
        assert args.hash_algorithm is None
        assert args.blend_probe_stride is None
        assert args.timeout_keep_alive is None


class TestCoordinatorCommandExecute:
    def test_overrides_applied(self, cmd):
        """chunk_size/hash_algorithm/blend_probe_stride flags override the config."""
        # First Party
        from lmcache.v1.mp_coordinator.config import MPCoordinatorConfig

        args = argparse.Namespace(
            host=None,
            port=None,
            instance_timeout=None,
            health_check_interval=None,
            eviction_check_interval=None,
            eviction_ratio=None,
            trigger_watermark=None,
            chunk_size=512,
            hash_algorithm="sha256",
            blend_probe_stride=2,
            timeout_keep_alive=None,
        )

        captured = {}

        def fake_create_app(config: MPCoordinatorConfig):
            captured["config"] = config
            return MagicMock()

        with (
            patch("uvicorn.run"),
            patch(
                "lmcache.v1.mp_coordinator.app.create_app",
                side_effect=fake_create_app,
            ),
        ):
            cmd.execute(args)

        assert captured["config"].chunk_size == 512
        assert captured["config"].hash_algorithm == "sha256"
        assert captured["config"].blend_probe_stride == 2
