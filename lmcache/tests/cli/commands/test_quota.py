# SPDX-License-Identifier: Apache-2.0
"""Tests for the ``lmcache quota`` CLI command."""

# Standard
from unittest.mock import MagicMock, patch
import argparse
import sys

# Inject a fake openai module so that the auto-discovery of
# lmcache.cli.commands (which imports bench -> openai) does not fail
# in environments where openai is not installed.
if "openai" not in sys.modules:
    _fake_openai = MagicMock()
    sys.modules["openai"] = _fake_openai

# Third Party
import pytest

# First Party
from lmcache.cli.commands.quota import QuotaCommand
from lmcache.cli.commands.quota._helpers import (
    DEFAULT_SALT_SENTINEL,
    escape_salt,
    normalize_url,
    unescape_salt,
)


@pytest.fixture
def cmd() -> QuotaCommand:
    return QuotaCommand()


@pytest.fixture
def parser(cmd: QuotaCommand) -> argparse.ArgumentParser:
    """An ArgumentParser with QuotaCommand registered."""
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="command")
    cmd.register(sub)
    return p


class TestHelpers:
    def test_normalize_url(self) -> None:
        assert normalize_url("localhost:8080") == "http://localhost:8080"
        assert normalize_url("https://host:443/") == "https://host:443"

    def test_escape_unescape_salt(self) -> None:
        assert escape_salt("") == DEFAULT_SALT_SENTINEL
        assert escape_salt("tenant1") == "tenant1"
        assert unescape_salt(DEFAULT_SALT_SENTINEL) == ""
        assert unescape_salt("tenant1") == "tenant1"


class TestQuotaCommandMetadata:
    def test_name_and_help(self, cmd: QuotaCommand) -> None:
        assert cmd.name() == "quota"
        assert "quota" in cmd.help().lower()


class TestQuotaCommandExecute:
    @patch("lmcache.cli.commands.quota.set_command.http_request")
    def test_set(
        self,
        mock_http,
        cmd,
        parser,
        capsys,
    ) -> None:
        mock_http.return_value = {
            "cache_salt": "tenant1",
            "limit_gb": 10.5,
            "status": "ok",
        }
        args = parser.parse_args(["quota", "set", "tenant1", "--limit-gb", "10.5"])
        cmd.execute(args)

        mock_http.assert_called_once_with(
            "PUT",
            "http://localhost:8080/quota/tenant1",
            data={"limit_gb": 10.5},
        )
        out = capsys.readouterr().out
        assert "Quota Set" in out and "tenant1" in out

    @patch("lmcache.cli.commands.quota.get_command.http_request")
    def test_get(
        self,
        mock_http,
        cmd,
        parser,
        capsys,
    ) -> None:
        mock_http.return_value = {
            "cache_salt": "tenant1",
            "limit_gb": 10.5,
            "current_usage_gb": 3.27,
            "exists": True,
        }
        args = parser.parse_args(["quota", "get", "tenant1"])
        cmd.execute(args)

        mock_http.assert_called_once_with("GET", "http://localhost:8080/quota/tenant1")
        out = capsys.readouterr().out
        assert "Quota Info" in out and "3.27" in out

    @patch("lmcache.cli.commands.quota.list_command.http_request")
    def test_list(
        self,
        mock_http,
        cmd,
        parser,
        capsys,
    ) -> None:
        mock_http.return_value = {
            "users": {
                "tenant1": {"limit_gb": 10.5, "current_usage_gb": 3.27},
                "_default": {"limit_gb": 5.0, "current_usage_gb": 1.82},
            }
        }
        args = parser.parse_args(["quota", "list"])
        cmd.execute(args)

        mock_http.assert_called_once_with("GET", "http://localhost:8080/quota")
        out = capsys.readouterr().out
        assert "tenant1" in out and "_default" in out

    @patch("lmcache.cli.commands.quota.delete_command.http_request")
    def test_delete(
        self,
        mock_http,
        cmd,
        parser,
        capsys,
    ) -> None:
        mock_http.return_value = {"cache_salt": "tenant1", "status": "removed"}
        args = parser.parse_args(["quota", "delete", "tenant1"])
        cmd.execute(args)

        mock_http.assert_called_once_with(
            "DELETE",
            "http://localhost:8080/quota/tenant1",
        )
        out = capsys.readouterr().out
        assert "Quota Delete" in out and "removed" in out

    @patch("lmcache.cli.commands.quota.set_command.http_request")
    def test_quiet_suppresses_output(
        self,
        mock_http,
        cmd,
        parser,
        capsys,
    ) -> None:
        mock_http.return_value = {"cache_salt": "t1", "limit_gb": 1.0, "status": "ok"}
        args = parser.parse_args(["quota", "set", "t1", "--limit-gb", "1", "-q"])
        cmd.execute(args)
        assert capsys.readouterr().out == ""
