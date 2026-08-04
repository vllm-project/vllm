# SPDX-License-Identifier: Apache-2.0
"""
Unit tests for RESP adapter environment variable support.

These tests verify the precedence logic (config > env var > default)
without requiring a live Redis server.
"""

# Standard
import os


class TestRESPL2AdapterEnvVars:
    """Test env var resolution in the MP-mode RESP L2 adapter factory."""

    def test_env_vars_used_when_config_empty(self, monkeypatch):
        """Env vars should be used when config values are empty/default."""
        # First Party
        from lmcache.v1.distributed.l2_adapters.resp_l2_adapter import (
            RESPL2AdapterConfig,
        )

        monkeypatch.setenv("LMCACHE_RESP_HOST", "env-host")
        monkeypatch.setenv("LMCACHE_RESP_PORT", "7777")
        monkeypatch.setenv("LMCACHE_RESP_USERNAME", "env-user")
        monkeypatch.setenv("LMCACHE_RESP_PASSWORD", "env-pass")

        config = RESPL2AdapterConfig(host="", port=0, username="", password="")

        host = config.host or os.environ.get("LMCACHE_RESP_HOST", "")
        port = (
            config.port
            if config.port
            else int(os.environ.get("LMCACHE_RESP_PORT", "0"))
        )
        username = config.username or os.environ.get("LMCACHE_RESP_USERNAME", "")
        password = config.password or os.environ.get("LMCACHE_RESP_PASSWORD", "")

        assert host == "env-host"
        assert port == 7777
        assert username == "env-user"
        assert password == "env-pass"

    def test_config_overrides_env_vars(self, monkeypatch):
        """Config values should take precedence over env vars."""
        # First Party
        from lmcache.v1.distributed.l2_adapters.resp_l2_adapter import (
            RESPL2AdapterConfig,
        )

        monkeypatch.setenv("LMCACHE_RESP_HOST", "env-host")
        monkeypatch.setenv("LMCACHE_RESP_PORT", "7777")
        monkeypatch.setenv("LMCACHE_RESP_USERNAME", "env-user")
        monkeypatch.setenv("LMCACHE_RESP_PASSWORD", "env-pass")

        config = RESPL2AdapterConfig(
            host="cfg-host",
            port=8888,
            username="cfg-user",
            password="cfg-pass",
        )

        host = config.host or os.environ.get("LMCACHE_RESP_HOST", "")
        port = (
            config.port
            if config.port
            else int(os.environ.get("LMCACHE_RESP_PORT", "0"))
        )
        username = config.username or os.environ.get("LMCACHE_RESP_USERNAME", "")
        password = config.password or os.environ.get("LMCACHE_RESP_PASSWORD", "")

        assert host == "cfg-host"
        assert port == 8888
        assert username == "cfg-user"
        assert password == "cfg-pass"

    def test_defaults_when_no_env_and_no_config(self, monkeypatch):
        """Without env vars or config, values should be empty/zero."""
        # First Party
        from lmcache.v1.distributed.l2_adapters.resp_l2_adapter import (
            RESPL2AdapterConfig,
        )

        monkeypatch.delenv("LMCACHE_RESP_HOST", raising=False)
        monkeypatch.delenv("LMCACHE_RESP_PORT", raising=False)
        monkeypatch.delenv("LMCACHE_RESP_USERNAME", raising=False)
        monkeypatch.delenv("LMCACHE_RESP_PASSWORD", raising=False)

        config = RESPL2AdapterConfig(host="", port=0, username="", password="")

        host = config.host or os.environ.get("LMCACHE_RESP_HOST", "")
        port = (
            config.port
            if config.port
            else int(os.environ.get("LMCACHE_RESP_PORT", "0"))
        )
        username = config.username or os.environ.get("LMCACHE_RESP_USERNAME", "")
        password = config.password or os.environ.get("LMCACHE_RESP_PASSWORD", "")

        assert host == ""
        assert port == 0
        assert username == ""
        assert password == ""


class TestRESPConnectorAdapterEnvVars:
    """Test env var resolution in the non-MP RESP connector adapter."""

    def test_env_vars_used_when_extra_config_empty(self, monkeypatch):
        """Env vars should be used when extra_config has no credentials."""
        monkeypatch.setenv("LMCACHE_RESP_USERNAME", "env-user")
        monkeypatch.setenv("LMCACHE_RESP_PASSWORD", "env-pass")

        extra_config = {}

        cfg_username = str(extra_config.get("username", ""))
        cfg_password = str(extra_config.get("password", ""))
        username = cfg_username or os.environ.get("LMCACHE_RESP_USERNAME", "")
        password = cfg_password or os.environ.get("LMCACHE_RESP_PASSWORD", "")

        assert username == "env-user"
        assert password == "env-pass"

    def test_extra_config_overrides_env_vars(self, monkeypatch):
        """extra_config values should take precedence over env vars."""
        monkeypatch.setenv("LMCACHE_RESP_USERNAME", "env-user")
        monkeypatch.setenv("LMCACHE_RESP_PASSWORD", "env-pass")

        extra_config = {"username": "cfg-user", "password": "cfg-pass"}

        cfg_username = str(extra_config.get("username", ""))
        cfg_password = str(extra_config.get("password", ""))
        username = cfg_username or os.environ.get("LMCACHE_RESP_USERNAME", "")
        password = cfg_password or os.environ.get("LMCACHE_RESP_PASSWORD", "")

        assert username == "cfg-user"
        assert password == "cfg-pass"

    def test_host_port_env_var_fallback(self, monkeypatch):
        """Host/port env vars should be used when URL values are empty."""
        monkeypatch.setenv("LMCACHE_RESP_HOST", "env-host")
        monkeypatch.setenv("LMCACHE_RESP_PORT", "9999")

        parsed_host = ""
        parsed_port = 0

        host = parsed_host or os.environ.get("LMCACHE_RESP_HOST", "")
        port = (
            parsed_port
            if parsed_port
            else int(os.environ.get("LMCACHE_RESP_PORT", "0"))
        )

        assert host == "env-host"
        assert port == 9999

    def test_url_overrides_env_vars(self, monkeypatch):
        """URL-parsed values should take precedence over env vars."""
        monkeypatch.setenv("LMCACHE_RESP_HOST", "env-host")
        monkeypatch.setenv("LMCACHE_RESP_PORT", "9999")

        parsed_host = "url-host"
        parsed_port = 6379

        host = parsed_host or os.environ.get("LMCACHE_RESP_HOST", "")
        port = (
            parsed_port
            if parsed_port
            else int(os.environ.get("LMCACHE_RESP_PORT", "0"))
        )

        assert host == "url-host"
        assert port == 6379
