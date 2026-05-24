# SPDX-License-Identifier: Apache-2.0
"""TDD for PN65 — Genesis structured API access log middleware."""
from __future__ import annotations

import json
import logging

import pytest

from vllm._genesis.wiring.middleware.patch_N65_access_log import (
    _extract_response_tokens,
    _format_duration_ms,
    _format_log_line,
    _quiet_paths,
    install_into_app,
)


# ─── Format helpers ─────────────────────────────────────────────────────


class TestDurationFormat:
    def test_sub_ms(self):
        assert _format_duration_ms(0.0001) == "<1ms"

    def test_single_digit_ms(self):
        assert _format_duration_ms(0.005) == "5.0ms"

    def test_three_digit_ms(self):
        assert _format_duration_ms(0.123) == "123ms"

    def test_seconds(self):
        assert _format_duration_ms(2.5) == "2.50s"


class TestQuietPaths:
    def test_default_quiet_paths_includes_health(self, monkeypatch):
        monkeypatch.delenv("GENESIS_PN65_QUIET_PATHS", raising=False)
        paths = _quiet_paths()
        assert "/health" in paths
        assert "/metrics" in paths

    def test_env_override_replaces_default(self, monkeypatch):
        monkeypatch.setenv("GENESIS_PN65_QUIET_PATHS", "/health,/admin")
        paths = _quiet_paths()
        assert "/health" in paths
        assert "/admin" in paths
        assert "/metrics" not in paths


# ─── Token extraction ───────────────────────────────────────────────────


class TestTokenExtraction:
    def test_chat_completion_response(self):
        body = json.dumps({
            "choices": [{"message": {"content": "hi", "tool_calls": []}}],
            "usage": {"prompt_tokens": 12, "completion_tokens": 5,
                      "total_tokens": 17},
        }).encode()
        tokens = _extract_response_tokens(body)
        assert tokens["prompt"] == 12
        assert tokens["completion"] == 5
        assert "tools" not in tokens  # no tool calls in this response

    def test_response_with_tool_calls(self):
        body = json.dumps({
            "choices": [{"message": {
                "content": None,
                "tool_calls": [
                    {"function": {"name": "f1", "arguments": "{}"}},
                    {"function": {"name": "f2", "arguments": "{}"}},
                ],
            }}],
            "usage": {"prompt_tokens": 30, "completion_tokens": 10},
        }).encode()
        tokens = _extract_response_tokens(body)
        assert tokens["tools"] == 2

    def test_streaming_response_marked(self):
        body = b'data: {"choices":[]}\n\ndata: [DONE]\n'
        tokens = _extract_response_tokens(body)
        assert tokens.get("stream") is True
        assert "prompt" not in tokens

    def test_empty_body_returns_empty(self):
        assert _extract_response_tokens(b"") == {}

    def test_garbage_body_does_not_raise(self):
        assert _extract_response_tokens(b"not json at all") == {}


# ─── Log line formatting ────────────────────────────────────────────────


class TestLogLineFormat:
    def test_simple_get_no_tokens(self):
        line = _format_log_line(
            "GET", "/v1/models", 401, 0.0005, "192.168.1.10", {}
        )
        assert "[Genesis-API]" in line
        assert "401" in line
        assert "GET" in line
        assert "/v1/models" in line
        assert "<1ms" in line
        assert "client=192.168.1.10" in line

    def test_chat_completion_with_full_tokens(self):
        line = _format_log_line(
            "POST", "/v1/chat/completions", 200, 0.034, "192.168.1.35",
            {"prompt": 46, "completion": 400, "tools": 1},
        )
        assert "200" in line
        assert "POST" in line
        assert "prompt=46t" in line
        assert "completion=400t" in line
        assert "tools=1" in line
        assert "34ms" in line

    def test_streaming_marked(self):
        line = _format_log_line(
            "POST", "/v1/chat/completions", 200, 1.5, "127.0.0.1",
            {"stream": True},
        )
        assert "stream=Y" in line


# ─── install_into_app ───────────────────────────────────────────────────


class _FakeApp:
    """Minimal FastAPI-like app for testing middleware install."""
    def __init__(self):
        self.middlewares: list = []
        # Mimic FastAPI's `.middleware('http')` decorator:
        # returns a decorator that captures the function

    def middleware(self, kind: str):
        assert kind == "http"
        def decorator(fn):
            self.middlewares.append(fn)
            return fn
        return decorator


class TestInstallIntoApp:
    def test_first_install_returns_true_and_marks_app(self):
        app = _FakeApp()
        installed = install_into_app(app)
        assert installed is True
        assert getattr(app, "__pn65_installed__", False) is True
        assert len(app.middlewares) == 1

    def test_idempotent_second_install_returns_false(self):
        app = _FakeApp()
        install_into_app(app)
        installed_again = install_into_app(app)
        assert installed_again is False
        assert len(app.middlewares) == 1  # not added twice


# ─── apply() integration ────────────────────────────────────────────────


class TestApplyFunction:
    def test_apply_skipped_when_env_disabled(self, monkeypatch):
        from vllm._genesis.wiring.middleware import patch_N65_access_log as p
        monkeypatch.delenv("GENESIS_ENABLE_PN65", raising=False)
        status, reason = p.apply()
        assert status == "skipped"

    def test_apply_skipped_when_api_server_absent(self, monkeypatch):
        from vllm._genesis.wiring.middleware import patch_N65_access_log as p
        monkeypatch.setenv("GENESIS_ENABLE_PN65", "1")
        import sys
        monkeypatch.setitem(sys.modules,
                            "vllm.entrypoints.openai.api_server", None)
        status, reason = p.apply()
        assert status == "skipped"
        assert "not importable" in reason or "api_server" in reason


# ─── G-POST-07: uvicorn.access dedup filter ─────────────────────────────
#
# Audit `genesis_post_fix_rescan_audit_2026-05-05` G-POST-07 mandated unit
# coverage for `_DropUvicornAccessInfo` and `_suppress_uvicorn_access_logger`.
# v2 install path was live-verified on 35B PROD but had no automated test
# until this section.


def _make_record(name: str, level: int, msg: str = "x") -> logging.LogRecord:
    return logging.LogRecord(
        name=name, level=level, pathname=__file__, lineno=1,
        msg=msg, args=(), exc_info=None,
    )


class TestDropUvicornAccessFilter:
    def test_drops_uvicorn_access_info(self):
        from vllm._genesis.wiring.middleware.patch_N65_access_log import (
            _DropUvicornAccessInfo,
        )
        f = _DropUvicornAccessInfo()
        rec = _make_record("uvicorn.access", logging.INFO,
                           '192.168.1.10 - "GET /v1/models" 401')
        assert f.filter(rec) is False

    def test_keeps_uvicorn_access_warning(self):
        from vllm._genesis.wiring.middleware.patch_N65_access_log import (
            _DropUvicornAccessInfo,
        )
        f = _DropUvicornAccessInfo()
        rec = _make_record("uvicorn.access", logging.WARNING, "boom")
        assert f.filter(rec) is True

    def test_keeps_uvicorn_access_error(self):
        from vllm._genesis.wiring.middleware.patch_N65_access_log import (
            _DropUvicornAccessInfo,
        )
        f = _DropUvicornAccessInfo()
        rec = _make_record("uvicorn.access", logging.ERROR, "5xx")
        assert f.filter(rec) is True

    def test_keeps_other_logger_info(self):
        from vllm._genesis.wiring.middleware.patch_N65_access_log import (
            _DropUvicornAccessInfo,
        )
        f = _DropUvicornAccessInfo()
        rec = _make_record("genesis.api", logging.INFO, "[Genesis-API] 200")
        assert f.filter(rec) is True

    def test_keeps_uvicorn_error_logger_info(self):
        # uvicorn.error is a separate logger; PN65 must not drop it.
        from vllm._genesis.wiring.middleware.patch_N65_access_log import (
            _DropUvicornAccessInfo,
        )
        f = _DropUvicornAccessInfo()
        rec = _make_record("uvicorn.error", logging.INFO, "startup")
        assert f.filter(rec) is True


@pytest.fixture
def _reset_pn65_filter_install():
    """Reset PN65 install marker + remove any installed filters before/after.

    The module holds module-level state (`_PN65_FILTER_INSTALLED`) plus the
    filter is attached to root + uvicorn.access loggers — leaks between
    tests would mask idempotency bugs.
    """
    from vllm._genesis.wiring.middleware import patch_N65_access_log as p

    def _strip():
        for logger in (logging.getLogger(), logging.getLogger("uvicorn.access")):
            for f in list(logger.filters):
                if isinstance(f, p._DropUvicornAccessInfo):
                    logger.removeFilter(f)
        p._PN65_FILTER_INSTALLED = False

    _strip()
    yield p
    _strip()


class TestSuppressUvicornAccessLogger:
    def test_install_attaches_filter_to_both_loggers(
        self, _reset_pn65_filter_install, monkeypatch,
    ):
        p = _reset_pn65_filter_install
        monkeypatch.delenv("GENESIS_PN65_KEEP_UVICORN_ACCESS", raising=False)
        p._suppress_uvicorn_access_logger()
        root_filters = [
            f for f in logging.getLogger().filters
            if isinstance(f, p._DropUvicornAccessInfo)
        ]
        uvicorn_filters = [
            f for f in logging.getLogger("uvicorn.access").filters
            if isinstance(f, p._DropUvicornAccessInfo)
        ]
        assert len(root_filters) == 1
        assert len(uvicorn_filters) == 1
        assert p._PN65_FILTER_INSTALLED is True

    def test_install_is_idempotent(
        self, _reset_pn65_filter_install, monkeypatch,
    ):
        p = _reset_pn65_filter_install
        monkeypatch.delenv("GENESIS_PN65_KEEP_UVICORN_ACCESS", raising=False)
        p._suppress_uvicorn_access_logger()
        p._suppress_uvicorn_access_logger()
        p._suppress_uvicorn_access_logger()
        root_filters = [
            f for f in logging.getLogger().filters
            if isinstance(f, p._DropUvicornAccessInfo)
        ]
        uvicorn_filters = [
            f for f in logging.getLogger("uvicorn.access").filters
            if isinstance(f, p._DropUvicornAccessInfo)
        ]
        assert len(root_filters) == 1, "no duplicate filters on root"
        assert len(uvicorn_filters) == 1, "no duplicate filters on uvicorn.access"

    def test_keep_env_skips_install(
        self, _reset_pn65_filter_install, monkeypatch,
    ):
        p = _reset_pn65_filter_install
        monkeypatch.setenv("GENESIS_PN65_KEEP_UVICORN_ACCESS", "1")
        p._suppress_uvicorn_access_logger()
        root_filters = [
            f for f in logging.getLogger().filters
            if isinstance(f, p._DropUvicornAccessInfo)
        ]
        uvicorn_filters = [
            f for f in logging.getLogger("uvicorn.access").filters
            if isinstance(f, p._DropUvicornAccessInfo)
        ]
        assert root_filters == []
        assert uvicorn_filters == []
        assert p._PN65_FILTER_INSTALLED is False

    @pytest.mark.parametrize("val", ["true", "yes", "Y", "ON", "1"])
    def test_keep_env_truthy_variants_skip_install(
        self, val, _reset_pn65_filter_install, monkeypatch,
    ):
        p = _reset_pn65_filter_install
        monkeypatch.setenv("GENESIS_PN65_KEEP_UVICORN_ACCESS", val)
        p._suppress_uvicorn_access_logger()
        assert p._PN65_FILTER_INSTALLED is False

    def test_keep_env_empty_does_not_skip(
        self, _reset_pn65_filter_install, monkeypatch,
    ):
        p = _reset_pn65_filter_install
        monkeypatch.setenv("GENESIS_PN65_KEEP_UVICORN_ACCESS", "")
        p._suppress_uvicorn_access_logger()
        assert p._PN65_FILTER_INSTALLED is True

    def test_filter_actually_drops_records_after_install(
        self, _reset_pn65_filter_install, monkeypatch, caplog,
    ):
        """End-to-end: install, emit a uvicorn.access INFO record, verify drop."""
        p = _reset_pn65_filter_install
        monkeypatch.delenv("GENESIS_PN65_KEEP_UVICORN_ACCESS", raising=False)
        p._suppress_uvicorn_access_logger()

        uv = logging.getLogger("uvicorn.access")
        uv.setLevel(logging.INFO)
        with caplog.at_level(logging.INFO, logger="uvicorn.access"):
            uv.info('192.168.1.10:45116 - "GET /v1/models HTTP/1.1" 401')
            uv.warning("503 backend down")

        # caplog records pass through filters too in pytest >= 7
        names_levels = [(r.name, r.levelno, r.getMessage())
                        for r in caplog.records]
        # INFO must be dropped, WARNING must remain
        info_records = [
            x for x in names_levels
            if x[0] == "uvicorn.access" and x[1] == logging.INFO
        ]
        warn_records = [
            x for x in names_levels
            if x[0] == "uvicorn.access" and x[1] == logging.WARNING
        ]
        assert info_records == []
        assert len(warn_records) == 1
