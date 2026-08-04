# SPDX-License-Identifier: Apache-2.0
"""Tests for the LMCache startup banner."""

# Standard
import io

# Third Party
import pytest

# First Party
from lmcache import banner
from lmcache.banner import (
    DISABLE_BANNER_ENV,
    LMCACHE_LINKEDIN,
    LMCACHE_RECIPES,
    LMCACHE_WEBSITE,
    print_banner_once,
)


class _TTYStream(io.StringIO):
    """In-memory stream that reports itself as a terminal."""

    def isatty(self) -> bool:
        return True


@pytest.fixture(autouse=True)
def _fresh_banner_state(monkeypatch):
    """Reset the once-per-process guard and opt-out env var between tests."""
    monkeypatch.setattr(banner, "_banner_printed", False)
    monkeypatch.delenv(DISABLE_BANNER_ENV, raising=False)


def test_banner_contains_version_links_and_opt_out_hint():
    stream = io.StringIO()
    print_banner_once(stream)
    output = stream.getvalue()
    assert "LMCache v" in output
    assert LMCACHE_WEBSITE in output
    assert LMCACHE_RECIPES in output
    assert LMCACHE_LINKEDIN in output
    assert DISABLE_BANNER_ENV in output


def test_banner_is_plain_on_non_tty_stream():
    stream = io.StringIO()
    print_banner_once(stream)
    assert "\x1b[" not in stream.getvalue()


def test_banner_is_colored_on_tty_stream():
    stream = _TTYStream()
    print_banner_once(stream)
    output = stream.getvalue()
    assert "\x1b[1;3;38;2;203;75;22m" in output  # bold italic solarized orange
    assert "\x1b[38;2;42;161;152m" in output  # solarized cyan


def test_banner_prints_at_most_once_per_process():
    first = io.StringIO()
    second = io.StringIO()
    print_banner_once(first)
    print_banner_once(second)
    assert first.getvalue() != ""
    assert second.getvalue() == ""


@pytest.mark.parametrize("value", ["1", "true", "YES"])
def test_banner_disabled_by_env_var(monkeypatch, value):
    monkeypatch.setenv(DISABLE_BANNER_ENV, value)
    stream = io.StringIO()
    print_banner_once(stream)
    assert stream.getvalue() == ""


def test_banner_not_disabled_by_falsy_env_var(monkeypatch):
    monkeypatch.setenv(DISABLE_BANNER_ENV, "0")
    stream = io.StringIO()
    print_banner_once(stream)
    assert stream.getvalue() != ""
