# SPDX-License-Identifier: Apache-2.0
"""Tests for 'go back' navigation in the interactive flow."""

# Standard
from typing import Any
from unittest import mock

# First Party
from lmcache.cli.commands.bench.engine_bench import interactive as flow
from lmcache.cli.commands.bench.engine_bench.interactive import terminal
from lmcache.cli.commands.bench.engine_bench.interactive.state import (
    InteractiveState,
)


def test_back_token_returns_sentinel_only_when_enabled() -> None:
    # The same key ("<") means "go back" everywhere, but only when offered.
    with mock.patch("builtins.input", return_value=terminal.BACK_TOKEN):
        assert terminal.prompt_text("L", allow_back=True) is terminal.GO_BACK
        assert terminal.prompt_text("L", allow_back=False) == terminal.BACK_TOKEN


def test_normalize_url_shorthand() -> None:
    assert flow._normalize_url("8000") == "http://localhost:8000"
    assert flow._normalize_url("localhost:8000") == "http://localhost:8000"
    assert flow._normalize_url("http://host:9000") == "http://host:9000"
    assert flow._normalize_url("") == ""


def _drive(monkeypatch: Any, answers: list[tuple[str, Any]]) -> InteractiveState:
    """Run the driver, feeding scripted answers keyed by question identity."""
    pending = iter(answers)

    def pull(tag: str) -> Any:
        expected, value = next(pending)
        assert expected == tag, f"expected {expected!r}, got {tag!r}"
        return value

    monkeypatch.setattr(
        flow, "_prompt_for_item", lambda item, allow_back: pull(item.key)
    )
    monkeypatch.setattr(
        flow,
        "_prompt_gate",
        lambda name, detail, allow_back: pull(
            "gate:workload" if "Workload" in name else "gate:general"
        ),
    )
    monkeypatch.setattr(flow, "_prompt_action", lambda allow_back: pull("action"))
    monkeypatch.setattr(flow, "_print_summary", lambda state: None)

    state = InteractiveState()
    state.set("__started", flow._gather_config(state))
    return state


def test_back_undoes_previous_answer(monkeypatch: Any) -> None:
    # has_lmcache=False makes tokens the next question; going back there must
    # un-set has_lmcache and re-ask it.
    state = _drive(
        monkeypatch,
        [
            ("engine_url", "http://x"),
            ("workload", "random-prefill"),
            ("has_lmcache", False),
            ("tokens_per_gb_kvcache", terminal.GO_BACK),  # back
            ("has_lmcache", True),  # re-answered
            ("lmcache_url", "http://y"),
            ("gate:general", "use defaults"),
            ("gate:workload", "use defaults"),
            ("action", "start"),
        ],
    )
    assert state.get("__started") is True
    assert state.get("has_lmcache") is True
    assert state.is_set("tokens_per_gb_kvcache") is False


def test_quit_returns_false(monkeypatch: Any) -> None:
    state = _drive(
        monkeypatch,
        [
            ("engine_url", "http://x"),
            ("workload", "random-prefill"),
            ("has_lmcache", True),
            ("lmcache_url", "http://y"),
            ("gate:general", "use defaults"),
            ("gate:workload", "use defaults"),
            ("action", "quit"),
        ],
    )
    assert state.get("__started") is False
