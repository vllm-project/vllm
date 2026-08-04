# SPDX-License-Identifier: Apache-2.0
"""Tests for InteractiveState."""

# Standard
import argparse
import json

# First Party
from lmcache.cli.commands.bench.engine_bench.interactive.state import (
    InteractiveState,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_full_state(**overrides: object) -> InteractiveState:
    """Build a state with all required fields set."""
    values: dict[str, object] = {
        "engine_url": "http://localhost:8000",
        "workload": "long-doc-qa",
        "tokens_per_gb_kvcache": 6553,
    }
    values.update(overrides)
    state = InteractiveState()
    for k, v in values.items():
        state.set(k, v)
    return state


def _make_cli_args(**overrides: object) -> argparse.Namespace:
    """Build a Namespace mimicking bench engine CLI output."""
    defaults: dict[str, object] = {
        "engine_url": None,
        "workload": None,
        "model": None,
        "lmcache_url": None,
        "tokens_per_gb_kvcache": None,
        "kv_cache_volume": 100.0,
        "seed": 42,
        "output_dir": ".",
        "no_csv": False,
        "json": False,
        "quiet": False,
        "ldqa_document_length": 10000,
        "ldqa_query_per_document": 2,
        "ldqa_shuffle_policy": "random",
        "ldqa_num_inflight_requests": 3,
        "mrc_shared_prompt_length": 2000,
        "mrc_chat_history_length": 10000,
        "mrc_user_input_length": 50,
        "mrc_output_length": 200,
        "mrc_qps": 1.0,
        "mrc_duration": 60.0,
        "rp_request_length": 10000,
        "rp_num_requests": 50,
        "format": None,
        "output": None,
        "config": None,
        "bench_target": "engine",
    }
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


# ---------------------------------------------------------------------------
# Basic accessors
# ---------------------------------------------------------------------------


class TestBasicAccessors:
    def test_empty_state(self) -> None:
        state = InteractiveState()
        assert not state.is_set("engine_url")
        assert state.get("engine_url") is None
        assert state.get("engine_url", "fallback") == "fallback"

    def test_set_and_get(self) -> None:
        state = InteractiveState()
        state.set("engine_url", "http://localhost:8000")
        assert state.is_set("engine_url")
        assert state.get("engine_url") == "http://localhost:8000"

    def test_values_copy(self) -> None:
        state = _make_full_state()
        vals = state.values
        vals["engine_url"] = "mutated"
        assert state.get("engine_url") == "http://localhost:8000"


# ---------------------------------------------------------------------------
# Readiness
# ---------------------------------------------------------------------------


class TestReadiness:
    def test_empty_not_ready(self) -> None:
        state = InteractiveState()
        assert not state.is_ready()

    def test_full_state_ready(self) -> None:
        state = _make_full_state()
        assert state.is_ready()

    def test_missing_engine_url(self) -> None:
        state = _make_full_state()
        state._values.pop("engine_url")
        assert not state.is_ready()

    def test_missing_workload(self) -> None:
        state = _make_full_state()
        state._values.pop("workload")
        assert not state.is_ready()

    def test_tokens_per_gb_not_required_when_lmcache_set(self) -> None:
        state = InteractiveState()
        state.set("engine_url", "http://localhost:8000")
        state.set("workload", "long-doc-qa")
        state.set("lmcache_url", "http://localhost:8080")
        # tokens_per_gb_kvcache not set, but lmcache_url is → condition
        # for tokens_per_gb is not met → not required
        assert state.is_ready()


# ---------------------------------------------------------------------------
# Missing required
# ---------------------------------------------------------------------------


class TestMissingRequired:
    def test_all_missing(self) -> None:
        state = InteractiveState()
        missing = state.get_missing_required()
        keys = [item.key for item in missing]
        assert "engine_url" in keys
        assert "workload" in keys

    def test_none_missing(self) -> None:
        state = _make_full_state()
        assert state.get_missing_required() == []

    def test_tokens_per_gb_skipped_with_lmcache(self) -> None:
        state = InteractiveState()
        state.set("engine_url", "http://localhost:8000")
        state.set("workload", "long-doc-qa")
        state.set("lmcache_url", "http://localhost:8080")
        missing = state.get_missing_required()
        keys = [item.key for item in missing]
        assert "tokens_per_gb_kvcache" not in keys


# ---------------------------------------------------------------------------
# Workload items
# ---------------------------------------------------------------------------


class TestWorkloadItems:
    def test_long_doc_qa(self) -> None:
        state = _make_full_state(workload="long-doc-qa")
        items = state.get_workload_items()
        keys = [item.key for item in items]
        assert "ldqa_document_length" in keys
        assert "mrc_qps" not in keys
        assert "rp_num_requests" not in keys

    def test_multi_round_chat(self) -> None:
        state = _make_full_state(workload="multi-round-chat")
        items = state.get_workload_items()
        keys = [item.key for item in items]
        assert "mrc_shared_prompt_length" in keys
        assert "ldqa_document_length" not in keys

    def test_random_prefill(self) -> None:
        state = _make_full_state(workload="random-prefill")
        items = state.get_workload_items()
        keys = [item.key for item in items]
        assert "rp_request_length" in keys
        assert "ldqa_document_length" not in keys

    def test_workload_items_all_default_initially(self) -> None:
        state = _make_full_state(workload="long-doc-qa")
        assert state.workload_items_all_default()

    def test_workload_items_not_default_after_set(self) -> None:
        state = _make_full_state(workload="long-doc-qa")
        state.set("ldqa_document_length", 5000)
        assert not state.workload_items_all_default()


# ---------------------------------------------------------------------------
# Fill defaults
# ---------------------------------------------------------------------------


class TestFillDefaults:
    def test_fills_kv_cache_volume(self) -> None:
        state = _make_full_state(workload="long-doc-qa")
        state.fill_defaults()
        assert state.get("kv_cache_volume") == 100.0

    def test_fills_workload_defaults(self) -> None:
        state = _make_full_state(workload="long-doc-qa")
        state.fill_defaults()
        assert state.get("ldqa_document_length") == 10000
        assert state.get("ldqa_query_per_document") == 2

    def test_does_not_overwrite_set_values(self) -> None:
        state = _make_full_state(workload="long-doc-qa")
        state.set("ldqa_document_length", 5000)
        state.fill_defaults()
        assert state.get("ldqa_document_length") == 5000


# ---------------------------------------------------------------------------
# from_cli_args
# ---------------------------------------------------------------------------


class TestFromCliArgs:
    def test_picks_up_explicit_values(self) -> None:
        args = _make_cli_args(
            engine_url="http://host:8000",
            workload="long-doc-qa",
            tokens_per_gb_kvcache=6553,
        )
        state = InteractiveState.from_cli_args(args)
        assert state.get("engine_url") == "http://host:8000"
        assert state.get("workload") == "long-doc-qa"
        assert state.get("tokens_per_gb_kvcache") == 6553

    def test_none_values_not_set(self) -> None:
        args = _make_cli_args()  # all defaults
        state = InteractiveState.from_cli_args(args)
        assert not state.is_set("engine_url")
        assert not state.is_set("workload")
        assert not state.is_set("model")

    def test_default_values_not_set(self) -> None:
        # kv_cache_volume defaults to 100.0 — should not be marked as set
        args = _make_cli_args(kv_cache_volume=100.0)
        state = InteractiveState.from_cli_args(args)
        assert not state.is_set("kv_cache_volume")

    def test_non_default_values_set(self) -> None:
        args = _make_cli_args(kv_cache_volume=50.0)
        state = InteractiveState.from_cli_args(args)
        assert state.is_set("kv_cache_volume")
        assert state.get("kv_cache_volume") == 50.0


# ---------------------------------------------------------------------------
# to_namespace
# ---------------------------------------------------------------------------


class TestToNamespace:
    def test_round_trip(self) -> None:
        state = _make_full_state(workload="long-doc-qa")
        ns = state.to_namespace()
        assert ns.engine_url == "http://localhost:8000"
        assert ns.workload == "long-doc-qa"
        assert ns.tokens_per_gb_kvcache == 6553
        # Defaults filled
        assert ns.kv_cache_volume == 100.0
        assert ns.ldqa_document_length == 10000
        assert ns.seed == 42

    def test_has_bench_target(self) -> None:
        state = _make_full_state()
        ns = state.to_namespace()
        assert ns.bench_target == "engine"

    def test_has_output_attrs(self) -> None:
        state = _make_full_state()
        ns = state.to_namespace()
        assert hasattr(ns, "output_dir")
        assert hasattr(ns, "no_csv")
        assert hasattr(ns, "quiet")
        assert hasattr(ns, "format")
        assert hasattr(ns, "output")


# ---------------------------------------------------------------------------
# JSON round-trip
# ---------------------------------------------------------------------------


class TestJsonRoundTrip:
    def test_to_json_and_back(self) -> None:
        state = _make_full_state(workload="long-doc-qa")
        state.set("ldqa_document_length", 5000)
        data = state.to_json()
        # engine_url excluded from export (environment-specific)
        assert "engine_url" not in data
        assert data["ldqa_document_length"] == 5000
        restored = InteractiveState.from_json(data)
        assert restored.get("ldqa_document_length") == 5000

    def test_save_and_load(self, tmp_path: object) -> None:
        # Standard
        import pathlib

        path = str(pathlib.Path(str(tmp_path)) / "test_config.json")
        state = _make_full_state(workload="long-doc-qa")
        state.save_json(path)

        loaded = InteractiveState.load_json(path)
        assert "engine_url" not in loaded.values
        assert loaded.get("workload") == "long-doc-qa"

    def test_json_file_is_valid_json(self, tmp_path: object) -> None:
        # Standard
        import pathlib

        path = str(pathlib.Path(str(tmp_path)) / "test.json")
        state = _make_full_state()
        state.save_json(path)

        with open(path) as f:
            data = json.load(f)
        assert isinstance(data, dict)
        assert "engine_url" not in data
        assert data["workload"] == "long-doc-qa"

    def test_merge_cli_args(self) -> None:
        state = _make_full_state(workload="long-doc-qa")
        args = _make_cli_args(
            engine_url="http://override:9000",
        )
        state.merge_cli_args(args)
        assert state.get("engine_url") == "http://override:9000"
        # Original values preserved
        assert state.get("workload") == "long-doc-qa"


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------


class TestSummary:
    def test_summary_lines_non_empty(self) -> None:
        state = _make_full_state(workload="long-doc-qa")
        state.fill_defaults()
        lines = state.summary_lines()
        assert len(lines) > 0

    def test_summary_includes_engine_url(self) -> None:
        state = _make_full_state(workload="long-doc-qa")
        state.fill_defaults()
        labels = [label for label, _ in state.summary_lines()]
        assert "Engine URL" in labels

    def test_summary_shows_auto_detect_for_empty_model(self) -> None:
        state = _make_full_state(workload="long-doc-qa")
        state.fill_defaults()
        for label, value in state.summary_lines():
            if label == "Model name":
                assert value == "(auto-detect)"
                break

    def test_summary_excludes_wrong_workload_items(self) -> None:
        state = _make_full_state(workload="long-doc-qa")
        state.fill_defaults()
        labels = [label for label, _ in state.summary_lines()]
        assert "Queries per second" not in labels  # mrc item
        assert "Number of requests" not in labels  # rp item
