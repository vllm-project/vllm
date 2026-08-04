# SPDX-License-Identifier: Apache-2.0
"""Tests for the config item schema."""

# First Party
from lmcache.cli.commands.bench.engine_bench.interactive.schema import (
    ALL_ITEMS,
    PHASE_GENERAL,
    PHASE_REQUIRED,
    PHASE_WORKLOAD,
    ConfigItem,
    get_item,
    get_items_by_phase,
)


class TestConfigItemStructure:
    def test_all_items_non_empty(self) -> None:
        assert len(ALL_ITEMS) > 0

    def test_all_items_have_unique_keys(self) -> None:
        keys = [item.key for item in ALL_ITEMS]
        assert len(keys) == len(set(keys))

    def test_required_items_in_phase_required(self) -> None:
        for item in ALL_ITEMS:
            if item.required:
                assert item.phase == PHASE_REQUIRED, (
                    f"{item.key} is required but in phase {item.phase}"
                )

    def test_choice_items_have_choices(self) -> None:
        for item in ALL_ITEMS:
            if item.input_type == "choice":
                assert len(item.choices) > 0, (
                    f"{item.key} is 'choice' type but has no choices"
                )

    def test_all_input_types_valid(self) -> None:
        valid = {"text", "int", "float", "bool", "choice"}
        for item in ALL_ITEMS:
            assert item.input_type in valid, (
                f"{item.key} has unknown input_type {item.input_type!r}"
            )


class TestGetItemsByPhase:
    def test_required_phase(self) -> None:
        items = get_items_by_phase(PHASE_REQUIRED)
        assert len(items) >= 3  # engine_url, workload, lmcache_url, tokens_per_gb
        for item in items:
            assert item.phase == PHASE_REQUIRED

    def test_general_phase(self) -> None:
        items = get_items_by_phase(PHASE_GENERAL)
        assert len(items) >= 1
        for item in items:
            assert item.phase == PHASE_GENERAL

    def test_workload_phase(self) -> None:
        items = get_items_by_phase(PHASE_WORKLOAD)
        assert len(items) >= 4  # long-doc-qa has 4
        for item in items:
            assert item.phase == PHASE_WORKLOAD


class TestGetItem:
    def test_known_key(self) -> None:
        item = get_item("engine_url")
        assert isinstance(item, ConfigItem)
        assert item.key == "engine_url"
        assert item.required is True

    def test_unknown_key_raises(self) -> None:
        # Third Party
        import pytest

        with pytest.raises(KeyError, match="no_such_key"):
            get_item("no_such_key")


class TestConditions:
    def test_tokens_per_gb_condition_no_lmcache(self) -> None:
        item = get_item("tokens_per_gb_kvcache")
        assert item.condition is not None
        # No lmcache_url → condition met → should show
        assert item.condition({}) is True
        assert item.condition({"lmcache_url": ""}) is True

    def test_tokens_per_gb_condition_with_lmcache(self) -> None:
        item = get_item("tokens_per_gb_kvcache")
        assert item.condition is not None
        # lmcache_url set → condition not met → skip
        assert item.condition({"lmcache_url": "http://localhost:8080"}) is False

    def test_ldqa_condition(self) -> None:
        item = get_item("ldqa_document_length")
        assert item.condition is not None
        assert item.condition({"workload": "long-doc-qa"}) is True
        assert item.condition({"workload": "multi-round-chat"}) is False
        assert item.condition({}) is False

    def test_mrc_condition(self) -> None:
        item = get_item("mrc_shared_prompt_length")
        assert item.condition is not None
        assert item.condition({"workload": "multi-round-chat"}) is True
        assert item.condition({"workload": "long-doc-qa"}) is False

    def test_rp_condition(self) -> None:
        item = get_item("rp_request_length")
        assert item.condition is not None
        assert item.condition({"workload": "random-prefill"}) is True
        assert item.condition({"workload": "long-doc-qa"}) is False
