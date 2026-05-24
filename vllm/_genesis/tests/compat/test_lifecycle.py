# SPDX-License-Identifier: Apache-2.0
"""Tests for vllm._genesis.compat.lifecycle."""
from __future__ import annotations


from vllm._genesis.compat.lifecycle import (
    KNOWN_STATES,
    audit_registry,
    format_audit_table,
    get_state,
    is_engageable,
)


class TestGetState:
    def test_explicit_lifecycle_returned(self):
        assert get_state({"lifecycle": "experimental"}) == "experimental"
        assert get_state({"lifecycle": "stable"}) == "stable"
        assert get_state({"lifecycle": "deprecated"}) == "deprecated"
        assert get_state({"lifecycle": "research"}) == "research"
        assert get_state({"lifecycle": "community"}) == "community"
        assert get_state({"lifecycle": "retired"}) == "retired"

    def test_legacy_deprecated_flag_maps_to_deprecated(self):
        """Legacy `deprecated: True` flag (pre-lifecycle field) → 'deprecated'."""
        assert get_state({"deprecated": True}) == "deprecated"

    def test_default_is_stable(self):
        assert get_state({}) == "stable"
        assert get_state({"title": "X"}) == "stable"


class TestAuditRegistry:
    def test_unknown_state_is_error(self):
        registry = {"P_X": {"lifecycle": "made_up_state"}}
        entries = audit_registry(registry)
        assert len(entries) == 1
        assert entries[0].severity == "error"
        assert "unknown lifecycle" in entries[0].note

    def test_experimental_warns(self):
        registry = {"P_X": {"lifecycle": "experimental"}}
        entries = audit_registry(registry)
        assert entries[0].severity == "warn"

    def test_deprecated_includes_superseded_by(self):
        registry = {"P56": {
            "lifecycle": "deprecated",
            "superseded_by": ["P65", "P67"],
            "removal_planned": "v8.0",
        }}
        entries = audit_registry(registry)
        assert "P65" in entries[0].note
        assert "P67" in entries[0].note
        assert "v8.0" in entries[0].note

    def test_research_is_ok(self):
        registry = {"P57": {
            "lifecycle": "research",
            "research_note": "memory blow-up unacceptable on 24GB",
        }}
        entries = audit_registry(registry)
        assert entries[0].severity == "ok"
        assert "memory blow-up" in entries[0].note

    def test_stable_is_ok(self):
        registry = {"PN14": {"lifecycle": "stable", "stable_since": "v7.62.18"}}
        entries = audit_registry(registry)
        assert entries[0].severity == "ok"

    def test_retired_warns(self):
        registry = {"P_DEAD": {"lifecycle": "retired"}}
        entries = audit_registry(registry)
        assert entries[0].severity == "warn"
        assert "retired" in entries[0].note


class TestIsEngageable:
    def test_stable_engageable(self):
        ok, _ = is_engageable({"lifecycle": "stable"})
        assert ok is True

    def test_experimental_engageable(self):
        ok, _ = is_engageable({"lifecycle": "experimental"})
        assert ok is True

    def test_deprecated_still_engageable(self):
        """Deprecated patches still work — they're just superseded."""
        ok, _ = is_engageable({"lifecycle": "deprecated"})
        assert ok is True

    def test_retired_NOT_engageable(self):
        ok, why = is_engageable({"lifecycle": "retired"})
        assert ok is False
        assert "retired" in why
        assert "allow-retired" in why

    def test_retired_engageable_with_override(self):
        ok, _ = is_engageable({"lifecycle": "retired"}, allow_gated=True)
        assert ok is True

    def test_research_engageable(self):
        """Research patches engage normally — operator opts in by env."""
        ok, _ = is_engageable({"lifecycle": "research"})
        assert ok is True


class TestKnownStates:
    def test_all_documented_states_recognised(self):
        for state in ("experimental", "stable", "deprecated",
                      "research", "community", "retired"):
            assert state in KNOWN_STATES


class TestFormatTable:
    def test_groups_by_state(self):
        registry = {
            "P_A": {"lifecycle": "stable"},
            "P_B": {"lifecycle": "experimental"},
            "P_C": {"lifecycle": "stable"},
        }
        entries = audit_registry(registry)
        lines = format_audit_table(entries)
        joined = "\n".join(lines)
        # Both states represented
        assert "stable" in joined
        assert "experimental" in joined
        # Counts in section headers
        assert "(2 patches)" in joined
        assert "(1 patches)" in joined

    def test_empty_registry(self):
        lines = format_audit_table([])
        assert "empty registry" in lines[0]
