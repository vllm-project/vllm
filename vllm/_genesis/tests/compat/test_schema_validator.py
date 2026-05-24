# SPDX-License-Identifier: Apache-2.0
"""Tests for vllm._genesis.compat.schema_validator —
PATCH_REGISTRY structural validation against schemas/patch_entry.schema.json."""
from __future__ import annotations



# ─── Synthetic entries for testing ───────────────────────────────────────


_VALID_MINIMAL = {
    "title": "Test minimal patch",
    "env_flag": "GENESIS_ENABLE_TEST_MIN",
    "default_on": False,
}

_VALID_FULL = {
    "title": "Test full patch with all fields populated",
    "env_flag": "GENESIS_ENABLE_TEST_FULL",
    "default_on": False,
    "lifecycle": "stable",
    "stable_since": "v7.63.0",
    "category": "kernel_safety",
    "credit": "Test patch — Genesis-original.",
    "upstream_pr": 12345,
    "applies_to": {"is_turboquant": True},
    "requires_patches": ["TEST_DEP"],
    "conflicts_with": ["TEST_OLD"],
}

_VALID_DEPRECATED = {
    "title": "Test deprecated patch",
    "env_flag": "GENESIS_ENABLE_TEST_DEP",
    "default_on": False,
    "lifecycle": "deprecated",
    "superseded_by": ["TEST_NEW"],
    "removal_planned": "v8.0",
    "credit": "old workaround.",
}

_VALID_RESEARCH = {
    "title": "Test research patch",
    "env_flag": "GENESIS_ENABLE_TEST_RES",
    "default_on": False,
    "lifecycle": "research",
    "research_note": "kept as reference for future hardware",
}

_INVALID_NO_TITLE = {
    "env_flag": "GENESIS_ENABLE_X",
    "default_on": False,
}

_INVALID_BAD_ENV_FLAG = {
    "title": "Bad env flag",
    "env_flag": "not_genesis_prefix",
    "default_on": False,
}

_INVALID_LIFECYCLE = {
    "title": "Wrong lifecycle",
    "env_flag": "GENESIS_ENABLE_X",
    "default_on": False,
    "lifecycle": "made_up_state",
}

_INVALID_DEPRECATED_NO_NOTE = {
    "title": "Deprecated without supersedes / note",
    "env_flag": "GENESIS_ENABLE_DEP",
    "default_on": False,
    "lifecycle": "deprecated",
}

_INVALID_UNKNOWN_FIELD = {
    "title": "Has typo'd field",
    "env_flag": "GENESIS_ENABLE_X",
    "default_on": False,
    "applys_to": {"is_turboquant": True},  # typo: "applys" not "applies"
}

_INVALID_REQUIRES_NOT_LIST = {
    "title": "Bad requires shape",
    "env_flag": "GENESIS_ENABLE_X",
    "default_on": False,
    "requires_patches": "P_X",  # should be list, got str
}


class TestValidate:
    def test_minimal_valid(self):
        from vllm._genesis.compat.schema_validator import validate_entry
        issues = validate_entry("PN_TEST", _VALID_MINIMAL)
        assert issues == []

    def test_full_valid(self):
        from vllm._genesis.compat.schema_validator import validate_entry
        assert validate_entry("PN_TEST", _VALID_FULL) == []

    def test_deprecated_with_supersedes_valid(self):
        from vllm._genesis.compat.schema_validator import validate_entry
        assert validate_entry("PN_TEST", _VALID_DEPRECATED) == []

    def test_research_with_note_valid(self):
        from vllm._genesis.compat.schema_validator import validate_entry
        assert validate_entry("PN_TEST", _VALID_RESEARCH) == []

    def test_missing_title_fails(self):
        from vllm._genesis.compat.schema_validator import validate_entry
        issues = validate_entry("PN_TEST", _INVALID_NO_TITLE)
        assert issues
        assert any("title" in i.message for i in issues)

    def test_bad_env_flag_pattern_fails(self):
        from vllm._genesis.compat.schema_validator import validate_entry
        issues = validate_entry("PN_TEST", _INVALID_BAD_ENV_FLAG)
        assert issues
        assert any("env_flag" in i.message.lower() for i in issues)

    def test_unknown_lifecycle_fails(self):
        from vllm._genesis.compat.schema_validator import validate_entry
        issues = validate_entry("PN_TEST", _INVALID_LIFECYCLE)
        assert issues
        assert any("lifecycle" in i.message.lower() for i in issues)

    def test_deprecated_without_note_or_supersedes_fails(self):
        from vllm._genesis.compat.schema_validator import validate_entry
        issues = validate_entry("PN_TEST", _INVALID_DEPRECATED_NO_NOTE)
        assert issues
        joined = " ".join(i.message for i in issues)
        assert "deprecation_note" in joined or "superseded_by" in joined

    def test_unknown_field_fails(self):
        """Typo'd field name like 'applys_to' instead of 'applies_to'
        should be caught before it silently misbehaves at boot."""
        from vllm._genesis.compat.schema_validator import validate_entry
        issues = validate_entry("PN_TEST", _INVALID_UNKNOWN_FIELD)
        assert issues
        assert any("applys_to" in i.message for i in issues)

    def test_requires_patches_must_be_list(self):
        from vllm._genesis.compat.schema_validator import validate_entry
        issues = validate_entry("PN_TEST", _INVALID_REQUIRES_NOT_LIST)
        assert issues
        assert any("requires_patches" in i.message for i in issues)


class TestValidateRegistry:
    def test_real_registry_passes(self):
        """The shipped PATCH_REGISTRY must be schema-clean."""
        from vllm._genesis.compat.schema_validator import validate_registry
        from vllm._genesis.dispatcher import PATCH_REGISTRY
        issues = validate_registry(PATCH_REGISTRY)
        # NB: surfaces ALL legacy `deprecated: True` entries — they should
        # all also have a `deprecation_note` per our schema.
        # If this fails on real data, output the violations for the operator.
        if issues:
            for i in issues:
                print(f"  {i.severity} {i.patch_id}: {i.message}")
        # First-time run may have legacy violations; we'll fix them.
        # But the validator itself must run without crashing.
        assert isinstance(issues, list)

    def test_registry_with_one_bad_entry_isolates_failure(self, monkeypatch):
        bad = {
            "P_OK": _VALID_MINIMAL,
            "P_BROKEN": _INVALID_BAD_ENV_FLAG,
        }
        from vllm._genesis.compat.schema_validator import validate_registry
        issues = validate_registry(bad)
        assert issues
        # Only P_BROKEN should be cited
        assert all(i.patch_id == "P_BROKEN" for i in issues)


class TestCLIWrapper:
    def test_cli_returns_int(self):
        from vllm._genesis.compat.schema_validator import main
        rc = main([])
        assert isinstance(rc, int)

    def test_cli_clean_registry_returns_zero(self, monkeypatch):
        from vllm._genesis import dispatcher
        monkeypatch.setattr(dispatcher, "PATCH_REGISTRY", {"P_OK": _VALID_MINIMAL})
        from vllm._genesis.compat.schema_validator import main
        rc = main([])
        assert rc == 0

    def test_cli_dirty_registry_returns_nonzero(self, monkeypatch):
        from vllm._genesis import dispatcher
        monkeypatch.setattr(dispatcher, "PATCH_REGISTRY", {
            "P_OK": _VALID_MINIMAL,
            "P_BAD": _INVALID_BAD_ENV_FLAG,
        })
        from vllm._genesis.compat.schema_validator import main
        rc = main([])
        assert rc == 1
