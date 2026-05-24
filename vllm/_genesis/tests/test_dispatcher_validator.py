# SPDX-License-Identifier: Apache-2.0
"""Unit tests for A3/D2 — PATCH_REGISTRY dependency / conflict validator.

Two distinct levels of validation:

  1. **Static registry validation** (`validate_registry`):
     Walks PATCH_REGISTRY and verifies that every `requires_patches` /
     `conflicts_with` reference resolves to a real patch_id, no patch
     references itself, and there are no simple A→B + B→A cycles.
     Runs at import time / boot.

  2. **Dynamic apply-plan validation** (`validate_apply_plan`):
     Given the live set of patch_ids that the dispatcher decided to APPLY
     this boot, returns issues for missing-required and present-conflict.
     Runs once after `apply_all` finishes.

Tests cover both.
"""
from __future__ import annotations


# Importing this also exercises module-load — if we accidentally introduce
# a cycle in the static registry, the test will surface it via fixture.
from vllm._genesis import dispatcher


# ─── Static registry validation ─────────────────────────────────────────────


class TestValidateRegistry:
    """Static checks on PATCH_REGISTRY shape (no dynamic state)."""

    def test_clean_registry_validates(self):
        """The shipped PATCH_REGISTRY must validate without errors.

        If this fails, a recent edit introduced an unknown patch_id reference
        or a cycle. Investigate the issue list before merging.
        """
        issues = dispatcher.validate_registry()
        assert issues == [], (
            f"PATCH_REGISTRY has structural problems:\n"
            + "\n".join(f"  - {i.severity}: {i.patch_id}: {i.message}" for i in issues)
        )

    def test_validate_registry_returns_list(self):
        """Always returns a list (never None / raises)."""
        out = dispatcher.validate_registry()
        assert isinstance(out, list)

    def test_unknown_required_patch_detected(self, monkeypatch):
        """If a patch declares requires=['P_NONEXISTENT'], surface as ERROR."""
        fake_registry = {
            "P_FOO": {
                "title": "test",
                "env_flag": "G_FOO",
                "default_on": False,
                "requires_patches": ["P_NONEXISTENT"],
            },
        }
        monkeypatch.setattr(dispatcher, "PATCH_REGISTRY", fake_registry)
        issues = dispatcher.validate_registry()
        assert len(issues) == 1
        assert issues[0].severity == "ERROR"
        assert issues[0].patch_id == "P_FOO"
        assert "P_NONEXISTENT" in issues[0].message
        assert "requires" in issues[0].message.lower()

    def test_unknown_conflict_patch_detected(self, monkeypatch):
        """If conflicts_with references a phantom patch, surface as ERROR."""
        fake_registry = {
            "P_FOO": {
                "title": "test",
                "env_flag": "G_FOO",
                "default_on": False,
                "conflicts_with": ["P_GHOST"],
            },
        }
        monkeypatch.setattr(dispatcher, "PATCH_REGISTRY", fake_registry)
        issues = dispatcher.validate_registry()
        assert len(issues) == 1
        assert issues[0].severity == "ERROR"
        assert issues[0].patch_id == "P_FOO"
        assert "P_GHOST" in issues[0].message
        assert "conflict" in issues[0].message.lower()

    def test_self_reference_in_requires_detected(self, monkeypatch):
        """A patch that requires itself is a clear bug."""
        fake_registry = {
            "P_FOO": {
                "title": "self-loop",
                "env_flag": "G_FOO",
                "default_on": False,
                "requires_patches": ["P_FOO"],
            },
        }
        monkeypatch.setattr(dispatcher, "PATCH_REGISTRY", fake_registry)
        issues = dispatcher.validate_registry()
        assert any("self" in i.message.lower() for i in issues), issues

    def test_self_reference_in_conflicts_detected(self, monkeypatch):
        """conflicts_with=[self] means the patch can never apply — bug."""
        fake_registry = {
            "P_FOO": {
                "title": "self-conflict",
                "env_flag": "G_FOO",
                "default_on": False,
                "conflicts_with": ["P_FOO"],
            },
        }
        monkeypatch.setattr(dispatcher, "PATCH_REGISTRY", fake_registry)
        issues = dispatcher.validate_registry()
        assert any("self" in i.message.lower() for i in issues), issues

    def test_simple_two_node_cycle_detected(self, monkeypatch):
        """A→B and B→A must be flagged as a requires-cycle."""
        fake_registry = {
            "P_A": {"env_flag": "GA", "default_on": False, "requires_patches": ["P_B"]},
            "P_B": {"env_flag": "GB", "default_on": False, "requires_patches": ["P_A"]},
        }
        monkeypatch.setattr(dispatcher, "PATCH_REGISTRY", fake_registry)
        issues = dispatcher.validate_registry()
        assert any("cycle" in i.message.lower() for i in issues), issues

    def test_three_node_cycle_detected(self, monkeypatch):
        """A→B→C→A must be flagged."""
        fake_registry = {
            "P_A": {"env_flag": "GA", "default_on": False, "requires_patches": ["P_B"]},
            "P_B": {"env_flag": "GB", "default_on": False, "requires_patches": ["P_C"]},
            "P_C": {"env_flag": "GC", "default_on": False, "requires_patches": ["P_A"]},
        }
        monkeypatch.setattr(dispatcher, "PATCH_REGISTRY", fake_registry)
        issues = dispatcher.validate_registry()
        assert any("cycle" in i.message.lower() for i in issues), issues

    def test_dag_no_cycle(self, monkeypatch):
        """Linear A→B→C is valid (DAG). No cycle issue."""
        fake_registry = {
            "P_A": {"env_flag": "GA", "default_on": False, "requires_patches": ["P_B"]},
            "P_B": {"env_flag": "GB", "default_on": False, "requires_patches": ["P_C"]},
            "P_C": {"env_flag": "GC", "default_on": False},
        }
        monkeypatch.setattr(dispatcher, "PATCH_REGISTRY", fake_registry)
        issues = dispatcher.validate_registry()
        cycle_issues = [i for i in issues if "cycle" in i.message.lower()]
        assert cycle_issues == [], cycle_issues


# ─── Dynamic apply-plan validation ─────────────────────────────────────────


class TestValidateApplyPlan:
    """Runtime validation given a set of actually-applied patch_ids."""

    def test_empty_plan_no_issues(self, monkeypatch):
        """No patches applied → no issues."""
        fake_registry = {
            "P_A": {"env_flag": "GA", "default_on": False},
        }
        monkeypatch.setattr(dispatcher, "PATCH_REGISTRY", fake_registry)
        issues = dispatcher.validate_apply_plan(applied=set())
        assert issues == []

    def test_required_satisfied(self, monkeypatch):
        """P_A requires P_B, both applied → no issues."""
        fake_registry = {
            "P_A": {"env_flag": "GA", "default_on": False,
                    "requires_patches": ["P_B"]},
            "P_B": {"env_flag": "GB", "default_on": False},
        }
        monkeypatch.setattr(dispatcher, "PATCH_REGISTRY", fake_registry)
        issues = dispatcher.validate_apply_plan(applied={"P_A", "P_B"})
        assert issues == []

    def test_missing_required_detected(self, monkeypatch):
        """P_A applied but P_B (required) is not → ERROR."""
        fake_registry = {
            "P_A": {"env_flag": "GA", "default_on": False,
                    "requires_patches": ["P_B"]},
            "P_B": {"env_flag": "GB", "default_on": False},
        }
        monkeypatch.setattr(dispatcher, "PATCH_REGISTRY", fake_registry)
        issues = dispatcher.validate_apply_plan(applied={"P_A"})
        assert len(issues) == 1
        assert issues[0].severity == "ERROR"
        assert issues[0].patch_id == "P_A"
        assert "P_B" in issues[0].message
        assert "missing" in issues[0].message.lower() or \
               "requires" in issues[0].message.lower()

    def test_missing_required_skipped_if_dependent_skipped(self, monkeypatch):
        """If P_A is NOT applied, missing P_B requirement is irrelevant."""
        fake_registry = {
            "P_A": {"env_flag": "GA", "default_on": False,
                    "requires_patches": ["P_B"]},
            "P_B": {"env_flag": "GB", "default_on": False},
        }
        monkeypatch.setattr(dispatcher, "PATCH_REGISTRY", fake_registry)
        issues = dispatcher.validate_apply_plan(applied=set())
        assert issues == []

    def test_conflict_both_applied(self, monkeypatch):
        """P_A conflicts with P_B, both applied → ERROR."""
        fake_registry = {
            "P_A": {"env_flag": "GA", "default_on": False,
                    "conflicts_with": ["P_B"]},
            "P_B": {"env_flag": "GB", "default_on": False},
        }
        monkeypatch.setattr(dispatcher, "PATCH_REGISTRY", fake_registry)
        issues = dispatcher.validate_apply_plan(applied={"P_A", "P_B"})
        assert len(issues) >= 1
        assert any(
            i.severity == "ERROR" and "conflict" in i.message.lower()
            for i in issues
        ), issues

    def test_conflict_only_one_applied_no_issue(self, monkeypatch):
        """Conflict declared but only one of pair applied → no issue."""
        fake_registry = {
            "P_A": {"env_flag": "GA", "default_on": False,
                    "conflicts_with": ["P_B"]},
            "P_B": {"env_flag": "GB", "default_on": False},
        }
        monkeypatch.setattr(dispatcher, "PATCH_REGISTRY", fake_registry)
        issues = dispatcher.validate_apply_plan(applied={"P_A"})
        assert issues == []

    def test_conflict_symmetry_not_double_reported(self, monkeypatch):
        """If P_A.conflicts=[P_B] AND P_B.conflicts=[P_A], do not report twice."""
        fake_registry = {
            "P_A": {"env_flag": "GA", "default_on": False,
                    "conflicts_with": ["P_B"]},
            "P_B": {"env_flag": "GB", "default_on": False,
                    "conflicts_with": ["P_A"]},
        }
        monkeypatch.setattr(dispatcher, "PATCH_REGISTRY", fake_registry)
        issues = dispatcher.validate_apply_plan(applied={"P_A", "P_B"})
        # One canonical conflict pair reported (in either direction), not two
        conflict_issues = [i for i in issues if "conflict" in i.message.lower()]
        assert len(conflict_issues) == 1, conflict_issues

    def test_unknown_patch_in_applied_set(self, monkeypatch):
        """Caller passes a patch_id not in registry → WARNING."""
        fake_registry = {"P_A": {"env_flag": "GA", "default_on": False}}
        monkeypatch.setattr(dispatcher, "PATCH_REGISTRY", fake_registry)
        issues = dispatcher.validate_apply_plan(applied={"P_GHOST"})
        assert len(issues) >= 1
        assert any("unknown" in i.message.lower() for i in issues)

    def test_validation_issue_dataclass_shape(self, monkeypatch):
        """Issue objects must expose .severity, .patch_id, .message attrs."""
        fake_registry = {
            "P_A": {"env_flag": "GA", "default_on": False,
                    "requires_patches": ["P_GHOST"]},
        }
        monkeypatch.setattr(dispatcher, "PATCH_REGISTRY", fake_registry)
        issues = dispatcher.validate_registry()
        for i in issues:
            assert hasattr(i, "severity")
            assert hasattr(i, "patch_id")
            assert hasattr(i, "message")
            assert i.severity in ("ERROR", "WARNING", "INFO")


# ─── Real-registry sanity: the relationships we just declared ──────────────


class TestRealRegistryRelationships:
    """Verify the natural dependencies / conflicts we've encoded.

    These pin specific design decisions so a refactor can't silently
    weaken the validator's coverage of known constraints.
    """

    def test_p60b_requires_p60(self):
        """P60b (Phase 2 Triton kernel) requires P60 (Phase 1 SSM pre-copy)."""
        meta = dispatcher.PATCH_REGISTRY.get("P60b")
        assert meta is not None
        assert "P60" in meta.get("requires_patches", []), (
            "P60b is Phase 2 of GDN+ngram fix; P60 (Phase 1) must apply first"
        )

    def test_p85_requires_p84(self):
        """P85 fine-shadow cache requires P84 fine hashes computed."""
        meta = dispatcher.PATCH_REGISTRY.get("P85")
        assert meta is not None
        assert "P84" in meta.get("requires_patches", []), (
            "P85 docstring explicitly states 'Requires P84 (fine hashes computed)'"
        )

    def test_p74_requires_p72(self):
        """P74 chunk-clamp is the safety-net for P72-unblocked batched_tokens."""
        meta = dispatcher.PATCH_REGISTRY.get("P74")
        assert meta is not None
        assert "P72" in meta.get("requires_patches", []), (
            "P74 is 'P72 companion' per its title; chunk-clamp guards "
            "P72-unblocked >4096 batched_tokens"
        )

    def test_p56_conflicts_p65(self):
        """P56 deprecated workaround vs P65 root-cause fix on TQ spec-decode CG."""
        meta = dispatcher.PATCH_REGISTRY.get("P56")
        assert meta is not None
        assert "P65" in meta.get("conflicts_with", []), (
            "P56 is 'deprecated — superseded by P65'; both engaged is contradictory"
        )

    def test_p57_conflicts_p65(self):
        """P57 +850 MiB capture-safe buffers superseded by P65 CG downgrade."""
        meta = dispatcher.PATCH_REGISTRY.get("P57")
        assert meta is not None
        assert "P65" in meta.get("conflicts_with", []), (
            "P57 deprecation_note states P65 achieves same correctness "
            "without memory blow-up"
        )

    def test_p67_conflicts_p65(self):
        """P67 multi-query kernel is 'proper fix replacing P65 workaround'."""
        meta = dispatcher.PATCH_REGISTRY.get("P67")
        assert meta is not None
        assert "P65" in meta.get("conflicts_with", []), (
            "P67 credit explicitly states 'replaces P65 workaround'"
        )
