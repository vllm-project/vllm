# SPDX-License-Identifier: Apache-2.0
"""Tests for vllm._genesis.compat.telemetry — opt-in anonymized stats.

Phase 5d completes the plugin story by adding optional telemetry that
helps the Genesis community see "which configs work in the wild" without
collecting any personally-identifiable information.

Strict design rules (must be enforced by tests):
  - Default OFF — telemetry collection requires GENESIS_ENABLE_TELEMETRY=1
  - No PII — never includes hostname, IP, user, paths, env values, or
    container names. Only stable categorical info (hw class, model class,
    patch IDs, version numbers).
  - Local-first — writes to ~/.genesis/telemetry/reports/ before any
    network. Network upload is a SEPARATE second opt-in
    (GENESIS_TELEMETRY_UPLOAD=1) and is deferred for now.
  - Inspectable — `genesis telemetry show` displays exactly what would
    be sent, before any actual send. Operator can review before opting in.
  - Disposable — `genesis telemetry clear` removes the local stash.
  - Stable anonymous ID — random UUID-shaped string per-machine,
    persisted locally. Not tied to user account, not transmitted with
    OS/network identifiers.
"""
from __future__ import annotations

import json
import re

import pytest


# ─── Fixtures ────────────────────────────────────────────────────────────


@pytest.fixture
def telemetry_dir(tmp_path, monkeypatch):
    """Redirect telemetry storage to tmp dir for hermetic tests."""
    monkeypatch.setenv("GENESIS_TELEMETRY_DIR", str(tmp_path / "telemetry"))
    return tmp_path / "telemetry"


@pytest.fixture
def telemetry_enabled(monkeypatch):
    monkeypatch.setenv("GENESIS_ENABLE_TELEMETRY", "1")


@pytest.fixture
def telemetry_disabled(monkeypatch):
    monkeypatch.delenv("GENESIS_ENABLE_TELEMETRY", raising=False)
    monkeypatch.delenv("GENESIS_TELEMETRY_UPLOAD", raising=False)


# ─── Opt-in gate (must be DEFAULT OFF) ───────────────────────────────────


class TestOptInGate:
    def test_default_off(self, telemetry_disabled):
        from vllm._genesis.compat.telemetry import is_enabled
        assert is_enabled() is False

    def test_env_on(self, telemetry_enabled):
        from vllm._genesis.compat.telemetry import is_enabled
        assert is_enabled() is True

    def test_upload_default_off(self, telemetry_disabled):
        from vllm._genesis.compat.telemetry import is_upload_enabled
        assert is_upload_enabled() is False

    def test_upload_requires_both_envs(self, monkeypatch):
        """Upload must require BOTH master gate AND upload gate."""
        from vllm._genesis.compat.telemetry import is_upload_enabled
        monkeypatch.setenv("GENESIS_ENABLE_TELEMETRY", "1")
        # Only master, no upload → still off
        monkeypatch.delenv("GENESIS_TELEMETRY_UPLOAD", raising=False)
        assert is_upload_enabled() is False

        # Only upload, no master → still off
        monkeypatch.delenv("GENESIS_ENABLE_TELEMETRY", raising=False)
        monkeypatch.setenv("GENESIS_TELEMETRY_UPLOAD", "1")
        assert is_upload_enabled() is False

        # Both on → enabled
        monkeypatch.setenv("GENESIS_ENABLE_TELEMETRY", "1")
        monkeypatch.setenv("GENESIS_TELEMETRY_UPLOAD", "1")
        assert is_upload_enabled() is True


# ─── Instance ID stability ──────────────────────────────────────────────


class TestInstanceID:
    def test_first_call_creates_id(self, telemetry_dir, telemetry_enabled):
        from vllm._genesis.compat.telemetry import get_or_create_instance_id
        instance_id = get_or_create_instance_id()
        assert isinstance(instance_id, str)
        assert len(instance_id) > 8  # at least UUID-ish

    def test_second_call_returns_same_id(self, telemetry_dir, telemetry_enabled):
        from vllm._genesis.compat.telemetry import get_or_create_instance_id
        a = get_or_create_instance_id()
        b = get_or_create_instance_id()
        assert a == b

    def test_id_persisted_to_file(self, telemetry_dir, telemetry_enabled):
        from vllm._genesis.compat.telemetry import get_or_create_instance_id
        instance_id = get_or_create_instance_id()
        # The file should exist
        files = list(telemetry_dir.rglob("instance_id*"))
        assert len(files) == 1
        assert instance_id in files[0].read_text()

    def test_id_not_recognizable_as_pii(self, telemetry_dir, telemetry_enabled):
        """Instance ID must not be derived from anything PII-flavored
        like hostname, username, mac address."""
        from vllm._genesis.compat.telemetry import get_or_create_instance_id
        import socket, getpass
        try:
            hostname = socket.gethostname()
        except Exception:
            hostname = "?"
        try:
            user = getpass.getuser()
        except Exception:
            user = "?"
        instance_id = get_or_create_instance_id()
        # Hostname / user must not appear in the instance_id
        if hostname and hostname != "?":
            assert hostname not in instance_id
        if user and user != "?":
            assert user not in instance_id


# ─── Report shape (no PII) ──────────────────────────────────────────────


class TestReportShape:
    def test_collect_report_returns_dict(self, telemetry_dir, telemetry_enabled):
        from vllm._genesis.compat.telemetry import collect_report
        report = collect_report()
        assert isinstance(report, dict)

    def test_required_top_level_keys(self, telemetry_dir, telemetry_enabled):
        from vllm._genesis.compat.telemetry import collect_report
        report = collect_report()
        for key in ("schema_version", "timestamp", "instance_id",
                     "genesis_version", "hardware", "software",
                     "model", "patches"):
            assert key in report, f"missing top-level key: {key}"

    def test_report_includes_no_PII(self, telemetry_dir, telemetry_enabled):
        """Strict PII check — report must not contain anything that
        could re-identify the user."""
        from vllm._genesis.compat.telemetry import collect_report
        report = collect_report()
        flat = json.dumps(report, default=str).lower()

        import socket, getpass
        try:
            hostname = socket.gethostname()
        except Exception:
            hostname = "?"
        try:
            user = getpass.getuser()
        except Exception:
            user = "?"

        # Hostname must not appear
        if hostname and hostname != "?" and len(hostname) > 3:
            assert hostname.lower() not in flat, (
                f"hostname {hostname!r} leaked into telemetry report"
            )
        # Username must not appear
        if user and user != "?" and len(user) > 2:
            assert user.lower() not in flat, (
                f"username {user!r} leaked into telemetry report"
            )
        # No absolute paths
        assert "/home/" not in flat
        assert "/users/" not in flat
        assert "/nfs/" not in flat
        # No env-style keys with values (the report should map env-flag
        # names → presence boolean, never include the env value itself)
        assert "=true" not in flat  # env strings like GENESIS_X=1 not included

    def test_patches_section_only_includes_ids(
        self, telemetry_dir, telemetry_enabled,
    ):
        from vllm._genesis.compat.telemetry import collect_report
        report = collect_report()
        applied = report["patches"].get("applied", [])
        assert isinstance(applied, list)
        for p in applied:
            assert isinstance(p, str)
            # Each entry should look like a patch ID (P\d+, PN\d+)
            assert re.match(r"^[A-Z]+\d+\w*$", p), (
                f"unexpected patch id format: {p!r}"
            )

    def test_plugin_section_count_only_by_default(
        self, telemetry_dir, telemetry_enabled, monkeypatch,
    ):
        """By default, plugin names are NOT included (could fingerprint
        an operator). Only count is sent. Operator can opt-in via
        GENESIS_TELEMETRY_INCLUDE_PLUGIN_NAMES=1."""
        monkeypatch.delenv("GENESIS_TELEMETRY_INCLUDE_PLUGIN_NAMES", raising=False)
        from vllm._genesis.compat.telemetry import collect_report
        report = collect_report()
        plugin = report.get("plugins", {})
        assert "count" in plugin
        assert "names" not in plugin or not plugin["names"]


# ─── Storage ─────────────────────────────────────────────────────────────


class TestStorage:
    def test_save_creates_report_file(self, telemetry_dir, telemetry_enabled):
        from vllm._genesis.compat.telemetry import collect_report, save_report
        report = collect_report()
        path = save_report(report)
        assert path.is_file()
        # Filename should be timestamp-based, not random
        assert re.match(r"^\d{4}-\d{2}-\d{2}T\d{2}", path.stem)

    def test_clear_removes_reports(self, telemetry_dir, telemetry_enabled):
        from vllm._genesis.compat.telemetry import (
            collect_report, save_report, clear,
        )
        save_report(collect_report())
        save_report(collect_report())
        n = clear()
        assert n >= 2
        # Reports dir should be empty (or non-existent)
        if (telemetry_dir / "reports").is_dir():
            assert list((telemetry_dir / "reports").iterdir()) == []

    def test_save_when_disabled_returns_None(
        self, telemetry_dir, telemetry_disabled,
    ):
        from vllm._genesis.compat.telemetry import save_report
        # Even if collect_report works, save_report should refuse when off
        report = {"schema_version": "1.0"}
        result = save_report(report)
        assert result is None


# ─── Network upload (deferred — must NOT actually send) ─────────────────


class TestUploadGuards:
    def test_upload_guarded_by_double_opt_in(
        self, telemetry_dir, telemetry_disabled,
    ):
        """upload_report must refuse when either gate is closed.
        For now (Phase 5d alpha), upload is also deferred — should
        always be a no-op + return None."""
        from vllm._genesis.compat.telemetry import upload_report
        result = upload_report({"x": 1})
        assert result is None  # no-op when gates closed


# ─── CLI ─────────────────────────────────────────────────────────────────


class TestCLI:
    def test_status_subcommand(self, telemetry_dir, telemetry_disabled, capsys):
        from vllm._genesis.compat.telemetry import main
        rc = main(["status"])
        captured = capsys.readouterr()
        assert rc == 0
        # Should clearly show OFF state
        joined = captured.out + captured.err
        assert "OFF" in joined or "disabled" in joined.lower()

    def test_status_when_enabled(self, telemetry_dir, telemetry_enabled, capsys):
        from vllm._genesis.compat.telemetry import main
        rc = main(["status"])
        captured = capsys.readouterr()
        assert rc == 0
        joined = captured.out + captured.err
        assert "ON" in joined or "enabled" in joined.lower()

    def test_show_subcommand_displays_report(
        self, telemetry_dir, telemetry_enabled, capsys,
    ):
        from vllm._genesis.compat.telemetry import main
        rc = main(["show"])
        captured = capsys.readouterr()
        assert rc == 0
        # Should include schema_version / instance_id markers
        assert "schema_version" in captured.out
        assert "instance_id" in captured.out

    def test_show_refuses_when_disabled(
        self, telemetry_dir, telemetry_disabled, capsys,
    ):
        from vllm._genesis.compat.telemetry import main
        rc = main(["show"])
        captured = capsys.readouterr()
        # Either prints help/refuse + nonzero, or zero with message
        assert rc != 0 or "OFF" in (captured.out + captured.err)

    def test_collect_subcommand(self, telemetry_dir, telemetry_enabled, capsys):
        from vllm._genesis.compat.telemetry import main
        rc = main(["collect"])
        assert rc == 0
        # File should exist
        files = list((telemetry_dir / "reports").iterdir()) \
            if (telemetry_dir / "reports").is_dir() else []
        assert len(files) >= 1

    def test_clear_subcommand(self, telemetry_dir, telemetry_enabled, capsys):
        from vllm._genesis.compat.telemetry import collect_report, save_report, main
        save_report(collect_report())
        rc = main(["clear"])
        assert rc == 0
