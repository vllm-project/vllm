# SPDX-License-Identifier: Apache-2.0
"""Tests for vllm._genesis.compat.cli — unified subcommand dispatcher.

Replaces 13 scattered `python3 -m vllm._genesis.compat.X` invocations
with a single entry-point: `python3 -m vllm._genesis.compat.cli <sub>`.

The dispatcher validates the subcommand, prints a helpful message on
unknown subs, and forwards remaining args to the actual sub-CLI's main.
Each sub-CLI keeps its own module (operators / scripts that already
call the per-module form keep working — backwards compat).
"""
from __future__ import annotations




# ─── List of subcommands the unified CLI must support ──────────────────


_EXPECTED_SUBCOMMANDS = {
    "doctor", "explain", "init",
    "list-models", "pull",
    "lifecycle-audit", "validate-schema",
    "categories", "migrate",
    "recipe", "plugins", "telemetry", "update-channel",
    "self-test", "bench",
}


class TestSubcommandRouting:
    def test_main_returns_int(self):
        from vllm._genesis.compat.cli import main
        # No args → help, return non-zero
        try:
            rc = main([])
            assert isinstance(rc, int)
        except SystemExit as e:
            # argparse may exit on no-args
            assert e.code is None or isinstance(e.code, int)

    def test_unknown_subcommand_returns_nonzero(self, capsys):
        from vllm._genesis.compat.cli import main
        try:
            rc = main(["totally-not-a-subcommand"])
            assert rc != 0
        except SystemExit as e:
            assert e.code != 0

    def test_help_shows_all_subcommands(self, capsys):
        from vllm._genesis.compat.cli import main
        try:
            main(["--help"])
        except SystemExit:
            # argparse calls sys.exit(0) after --help — expected, capture continues
            pass
        captured = capsys.readouterr()
        # All key subcommands referenced
        for sub in ("doctor", "explain", "categories", "recipe",
                    "plugins", "telemetry"):
            assert sub in captured.out

    def test_known_subcommands_advertised(self):
        """The unified CLI must offer at least the 13 documented subs."""
        from vllm._genesis.compat.cli import KNOWN_SUBCOMMANDS
        for s in _EXPECTED_SUBCOMMANDS:
            assert s in KNOWN_SUBCOMMANDS, (
                f"unified CLI missing subcommand {s!r}"
            )

    def test_doctor_subcommand_routes(self, monkeypatch, capsys):
        """`cli doctor --quiet` should reach doctor.main with --quiet."""
        called_with = []
        from vllm._genesis.compat import doctor
        def fake_doctor_main(argv):
            called_with.append(argv)
            return 0
        monkeypatch.setattr(doctor, "main", fake_doctor_main)
        from vllm._genesis.compat.cli import main
        rc = main(["doctor", "--quiet"])
        assert rc == 0
        assert called_with == [["--quiet"]]

    def test_explain_subcommand_routes_with_args(self, monkeypatch):
        called_with = []
        from vllm._genesis.compat import explain
        def fake_explain_main(argv):
            called_with.append(argv)
            return 0
        monkeypatch.setattr(explain, "main", fake_explain_main)
        from vllm._genesis.compat.cli import main
        rc = main(["explain", "PN14", "--json"])
        assert rc == 0
        assert called_with == [["PN14", "--json"]]

    def test_categories_subcommand_routes(self, monkeypatch):
        called_with = []
        from vllm._genesis.compat import categories
        def fake_main(argv):
            called_with.append(argv)
            return 0
        monkeypatch.setattr(categories, "main", fake_main)
        from vllm._genesis.compat.cli import main
        rc = main(["categories", "--json"])
        assert rc == 0
        assert called_with == [["--json"]]


class TestAliasing:
    """Some subcommands use hyphens externally (`lifecycle-audit`) but
    map to module names with underscores. Verify both forms work."""

    def test_lifecycle_audit_subcommand(self, monkeypatch):
        called = []
        from vllm._genesis.compat import lifecycle_audit_cli
        def fake(argv):
            called.append(argv)
            return 0
        monkeypatch.setattr(lifecycle_audit_cli, "main", fake)
        from vllm._genesis.compat.cli import main
        rc = main(["lifecycle-audit", "--quiet"])
        assert rc == 0
        assert called == [["--quiet"]]

    def test_validate_schema_subcommand(self, monkeypatch):
        called = []
        from vllm._genesis.compat import schema_validator
        def fake(argv):
            called.append(argv)
            return 0
        monkeypatch.setattr(schema_validator, "main", fake)
        from vllm._genesis.compat.cli import main
        rc = main(["validate-schema"])
        assert rc == 0

    def test_list_models_subcommand(self, monkeypatch):
        called = []
        from vllm._genesis.compat.models import list_cli
        def fake(argv):
            called.append(argv)
            return 0
        monkeypatch.setattr(list_cli, "main", fake)
        from vllm._genesis.compat.cli import main
        rc = main(["list-models"])
        assert rc == 0

    def test_update_channel_subcommand(self, monkeypatch):
        called = []
        from vllm._genesis.compat import update_channel
        def fake(argv):
            called.append(argv)
            return 0
        monkeypatch.setattr(update_channel, "main", fake)
        from vllm._genesis.compat.cli import main
        rc = main(["update-channel", "status"])
        assert rc == 0
        assert called == [["status"]]


class TestExitCodePropagation:
    def test_subcommand_exit_code_propagates(self, monkeypatch):
        """If sub returns 2, unified CLI should also return 2."""
        from vllm._genesis.compat import doctor
        monkeypatch.setattr(doctor, "main", lambda argv: 2)
        from vllm._genesis.compat.cli import main
        rc = main(["doctor"])
        assert rc == 2


class TestNoArgs:
    def test_no_args_prints_usage(self, capsys):
        from vllm._genesis.compat.cli import main
        try:
            main([])
        except SystemExit:
            # argparse exits when no subcommand provided — expected, output captured
            pass
        captured = capsys.readouterr()
        joined = captured.out + captured.err
        # Should mention available subs or help
        assert "doctor" in joined or "subcommand" in joined.lower() \
            or "usage" in joined.lower()


class TestHelpForwarding:
    def test_subcommand_help_forwarded(self, monkeypatch):
        """`cli doctor --help` should call doctor.main with --help, not
        intercept the help at the dispatcher level."""
        called = []
        from vllm._genesis.compat import doctor
        def fake(argv):
            called.append(argv)
            # argparse usually exits 0 on --help
            return 0
        monkeypatch.setattr(doctor, "main", fake)
        from vllm._genesis.compat.cli import main
        rc = main(["doctor", "--help"])
        assert rc == 0
        assert called == [["--help"]]
