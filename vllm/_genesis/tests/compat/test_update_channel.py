# SPDX-License-Identifier: Apache-2.0
"""Tests for vllm._genesis.compat.update_channel — channel management
+ check-for-updates against the Genesis GitHub repo.

Phase 3.x scope (this commit): channel selection (stable / beta / dev),
local commit detection (via git rev-parse), upstream check (via GitHub
REST API), result caching to avoid rate-limiting. The actual file-
update apply pass is DEFERRED — operators can `git pull` manually
based on the check tool's output.
"""
from __future__ import annotations


import pytest


# ─── Fixtures ────────────────────────────────────────────────────────────


@pytest.fixture
def update_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("GENESIS_UPDATE_DIR", str(tmp_path / "update"))
    return tmp_path / "update"


# ─── Channel selection ──────────────────────────────────────────────────


class TestChannelSelection:
    def test_default_channel_is_stable(self, update_dir):
        from vllm._genesis.compat.update_channel import get_channel
        assert get_channel() == "stable"

    def test_set_channel_persists(self, update_dir):
        from vllm._genesis.compat.update_channel import (
            get_channel, set_channel,
        )
        set_channel("beta")
        assert get_channel() == "beta"
        set_channel("dev")
        assert get_channel() == "dev"
        set_channel("stable")
        assert get_channel() == "stable"

    def test_set_channel_rejects_unknown(self, update_dir):
        from vllm._genesis.compat.update_channel import set_channel
        with pytest.raises(ValueError):
            set_channel("nightly")
        with pytest.raises(ValueError):
            set_channel("")

    def test_env_override(self, update_dir, monkeypatch):
        """GENESIS_UPDATE_CHANNEL env overrides the persisted choice."""
        from vllm._genesis.compat.update_channel import (
            get_channel, set_channel,
        )
        set_channel("dev")
        monkeypatch.setenv("GENESIS_UPDATE_CHANNEL", "stable")
        # Env wins
        assert get_channel() == "stable"


# ─── Local commit detection ─────────────────────────────────────────────


class TestLocalCommitDetection:
    def test_returns_string_or_none(self):
        from vllm._genesis.compat.update_channel import detect_local_commit
        # Doesn't matter if there's no .git — should not raise
        result = detect_local_commit()
        assert result is None or isinstance(result, str)


# ─── Upstream check (mocked HTTP) ───────────────────────────────────────


class TestUpstreamCheck:
    def test_check_for_updates_returns_dict(
        self, update_dir, monkeypatch,
    ):
        # Mock the GitHub API call
        fake_response = {
            "sha": "abc1234567890",
            "commit": {
                "author": {"date": "2026-04-30T22:00:00Z"},
                "message": "v7.64.0 release",
            },
        }
        from vllm._genesis.compat import update_channel as uc
        monkeypatch.setattr(uc, "_fetch_github_ref",
                            lambda channel: fake_response)
        result = uc.check_for_updates()
        assert isinstance(result, dict)
        assert "channel" in result
        assert "upstream_sha" in result
        assert result["upstream_sha"].startswith("abc1234")

    def test_check_handles_network_failure(self, update_dir, monkeypatch):
        from vllm._genesis.compat import update_channel as uc
        def raises(channel):
            raise RuntimeError("network down")
        monkeypatch.setattr(uc, "_fetch_github_ref", raises)
        result = uc.check_for_updates()
        assert "error" in result

    def test_check_caches_result(self, update_dir, monkeypatch):
        """Repeated calls within cache TTL must NOT hit network again."""
        from vllm._genesis.compat import update_channel as uc
        call_count = [0]
        def fetch(channel):
            call_count[0] += 1
            return {"sha": "abc123", "commit": {
                "author": {"date": "2026-04-30T22:00:00Z"},
                "message": "test",
            }}
        monkeypatch.setattr(uc, "_fetch_github_ref", fetch)
        # First call hits network
        uc.check_for_updates()
        # Second call (within TTL) uses cache
        uc.check_for_updates()
        assert call_count[0] == 1, "network should only be hit once"

    def test_force_refresh_bypasses_cache(self, update_dir, monkeypatch):
        from vllm._genesis.compat import update_channel as uc
        call_count = [0]
        def fetch(channel):
            call_count[0] += 1
            return {"sha": "abc123", "commit": {
                "author": {"date": "2026-04-30T22:00:00Z"},
                "message": "test",
            }}
        monkeypatch.setattr(uc, "_fetch_github_ref", fetch)
        uc.check_for_updates()
        uc.check_for_updates(force_refresh=True)
        assert call_count[0] == 2


# ─── Update-availability decision ───────────────────────────────────────


class TestUpdateAvailable:
    def test_local_matches_upstream_no_update(self, update_dir, monkeypatch):
        from vllm._genesis.compat import update_channel as uc
        monkeypatch.setattr(uc, "detect_local_commit", lambda: "abc1234567")
        monkeypatch.setattr(
            uc, "_fetch_github_ref",
            lambda c: {"sha": "abc1234567",
                       "commit": {
                           "author": {"date": "2026-04-30T22:00:00Z"},
                           "message": "current"}},
        )
        result = uc.check_for_updates(force_refresh=True)
        assert result["update_available"] is False

    def test_local_differs_from_upstream_update_available(
        self, update_dir, monkeypatch,
    ):
        from vllm._genesis.compat import update_channel as uc
        monkeypatch.setattr(uc, "detect_local_commit", lambda: "old1234567")
        monkeypatch.setattr(
            uc, "_fetch_github_ref",
            lambda c: {"sha": "new1234567",
                       "commit": {
                           "author": {"date": "2026-05-01T10:00:00Z"},
                           "message": "v7.64 release"}},
        )
        result = uc.check_for_updates(force_refresh=True)
        assert result["update_available"] is True
        assert result["upstream_sha"].startswith("new")

    def test_local_unknown_treated_as_unknown(self, update_dir, monkeypatch):
        from vllm._genesis.compat import update_channel as uc
        monkeypatch.setattr(uc, "detect_local_commit", lambda: None)
        monkeypatch.setattr(
            uc, "_fetch_github_ref",
            lambda c: {"sha": "new1234567",
                       "commit": {"author": {"date": "now"},
                                  "message": "test"}},
        )
        result = uc.check_for_updates(force_refresh=True)
        # When we don't know local commit, can't make a decision either way
        assert result.get("update_available") in (None, False)


# ─── CLI ─────────────────────────────────────────────────────────────────


class TestCLI:
    def test_status_subcommand(self, update_dir, monkeypatch, capsys):
        from vllm._genesis.compat import update_channel as uc
        monkeypatch.setattr(uc, "_fetch_github_ref",
                            lambda c: {"sha": "abc1234567",
                                       "commit": {"author": {"date": "now"},
                                                  "message": "test"}})
        rc = uc.main(["status"])
        captured = capsys.readouterr()
        assert rc == 0
        assert "stable" in captured.out or "channel" in captured.out.lower()

    def test_check_subcommand(self, update_dir, monkeypatch, capsys):
        from vllm._genesis.compat import update_channel as uc
        monkeypatch.setattr(uc, "_fetch_github_ref",
                            lambda c: {"sha": "abc123",
                                       "commit": {"author": {"date": "now"},
                                                  "message": "test"}})
        rc = uc.main(["check"])
        assert rc in (0, 1)
        captured = capsys.readouterr()
        assert "abc123" in captured.out or "upstream" in captured.out.lower()

    def test_channel_set_subcommand(self, update_dir, capsys):
        from vllm._genesis.compat import update_channel as uc
        rc = uc.main(["channel", "set", "beta"])
        assert rc == 0
        captured = capsys.readouterr()
        assert "beta" in captured.out

    def test_channel_set_invalid_returns_nonzero(self, update_dir, capsys):
        from vllm._genesis.compat import update_channel as uc
        # argparse with choices= raises SystemExit(2) on invalid choice;
        # both that AND a clean nonzero return are acceptable rejection
        # signals
        try:
            rc = uc.main(["channel", "set", "nightly"])
            assert rc != 0
        except SystemExit as e:
            assert e.code != 0

    def test_channel_get_subcommand(self, update_dir, capsys):
        from vllm._genesis.compat import update_channel as uc
        uc.main(["channel", "set", "beta"])
        capsys.readouterr()  # clear
        rc = uc.main(["channel", "get"])
        assert rc == 0
        captured = capsys.readouterr()
        assert "beta" in captured.out

    def test_apply_subcommand_returns_message(
        self, update_dir, monkeypatch, capsys,
    ):
        """`apply` is currently deferred — should print a clear
        operator-action message rather than fail silently."""
        from vllm._genesis.compat import update_channel as uc
        monkeypatch.setattr(uc, "_fetch_github_ref",
                            lambda c: {"sha": "abc123",
                                       "commit": {"author": {"date": "now"},
                                                  "message": "test"}})
        rc = uc.main(["apply"])
        captured = capsys.readouterr()
        # Either it gives the manual command, or it succeeds
        # (depends on impl). Must not crash.
        assert isinstance(rc, int)
        joined = captured.out + captured.err
        # Should mention git pull or manual instruction
        assert "git" in joined.lower() or "manual" in joined.lower() or rc == 0
