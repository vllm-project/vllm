# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the `vllm launch` CLI subcommand."""

import argparse
import json
import sys
from unittest.mock import patch

import pytest

from vllm.entrypoints.cli.launch import (
    LaunchSubcommand,
    RenderSubcommand,
    cmd_init,
)
from vllm.entrypoints.cli.snapshot import SnapshotSubcommand
from vllm.entrypoints.snapshot import (
    creation_env,
    environment_miss,
    environment_record,
    key_from,
    maybe_restore_serve,
)
from vllm.utils.argparse_utils import FlexibleArgumentParser


@pytest.fixture
def launch_parser():
    parser = FlexibleArgumentParser(description="test")
    subparsers = parser.add_subparsers(required=False, dest="subparser")
    LaunchSubcommand().subparser_init(subparsers)
    return parser


def test_subcommand_name():
    assert LaunchSubcommand().name == "launch"


def test_cmd_init_returns_subcommand():
    result = cmd_init()
    assert len(result) == 1
    assert isinstance(result[0], LaunchSubcommand)


# -- Parsing: `vllm launch render` --


def test_parse_launch_render(launch_parser):
    args = launch_parser.parse_args(["launch", "render", "--model", "test-model"])
    assert args.launch_component == "render"


def test_parse_launch_requires_component(launch_parser):
    with pytest.raises(SystemExit):
        launch_parser.parse_args(["launch", "--model", "test-model"])


def test_parse_launch_invalid_component(launch_parser):
    with pytest.raises(SystemExit):
        launch_parser.parse_args(["launch", "unknown", "--model", "test-model"])


# -- Dispatch --


def test_cmd_launch_render_calls_run():
    args = argparse.Namespace(model_tag=None, model="test-model")
    with patch("vllm.entrypoints.cli.launch.uvloop.run") as mock_uvloop_run:
        RenderSubcommand.cmd(args)
        mock_uvloop_run.assert_called_once()


def test_cmd_launch_model_tag_overrides():
    args = argparse.Namespace(
        model_tag="tag-model",
        model="original-model",
        launch_command=lambda a: None,
    )
    LaunchSubcommand.cmd(args)
    assert args.model == "tag-model"


def test_cmd_launch_model_tag_none():
    args = argparse.Namespace(
        model_tag=None,
        model="original-model",
        launch_command=lambda a: None,
    )
    LaunchSubcommand.cmd(args)
    assert args.model == "original-model"


def test_cmd_dispatches():
    called = {}

    def fake_dispatch(args):
        called["args"] = args

    args = argparse.Namespace(launch_command=fake_dispatch)
    LaunchSubcommand.cmd(args)
    assert "args" in called


# -- Module registration --


def test_subparser_init_returns_parser():
    parser = FlexibleArgumentParser(description="test")
    subparsers = parser.add_subparsers(required=False, dest="subparser")
    result = LaunchSubcommand().subparser_init(subparsers)
    assert isinstance(result, FlexibleArgumentParser)


def test_launch_registered_in_main():
    """Verify that launch module is importable as a CLI module."""
    import vllm.entrypoints.cli.launch as launch_module

    assert hasattr(launch_module, "cmd_init")
    subcmds = launch_module.cmd_init()
    assert any(s.name == "launch" for s in subcmds)


# -- `vllm snapshot` subcommand (folded here per the no-new-test-file rule; this
#    file now also covers the snapshot CLI surface) --


@pytest.fixture
def snapshot_parser():
    parser = FlexibleArgumentParser(description="test")
    subparsers = parser.add_subparsers(required=False, dest="subparser")
    SnapshotSubcommand().subparser_init(subparsers)
    return parser


def test_snapshot_registered_in_main():
    import vllm.entrypoints.cli.snapshot as snapshot_module

    assert hasattr(snapshot_module, "cmd_init")
    subcmds = snapshot_module.cmd_init()
    assert any(s.name == "snapshot" for s in subcmds)


def test_parse_snapshot_create_flags(snapshot_parser):
    args = snapshot_parser.parse_args(["snapshot", "create", "--dry-run", "--force"])
    assert args.dry_run is True
    assert args.force is True


def test_restore_hook_noop_when_disabled(monkeypatch, caplog):
    monkeypatch.delenv("VLLM_SNAPSHOT", raising=False)
    with caplog.at_level("INFO", logger="vllm.entrypoints.snapshot"):
        assert maybe_restore_serve() is None
    assert not any(
        "snapshot restore" in record.getMessage() for record in caplog.records
    )


def test_restore_hook_cold_fallback_logs_miss(monkeypatch, caplog, tmp_path):
    # Enabled + `serve` + no matching snapshot: the hook must fall back to a cold
    # start (return None) and log exactly one miss line. Platform is pinned so
    # the linux-only gate runs off-linux too, and the lookup key is stubbed so
    # the test stays on the no-snapshot path even in editable or RECORD-less
    # environments where key computation itself would refuse.
    import vllm.entrypoints.snapshot as snapshot_module

    monkeypatch.setattr(snapshot_module, "_entry_state", {})
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(sys, "argv", ["vllm", "serve", "some-model"])
    monkeypatch.setenv("VLLM_SNAPSHOT", "1")
    monkeypatch.delenv("VLLM_SNAPSHOT_RESTORED", raising=False)
    monkeypatch.delenv("PYTHONHASHSEED", raising=False)
    monkeypatch.setenv("VLLM_SNAPSHOT_ROOT", str(tmp_path))
    monkeypatch.setattr(snapshot_module, "lookup_key", lambda env: {"stub": 1})
    with caplog.at_level("INFO", logger="vllm.entrypoints.snapshot"):
        assert maybe_restore_serve() is None
    misses = [
        record
        for record in caplog.records
        if "snapshot restore miss" in record.getMessage()
    ]
    assert len(misses) == 1
    assert "no snapshot" in misses[0].getMessage()


def test_restore_hook_refuses_pythonhashseed(monkeypatch, caplog):
    # A restored interpreter keeps its create-time hash seed, so a requested
    # PYTHONHASHSEED can never be honored: the hook must miss explicitly,
    # before any key lookup.
    import vllm.entrypoints.snapshot as snapshot_module

    monkeypatch.setattr(snapshot_module, "_entry_state", {})
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(sys, "argv", ["vllm", "serve", "some-model"])
    monkeypatch.setenv("VLLM_SNAPSHOT", "1")
    monkeypatch.delenv("VLLM_SNAPSHOT_RESTORED", raising=False)
    monkeypatch.setenv("PYTHONHASHSEED", "0")
    with caplog.at_level("INFO", logger="vllm.entrypoints.snapshot"):
        assert maybe_restore_serve() is None
    misses = [
        record
        for record in caplog.records
        if "snapshot restore miss" in record.getMessage()
    ]
    assert len(misses) == 1
    assert "PYTHONHASHSEED" in misses[0].getMessage()


def test_creation_env_drops_secrets_keeps_policy_vars():
    # Credentials never reach the dumped helper; policy vars that merely end
    # in _TOKEN (import-affecting) stay in the keyed env. Shell bookkeeping
    # (SHLVL moves between a wrapper's create child and its exec'd serve)
    # is scrubbed so it can never key or miss a snapshot.
    env = {
        "HF_TOKEN": "hf_secret",
        "HF_HUB_DISABLE_IMPLICIT_TOKEN": "1",
        "PATH": "/usr/bin",
        "SHLVL": "1",
        "_": "/usr/local/bin/vllm",
        "HOSTNAME": "9f2c81d0e4a7",
    }
    values = creation_env(env)
    assert "HF_TOKEN" not in values
    assert values["HF_HUB_DISABLE_IMPLICIT_TOKEN"] == "1"
    assert "SHLVL" not in values
    assert "_" not in values
    assert "HOSTNAME" not in values


def test_environment_miss_ignores_secrets_not_policy_vars():
    # A live secret absent from the create-side record must not cold-fallback
    # the restore; a policy-var difference must still be named.
    recorded = environment_record({"PATH": "/usr/bin"})["values"]
    live_with_secret = {"PATH": "/usr/bin", "HF_TOKEN": "hf_live"}
    assert environment_miss(recorded, live_with_secret, frozenset()) is None
    live_with_policy = {
        "PATH": "/usr/bin",
        "HF_HUB_DISABLE_IMPLICIT_TOKEN": "1",
    }
    assert (
        environment_miss(recorded, live_with_policy, frozenset())
        == "env.HF_HUB_DISABLE_IMPLICIT_TOKEN"
    )


def test_pgid_empty_reads_the_process_table(monkeypatch):
    # Emptiness comes from /proc, not pgrep (absent on slim images); a zombie
    # still occupies its group until reaped.
    import vllm.entrypoints.snapshot as snapshot_module

    table = {10: (1, 42, 100, "Z")}
    monkeypatch.setattr(snapshot_module, "process_table", lambda: table)
    assert snapshot_module.pgid_empty(42) is False
    assert snapshot_module.pgid_empty(43) is True


def _prime_snapshot_dir(root, key, shared_objects=()):
    # A minimal on-disk snapshot the module treats as restorable: exact-key
    # manifest whose env record matches the live creation env and whose work
    # assets exist.
    directory = root / key
    (directory / "work").mkdir(parents=True)
    # The trust check stats both, and pytest creates these at 0777 & ~umask,
    # so pin them rather than let the runner's umask decide.
    root.chmod(0o700)
    directory.chmod(0o700)
    (directory / "work" / "stdin.null").touch()
    manifest = {
        "env": environment_record(creation_env()),
        "shared_objects": list(shared_objects),
        "work_assets": ["stdin.null"],
    }
    (directory / "MANIFEST.json").write_text(json.dumps(manifest))
    return directory


def test_snapshot_create_early_exit_when_snapshot_current(monkeypatch, tmp_path):
    # Bare `vllm snapshot create` with a layer2-valid exact-key snapshot on
    # disk must exit 3 ("already exists") before the eager CLI imports; the
    # lookup key is stubbed as in the cold-fallback test so key computation
    # never refuses in editable environments.
    import vllm.entrypoints.snapshot as snapshot_module

    monkeypatch.setattr(snapshot_module, "_entry_state", {})
    monkeypatch.setattr(sys, "argv", ["vllm", "snapshot", "create"])
    monkeypatch.delenv("VLLM_SNAPSHOT", raising=False)
    monkeypatch.setenv("VLLM_SNAPSHOT_ROOT", str(tmp_path))
    monkeypatch.setattr(snapshot_module, "lookup_key", lambda env: {"stub": 1})
    # 0770: group-writable must stay trusted (arbitrary-UID pods need it), so
    # re-adding a group-write refusal fails this test. The owner half of that
    # narrowing is uncoverable without a second uid.
    _prime_snapshot_dir(tmp_path, key_from({"stub": 1})).chmod(0o770)
    with pytest.raises(SystemExit) as excinfo:
        maybe_restore_serve()
    assert excinfo.value.code == 3


def test_snapshot_create_falls_through_on_stale_manifest(monkeypatch, tmp_path):
    # A stale exact-key manifest (a recorded .so that no longer matches) must
    # NOT early-exit: the hook returns and the slow path re-primes in place.
    import vllm.entrypoints.snapshot as snapshot_module

    monkeypatch.setattr(snapshot_module, "_entry_state", {})
    monkeypatch.setattr(sys, "argv", ["vllm", "snapshot", "create"])
    monkeypatch.delenv("VLLM_SNAPSHOT", raising=False)
    monkeypatch.setenv("VLLM_SNAPSHOT_ROOT", str(tmp_path))
    monkeypatch.setattr(snapshot_module, "lookup_key", lambda env: {"stub": 1})
    _prime_snapshot_dir(
        tmp_path,
        key_from({"stub": 1}),
        shared_objects=[{"path": "/nonexistent.so", "id": "sha256:0"}],
    )
    assert maybe_restore_serve() is None


def test_restore_refuses_world_writable_snapshot_dir(
    monkeypatch, caplog_vllm, tmp_path
):
    # criu restore executes the images, so a world-writable snapshot directory
    # must not be trusted. The same fixture restores at 0700, so the loosened
    # mode is the only variable: it must refuse, and say so at warning level
    # rather than as an ordinary miss.
    import vllm.entrypoints.snapshot as snapshot_module

    monkeypatch.setattr(snapshot_module, "_entry_state", {})
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(sys, "argv", ["vllm", "serve", "some-model"])
    monkeypatch.setenv("VLLM_SNAPSHOT", "1")
    monkeypatch.delenv("VLLM_SNAPSHOT_RESTORED", raising=False)
    monkeypatch.delenv("PYTHONHASHSEED", raising=False)
    monkeypatch.setenv("VLLM_SNAPSHOT_ROOT", str(tmp_path))
    monkeypatch.setattr(snapshot_module, "lookup_key", lambda env: {"stub": 1})
    _prime_snapshot_dir(tmp_path, key_from({"stub": 1})).chmod(0o707)
    with caplog_vllm.at_level("WARNING", logger="vllm.entrypoints.snapshot"):
        assert maybe_restore_serve() is None
    refusals = [r for r in caplog_vllm.records if "restore refused" in r.getMessage()]
    assert len(refusals) == 1
    assert "trust.mode" in refusals[0].getMessage()


def _serve_restore_env(monkeypatch, root):
    # The shared restore-hook harness: enabled, linux-pinned, stubbed key.
    import vllm.entrypoints.snapshot as snapshot_module

    monkeypatch.setattr(snapshot_module, "_entry_state", {})
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(sys, "argv", ["vllm", "serve", "some-model"])
    monkeypatch.setenv("VLLM_SNAPSHOT", "1")
    monkeypatch.delenv("VLLM_SNAPSHOT_RESTORED", raising=False)
    monkeypatch.delenv("PYTHONHASHSEED", raising=False)
    monkeypatch.setenv("VLLM_SNAPSHOT_ROOT", str(root))
    monkeypatch.setattr(snapshot_module, "lookup_key", lambda env: {"stub": 1})
    return snapshot_module


def test_snapshot_create_refuses_instead_of_repriming_untrusted_dir(
    monkeypatch, tmp_path
):
    # A world-writable key dir must refuse creation, never re-prime: the
    # surviving manifest is the discriminating assertion (a re-priming path
    # rmtrees the directory before the refusal could fire).
    import vllm.entrypoints.snapshot as snapshot_module

    monkeypatch.setattr(snapshot_module, "_entry_state", {})
    monkeypatch.setenv("VLLM_SNAPSHOT_ROOT", str(tmp_path))
    monkeypatch.setattr(snapshot_module, "lookup_key", lambda env: {"stub": 1})
    monkeypatch.setattr(snapshot_module, "require_dump_host", lambda: None)
    directory = _prime_snapshot_dir(tmp_path, key_from({"stub": 1}))
    directory.chmod(0o707)
    with pytest.raises(RuntimeError, match="trust.mode"):
        snapshot_module.create_snapshot()
    assert (directory / "MANIFEST.json").exists()


def test_restore_trust_gate_runs_before_lock(monkeypatch, caplog_vllm, tmp_path):
    # A world-writable ROOT must be refused before _acquire_lock creates a
    # lock file inside it; the absent lock file pins the order, not just the
    # refusal.
    _serve_restore_env(monkeypatch, tmp_path)
    key = key_from({"stub": 1})
    _prime_snapshot_dir(tmp_path, key)
    tmp_path.chmod(0o777)
    with caplog_vllm.at_level("WARNING", logger="vllm.entrypoints.snapshot"):
        assert maybe_restore_serve() is None
    assert any("restore refused" in r.getMessage() for r in caplog_vllm.records)
    assert not (tmp_path / f"{key}.lock").exists()


def test_restore_refuses_symlinked_key_dir_before_reading_it(
    monkeypatch, caplog_vllm, tmp_path
):
    # root/<key> is created by us at 0700 and is never legitimately a
    # symlink, so a planted one is refused on the LINK. The malformed
    # manifest behind it discriminates check-before-read: a parse-first path
    # would report an unreadable-manifest miss instead.
    root = tmp_path / "root"
    _serve_restore_env(monkeypatch, root)
    key = key_from({"stub": 1})
    root.mkdir()
    root.chmod(0o700)
    target = tmp_path / "elsewhere"
    target.mkdir()
    target.chmod(0o700)
    (target / "MANIFEST.json").write_text("not json{")
    (root / key).symlink_to(target)
    with caplog_vllm.at_level("INFO", logger="vllm.entrypoints.snapshot"):
        assert maybe_restore_serve() is None
    refusals = [r for r in caplog_vllm.records if "restore refused" in r.getMessage()]
    assert len(refusals) == 1
    assert "trust.link" in refusals[0].getMessage()
    assert not any("restore miss" in r.getMessage() for r in caplog_vllm.records)


@pytest.mark.parametrize(("parent_mode", "refused"), [(0o777, True), (0o1777, False)])
def test_restore_judges_world_writable_ancestors(
    monkeypatch, caplog_vllm, tmp_path, parent_mode, refused
):
    # A 0777 parent of the root grants the image-swap primitive and must be
    # refused by name; the same layout with a sticky, euid-owned parent and
    # an euid-owned child is a legitimate /tmp-style layout and is not. The
    # malformed manifest stops the trusted case at a parse miss, after the
    # gate, without invoking criu.
    parent = tmp_path / "parent"
    root = parent / "root"
    _serve_restore_env(monkeypatch, root)
    directory = _prime_snapshot_dir(root, key_from({"stub": 1}))
    (directory / "MANIFEST.json").write_text("not json{")
    parent.chmod(parent_mode)
    with caplog_vllm.at_level("INFO", logger="vllm.entrypoints.snapshot"):
        assert maybe_restore_serve() is None
    refusals = [r for r in caplog_vllm.records if "restore refused" in r.getMessage()]
    if refused:
        assert len(refusals) == 1
        assert "trust.mode.parent" in refusals[0].getMessage()
    else:
        assert not refusals
        assert any("restore miss" in r.getMessage() for r in caplog_vllm.records)


def test_restore_lock_symlink_does_not_truncate_target(monkeypatch, tmp_path):
    # A planted symlink at the lock name must fail the open (cold-start
    # miss), not truncate its target as root; the surviving content is the
    # discriminating assertion.
    _serve_restore_env(monkeypatch, tmp_path)
    key = key_from({"stub": 1})
    _prime_snapshot_dir(tmp_path, key)
    victim = tmp_path / "victim"
    victim.write_text("precious")
    (tmp_path / f"{key}.lock").symlink_to(victim)
    assert maybe_restore_serve() is None
    assert victim.read_text() == "precious"


def test_snapshot_create_refuses_hostile_parent_before_creating_root(
    monkeypatch, tmp_path
):
    # First create under a 0777 non-sticky parent: the missing root must not
    # exempt the check, and nothing may be created beneath the parent.
    import vllm.entrypoints.snapshot as snapshot_module

    parent = tmp_path / "parent"
    parent.mkdir()
    parent.chmod(0o777)
    root = parent / "root"
    monkeypatch.setattr(snapshot_module, "_entry_state", {})
    monkeypatch.setenv("VLLM_SNAPSHOT_ROOT", str(root))
    monkeypatch.setattr(snapshot_module, "lookup_key", lambda env: {"stub": 1})
    monkeypatch.setattr(snapshot_module, "require_dump_host", lambda: None)
    with pytest.raises(RuntimeError, match="trust.mode"):
        snapshot_module.create_snapshot()
    assert list(parent.iterdir()) == []


def test_snapshot_create_fast_path_distrusts_world_writable_dir(monkeypatch, tmp_path):
    # A world-writable key dir holding a VALID manifest: the bare-create fast
    # path must neither exit 3 nor crash (None, not False, from the
    # validity probe); the slow path refuses loudly.
    import vllm.entrypoints.snapshot as snapshot_module

    monkeypatch.setattr(snapshot_module, "_entry_state", {})
    monkeypatch.setattr(sys, "argv", ["vllm", "snapshot", "create"])
    monkeypatch.delenv("VLLM_SNAPSHOT", raising=False)
    monkeypatch.setenv("VLLM_SNAPSHOT_ROOT", str(tmp_path))
    monkeypatch.setattr(snapshot_module, "lookup_key", lambda env: {"stub": 1})
    _prime_snapshot_dir(tmp_path, key_from({"stub": 1})).chmod(0o707)
    assert maybe_restore_serve() is None
    monkeypatch.setattr(snapshot_module, "require_dump_host", lambda: None)
    with pytest.raises(RuntimeError, match="trust.mode"):
        snapshot_module.create_snapshot()


def test_restore_accepts_trusted_symlinked_root(monkeypatch, tmp_path):
    # A root that IS a symlink (cache moved to a bigger disk) is a legitimate
    # layout: the lock file appearing on the target side proves the restore
    # got through the trust gate to _acquire_lock; the malformed manifest
    # then stops it before criu.
    real = tmp_path / "real"
    link = tmp_path / "link"
    _serve_restore_env(monkeypatch, link)
    key = key_from({"stub": 1})
    directory = _prime_snapshot_dir(real, key)
    (directory / "MANIFEST.json").write_text("not json{")
    link.symlink_to(real)
    assert maybe_restore_serve() is None
    assert (real / f"{key}.lock").exists()


def test_restore_cold_boot_under_sticky_parent_is_quiet_miss(
    monkeypatch, caplog_vllm, tmp_path
):
    # First boot with no root yet under a sticky /tmp-style parent: an
    # ordinary INFO miss, never a trust warning. The unclaimed-missing rule
    # (a missing component under a sticky parent is untrusted for unlocked
    # readers) must not make legitimate cold starts noisy.
    tmp_path.chmod(0o1777)
    root = tmp_path / "root"
    _serve_restore_env(monkeypatch, root)
    with caplog_vllm.at_level("INFO", logger="vllm.entrypoints.snapshot"):
        assert maybe_restore_serve() is None
    assert not any("restore refused" in r.getMessage() for r in caplog_vllm.records)
    assert any("restore miss" in r.getMessage() for r in caplog_vllm.records)


def test_snapshot_create_flags_skip_early_exit(monkeypatch, tmp_path):
    # Any token beyond the bare two-token form (--force here, but equally
    # --dry-run/--help/typos) must reach argparse on the slow path even when
    # a current snapshot exists.
    import vllm.entrypoints.snapshot as snapshot_module

    monkeypatch.setattr(snapshot_module, "_entry_state", {})
    monkeypatch.setattr(sys, "argv", ["vllm", "snapshot", "create", "--force"])
    monkeypatch.delenv("VLLM_SNAPSHOT", raising=False)
    monkeypatch.setenv("VLLM_SNAPSHOT_ROOT", str(tmp_path))
    monkeypatch.setattr(snapshot_module, "lookup_key", lambda env: {"stub": 1})
    _prime_snapshot_dir(tmp_path, key_from({"stub": 1}))
    assert maybe_restore_serve() is None


def test_lookup_key_keys_dpkg_dists_by_deb_revision(monkeypatch):
    # A RECORD-less dpkg-managed dist (Debian ships apt python packages
    # without RECORD) must not refuse the snapshot key, and must be keyed by
    # the dpkg package version: a security update bumps only the Debian
    # revision, so the digest has to change with it.
    import pathlib

    import vllm.entrypoints.snapshot as snapshot_module

    class FakeDpkgDist:
        name = "protobuf"
        version = "3.12.4"
        metadata = {"Name": "protobuf"}
        _path = pathlib.Path("/usr/lib/python3/dist-packages/protobuf-3.12.4.egg-info")

        def read_text(self, _name):
            return None

    deb = {"version": "3.12.4-1ubuntu7"}

    def fake_run(argv, **_kwargs):
        class Result:
            returncode = 0
            stderr = ""
            stdout = ""

        result = Result()
        if argv[0] == "criu":
            result.stdout = "Version: 4.2"
        elif argv[0] == "dpkg":
            result.stdout = f"python3-protobuf: {argv[-1]}\n"
        else:  # dpkg-query
            result.stdout = f"python3-protobuf {deb['version']}\n"
        return result

    monkeypatch.setattr(snapshot_module, "_entry_state", {})
    monkeypatch.setattr(
        snapshot_module.importlib.metadata,
        "distributions",
        lambda **_kwargs: [FakeDpkgDist()],
    )
    monkeypatch.setattr(snapshot_module.subprocess, "run", fake_run)
    first = snapshot_module.lookup_key(creation_env())["dists_digest"]
    deb["version"] = "3.12.4-1ubuntu7.22.04.2"
    second = snapshot_module.lookup_key(creation_env())["dists_digest"]
    assert first != second


def _make_dist(site, name):
    info = site / f"{name}-1.0.dist-info"
    info.mkdir(parents=True)
    (info / "METADATA").write_text(
        f"Metadata-Version: 2.1\nName: {name}\nVersion: 1.0\n"
    )
    (info / "RECORD").write_text(f"{name}/__init__.py,,\n")


def test_lookup_key_ignores_sys_path_growth_after_entry(monkeypatch, tmp_path):
    # `snapshot create` keys after the eager CLI imports while the restore
    # hook keys before them, and importing the serve envelope grows sys.path
    # (setuptools appends its _vendor directory). The dists digest must walk
    # the captured entry path, or create and restore disagree forever.
    import vllm.entrypoints.snapshot as snapshot_module

    base = tmp_path / "base-site"
    extra = tmp_path / "extra-site"
    _make_dist(base, "basepkg")
    _make_dist(extra, "extrapkg")
    monkeypatch.setattr(sys, "path", [str(base)])
    monkeypatch.setattr(sys, "argv", ["vllm", "snapshot"])
    monkeypatch.setattr(snapshot_module, "_entry_state", {})
    maybe_restore_serve()  # captures entry state, the public hook's job
    first = snapshot_module.lookup_key(creation_env())["dists_digest"]
    sys.path.append(str(extra))
    second = snapshot_module.lookup_key(creation_env())["dists_digest"]
    assert first == second
    # negative control: the extra dir does change the digest once it is part
    # of the captured entry state, so the equality above is not vacuous
    monkeypatch.setattr(snapshot_module, "_entry_state", {})
    maybe_restore_serve()
    third = snapshot_module.lookup_key(creation_env())["dists_digest"]
    assert third != first
