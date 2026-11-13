# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the pip-compile blessed-hash wrapper.

A parsing regression would silently drop a file's content from the hash and
serve stale lockfiles as blessed. These tests lock the argv parsing against
the real hook shapes and the hash against input changes.
"""

import json

import pytest

from tools.pre_commit.pip_compile_blessed import (
    compute_hash,
    is_network_error,
    parse_args,
)

CUDA_ARGS = [
    "--generate-hashes",
    "requirements/test/cuda.in",
    "-c",
    "requirements/cuda.txt",
    "-o",
    "requirements/test/cuda.txt",
    "--index-strategy",
    "unsafe-best-match",
    "--extra-index-url",
    "https://download.pytorch.org/whl/cu132",
    "--emit-index-url",
    "--python-platform",
    "x86_64-manylinux_2_28",
    "--python-version",
    "3.12",
]

KV_CONNECTORS_ARGS = [
    "requirements/test/kv_connectors_cuda.in",
    "-c",
    "requirements/cuda.txt",
    "-c",
    "requirements/test/cuda.txt",
    "-o",
    "requirements/test/kv_connectors_cuda.txt",
    "--override",
    "requirements/test/kv_connectors_cuda.overrides",
    "--generate-hashes",
    "--no-emit-package",
    "nixl-cu12",
    "--no-emit-package",
    "nvidia-nccl-cu13",
    "--index-strategy",
    "unsafe-best-match",
    "--torch-backend",
    "cu132",
    "--python-platform",
    "x86_64-manylinux_2_28",
    "--python-version",
    "3.12",
]


def test_parse_cuda_hook_boolean_flag_not_swallowed():
    # --emit-index-url takes no value; treating it as value-taking would
    # swallow --python-platform and parse "3.12" as the source, dropping
    # the .in content from the hash.
    src, out, constraints, overrides = parse_args(CUDA_ARGS)
    assert src == "requirements/test/cuda.in"
    assert out == "requirements/test/cuda.txt"
    assert constraints == ["requirements/cuda.txt"]
    assert overrides == []


def test_parse_kv_connectors_hook():
    src, out, constraints, overrides = parse_args(KV_CONNECTORS_ARGS)
    assert src == "requirements/test/kv_connectors_cuda.in"
    assert out == "requirements/test/kv_connectors_cuda.txt"
    assert constraints == ["requirements/cuda.txt", "requirements/test/cuda.txt"]
    assert overrides == ["requirements/test/kv_connectors_cuda.overrides"]


def test_parse_source_first_flag_only():
    src, out, constraints, _ = parse_args(["docs.in", "-o", "docs.txt"])
    assert src == "docs.in"
    assert out == "docs.txt"
    assert constraints == []


def test_parse_unknown_boolean_flag_skips_one_token():
    src, out, _, _ = parse_args(["a.in", "--future-flag", "-o", "a.txt"])
    assert src == "a.in"
    assert out == "a.txt"


def test_parse_unlisted_value_flag_fails_loudly(capsys):
    # A value-taking flag missing from VALUE_FLAGS leaks its value as a
    # second positional; this must exit loudly, not corrupt the hash.
    with pytest.raises(SystemExit):
        parse_args(["a.in", "--future-with-value", "x", "-o", "a.txt"])
    assert "VALUE_FLAGS" in capsys.readouterr().err


def test_parse_missing_value_fails_loudly(capsys):
    with pytest.raises(SystemExit):
        parse_args(["a.in", "-o"])
    assert "missing its value" in capsys.readouterr().err


def test_parse_requires_source_and_output(capsys):
    with pytest.raises(SystemExit):
        parse_args(["-o", "a.txt"])
    assert "cannot parse" in capsys.readouterr().err


def test_compute_hash_covers_every_input(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    src = tmp_path / "a.in"
    lock = tmp_path / "a.txt"
    constraint = tmp_path / "c.txt"
    override = tmp_path / "o.txt"
    for path, text in (
        (src, "deps\n"),
        (lock, "lock\n"),
        (constraint, "pin\n"),
        (override, "keep\n"),
    ):
        path.write_text(text)
    argv = ["a.in", "-c", "c.txt", "--override", "o.txt", "-o", "a.txt"]

    def digest():
        return compute_hash(argv, "a.in", "a.txt", ["c.txt"], ["o.txt"])

    baseline = digest()
    assert digest() == baseline
    src.write_text("deps2\n")
    assert digest() != baseline
    src.write_text("deps\n")
    constraint.write_text("pin2\n")
    assert digest() != baseline
    constraint.write_text("pin\n")
    override.write_text("keep2\n")
    assert digest() != baseline
    override.write_text("keep\n")
    lock.write_text("lock2\n")
    assert digest() != baseline
    lock.write_text("lock\n")
    assert digest() == baseline
    # Any argv change (e.g. a new flag) changes the digest.
    assert (
        compute_hash(argv + ["--quiet"], "a.in", "a.txt", ["c.txt"], ["o.txt"])
        != baseline
    )


def test_compute_hash_covers_transitive_includes(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "a.in").write_text("-r base.txt\npkg==1\n")
    (tmp_path / "base.txt").write_text("common==1\n")
    (tmp_path / "a.txt").write_text("lock\n")
    argv = ["a.in", "-o", "a.txt"]

    def digest():
        return compute_hash(argv, "a.in", "a.txt", [], [])

    baseline = digest()
    # Editing the included file is a resolution input even though it never
    # appears in the hook argv.
    (tmp_path / "base.txt").write_text("common==2\n")
    assert digest() != baseline
    (tmp_path / "base.txt").write_text("common==1\n")
    assert digest() == baseline


def test_compute_hash_covers_pyproject(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "a.in").write_text("pkg==1\n")
    (tmp_path / "a.txt").write_text("lock\n")
    argv = ["a.in", "-o", "a.txt"]
    baseline = compute_hash(argv, "a.in", "a.txt", [], [])
    (tmp_path / "pyproject.toml").write_text(
        "[tool.uv]\nexclude-dependencies = ['x']\n"
    )
    assert compute_hash(argv, "a.in", "a.txt", [], []) != baseline


def test_is_network_error_classification():
    for output in (
        "error: failed to fetch https://example.com/simple/",
        "HTTP status server error 503",
        "error sending request for url (...)",
        "Temporary failure in name resolution",
    ):
        assert is_network_error(output)
    assert not is_network_error("no solution found because two deps conflict")
    assert not is_network_error("")


def test_store_round_trip(tmp_path, monkeypatch):
    import tools.pre_commit.pip_compile_blessed as blessed

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(blessed, "STORE", tmp_path / "requirements" / "store.json")
    assert blessed.load_store() == {}
    blessed.STORE.parent.mkdir(parents=True)
    blessed.STORE.write_text(json.dumps({"a.txt": "deadbeef"}))
    assert blessed.load_store() == {"a.txt": "deadbeef"}
