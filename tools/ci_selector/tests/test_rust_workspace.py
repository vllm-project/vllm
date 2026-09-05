# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Rust workspace parse: crate closures from the two shipped artifacts.

The two-root discriminator rests on which artifact a crate feeds; these pin
the parse floor and the fail-open direction so a moved workspace reads as
loud failure or as the widest RUST answer, not as the image union.
"""

import pytest
from ci_selector.codemap.rust_workspace import RustWorkspace


@pytest.fixture(scope="module")
def ws(vllm_repo):
    return RustWorkspace.build(vllm_repo)


def test_workspace_parse_floor(ws):
    """Derivation collapsing to nothing must fail loudly: ten members and
    both artifact closures are far below today's fourteen, so ordinary
    churn passes while a moved workspace or renamed root does not."""
    assert len(ws.members) >= 10, sorted(ws.members)
    assert len(ws.binary_crates) >= 8, sorted(ws.binary_crates)
    assert ws.cdylib_crates == {
        "rust/src/parser/python",
        "rust/src/parser",
        "rust/src/tokenizer",
    }


def test_buckets_follow_artifact_reach(ws):
    assert ws.bucket_of("rust/src/server/src/lib.rs") == "binary"
    assert ws.bucket_of("rust/src/tokenizer/src/lib.rs") == "cdylib"
    assert ws.bucket_of("rust/Cargo.lock") == "root"
    assert ws.bucket_of("rust-toolchain.toml") == "root"
    assert ws.bucket_of("build_rust.sh") == "root"
    assert ws.bucket_of("tools/build_rust.py") == "root"


def test_mock_engine_feeds_no_artifact_but_stays_binary(ws):
    """A dev fixture only cargo compiles; calling it nothing-to-run would
    silence a cargo-visible crate for a saving of zero steps, since the
    cargo steps ride in every bucket anyway."""
    assert ws.bucket_of("rust/src/mock-engine/src/lib.rs") == "binary"


def test_unknown_rust_path_fails_open_to_root_bucket(ws):
    """A new crate the parser has not met takes the WIDEST rust answer, not the
    image union, which would balloon the answer for exactly the files most
    likely to hit the fail-open (new and moved members)."""
    assert ws.bucket_of("rust/src/brand_new_crate/src/lib.rs") == "root"
    assert ws.bucket_of("rust/proto/engine.proto") == "root"


def test_nested_member_longest_prefix(ws):
    """rust/src/parser/python/ must beat rust/src/parser/ or the cdylib's
    own sources bucket as the parent crate."""
    assert ws.bucket_of("rust/src/parser/python/src/lib.rs") == "cdylib"
    assert ws.bucket_of("rust/src/parser/src/lib.rs") == "cdylib"


def test_owns_covers_the_toolchain_trio_and_nothing_python(ws):
    for p in ("rust/src/cmd/src/main.rs", "rust-toolchain.toml", "build_rust.sh"):
        assert ws.owns(p), p
    for p in ("vllm/envs.py", "tools/ci_selector/pyproject.toml", "rustfmt.toml"):
        assert not ws.owns(p), p
