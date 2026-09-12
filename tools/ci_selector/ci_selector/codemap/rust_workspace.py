# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Which shipped artifact a rust file feeds, from the workspace's own metadata.

tools/build_rust.py ships exactly two artifacts: the vllm-rs binary (opt-in
behind the gate env vars) and the _rust_tool_parser PyO3 cdylib (not opt-in).
A rust file's honest reach is the closure of the crate it lives in, so the
bucket decides which routing legs the rust rule composes. Everything is a
static tomllib parse: no cargo binary, no network, deterministic at
state-build time.

Fail-open direction: anything unparsable or unrecognized buckets as "root",
the widest RUST answer (union of both artifact routes) — never back to the
image union, which would resurrect the 268-step balloon for exactly the files
most likely to hit the fail-open (new or moved crates).
"""

from __future__ import annotations

import posixpath
from dataclasses import dataclass, field
from pathlib import Path

import tomllib

from ..handwritten import RUST_ARTIFACT_ROOTS, RUST_TOOLCHAIN_FILES

WORKSPACE_MANIFEST = "rust/Cargo.toml"


@dataclass
class RustWorkspace:
    members: dict[str, str] = field(default_factory=dict)  # dir -> crate name
    deps: dict[str, set[str]] = field(default_factory=dict)  # dir -> member dirs
    binary_crates: set[str] = field(default_factory=set)
    cdylib_crates: set[str] = field(default_factory=set)
    parse_failed: bool = False

    @classmethod
    def build(cls, repo: Path) -> RustWorkspace:
        ws = cls()
        try:
            manifest = tomllib.loads((repo / WORKSPACE_MANIFEST).read_text())
            member_dirs = [f"rust/{rel}" for rel in manifest["workspace"]["members"]]
        except (OSError, tomllib.TOMLDecodeError, KeyError, TypeError):
            ws.parse_failed = True
            return ws
        raw_deps: dict[str, dict] = {}
        name_to_dir: dict[str, str] = {}
        for mdir in member_dirs:
            try:
                mdata = tomllib.loads((repo / mdir / "Cargo.toml").read_text())
                name = mdata["package"]["name"]
            except (OSError, tomllib.TOMLDecodeError, KeyError, TypeError):
                continue  # its files fall to the root bucket via no-member-match
            ws.members[mdir] = name
            name_to_dir[name] = mdir
            raw_deps[mdir] = mdata.get("dependencies", {}) or {}
        for mdir, dep_table in raw_deps.items():
            resolved: set[str] = set()
            for dep_name, spec in dep_table.items():
                if dep_name in name_to_dir:
                    resolved.add(name_to_dir[dep_name])
                elif isinstance(spec, dict) and "path" in spec:
                    resolved.add(posixpath.normpath(f"{mdir}/{spec['path']}"))
            ws.deps[mdir] = resolved & set(ws.members)
        binary_root, cdylib_root = RUST_ARTIFACT_ROOTS
        ws.binary_crates = ws._closure(binary_root)
        ws.cdylib_crates = ws._closure(cdylib_root)
        # An empty closure means an artifact root is no longer a member;
        # bucketing on it would under-scope, so widen everything instead.
        if not ws.binary_crates or not ws.cdylib_crates:
            ws.parse_failed = True
        return ws

    def _closure(self, root: str) -> set[str]:
        if root not in self.members:
            return set()
        seen = {root}
        frontier = [root]
        while frontier:
            for dep in self.deps.get(frontier.pop(), set()):
                if dep not in seen:
                    seen.add(dep)
                    frontier.append(dep)
        return seen

    def owns(self, path: str) -> bool:
        return path.startswith("rust/") or path in RUST_TOOLCHAIN_FILES

    def bucket_of(self, path: str) -> str | None:
        """ "binary" | "cdylib" | "root" for rust files, None for the rest.

        cdylib wins over binary for the crates both artifacts reach (parser,
        tokenizer): those files affect the not-opt-in parser too, so they
        need the bridge leg on top of the binary route.
        """
        if not self.owns(path):
            return None
        if self.parse_failed or path in RUST_TOOLCHAIN_FILES:
            return "root"
        best = ""
        for mdir in self.members:
            if path.startswith(mdir + "/") and len(mdir) > len(best):
                best = mdir
        if not best:
            return "root"  # workspace files, proto/, a crate we never parsed
        if best in self.cdylib_crates:
            return "cdylib"
        if best in self.binary_crates:
            return "binary"
        # A member feeding neither artifact (mock-engine) is a dev fixture
        # only the cargo steps compile; the base route is safe and calling it
        # nothing-to-run would buy zero steps.
        return "binary"
