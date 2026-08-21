#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Header dependency ingestion for the build-graph exporter (D1).

Primary source: `ninja -t deps` output captured while the build tree still
exists. CMake+Ninja consumes compiler .d files into the binary .ninja_deps
database and deletes them (verified on a real build), so on a completed
Ninja build the tool dump is the only depfile source. Classic on-disk *.d
scanning is retained solely as the make-generator fallback.

Scope rules (measured, not silent): dependencies inside the build tree
(generated files) and outside the source root (system headers, toolchain)
are dropped and counted; rules whose object path yields no
CMakeFiles/<name>.dir target segment (e.g. link rules) are counted as
unattributed.
"""

import pathlib
from collections import defaultdict

import regex as re

_SPLIT = re.compile(r"(?<!\\)[ \t\n]+")
_NINJA_HEADER = re.compile(r"^(\S+): #deps \d+, deps mtime \d+ \((\w+)\)$")


def parse_depfile(text):
    r"""Parse make-style depfile text -> {rule_target: [dep, ...]}.

    Handles backslash-newline continuations, escaped spaces (``\ ``),
    multiple rules per file, and duplicate deps (preserved; caller dedups).
    """
    rules = {}
    text = text.replace("\\\n", " ")
    for line in text.split("\n"):
        if ":" not in line:
            continue
        target, _, deps = line.partition(":")
        target = target.strip()
        if not target:
            continue
        items = [d.replace("\\ ", " ") for d in _SPLIT.split(deps.strip()) if d]
        rules.setdefault(target, []).extend(items)
    return rules


def parse_ninja_deps(text):
    """Parse `ninja -t deps` output -> {object_path: [dep, ...]}.

    STALE entries are skipped; link rules (no CMakeFiles/<t>.dir segment)
    are kept here and filtered by the caller's target extraction.
    """
    rules = {}
    current = None
    for line in text.splitlines():
        m = _NINJA_HEADER.match(line)
        if m:
            current = m.group(1) if m.group(2) == "VALID" else None
            if current is not None:
                rules.setdefault(current, [])
            continue
        if current is not None and line.startswith((" ", "\t")):
            dep = line.strip()
            if dep:
                rules[current].append(dep)
        elif not line.strip():
            current = None
    return rules


def target_from_object_path(path):
    """CMakeFiles/<name>.dir/... -> <name>, else None."""
    for part in pathlib.PurePath(path).parts:
        if part.endswith(".dir"):
            return part[: -len(".dir")]
    return None


def collect_file_target_pairs(rules, build_dir, source_root):
    """Resolve dep rules to sorted (repo-relative file, target) pairs.

    Relative deps resolve against the build dir (ninja invokes the compiler
    from the build root). Returns (pairs, stats).
    """
    build_dir = pathlib.Path(build_dir).resolve()
    source_root = pathlib.Path(source_root).resolve()
    pairs = set()
    stats = defaultdict(int)
    for obj, deps in rules.items():
        stats["deps_rules"] += 1
        target = target_from_object_path(obj)
        if target is None:
            stats["rules_unattributed"] += 1
            continue
        for dep in deps:
            p = pathlib.Path(dep)
            if not p.is_absolute():
                p = build_dir / p
            resolved = p.resolve()
            try:
                resolved.relative_to(build_dir)
            except ValueError:
                pass
            else:
                stats["deps_in_build_tree"] += 1
                continue
            try:
                rel = resolved.relative_to(source_root)
            except ValueError:
                stats["deps_outside_source_root"] += 1
                continue
            pairs.add((rel.as_posix(), target))
    stats["file_target_pairs"] = len(pairs)
    return sorted(pairs), dict(stats)


def load_rules(ninja_deps_path, build_dir):
    """Rules from the captured `ninja -t deps` dump, else *.d scan fallback.

    Returns (rules, source) where source is "ninja_deps" or "depfile_scan".
    """
    if ninja_deps_path is not None:
        text = pathlib.Path(ninja_deps_path).read_text(errors="replace")
        return parse_ninja_deps(text), "ninja_deps"
    rules = {}
    for depfile in pathlib.Path(build_dir).rglob("*.d"):
        for obj, deps in parse_depfile(depfile.read_text(errors="replace")).items():
            rules.setdefault(obj, []).extend(deps)
    return rules, "depfile_scan"
