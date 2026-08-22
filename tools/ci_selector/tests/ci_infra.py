# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Reading vllm-project/ci-infra, which holds the pipeline generator.

We reproduce parts of that generator by hand. Until 2026-08-21 nothing watched
it, on the belief that a vLLM checkout could not reach it; it is public and
fetches from `raw.githubusercontent.com` with no auth.

Two jobs, and only the first needs network:

  download it     `pytest --sync`, which fetches and nothing else
  check it        ordinary offline tests, against what --sync wrote

`sync()` asserts nothing. It refuses to overwrite a good snapshot with a short
read, and otherwise just writes what it found. Every pass/fail lives offline in
`test_ci_infra_snapshot.py`.

The checks compare behaviour, not text. Constants are compared to the value the
generator assigns, and the functions we reproduce are **executed** out of the
snapshot and run against ours on generated inputs. That matters because the two
are written in different names and types, so no textual comparison could ever
say they agree, and a stored "someone approved this" baseline would need a
command to update it that could only ever rubber-stamp whatever was on disk.
"""

from __future__ import annotations

import ast
import json
import urllib.request
from pathlib import Path

SNAPSHOT = Path(__file__).parent / "ci_infra_snapshot"
MANIFEST = SNAPSHOT / "manifest.json"
VALUES = SNAPSHOT / "values.json"
# Not `.py`: this is somebody else's source, recorded verbatim. Naming it as
# Python would put ruff and mypy on it, and they would be right to complain
# about names it imports from modules we do not have.
SUFFIX = ".py.txt"

REPO = "vllm-project/ci-infra"
PKG = "buildkite/pipeline_generator"
# Both take a ref. Fetching at `main` and recording a commit resolved
# separately would let the two disagree, and did: the recorded sha was the last
# one to touch the generator while the bytes came from whatever main held.
RAW = "https://raw.githubusercontent.com/{repo}/{ref}/{pkg}/{name}"
CONTENTS = "https://api.github.com/repos/{repo}/contents/{pkg}?ref={ref}"
COMMITS = "https://api.github.com/repos/{repo}/commits?path={pkg}&per_page=1"

# Functions we transcribe or read a value out of, and our counterpart. None
# means we do not reimplement it: either we depend on the behaviour, or we only
# want a literal from its body.
ANCHORS: dict[str, tuple[str, str | None]] = {
    "_step_should_run": (
        "buildkite_step.py",
        "ci_selector.validate.generator_replica:step_should_run",
    ),
    # Upstream splits dep matching in two: the outer one applies the
    # include/exclude split across the diff, the inner one matches one dep
    # against one path.
    "_source_file_dependencies_match": (
        "buildkite_step.py",
        "ci_selector.codemap.claim:step_declares",
    ),
    "_matches_source_dependency": (
        "buildkite_step.py",
        "ci_selector.codemap.claim:matches_source_dependency",
    ),
    "is_docs_only_change": (
        "pipeline_generator.py",
        "ci_selector.codemap.claim:docs_only",
    ),
    # `--emit-keys` rests on this: we name tests and the generator pulls in
    # their prerequisites. We do not reimplement it, so watch it and no more.
    "select_steps_and_dependencies": ("pipeline_generator.py", None),
    # Anchored only for the working-dir default it assigns.
    "read_steps_from_job_dir": ("step.py", None),
}

# Anchors whose source runs on its own, so our version can be checked against
# real behaviour rather than against a diff someone once read. The rest depend
# on ci-infra's own imports; `_step_should_run` gets there with stubs.
SELF_CONTAINED = (
    "_matches_source_dependency",
    "_source_file_dependencies_match",
    "is_docs_only_change",
)

# Floors on the fetch. A short read would satisfy every offline check by
# leaving nothing to disagree with, so it must never reach disk.
MIN_FILES = 6
MIN_BYTES = 40_000
CONTROL_LITERAL = "def _step_should_run"


def normalized(source: str, name: str) -> str:
    """One function's source, reduced to what a logic change would move.

    Reparsed and unparsed, so formatting, comments and quote style do not
    fire, and docstrings are dropped for the same reason.
    """
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            if node.name != name:
                continue
            body = node.body
            if (
                body
                and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)
            ):
                node.body = body[1:] or [ast.Pass()]
            return ast.unparse(node)
    raise LookupError(name)


def constants_in(source: str) -> dict[str, str | list[str]]:
    """Every string-valued constant a module defines, class attributes
    included, keyed `NAME` or `Class.NAME`.

    Deliberately generic. Extracting only the ten we care about would put
    knowledge of what we watch into the download step, and watching an
    eleventh would then need a re-sync rather than just a new test.

    Values are evaluated, not read literally, so a constant assembled from
    earlier ones resolves: upstream's `AMD_ALWAYS_RUN_STEP_KEYS` is a
    `frozenset` holding one literal and one name.
    """
    env: dict[str, object] = {}
    out: dict[str, str | list[str]] = {}
    safe = {"frozenset": frozenset, "set": set, "tuple": tuple, "list": list}

    def visit(body, prefix=""):
        for node in body:
            if isinstance(node, ast.ClassDef):
                visit(node.body, f"{node.name}.")
                continue
            targets = (
                node.targets
                if isinstance(node, ast.Assign)
                else [node.target]
                if isinstance(node, ast.AnnAssign) and node.value
                else []
            )
            for target in targets:
                if not isinstance(target, ast.Name):
                    continue
                try:
                    value = eval(  # noqa: S307 - restricted env, upstream source
                        compile(ast.Expression(node.value), "<snapshot>", "eval"),
                        safe,
                        dict(env),
                    )
                except Exception:
                    continue
                env[target.id] = value
                if isinstance(value, str):
                    out[prefix + target.id] = value
                elif (
                    value
                    and isinstance(value, list | tuple | set | frozenset)
                    and all(isinstance(v, str) for v in value)
                ):
                    out[prefix + target.id] = sorted(value)

    visit(ast.parse(source).body)
    return out


def our_source(target: str) -> str:
    """The normalized source of one of our own functions, from `mod:name`."""
    import importlib

    module, _, name = target.partition(":")
    path = Path(importlib.import_module(module).__file__)
    return normalized(path.read_text(), name)


# ---------------------------------------------------------------- reading it


def read_manifest() -> dict:
    return json.loads(MANIFEST.read_text())


def read_values() -> dict[str, dict[str, str | list[str]]]:
    """`{upstream file: {constant name: value}}`, as the last sync found it."""
    return json.loads(VALUES.read_text())


def anchor_source(name: str) -> str:
    return (SNAPSHOT / f"{name}{SUFFIX}").read_text().strip()


def constant(name: str) -> str | list[str]:
    """One upstream constant by name, from anywhere in the package.

    Never scoped to a named file. `ONLY_STEP_KEYS_ENV_VAR` lives in
    `global_config.py`, not in any of the files you would guess, and a
    per-constant file mapping would have raised a false alarm on day one.
    """
    hits = {f: v[name] for f, v in read_values().items() if name in v}
    if not hits:
        raise LookupError(name)
    return next(iter(hits.values()))


def literals_in(anchor: str) -> set[str]:
    """Every string literal in an anchored function."""
    return {
        node.value
        for node in ast.walk(ast.parse(anchor_source(anchor)))
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and node.value
    }


def method_arg(anchor: str, method: str) -> str:
    """The literal argument of `<anything>.method("...")` in an anchored
    function. Raises when the call is gone, so the query cannot fail open."""
    for node in ast.walk(ast.parse(anchor_source(anchor))):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == method
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and isinstance(node.args[0].value, str)
        ):
            return node.args[0].value
    raise LookupError(f"{anchor}: no .{method}(<literal>) call")


def attr_assignment(anchor: str, attr: str) -> str:
    """The literal assigned to `<anything>.attr` in an anchored function."""
    for node in ast.walk(ast.parse(anchor_source(anchor))):
        if (
            isinstance(node, ast.Assign)
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str)
            and any(
                isinstance(t, ast.Attribute) and t.attr == attr for t in node.targets
            )
        ):
            return node.value.value
    raise LookupError(f"{anchor}: nothing assigned to .{attr}")


# ---------------------------------------------------------------- writing it


def generator_commit() -> str:
    """The last commit that touched the generator package.

    Not the tip of the default branch, which moves for reasons that have
    nothing to do with us: the tip was four days and one AMI change ahead of
    the generator when this was written.
    """
    url = COMMITS.format(repo=REPO, pkg=PKG)
    return json.load(urllib.request.urlopen(url, timeout=20))[0]["sha"]


def fetch_package(ref: str) -> dict[str, str]:
    """Every .py in ci-infra's generator package, at one commit.

    Pinned rather than taken from a branch, so the bytes and the sha we record
    beside them cannot describe different things, and so anyone can fetch that
    sha and get this snapshot back.
    """
    listing = json.load(
        urllib.request.urlopen(CONTENTS.format(repo=REPO, pkg=PKG, ref=ref), timeout=20)
    )
    names = sorted(e["name"] for e in listing if e["name"].endswith(".py"))
    return {
        name: urllib.request.urlopen(
            RAW.format(repo=REPO, ref=ref, pkg=PKG, name=name), timeout=20
        )
        .read()
        .decode()
        for name in names
    }


def sync() -> None:
    """Download, check it is real, write it down. No judgement of any kind.

    Raises on an unreachable network or a short read rather than leaving a
    half-written snapshot, because the offline tests cannot tell a truthful
    empty answer from a failed download.
    """
    commit = generator_commit()
    package = fetch_package(commit)
    body = "".join(package.values())
    if len(package) < MIN_FILES:
        raise RuntimeError(
            f"fetched {len(package)} files from {PKG}, expected at least "
            f"{MIN_FILES}. Refusing to overwrite the snapshot with a short read."
        )
    if len(body) < MIN_BYTES:
        raise RuntimeError(
            f"fetched {len(body)} bytes, expected at least {MIN_BYTES}. "
            "Refusing to overwrite the snapshot with a short read."
        )
    if CONTROL_LITERAL not in body:
        raise RuntimeError(
            f"{CONTROL_LITERAL!r} is missing, so this is not the pipeline "
            "generator. Refusing to overwrite the snapshot."
        )

    SNAPSHOT.mkdir(exist_ok=True)
    for name, (file, _target) in ANCHORS.items():
        (SNAPSHOT / f"{name}{SUFFIX}").write_text(
            normalized(package[file], name) + "\n"
        )
    VALUES.write_text(
        json.dumps(
            {f: constants_in(src) for f, src in sorted(package.items()) if src.strip()},
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    # Written whole, not merged: sync is the only writer, and a leftover key
    # from an older shape would sit there looking authoritative.
    MANIFEST.write_text(
        json.dumps(
            {"upstream_commit": commit, "files": sorted(package)},
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    print(f"synced {len(package)} files from {REPO}@{commit[:12]} into {SNAPSHOT}")


def upstream_callables() -> dict:
    """The self-contained anchors, executed out of the snapshot.

    Running somebody else's source is the point: it is the only way to compare
    behaviour rather than text. It is safe enough here because the bytes are
    whatever `sync()` wrote, and `sync()` refuses anything that is not
    ci-infra's generator.
    """
    # Upstream still annotates with the old typing aliases, so its source will
    # not compile without them in scope. Fetched by name rather than written
    # out, which also covers an alias it has not used yet.
    import typing

    namespace: dict = {
        alias: getattr(typing, alias)
        for alias in ("List", "Optional", "Dict", "Set", "FrozenSet", "Tuple")
    }
    for name in SELF_CONTAINED:
        exec(anchor_source(name), namespace)  # noqa: S102 - see docstring
    return namespace


if __name__ == "__main__":
    sync()
