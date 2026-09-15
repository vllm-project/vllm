# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Derived facts about CI infra we do not model.

Both sets are recomputed every build from repo text, so a moved script or a new
COPY line is picked up on its own. Neither is a routing map: each gates one
narrow rule. A script only the release pipeline runs selects nothing, and a
docker build input keeps its run-all but names itself as the reason.
"""

from __future__ import annotations

from pathlib import Path

import regex as re

from ..handwritten import RELEASE_PIPELINE_FILES

DOCKER_DIR = "docker"
DOCKERFILE_GLOB = "Dockerfile*"

# Repo paths inside a yaml or script, anchored on a top-level dir so a bare word
# or an image tag is not read as a file. docker/ and vllm/ are left out on
# purpose: neither should ever be silenced as release-only.
_REPO_PATH_RE = re.compile(
    r"(?<![\w/])((?:\.buildkite|tools|csrc|cmake|examples|benchmarks|"
    r"tests|requirements)/[\w./=,-]+)"
)


def release_pipeline_refs(repo: Path) -> frozenset[str]:
    """Repo files the release pipeline references: the scripts it runs and the
    artifacts it reads, restricted to paths that exist. Follows one script into
    the next, so a lib a release script sources is release-only too. A file a
    live step also uses is still claimed by that step first."""
    refs: set[str] = set()
    seen: set[str] = set()
    frontier = list(RELEASE_PIPELINE_FILES)
    while frontier:
        rel = frontier.pop()
        if rel in seen:
            continue
        seen.add(rel)
        try:
            text = (repo / rel).read_text()
        except OSError:
            continue
        for match in _REPO_PATH_RE.findall(text):
            if (repo / match).is_file() and match not in refs:
                refs.add(match)
                frontier.append(match)
    return frozenset(refs)


def _copy_sources(text: str) -> list[str]:
    """Sources of every COPY/ADD, minus `--from=` stage copies, whose sources
    are image paths and not repo files."""
    joined = text.replace("\\\n", " ")
    out: list[str] = []
    for raw in joined.splitlines():
        line = raw.strip()
        upper = line.upper()
        if not (upper.startswith("COPY ") or upper.startswith("ADD ")):
            continue
        if "--from=" in line:
            continue
        tokens = [t for t in line.split()[1:] if not t.startswith("--")]
        if len(tokens) < 2:
            continue
        for src in tokens[:-1]:  # last token is the destination
            out.append(src[2:] if src.startswith("./") else src)
    return out


def docker_image_inputs(repo: Path) -> dict[str, str]:
    """Repo file -> the Dockerfile that copies it in. File sources only: a
    directory blanket would relabel every fail-open and say nothing."""
    out: dict[str, str] = {}
    for dockerfile in sorted((repo / DOCKER_DIR).glob(DOCKERFILE_GLOB)):
        rel_df = dockerfile.relative_to(repo).as_posix()
        try:
            text = dockerfile.read_text()
        except OSError:
            continue
        for src in _copy_sources(text):
            if (repo / src).is_file():
                out.setdefault(src, rel_df)
    return out


def copy_inputs(
    repo: Path, dockerfiles
) -> tuple[dict[str, set[str]], dict[str, set[str]], set[str]]:
    """What each Dockerfile copies IN: (file sources, dir sources, blankets).

    Separate from `docker_image_inputs` on purpose. That one answers "which
    Dockerfile do I name in a fail-open message" and keeps one winner per file;
    this one answers "which images does a change here rebuild", where a file
    three Dockerfiles copy must reach all three.

    Directory sources are returned rather than dropped, since the build-layer
    callers arrive that way. Which of them are safe to use needs the import
    graph, which this module does not have, so that call is the caller's.

    A whole-context `COPY . .` is reported as the third element rather than
    expanded. Expanding it would make every tree the graph cannot route a build
    input and send a docs edit to the full image closure.
    """
    files: dict[str, set[str]] = {}
    dirs: dict[str, set[str]] = {}
    blanket: set[str] = set()
    for rel in dockerfiles:
        try:
            text = (repo / rel).read_text()
        except OSError:
            continue
        for src in _copy_sources(text):
            src = src.rstrip("/")
            if src in ("", ".", "./"):
                blanket.add(rel)
                continue
            if not (repo / src).exists():
                continue
            target = files if (repo / src).is_file() else dirs
            target.setdefault(src, set()).add(rel)
    return files, dirs, blanket
