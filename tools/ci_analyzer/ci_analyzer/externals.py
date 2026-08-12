# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Derived facts about non-modeled CI infra.

Both sets are recomputed every build from repo text, so a moved script or a new
COPY line is picked up without hand-maintenance. Neither is a routing map: each
gates one narrow selection rule; a script referenced only by the release
pipeline selects nothing, and a docker-image build input keeps its run-all but
labels itself as the reason.
"""

from __future__ import annotations

from pathlib import Path

import regex as re

from .curated import RELEASE_PIPELINE_FILES

# Repo-relative paths embedded in a yaml/script, anchored on a top-level dir so
# a bare word or an image tag is not mistaken for a file. docker/ and vllm/ are
# deliberately absent: a Dockerfile the release pipeline builds from is world
# (it self-classifies as run-all), and release infra does not consume vllm
# source; neither should be silenced as "release-only".
_REPO_PATH_RE = re.compile(
    r"(?<![\w/])((?:\.buildkite|tools|csrc|cmake|examples|benchmarks|"
    r"tests|requirements)/[\w./=,-]+)"
)


def release_pipeline_refs(repo: Path) -> frozenset[str]:
    """Repo files referenced by the release/nightly pipeline yaml(s): the
    scripts it runs and artifacts it reads, restricted to paths that exist.
    Follows one script into the next (a release script that sources
    manylinux.sh makes that lib release-only too); a file also consumed by a
    live step is still claimed by that step first (rule order in select.py)."""
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
    """Repo-relative sources of every COPY/ADD instruction, minus --from= stage
    copies (their sources are image paths, not repo files)."""
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
    """Repo file -> the Dockerfile that COPY/ADDs it. FILE sources only:
    directory blankets (`COPY . /workspace`) are excluded by the is_file check,
    else the relabel would fire on every terminal fail-open and say nothing."""
    out: dict[str, str] = {}
    for dockerfile in sorted((repo / "docker").glob("Dockerfile*")):
        rel_df = dockerfile.relative_to(repo).as_posix()
        try:
            text = dockerfile.read_text()
        except OSError:
            continue
        for src in _copy_sources(text):
            if (repo / src).is_file():
                out.setdefault(src, rel_df)
    return out
