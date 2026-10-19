# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The ROCm image keys in .buildkite/scripts/rocm/build.sh must cover every
build-context file the keyed Dockerfile parts read, or editing that file would
ship a stale image."""

from pathlib import Path

import regex as re

ROOT = Path(__file__).parents[2]
MARKER = "# ==== per-commit stages: nothing below is part of the runtime key ===="


def context_sources(dockerfile: str) -> set[str]:
    text = dockerfile.replace("\\\n", " ")
    srcs = set()
    for args in re.findall(r"^\s*(?:COPY|ADD)\s+(.*)$", text, re.M):
        words = [w for w in args.split() if not w.startswith("--")]
        if "--from=" not in args and not words[0].startswith("http"):
            srcs.update(words[:-1])
    for opts in re.findall(r"--mount=type=bind,(\S*)", text):
        kv = dict(o.partition("=")[::2] for o in opts.split(","))
        if "from" not in kv:
            srcs.add(kv.get("source", kv.get("src", ".")))
    return srcs


def key_paths(name: str, stack: str = "") -> set[str]:
    script = (ROOT / ".buildkite/scripts/rocm/build.sh").read_text()
    block_match = re.search(rf"^{name}=\$\(.*?sha256sum", script, re.M | re.S)
    assert block_match is not None, f"{name}= block not found in build.sh"
    pattern = r"\b(?:requirements|tools|tests)/[\w./-]+|\brust-toolchain\.toml\b"
    paths = set(re.findall(pattern, block_match.group(0)))
    # therock_wheels.py is injected into Ks via a helper variable (rocm stack only)
    if name == "Ks" and stack == "rocm":
        m = re.search(r"_[Tt][Rr][Ww]=.*?tools/[\w./-]+", script, re.S)
        if m:
            paths.update(re.findall(r"\b(?:requirements|tools)/[\w./-]+", m.group(0)))
    return paths


def test_runtime_key_covers_runtime_inputs():
    dockerfile = (ROOT / "docker/Dockerfile.rocm").read_text()
    assert dockerfile.count(MARKER) == 1
    assert context_sources(dockerfile.split(MARKER)[0]) <= key_paths("Kr")


def test_stack_key_covers_stack_inputs():
    dockerfile = (ROOT / "docker/Dockerfile.rocm_base").read_text()
    assert context_sources(dockerfile) <= key_paths("Ks", stack="rocm")
