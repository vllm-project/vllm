# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The ROCm image keys in .buildkite/scripts/rocm/build.sh must cover every
build-context file the keyed Dockerfile parts read, or editing that file would
ship a stale image."""

import os
import subprocess
from pathlib import Path

import pytest
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


def run_ci_build(tmp_path: Path, **env: str) -> tuple[int, str, list[str]]:
    """build.sh ci against fake git, docker and buildkite-agent; returns (status,
    stderr, calls). FAILURE=bake|missing|upload breaks the kernel symbol map."""
    (tmp_path / "docker").symlink_to(ROOT / "docker")
    fake = tmp_path / "bin"
    fake.mkdir()
    calls = tmp_path / "calls"
    (fake / "git").write_text(
        """#!/bin/bash
case $1 in
  describe) echo v0.0.0-0-g0 ;;
  rev-parse|log) echo 0 ;;
esac
"""
    )
    (fake / "docker").write_text(
        """#!/bin/bash
[[ $1 == buildx && $2 == bake ]] || exit 0
target=${@: -1}
echo "bake $target" >> calls
case $target in
  ci) mkdir -p .rocm-wheel && touch .rocm-wheel/vllm-0.whl ;;
  kernel-symbol-map)
    [[ $FAILURE == bake ]] && exit 1
    mkdir -p .rocm-kernel-map && cd .rocm-kernel-map
    [[ $FAILURE == missing ]] || echo map > kernel_symbol_map.rocm.json.gz ;;
esac
"""
    )
    (fake / "buildkite-agent").write_text(
        """#!/bin/bash
[[ $1 == meta-data ]] && { echo "ref@sha256:0"; exit 0; }
echo "upload ${PWD##*/}/$3" >> "$CALLS"
[[ $FAILURE == upload && $3 == kernel_symbol_map.rocm.json.gz ]] && exit 1
exit 0
"""
    )
    for tool in fake.iterdir():
        tool.chmod(0o755)
    # A map left by an earlier build in this checkout must never be uploaded.
    (tmp_path / ".rocm-kernel-map").mkdir()
    (tmp_path / ".rocm-kernel-map/kernel_symbol_map.rocm.json.gz").write_text("stale")
    run_env = {
        "PATH": f"{fake}:{os.environ['PATH']}",
        "ROCM_CI_REPO": "rocm/vllm-ci",
        "BUILDKITE_BUILD_ID": "b1",
        "CALLS": str(calls),
        "FAILURE": "",
        **env,
    }
    result = subprocess.run(
        ["bash", str(ROOT / ".buildkite/scripts/rocm/build.sh"), "rocm", "ci"],
        cwd=tmp_path,
        env=run_env,
        capture_output=True,
        text=True,
    )
    return result.returncode, result.stderr, calls.read_text().splitlines()


def test_ci_build_uploads_the_kernel_symbol_map_when_asked(tmp_path):
    status, stderr, calls = run_ci_build(tmp_path, VLLM_KERNEL_SYMBOL_MAP="1")
    assert status == 0, stderr
    assert calls == [
        "bake ci",
        "upload tmp_path/.rocm-wheel/vllm-*.whl".replace("tmp_path", tmp_path.name),
        "bake kernel-symbol-map",
        "upload .rocm-kernel-map/kernel_symbol_map.rocm.json.gz",
    ]


def test_ci_build_skips_the_kernel_symbol_map_by_default(tmp_path):
    status, stderr, calls = run_ci_build(tmp_path)
    assert status == 0, stderr
    assert "bake kernel-symbol-map" not in calls


@pytest.mark.parametrize("failure", ["bake", "missing", "upload"])
def test_a_kernel_symbol_map_failure_never_fails_the_image_build(tmp_path, failure):
    status, stderr, calls = run_ci_build(
        tmp_path, VLLM_KERNEL_SYMBOL_MAP="1", FAILURE=failure
    )
    assert status == 0, stderr
    assert "continuing without it" in stderr
    uploaded = "upload .rocm-kernel-map/kernel_symbol_map.rocm.json.gz" in calls
    assert uploaded == (failure == "upload")  # the stale map is never uploaded
