# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A PR may skip the image bake only if its files alone make up the difference."""

import importlib.util
import subprocess
from pathlib import Path

import pytest

SCRIPT = (
    Path(__file__).resolve().parents[2]
    / ".buildkite/image_build/python_only_overlay.py"
)
spec = importlib.util.spec_from_file_location("python_only_overlay", SCRIPT)
assert spec is not None and spec.loader is not None
overlay = importlib.util.module_from_spec(spec)
spec.loader.exec_module(overlay)


@pytest.mark.parametrize(
    ("status", "path"),
    [
        ("M", "vllm/v1/worker/gpu_worker.py"),
        ("A", "vllm/v1/worker/new_module.py"),
        ("M", "vllm/model_executor/layers/fused_moe/configs/E=8.json"),
        ("A", "tests/basic_correctness/test_new.py"),
        ("M", ".buildkite/test_areas/basic_correctness.yaml"),
        ("M", "docs/usage/README.md"),
        ("M", "README.md"),
    ],
)
def test_thin_build(status, path):
    assert overlay.blocking_reason(status, path) is None


@pytest.mark.parametrize(
    ("status", "path"),
    [
        ("M", "csrc/attention/attention_kernels.cu"),
        ("M", "CMakeLists.txt"),
        ("M", "setup.py"),
        ("M", "requirements/common.txt"),
        ("M", "docker/Dockerfile"),
        ("M", "rust/src/main.rs"),
        ("M", "tools/build_rust.py"),
        ("M", "vllm/envs.py"),
        ("M", "vllm/utils/numa_wrapper.sh"),
        ("A", "vllm/model_executor/layers/fused_moe/configs/E=16.json"),
        ("M", ".buildkite/image_build/image_build.sh"),
        ("M", "tests/vllm_test_utils/setup.py"),
        ("D", "vllm/v1/worker/old_module.py"),
    ],
)
def test_full_build(status, path):
    assert overlay.blocking_reason(status, path) is not None


def _git(repo: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True)


def _commit(repo: Path, files: dict[str, str]) -> None:
    for rel, text in files.items():
        (repo / rel).parent.mkdir(parents=True, exist_ok=True)
        (repo / rel).write_text(text)
    _git(repo, "add", "-A")
    _git(repo, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "c")


@pytest.mark.parametrize(
    ("change", "thin"),
    [({"vllm/a.py": "new", "tests/test_a.py": "t"}, True), ({"csrc/k.cu": "x"}, False)],
)
def test_stages_files_at_image_paths(tmp_path, monkeypatch, change, thin):
    repo, context = tmp_path / "repo", tmp_path / "context"
    repo.mkdir()
    _git(repo, "init", "-q")
    _commit(repo, {"vllm/a.py": "old", "csrc/k.cu": "k"})
    _commit(repo, change)
    monkeypatch.chdir(repo)

    assert overlay.main("HEAD~1", context) == (0 if thin else 1)

    staged = sorted(
        str(p.relative_to(context)) for p in context.rglob("*") if p.is_file()
    )
    if thin:
        assert staged == [
            "vllm-workspace/src/vllm/a.py",
            "vllm-workspace/tests/test_a.py",
        ]
        assert (context / "vllm-workspace/src/vllm/a.py").read_text() == "new"
    else:
        assert staged == []
