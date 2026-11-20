# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for C7 packaging: offline lockfile freshness, Dockerfile, Buildkite."""

from __future__ import annotations

import shutil
from pathlib import Path

import yaml

from tools.byog.compile_requirements import (
    BYOG_DIR,
    LOCK_FILE,
    REPO_ROOT,
    REQ_DIR,
    check_lockfile,
)


def test_requirements_lock_exists_and_is_fresh_offline() -> None:
    ok, msg = check_lockfile(REQ_DIR, LOCK_FILE)
    assert ok, msg
    lock_text = LOCK_FILE.read_text(encoding="utf-8")
    assert "pydantic==" in lock_text
    assert "pyyaml==" in lock_text
    # Dev-only test runner must not bloat the runtime lockfile.
    assert "pytest==" not in lock_text


def test_requirements_lock_detects_stale_inputs(tmp_path: Path) -> None:
    req_copy = tmp_path / "requirements.d"
    shutil.copytree(REQ_DIR, req_copy)
    lock_copy = tmp_path / "requirements.lock"
    shutil.copy2(LOCK_FILE, lock_copy)

    ok_before, _ = check_lockfile(req_copy, lock_copy)
    assert ok_before

    # Modifying a requirement file without recompiling must fail deterministically.
    (req_copy / "new_plugin.txt").write_text("httpx>=0.27\n", encoding="utf-8")
    ok_after, msg_after = check_lockfile(req_copy, lock_copy)
    assert not ok_after
    assert "stale" in msg_after.lower()


def test_dockerfile_byog_stages_and_pinning() -> None:
    df = (REPO_ROOT / "docker" / "Dockerfile.byog").read_text(encoding="utf-8")
    assert "AS byog-tools" in df
    assert "AS byog" in df
    assert "AS prebuilt-tools" in df
    assert "AS byog-prebuilt" in df
    assert "COPY --from=prebuilt-tools /opt/byog /opt/byog" in df
    assert "uv==${UV_VERSION}" in df
    assert "vllm/vllm-openai:latest" not in df
    assert "/opt/byog/build-manifest.json" in df


def test_buildkite_yaml_configs_are_valid() -> None:
    test_yaml = yaml.safe_load(
        (REPO_ROOT / ".buildkite" / "test_areas" / "byog.yaml").read_text(
            encoding="utf-8"
        )
    )
    keys = {s["key"] for s in test_yaml["steps"]}
    assert {"byog-cpu-tests", "byog-nvidia-gpu-smoke"} <= keys
    gpu_step = next(
        s for s in test_yaml["steps"] if s["key"] == "byog-nvidia-gpu-smoke"
    )
    assert "artifact_paths" in gpu_step

    img_yaml = yaml.safe_load(
        (REPO_ROOT / ".buildkite" / "image_build" / "byog_image.yaml").read_text(
            encoding="utf-8"
        )
    )
    assert any(s["key"] == "byog-image-smoke" for s in img_yaml["steps"])
    assert BYOG_DIR.is_dir()
