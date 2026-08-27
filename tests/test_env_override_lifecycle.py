# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import importlib
import subprocess
import sys

import pytest


def test_env_override_import_does_not_import_torch():
    script = """
import importlib.abc
import importlib
import sys

class RejectTorch(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "torch" or fullname.startswith("torch."):
            raise AssertionError(f"unexpected import: {fullname}")
        return None

sys.meta_path.insert(0, RejectTorch())
importlib.import_module("vllm.env_override")
"""
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True
    )

    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
    ("torch_version", "expected_targets"),
    [
        pytest.param(
            "2.9.0",
            {
                "torch",
                "torch._inductor.config",
                "torch._inductor.codegen.wrapper",
                "torch._inductor.graph",
                "torch._inductor.lowering",
            },
            id="2.9",
        ),
        pytest.param(
            "2.10.0",
            {
                "torch._dynamo.convert_frame",
                "torch._inductor.codecache",
                "torch._inductor.lowering",
            },
            id="2.10",
        ),
        pytest.param(
            "2.11.0",
            {
                "torch._dynamo.convert_frame",
                "torch._inductor.codegen.cpp",
                "torch._inductor.lowering",
            },
            id="2.11",
        ),
        pytest.param("2.13.0", {"torch._inductor.lowering"}, id="current"),
    ],
)
def test_post_import_targets_match_torch_version(
    torch_version: str, expected_targets: set[str]
):
    script = f"""
import importlib.abc
import importlib.metadata

class RejectTorch(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "torch" or fullname.startswith("torch."):
            raise AssertionError(f"unexpected import: {{fullname}}")
        return None

importlib.metadata.version = lambda package: {torch_version!r}
import sys
sys.meta_path.insert(0, RejectTorch())

import vllm.env_override as env_override

assert set(env_override._POST_IMPORT_PATCHES) == set({sorted(expected_targets)!r})
"""
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True
    )

    assert result.returncode == 0, result.stderr


def test_post_import_callback_retries_after_target_import_failure(
    tmp_path, monkeypatch
):
    from vllm import env_override

    target_name = "vllm_env_override_retry_target"
    state_name = "vllm_env_override_retry_state"
    (tmp_path / f"{state_name}.py").write_text("attempts = 0\n")
    (tmp_path / f"{target_name}.py").write_text(
        f"import {state_name}\n"
        f"{state_name}.attempts += 1\n"
        f"if {state_name}.attempts == 1:\n"
        '    raise RuntimeError("target import failure")\n'
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    calls = []

    env_override._register_post_import_patch(target_name, lambda: calls.append("patch"))

    with pytest.raises(RuntimeError, match="target import failure"):
        importlib.import_module(target_name)

    target = importlib.import_module(target_name)

    assert calls == ["patch"]
    assert target.__loader__ is not None
    assert target.__spec__ is not None
    assert target.__spec__.loader is target.__loader__
    assert target_name not in env_override._POST_IMPORT_PATCHES


def test_post_import_callback_retries_after_callback_failure(tmp_path, monkeypatch):
    from vllm import env_override

    target_name = "vllm_env_override_callback_target"
    (tmp_path / f"{target_name}.py").write_text("value = 1\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    calls = []

    def patch():
        calls.append("patch")
        if len(calls) == 1:
            raise RuntimeError("callback failure")

    env_override._register_post_import_patch(target_name, patch)

    with pytest.raises(RuntimeError, match="callback failure"):
        importlib.import_module(target_name)

    target = importlib.import_module(target_name)

    assert target.value == 1
    assert calls == ["patch", "patch"]
    assert target.__spec__ is not None
    assert target.__spec__.loader is target.__loader__
    assert target_name not in env_override._POST_IMPORT_PATCHES


def test_post_import_callbacks_repeat_after_later_callback_failure(
    tmp_path, monkeypatch
):
    from vllm import env_override

    target_name = "vllm_env_override_multiple_callbacks_target"
    (tmp_path / f"{target_name}.py").write_text("value = 1\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    calls = []

    def first_patch():
        calls.append("first")

    def second_patch():
        calls.append("second")
        if calls == ["first", "second"]:
            raise RuntimeError("later callback failure")

    env_override._register_post_import_patch(target_name, first_patch)
    env_override._register_post_import_patch(target_name, second_patch)

    with pytest.raises(RuntimeError, match="later callback failure"):
        importlib.import_module(target_name)

    importlib.import_module(target_name)

    assert calls == ["first", "second", "first", "second"]
    assert target_name not in env_override._POST_IMPORT_PATCHES
