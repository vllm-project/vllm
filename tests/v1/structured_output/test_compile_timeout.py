# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import time

import pytest

import vllm.envs as envs
from vllm.v1.structured_output.utils import compile_with_timeout


def test_compile_with_timeout_raises(monkeypatch):
    monkeypatch.setattr(envs, "VLLM_REGEX_COMPILATION_TIMEOUT_S", 1)
    with pytest.raises(ValueError, match="timed out after 1s"):
        compile_with_timeout(lambda: time.sleep(5), what="Test")


def test_compile_with_timeout_passthrough(monkeypatch):
    monkeypatch.setattr(envs, "VLLM_REGEX_COMPILATION_TIMEOUT_S", 30)
    assert compile_with_timeout(lambda x, y=0: x + y, 1, y=2) == 3


def test_compile_with_timeout_disabled(monkeypatch):
    monkeypatch.setattr(envs, "VLLM_REGEX_COMPILATION_TIMEOUT_S", 0)
    assert compile_with_timeout(lambda: "ok") == "ok"