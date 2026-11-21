# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The offline `LLM` keeps one renderer worker unless asked otherwise.

`ModelConfig` resolves an unset `renderer_num_workers` to 4 for the async
renderer, which the synchronous `LLM` never uses.
"""

from types import SimpleNamespace

import pytest

from vllm.engine.arg_utils import EngineArgs
from vllm.entrypoints import llm as llm_module
from vllm.entrypoints.llm import LLM


class _Stop(Exception):
    pass


@pytest.fixture
def captured_engine_args(monkeypatch):
    captured: dict = {}

    def fake_engine_args(**kwargs):
        captured.update(kwargs)
        raise _Stop

    monkeypatch.setattr(llm_module, "EngineArgs", fake_engine_args)
    return captured


def test_default_is_one(captured_engine_args):
    with pytest.raises(_Stop):
        LLM(model="dummy")
    assert captured_engine_args["renderer_num_workers"] == 1


def test_explicit_none_is_one(captured_engine_args):
    with pytest.raises(_Stop):
        LLM(model="dummy", renderer_num_workers=None)
    assert captured_engine_args["renderer_num_workers"] == 1


def test_explicit_value_is_kept(captured_engine_args):
    with pytest.raises(_Stop):
        LLM(model="dummy", renderer_num_workers=8)
    assert captured_engine_args["renderer_num_workers"] == 8


def test_from_engine_args_default_is_one(monkeypatch):
    # `from_engine_args` forwards every EngineArgs field, including an unset
    # `renderer_num_workers=None`, as an explicit keyword.
    assert EngineArgs.renderer_num_workers is None
    engine_args = SimpleNamespace(model="dummy", renderer_num_workers=None)
    captured: dict = {}

    def fake_engine_args(**kwargs):
        captured.update(kwargs)
        raise _Stop

    monkeypatch.setattr(llm_module, "EngineArgs", fake_engine_args)
    with pytest.raises(_Stop):
        LLM.from_engine_args(engine_args)  # type: ignore[arg-type]
    assert captured["renderer_num_workers"] == 1
