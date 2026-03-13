# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for gpu_memory_utilization None-sentinel semantics."""

import pytest

from vllm import LLM
from vllm.engine.arg_utils import EngineArgs
from vllm.platforms import current_platform
from vllm.usage.usage_lib import UsageContext
from vllm.utils.argparse_utils import FlexibleArgumentParser


def test_cli_unset_defaults_to_0_9():
    parser = EngineArgs.add_cli_args(FlexibleArgumentParser())
    args = parser.parse_args([])
    assert args.gpu_memory_utilization is None

    vllm_config = EngineArgs.from_cli_args(args=args).create_engine_config()
    assert vllm_config.cache_config.gpu_memory_utilization == 0.9


@pytest.mark.parametrize("value", [0.9, 0.5])
def test_cli_explicit_value_respected(value: float):
    parser = EngineArgs.add_cli_args(FlexibleArgumentParser())
    args = parser.parse_args(["--gpu-memory-utilization", str(value)])
    assert args.gpu_memory_utilization == value

    vllm_config = EngineArgs.from_cli_args(args=args).create_engine_config()
    assert vllm_config.cache_config.gpu_memory_utilization == value


def test_engine_args_unset_defaults_to_0_9():
    vllm_config = EngineArgs(model="facebook/opt-125m").create_engine_config(
        UsageContext.LLM_CLASS
    )
    assert vllm_config.cache_config.gpu_memory_utilization == 0.9


def test_engine_args_explicit_value_respected():
    vllm_config = EngineArgs(
        model="facebook/opt-125m",
        gpu_memory_utilization=0.5,
    ).create_engine_config(UsageContext.LLM_CLASS)
    assert vllm_config.cache_config.gpu_memory_utilization == 0.5


def test_platform_hook_sees_none_and_can_override(monkeypatch):
    seen_none = False

    def fake_check_and_update_config(vllm_config):
        nonlocal seen_none
        seen_none = vllm_config.cache_config.gpu_memory_utilization is None
        vllm_config.cache_config.gpu_memory_utilization = 0.42

    monkeypatch.setattr(
        current_platform,
        "check_and_update_config",
        fake_check_and_update_config,
    )

    vllm_config = EngineArgs(model="facebook/opt-125m").create_engine_config(
        UsageContext.LLM_CLASS
    )

    assert seen_none
    # Platform set 0.42, so fallback 0.9 should NOT overwrite
    assert vllm_config.cache_config.gpu_memory_utilization == 0.42


def test_llm_omitted_gpu_memory_utilization_passes_none(monkeypatch):
    captured = {}

    class DummyEngine:
        def get_supported_tasks(self):
            return ["generate"]

    def fake_from_engine_args(*, engine_args, usage_context):
        captured["gpu_memory_utilization"] = engine_args.gpu_memory_utilization
        captured["usage_context"] = usage_context
        return DummyEngine()

    monkeypatch.setattr(
        "vllm.entrypoints.llm.LLMEngine.from_engine_args", fake_from_engine_args
    )

    LLM(model="facebook/opt-125m", enforce_eager=True)

    assert captured["gpu_memory_utilization"] is None
    assert captured["usage_context"] == UsageContext.LLM_CLASS


def test_llm_explicit_gpu_memory_utilization_is_preserved(monkeypatch):
    captured = {}

    class DummyEngine:
        def get_supported_tasks(self):
            return ["generate"]

    def fake_from_engine_args(*, engine_args, usage_context):
        captured["gpu_memory_utilization"] = engine_args.gpu_memory_utilization
        captured["usage_context"] = usage_context
        return DummyEngine()

    monkeypatch.setattr(
        "vllm.entrypoints.llm.LLMEngine.from_engine_args", fake_from_engine_args
    )

    LLM(model="facebook/opt-125m", gpu_memory_utilization=0.5, enforce_eager=True)

    assert captured["gpu_memory_utilization"] == 0.5
    assert captured["usage_context"] == UsageContext.LLM_CLASS
