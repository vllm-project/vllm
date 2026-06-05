# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

from vllm.config.vllm import _should_auto_enable_deepseek_v4_breakable_cudagraph
from vllm.platforms import current_platform


def _model_config(*architectures: str):
    return SimpleNamespace(architectures=list(architectures))


def test_deepseek_v4_auto_enables_breakable_cudagraph_off_sm121(monkeypatch):
    monkeypatch.setattr(
        current_platform,
        "is_device_capability",
        lambda capability, device_id=0: False,
    )

    assert _should_auto_enable_deepseek_v4_breakable_cudagraph(
        _model_config("DeepseekV4ForCausalLM")
    )
    assert _should_auto_enable_deepseek_v4_breakable_cudagraph(
        _model_config("DeepSeekV4MTPModel")
    )


def test_deepseek_v4_skips_breakable_cudagraph_on_sm121(monkeypatch):
    monkeypatch.setattr(
        current_platform,
        "is_device_capability",
        lambda capability, device_id=0: capability == 121,
    )

    assert not _should_auto_enable_deepseek_v4_breakable_cudagraph(
        _model_config("DeepseekV4ForCausalLM")
    )


def test_non_deepseek_v4_does_not_auto_enable_breakable_cudagraph(monkeypatch):
    monkeypatch.setattr(
        current_platform,
        "is_device_capability",
        lambda capability, device_id=0: False,
    )

    assert not _should_auto_enable_deepseek_v4_breakable_cudagraph(
        _model_config("Qwen3ForCausalLM")
    )
