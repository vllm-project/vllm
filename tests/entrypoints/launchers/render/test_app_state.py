# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from argparse import Namespace

import pytest
from starlette.datastructures import State

import vllm.entrypoints.launchers.render.app_state as app_state_mod
from vllm.config import ModelConfig, VllmConfig


class _CaptureKwargs:
    """Stands in for OnlineRenderer/OnlineDerenderer; records init kwargs."""

    captured: list[dict]

    def __init__(self, **kwargs):
        type(self).captured.append(kwargs)

    def warmup(self):
        pass


@pytest.mark.asyncio
async def test_render_app_state_uses_config_resolved_reasoning_parser(monkeypatch):
    """gpt_oss defaults its reasoning parser through verify_and_update_config;
    the render server must pick up that resolved value like the main API
    server does, otherwise harmony markup leaks unparsed into derender output.
    """
    model_config = ModelConfig("openai/gpt-oss-20b")
    vllm_config = VllmConfig(model_config=model_config)
    assert vllm_config.structured_outputs_config.reasoning_parser == "openai_gptoss"

    class _Renderer(_CaptureKwargs):
        captured = []

    class _Derenderer(_CaptureKwargs):
        captured = []

    async def _noop_async(*args, **kwargs):
        return None

    monkeypatch.setattr(app_state_mod, "renderer_from_config", lambda cfg: object())
    monkeypatch.setattr(app_state_mod, "OnlineRenderer", _Renderer)
    monkeypatch.setattr(app_state_mod, "OnlineDerenderer", _Derenderer)
    monkeypatch.setattr(app_state_mod, "ServingTokenization", lambda *a, **kw: object())
    monkeypatch.setattr(app_state_mod, "init_render_state", lambda *a, **kw: None)
    monkeypatch.setattr(app_state_mod, "init_endpoint_plugins_state", _noop_async)

    args = Namespace(
        model="openai/gpt-oss-20b",
        served_model_name=None,
        enable_log_requests=False,
        chat_template=None,
        chat_template_content_format="auto",
        trust_request_chat_template=False,
        enable_auto_tool_choice=False,
        exclude_tools_when_tool_choice_none=False,
        tool_call_parser=None,
        reasoning_parser="",
        default_chat_template_kwargs=None,
        log_error_stack=False,
    )

    await app_state_mod.init_render_app_state(vllm_config, State(), args)

    assert _Renderer.captured[0]["reasoning_parser"] == "openai_gptoss"
    assert _Derenderer.captured[0]["reasoning_parser"] == "openai_gptoss"
