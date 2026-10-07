# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from argparse import Namespace

import pytest
from starlette.datastructures import State

import vllm.entrypoints.launchers.render.app_state as app_state_mod
from vllm.config import ModelConfig, VllmConfig
from vllm.entrypoints.launchers.cli_args import make_arg_parser
from vllm.utils.argparse_utils import FlexibleArgumentParser


class _CaptureKwargs:
    """Stands in for OnlineRenderer/OnlineDerenderer; records init kwargs."""

    captured: list[dict]

    def __init__(self, **kwargs):
        type(self).captured.append(kwargs)

    def warmup(self):
        pass


def _render_cli_args(*argv: str) -> Namespace:
    """Parse ``argv`` the way ``vllm launch render`` does, so new serve flags
    carry their defaults instead of needing a hand-maintained ``Namespace``."""
    args = make_arg_parser(FlexibleArgumentParser()).parse_args(list(argv))
    if args.model_tag is not None:
        args.model = args.model_tag
    return args


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

    args = _render_cli_args("openai/gpt-oss-20b")

    await app_state_mod.init_render_app_state(vllm_config, State(), args)

    assert _Renderer.captured[0]["reasoning_parser"] == "openai_gptoss"
    assert _Derenderer.captured[0]["reasoning_parser"] == "openai_gptoss"


def test_explicit_reasoning_parser_flag_wins_over_model_default():
    """An explicit --reasoning-parser must survive VllmConfig construction
    for models that define their own default (gpt_oss); the default only
    applies when the resolved value is empty."""
    from vllm.engine.arg_utils import AsyncEngineArgs

    engine_args = AsyncEngineArgs(
        model="openai/gpt-oss-20b", reasoning_parser="deepseek_r1"
    )
    model_config = engine_args.create_model_config()
    vllm_config = VllmConfig(
        model_config=model_config,
        structured_outputs_config=engine_args.create_structured_outputs_config(),
    )

    assert vllm_config.structured_outputs_config.reasoning_parser == "deepseek_r1"
