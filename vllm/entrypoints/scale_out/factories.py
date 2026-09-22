# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from argparse import Namespace
from typing import TYPE_CHECKING

from fastapi import FastAPI

from vllm.engine.protocol import EngineClient
from vllm.tasks import SupportedTask

if TYPE_CHECKING:
    from starlette.datastructures import State

    from vllm.entrypoints.serve.utils.request_logger import RequestLogger
else:
    RequestLogger = object


def init_render_state(
    state: "State",
    request_logger: RequestLogger | None,
):
    from .derender.serving import ServingDerender
    from .render.serving import ServingRender

    state.serving_render = ServingRender(
        state.openai_serving_models,
        state.online_renderer,
        request_logger=request_logger,
    )

    state.serving_derender = ServingDerender(
        state.openai_serving_models,
        state.online_derenderer,
        request_logger=request_logger,
    )


def init_scale_out_state(
    state: "State",
    args: "Namespace",
    engine_client: "EngineClient",
    request_logger: RequestLogger | None,
):
    init_render_state(state, request_logger)

    from vllm.entrypoints.scale_out.token_in_token_out.serving import ServingTokens

    state.serving_tokens = ServingTokens(
        engine_client,
        state.openai_serving_models,
        state.online_renderer,
        request_logger=request_logger,
        return_tokens_as_token_ids=args.return_tokens_as_token_ids,
        enable_prompt_tokens_details=args.enable_prompt_tokens_details,
        enable_log_outputs=args.enable_log_outputs,
        force_no_detokenize=args.tokens_only,
    )


def register_scale_out_api_routers(
    app: FastAPI,
    supported_tasks: tuple["SupportedTask", ...],
):
    from .render.api_router import router as render_render

    app.include_router(render_render)

    from .derender.api_router import router as derender_render

    app.include_router(derender_render)

    if "generate" in supported_tasks:
        from .token_in_token_out.api_router import (
            attach_router as attach_disagg_router,
        )

        attach_disagg_router(app)
