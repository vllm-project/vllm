# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from argparse import Namespace
from http import HTTPStatus
from typing import TYPE_CHECKING

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.routing import APIRoute

from vllm.engine.protocol import EngineClient
from vllm.entrypoints.serve.exception_handling.error_response import (
    create_error_response,
)
from vllm.logger import init_logger
from vllm.tasks import SupportedTask

if TYPE_CHECKING:
    from starlette.datastructures import State

    from vllm.entrypoints.anthropic.protocol import (
        AnthropicDisabledThinkingEffortOption,
    )
    from vllm.entrypoints.serve.utils.request_logger import RequestLogger
else:
    RequestLogger = object

logger = init_logger(__name__)


def init_render_state(
    state: "State",
    request_logger: RequestLogger | None,
    *,
    disabled_thinking_effort: "AnthropicDisabledThinkingEffortOption" = "auto",
):
    from .derender.serving import ServingDerender
    from .render.serving import ServingRender

    state.serving_render = ServingRender(
        state.openai_serving_models,
        state.online_renderer,
        request_logger=request_logger,
        tool_server=state.tool_server,
        disabled_thinking_effort=disabled_thinking_effort,
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
    init_render_state(
        state,
        request_logger,
        disabled_thinking_effort=args.anthropic_disabled_thinking_effort,
    )

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
    from .derender.api_router import router as derender_render
    from .render.api_router import router as render_render

    args = getattr(app.state, "args", None)
    # `vllm launch render` and `vllm serve --tokens-only` are dedicated to
    # serving these endpoints, so they always register them regardless of
    # `--enable-scale-out`.
    dedicated_mode = "render" in supported_tasks or getattr(args, "tokens_only", False)
    enabled = dedicated_mode or getattr(args, "enable_scale_out", False)
    if not enabled:
        message = (
            "Scale-out endpoints are disabled. Restart vllm serve with "
            "--enable-scale-out to enable them."
        )
        logger.info(message)

        async def disabled_scale_out() -> JSONResponse:
            error = create_error_response(
                message, "PermissionDeniedError", HTTPStatus.FORBIDDEN
            )
            return JSONResponse(error.model_dump(), status_code=HTTPStatus.FORBIDDEN)

        from .token_in_token_out.api_router import router as token_router

        for route in (
            *render_render.routes,
            *derender_render.routes,
            *token_router.routes,
        ):
            if isinstance(route, APIRoute):
                app.add_api_route(
                    route.path,
                    disabled_scale_out,
                    methods=sorted(route.methods),
                    include_in_schema=False,
                )
        return

    app.include_router(render_render)
    app.include_router(derender_render)

    if "generate" in supported_tasks:
        from .token_in_token_out.api_router import (
            attach_router as attach_disagg_router,
        )

        attach_disagg_router(app)
