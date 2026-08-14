# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import importlib
import inspect
from argparse import Namespace

from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware

from vllm import envs
from vllm.logger import init_logger
from vllm.tasks import SupportedTask

from .log_response import log_response

logger = init_logger(__name__)


def init_entrypoints_middleware(
    args: Namespace,
    app: FastAPI,
    supported_tasks: tuple["SupportedTask", ...] | None = None,
):
    app.add_middleware(
        CORSMiddleware,
        allow_origins=args.allowed_origins,
        allow_credentials=args.allow_credentials,
        allow_methods=args.allowed_methods,
        allow_headers=args.allowed_headers,
    )

    # Ensure --api-key option from CLI takes precedence over VLLM_API_KEY
    if tokens := [key for key in (args.api_key or [envs.VLLM_API_KEY]) if key]:
        from .authenticate import AuthenticationMiddleware

        app.add_middleware(AuthenticationMiddleware, tokens=tokens)

    if args.enable_request_id_headers:
        from .x_request_id import XRequestIdMiddleware

        app.add_middleware(XRequestIdMiddleware)

    if "generate" in supported_tasks:
        # Add scaling middleware to check for scaling state
        from vllm.entrypoints.serve.elastic_ep.middleware import ScalingMiddleware

        app.add_middleware(ScalingMiddleware)

    if "realtime" in supported_tasks:
        # Add WebSocket metrics middleware
        from vllm.entrypoints.speech_to_text.realtime.metrics import (
            WebSocketMetricsMiddleware,
        )

        app.add_middleware(WebSocketMetricsMiddleware)

    if envs.VLLM_DEBUG_LOG_API_SERVER_RESPONSE:
        logger.warning(
            "CAUTION: Enabling log response in the API Server. "
            "This can include sensitive information and should be "
            "avoided in production."
        )
        app.middleware("http")(log_response)

    for middleware in args.middleware:
        module_path, object_name = middleware.rsplit(".", 1)
        imported = getattr(importlib.import_module(module_path), object_name)
        if inspect.isclass(imported):
            app.add_middleware(imported)  # type: ignore[arg-type]
        elif inspect.iscoroutinefunction(imported):
            app.middleware("http")(imported)
        else:
            raise ValueError(
                f"Invalid middleware {middleware}. Must be a function or a class."
            )
