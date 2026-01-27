# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import TYPE_CHECKING, Any

from fastapi import FastAPI

if TYPE_CHECKING:
    from argparse import Namespace

    from starlette.datastructures import State

    from vllm.engine.protocol import EngineClient
    from vllm.entrypoints.logger import RequestLogger
    from vllm.tasks import SupportedTask
else:
    RequestLogger = object
    SupportedTask = object


def register_pooling_api_routers(
    app: FastAPI, supported_tasks: tuple["SupportedTask", ...]
):
    from vllm.entrypoints.pooling.pooling.api_router import router as pooling_router

    app.include_router(pooling_router)

    if "classify" in supported_tasks:
        from vllm.entrypoints.pooling.classify.api_router import (
            router as classify_router,
        )

        app.include_router(classify_router)

    if "embed" in supported_tasks:
        from vllm.entrypoints.pooling.embed.api_router import router as embed_router

        app.include_router(embed_router)

    if "score" in supported_tasks or "embed" in supported_tasks:
        from vllm.entrypoints.pooling.score.api_router import router as score_router

        app.include_router(score_router)


def init_pooling_state(
    engine_client: "EngineClient",
    state: "State",
    args: "Namespace",
    request_logger: RequestLogger | None,
    supported_tasks: tuple["SupportedTask", ...],
):
    from vllm.entrypoints.chat_utils import load_chat_template
    from vllm.entrypoints.pooling.classify.serving import ServingClassification
    from vllm.entrypoints.pooling.embed.serving import OpenAIServingEmbedding
    from vllm.entrypoints.pooling.pooling.serving import OpenAIServingPooling
    from vllm.entrypoints.pooling.score.serving import ServingScores
    from vllm.tasks import POOLING_TASKS

    resolved_chat_template = load_chat_template(args.chat_template)

    state.openai_serving_pooling = (
        (
            OpenAIServingPooling(
                engine_client,
                state.openai_serving_models,
                supported_tasks=supported_tasks,
                request_logger=request_logger,
                chat_template=resolved_chat_template,
                chat_template_content_format=args.chat_template_content_format,
                trust_request_chat_template=args.trust_request_chat_template,
                log_error_stack=args.log_error_stack,
            )
        )
        if any(task in POOLING_TASKS for task in supported_tasks)
        else None
    )
    state.openai_serving_embedding = (
        OpenAIServingEmbedding(
            engine_client,
            state.openai_serving_models,
            request_logger=request_logger,
            chat_template=resolved_chat_template,
            chat_template_content_format=args.chat_template_content_format,
            trust_request_chat_template=args.trust_request_chat_template,
            log_error_stack=args.log_error_stack,
        )
        if "embed" in supported_tasks
        else None
    )
    state.openai_serving_classification = (
        ServingClassification(
            engine_client,
            state.openai_serving_models,
            request_logger=request_logger,
            chat_template=resolved_chat_template,
            chat_template_content_format=args.chat_template_content_format,
            trust_request_chat_template=args.trust_request_chat_template,
            log_error_stack=args.log_error_stack,
        )
        if "classify" in supported_tasks
        else None
    )
    state.openai_serving_scores = (
        ServingScores(
            engine_client,
            state.openai_serving_models,
            request_logger=request_logger,
            score_template=resolved_chat_template,
            log_error_stack=args.log_error_stack,
        )
        if ("embed" in supported_tasks or "score" in supported_tasks)
        else None
    )
