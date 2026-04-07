# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import TYPE_CHECKING

from fastapi import FastAPI

from vllm.config import ModelConfig
from vllm.entrypoints.pooling.utils import enable_scoring_api
from vllm.logger import init_logger

if TYPE_CHECKING:
    from argparse import Namespace

    from starlette.datastructures import State

    from vllm.engine.protocol import EngineClient
    from vllm.entrypoints.logger import RequestLogger
    from vllm.tasks import SupportedTask
else:
    RequestLogger = object
    SupportedTask = object

logger = init_logger(__name__)


def register_pooling_api_routers(
    app: FastAPI,
    supported_tasks: tuple["SupportedTask", ...],
    model_config: ModelConfig | None = None,
):
    if model_config is None:
        return

    pooling_task = model_config.get_pooling_task(supported_tasks)

    if pooling_task is not None:
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

    if enable_scoring_api(supported_tasks, model_config):
        from vllm.entrypoints.pooling.scoring.api_router import router as score_router

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
    from vllm.entrypoints.pooling.embed.serving import ServingEmbedding
    from vllm.entrypoints.pooling.pooling.serving import OpenAIServingPooling
    from vllm.entrypoints.pooling.scoring.serving import ServingScores
    from vllm.tasks import POOLING_TASKS

    model_config = engine_client.model_config

    resolved_chat_template = load_chat_template(args.chat_template)

    state.serving_pooling = (
        (
            OpenAIServingPooling(
                engine_client,
                state.openai_serving_models,
                state.openai_serving_render,
                supported_tasks=supported_tasks,
                request_logger=request_logger,
                chat_template=resolved_chat_template,
                chat_template_content_format=args.chat_template_content_format,
                trust_request_chat_template=args.trust_request_chat_template,
            )
        )
        if any(t in supported_tasks for t in POOLING_TASKS)
        else None
    )
    state.serving_embedding = (
        ServingEmbedding(
            engine_client,
            state.openai_serving_models,
            request_logger=request_logger,
            chat_template=resolved_chat_template,
            chat_template_content_format=args.chat_template_content_format,
            trust_request_chat_template=args.trust_request_chat_template,
        )
        if "embed" in supported_tasks
        else None
    )
    state.serving_classification = (
        ServingClassification(
            engine_client,
            state.openai_serving_models,
            request_logger=request_logger,
            chat_template=resolved_chat_template,
            chat_template_content_format=args.chat_template_content_format,
            trust_request_chat_template=args.trust_request_chat_template,
        )
        if "classify" in supported_tasks
        else None
    )
    state.serving_scores = (
        ServingScores(
            engine_client,
            state.openai_serving_models,
            request_logger=request_logger,
            chat_template=resolved_chat_template,
            chat_template_content_format=args.chat_template_content_format,
            trust_request_chat_template=args.trust_request_chat_template,
            enable_flash_late_interaction=getattr(
                args, "enable_flash_late_interaction", True
            ),
        )
        if enable_scoring_api(supported_tasks, model_config)
        else None
    )
