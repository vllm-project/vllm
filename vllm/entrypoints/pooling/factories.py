# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import TYPE_CHECKING

from fastapi import FastAPI

from vllm.config import ModelConfig, VllmConfig
from vllm.entrypoints.chat_utils import ChatTemplateConfig
from vllm.logger import init_logger
from vllm.plugins.io_processors import has_io_processor
from vllm.renderers import BaseRenderer
from vllm.tasks import POOLING_TASKS, SCORE_TYPE_MAP, SupportedTask

from .base.io_processor import PoolingIOProcessor
from .utils import enable_scoring_api

if TYPE_CHECKING:
    from argparse import Namespace

    from starlette.datastructures import State

    from vllm.engine.protocol import EngineClient
    from vllm.entrypoints.logger import RequestLogger
    from vllm.entrypoints.sagemaker.api_router import (
        EndpointFn,
        GetHandlerFn,
        RequestType,
    )

else:
    RequestLogger = object


logger = init_logger(__name__)


def init_pooling_io_processors(
    supported_tasks: tuple[SupportedTask, ...],
    vllm_config: VllmConfig,
    renderer: BaseRenderer,
    chat_template_config: ChatTemplateConfig,
) -> dict[str, PoolingIOProcessor]:
    model_config = vllm_config.model_config
    processors: dict[str, type[PoolingIOProcessor]] = {}
    pooling_task = model_config.get_pooling_task(supported_tasks)

    if pooling_task == "classify":
        from .classify.io_processor import ClassifyIOProcessor

        processors["classify"] = ClassifyIOProcessor

    if pooling_task == "token_classify":
        from .classify.io_processor import TokenClassifyIOProcessor

        processors["token_classify"] = TokenClassifyIOProcessor

    if pooling_task == "embed":
        from .embed.io_processor import EmbedIOProcessor

        processors["embed"] = EmbedIOProcessor

    if pooling_task == "token_embed":
        from .embed.io_processor import TokenEmbedIOProcessor

        processors["token_embed"] = TokenEmbedIOProcessor

    if has_io_processor(
        vllm_config,
        model_config.io_processor_plugin,
    ):
        from .pooling.io_processor import PluginWithIOProcessorPlugins

        processors["plugin"] = PluginWithIOProcessorPlugins
    elif pooling_task == "plugin":
        from .pooling.io_processor import PluginWithoutIOProcessorPlugins

        processors["plugin"] = PluginWithoutIOProcessorPlugins

    if enable_scoring_api(supported_tasks, model_config):
        from .scoring.io_processor import ScoringIOProcessors

        score_type: str | None = SCORE_TYPE_MAP.get(pooling_task, None)  # type: ignore[arg-type]
        if score_type is not None and score_type in ScoringIOProcessors:
            processors[score_type] = ScoringIOProcessors[score_type]

    if model_config.architecture == "JinaForRanking":
        from .embed.io_processor import JinaRankingTokenEmbedIOProcessor
        from .scoring.io_processor import ScoringIOProcessors

        processors["token_embed"] = JinaRankingTokenEmbedIOProcessor
        processors["late-interaction"] = ScoringIOProcessors["jina-reranking-scoring"]

    return {
        task: processor_cls(
            vllm_config=vllm_config,
            renderer=renderer,
            chat_template_config=chat_template_config,
        )
        for task, processor_cls in processors.items()
    }


def register_pooling_api_routers(
    app: FastAPI,
    supported_tasks: tuple["SupportedTask", ...],
    model_config: ModelConfig | None = None,
):
    if model_config is None:
        return

    pooling_task = model_config.get_pooling_task(supported_tasks)

    if pooling_task is not None:
        from .pooling.api_router import router as pooling_router

        app.include_router(pooling_router)

    if "classify" in supported_tasks:
        from .classify.api_router import (
            router as classify_router,
        )

        app.include_router(classify_router)

    if "embed" in supported_tasks:
        from .embed.api_router import router as embed_router

        app.include_router(embed_router)

    if enable_scoring_api(supported_tasks, model_config):
        from .scoring.api_router import router as score_router

        app.include_router(score_router)


def init_pooling_state(
    engine_client: "EngineClient",
    state: "State",
    args: "Namespace",
    request_logger: RequestLogger | None,
    supported_tasks: tuple["SupportedTask", ...],
):
    model_config = engine_client.model_config
    if model_config is None:
        return

    from vllm.entrypoints.chat_utils import load_chat_template
    from vllm.tasks import POOLING_TASKS

    from .classify.serving import ServingClassification
    from .embed.serving import ServingEmbedding
    from .pooling.serving import ServingPooling
    from .scoring.serving import ServingScores

    resolved_chat_template = load_chat_template(args.chat_template)
    pooling_task = model_config.get_pooling_task(supported_tasks)

    chat_template_config = ChatTemplateConfig(
        chat_template=resolved_chat_template,
        chat_template_content_format=args.chat_template_content_format,
        trust_request_chat_template=args.trust_request_chat_template,
    )

    state.serving_pooling = (
        (
            ServingPooling(
                engine_client,
                state.openai_serving_models,
                supported_tasks=supported_tasks,
                request_logger=request_logger,
                chat_template_config=chat_template_config,
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
            chat_template_config=chat_template_config,
        )
        if pooling_task == "embed"
        else None
    )
    state.serving_classification = (
        ServingClassification(
            engine_client,
            state.openai_serving_models,
            request_logger=request_logger,
            chat_template_config=chat_template_config,
        )
        if pooling_task == "classify"
        else None
    )
    state.serving_scores = (
        ServingScores(
            engine_client,
            state.openai_serving_models,
            supported_tasks=supported_tasks,
            request_logger=request_logger,
            chat_template_config=chat_template_config,
            enable_flash_late_interaction=getattr(
                args, "enable_flash_late_interaction", True
            ),
        )
        if enable_scoring_api(supported_tasks, model_config)
        else None
    )


def get_pooling_invocation_types(
    supported_tasks: tuple["SupportedTask", ...],
    model_config: ModelConfig | None = None,
):
    # NOTE: Items defined earlier take higher priority
    invocation_types: list[tuple[RequestType, tuple[GetHandlerFn, EndpointFn]]] = []

    if model_config is None:
        return invocation_types

    pooling_task = model_config.get_pooling_task(supported_tasks)

    if pooling_task == "embed":
        from .embed.api_router import create_embedding, embedding
        from .embed.protocol import EmbeddingRequest

        invocation_types += [
            (EmbeddingRequest, (embedding, create_embedding)),
        ]

    if pooling_task == "classify":
        from .classify.api_router import classify, create_classify
        from .classify.protocol import ClassificationRequest

        invocation_types += [
            (ClassificationRequest, (classify, create_classify)),
        ]

    if enable_scoring_api(supported_tasks, model_config):
        from .scoring.api_router import do_rerank, rerank
        from .scoring.protocol import RerankRequest

        invocation_types += [
            (RerankRequest, (rerank, do_rerank)),
        ]

        from .scoring.api_router import create_score, score
        from .scoring.protocol import ScoreRequest

        invocation_types += [
            (ScoreRequest, (score, create_score)),
        ]

    if any(task in POOLING_TASKS for task in supported_tasks):
        from .pooling.api_router import create_pooling, pooling
        from .pooling.protocol import PoolingRequest

        invocation_types += [
            (PoolingRequest, (pooling, create_pooling)),
        ]

    return invocation_types
