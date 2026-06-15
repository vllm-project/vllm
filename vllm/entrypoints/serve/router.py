# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from http import HTTPStatus

from fastapi import APIRouter, Depends, FastAPI
from fastapi.responses import Response

from vllm import envs
from vllm.config import ModelConfig
from vllm.entrypoints.openai.engine.protocol import ErrorResponse
from vllm.entrypoints.serve.utils.api_utils import validate_json_request
from vllm.logger import init_logger
from vllm.tasks import POOLING_TASKS, SupportedTask

logger = init_logger(__name__)


def init_api_router(
    app: FastAPI,
    supported_tasks: tuple["SupportedTask", ...],
    model_config: ModelConfig | None = None,
) -> APIRouter:
    router = APIRouter()

    #############################################################
    ## OpenAI-Compatible Server

    if "generate" in supported_tasks:
        ### Completions API
        from vllm.entrypoints.openai.completion.api_router import create_completion

        router.post(
            "/v1/completions",
            dependencies=[Depends(validate_json_request)],
            responses={
                HTTPStatus.OK.value: {"content": {"text/event-stream": {}}},
                HTTPStatus.BAD_REQUEST.value: {"model": ErrorResponse},
                HTTPStatus.NOT_FOUND.value: {"model": ErrorResponse},
                HTTPStatus.INTERNAL_SERVER_ERROR.value: {"model": ErrorResponse},
            },
        )(create_completion)

        ### Chat Completions API
        from vllm.entrypoints.openai.chat_completion.api_router import (
            create_batch_chat_completion,
            create_chat_completion,
        )

        router.post(
            "/v1/chat/completions",
            dependencies=[Depends(validate_json_request)],
            responses={
                HTTPStatus.OK.value: {"content": {"text/event-stream": {}}},
                HTTPStatus.BAD_REQUEST.value: {"model": ErrorResponse},
                HTTPStatus.NOT_FOUND.value: {"model": ErrorResponse},
                HTTPStatus.INTERNAL_SERVER_ERROR.value: {"model": ErrorResponse},
                HTTPStatus.NOT_IMPLEMENTED.value: {"model": ErrorResponse},
            },
        )(create_chat_completion)
        router.post(
            "/v1/chat/completions/batch",
            dependencies=[Depends(validate_json_request)],
            responses={
                HTTPStatus.OK.value: {},
                HTTPStatus.BAD_REQUEST.value: {"model": ErrorResponse},
                HTTPStatus.NOT_FOUND.value: {"model": ErrorResponse},
                HTTPStatus.INTERNAL_SERVER_ERROR.value: {"model": ErrorResponse},
                HTTPStatus.NOT_IMPLEMENTED.value: {"model": ErrorResponse},
            },
        )(create_batch_chat_completion)

        ### Responses API
        from vllm.entrypoints.openai.responses.api_router import (
            cancel_responses,
            create_responses,
            retrieve_responses,
        )

        router.post(
            "/v1/responses",
            dependencies=[Depends(validate_json_request)],
            responses={
                HTTPStatus.OK.value: {"content": {"text/event-stream": {}}},
                HTTPStatus.BAD_REQUEST.value: {"model": ErrorResponse},
                HTTPStatus.NOT_FOUND.value: {"model": ErrorResponse},
                HTTPStatus.INTERNAL_SERVER_ERROR.value: {"model": ErrorResponse},
            },
        )(create_responses)
        router.get("/v1/responses/{response_id}")(retrieve_responses)
        router.post("/v1/responses/{response_id}/cancel")(cancel_responses)

    #############################################################
    ## Anthropic APIs

    if "generate" in supported_tasks:
        from vllm.entrypoints.anthropic.api_router import count_tokens, create_messages
        from vllm.entrypoints.anthropic.protocol import (
            AnthropicCountTokensResponse,
            AnthropicErrorResponse,
        )

        router.post(
            "/v1/messages",
            dependencies=[Depends(validate_json_request)],
            responses={
                HTTPStatus.OK.value: {"content": {"text/event-stream": {}}},
                HTTPStatus.BAD_REQUEST.value: {"model": AnthropicErrorResponse},
                HTTPStatus.NOT_FOUND.value: {"model": AnthropicErrorResponse},
                HTTPStatus.INTERNAL_SERVER_ERROR.value: {
                    "model": AnthropicErrorResponse
                },
            },
        )(create_messages)

        router.post(
            "/v1/messages/count_tokens",
            dependencies=[Depends(validate_json_request)],
            responses={
                HTTPStatus.OK.value: {"model": AnthropicCountTokensResponse},
                HTTPStatus.BAD_REQUEST.value: {"model": AnthropicErrorResponse},
                HTTPStatus.NOT_FOUND.value: {"model": AnthropicErrorResponse},
                HTTPStatus.INTERNAL_SERVER_ERROR.value: {
                    "model": AnthropicErrorResponse
                },
            },
        )(count_tokens)

    #############################################################
    ## Generative Scoring API

    if "generate" in supported_tasks:
        from vllm.entrypoints.generate.generative_scoring.api_router import (
            create_generative_scoring,
        )

        router.post(
            "/generative_scoring",
            dependencies=[Depends(validate_json_request)],
            responses={
                HTTPStatus.BAD_REQUEST.value: {"model": ErrorResponse},
                HTTPStatus.INTERNAL_SERVER_ERROR.value: {"model": ErrorResponse},
            },
        )(create_generative_scoring)

    #############################################################
    ## Pooling APIs

    if (
        any(task in POOLING_TASKS for task in supported_tasks)
        and model_config is not None
    ):
        from vllm.entrypoints.pooling.pooling.api_router import create_pooling
        from vllm.entrypoints.pooling.utils import enable_scoring_api

        router.post(
            "/pooling",
            dependencies=[Depends(validate_json_request)],
            responses={
                HTTPStatus.BAD_REQUEST.value: {"model": ErrorResponse},
                HTTPStatus.INTERNAL_SERVER_ERROR.value: {"model": ErrorResponse},
            },
        )(create_pooling)

        if "classify" in supported_tasks:
            from vllm.entrypoints.pooling.classify.api_router import create_classify

            router.post("/classify", dependencies=[Depends(validate_json_request)])(
                create_classify
            )

        if "embed" in supported_tasks:
            from vllm.entrypoints.pooling.embed.api_router import (
                create_cohere_embedding,
                create_embedding,
            )

            router.post(
                "/v1/embeddings",
                dependencies=[Depends(validate_json_request)],
                responses={
                    HTTPStatus.BAD_REQUEST.value: {"model": ErrorResponse},
                    HTTPStatus.INTERNAL_SERVER_ERROR.value: {"model": ErrorResponse},
                },
            )(create_embedding)
            router.post(
                "/v2/embed",
                dependencies=[Depends(validate_json_request)],
                responses={
                    HTTPStatus.BAD_REQUEST.value: {"model": ErrorResponse},
                    HTTPStatus.INTERNAL_SERVER_ERROR.value: {"model": ErrorResponse},
                },
            )(create_cohere_embedding)

        if enable_scoring_api(supported_tasks, model_config):
            from vllm.entrypoints.pooling.scoring.api_router import (
                create_score,
                create_score_v1,
                do_rerank,
                do_rerank_v1,
                do_rerank_v2,
            )

            router.post(
                "/score",
                dependencies=[Depends(validate_json_request)],
                responses={
                    HTTPStatus.BAD_REQUEST.value: {"model": ErrorResponse},
                    HTTPStatus.INTERNAL_SERVER_ERROR.value: {"model": ErrorResponse},
                },
            )(create_score)
            router.post(
                "/v1/score",
                dependencies=[Depends(validate_json_request)],
                responses={
                    HTTPStatus.BAD_REQUEST.value: {"model": ErrorResponse},
                    HTTPStatus.INTERNAL_SERVER_ERROR.value: {"model": ErrorResponse},
                },
            )(create_score_v1)
            router.post(
                "/rerank",
                dependencies=[Depends(validate_json_request)],
                responses={
                    HTTPStatus.BAD_REQUEST.value: {"model": ErrorResponse},
                    HTTPStatus.INTERNAL_SERVER_ERROR.value: {"model": ErrorResponse},
                },
            )(do_rerank)
            router.post(
                "/v1/rerank",
                dependencies=[Depends(validate_json_request)],
                responses={
                    HTTPStatus.BAD_REQUEST.value: {"model": ErrorResponse},
                    HTTPStatus.INTERNAL_SERVER_ERROR.value: {"model": ErrorResponse},
                },
            )(do_rerank_v1)
            router.post(
                "/v2/rerank",
                dependencies=[Depends(validate_json_request)],
                responses={
                    HTTPStatus.BAD_REQUEST.value: {"model": ErrorResponse},
                    HTTPStatus.INTERNAL_SERVER_ERROR.value: {"model": ErrorResponse},
                },
            )(do_rerank_v2)

    #############################################################
    ## Speech to Text APIs

    if "transcription" in supported_tasks or "realtime" in supported_tasks:
        ### Transcriptions API
        from vllm.entrypoints.speech_to_text.transcription.api_router import (
            create_transcriptions,
        )

        router.post(
            "/v1/audio/transcriptions",
            responses={
                HTTPStatus.OK.value: {"content": {"text/event-stream": {}}},
                HTTPStatus.BAD_REQUEST.value: {"model": ErrorResponse},
                HTTPStatus.UNPROCESSABLE_ENTITY.value: {"model": ErrorResponse},
                HTTPStatus.INTERNAL_SERVER_ERROR.value: {"model": ErrorResponse},
            },
        )(create_transcriptions)

        ### Translation API
        from vllm.entrypoints.speech_to_text.translation.api_router import (
            create_translations,
        )

        router.post(
            "/v1/audio/translations",
            responses={
                HTTPStatus.OK.value: {"content": {"text/event-stream": {}}},
                HTTPStatus.BAD_REQUEST.value: {"model": ErrorResponse},
                HTTPStatus.UNPROCESSABLE_ENTITY.value: {"model": ErrorResponse},
                HTTPStatus.INTERNAL_SERVER_ERROR.value: {"model": ErrorResponse},
            },
        )(create_translations)

        ### Realtime API
        if "realtime" in supported_tasks:
            from vllm.entrypoints.speech_to_text.realtime.api_router import (
                realtime_endpoint,
            )

            router.websocket("/v1/realtime")(realtime_endpoint)

    #############################################################
    ## Instrumentator APIs

    ### Basic APIs
    from .instrumentator.basic import get_server_load_metrics, show_version

    router.get("/load")(get_server_load_metrics)
    router.get("/version")(show_version)

    ### Models API
    from vllm.entrypoints.openai.models.api_router import models

    router.get("/v1/models")(models)

    ### Health API
    from .instrumentator.health import health

    router.get("/health", response_class=Response)(health)

    ### Metrics APIs
    from .instrumentator.metrics import attach_router as metrics_attach_router

    metrics_attach_router(app)

    ### Offline docs
    from .instrumentator.offline_docs import attach_router as offline_docs_attach_router

    offline_docs_attach_router(app)

    ### LoRA dynamic loading
    if envs.VLLM_ALLOW_RUNTIME_LORA_UPDATING:
        logger.warning(
            "LoRA dynamic loading & unloading is enabled in the API server. "
            "This should ONLY be used for local development!"
        )

        from .lora.api_router import load_lora_adapter, unload_lora_adapter

        router.post(
            "/v1/load_lora_adapter", dependencies=[Depends(validate_json_request)]
        )(load_lora_adapter)

        router.post(
            "/v1/unload_lora_adapter", dependencies=[Depends(validate_json_request)]
        )(unload_lora_adapter)

    ### Profiler
    from vllm.config import ProfilerConfig

    profiler_config = getattr(app.state.args, "profiler_config", None)
    assert profiler_config is None or isinstance(profiler_config, ProfilerConfig)
    if profiler_config is not None and profiler_config.profiler is not None:
        logger.warning_once(
            "Profiler with mode '%s' is enabled in the "
            "API server. This should ONLY be used for local development!",
            profiler_config.profiler,
        )

        from .profile.api_router import start_profile, stop_profile

        router.post("/start_profile")(start_profile)
        router.post("/stop_profile")(stop_profile)

    #############################################################
    ## SageMaker APIs

    from .sagemaker.api_router import invocations, ping

    router.post("/ping", response_class=Response)(ping)
    router.get("/ping", response_class=Response)(ping)

    router.post(
        "/invocations",
        dependencies=[Depends(validate_json_request)],
        responses={
            HTTPStatus.BAD_REQUEST.value: {"model": ErrorResponse},
            HTTPStatus.UNSUPPORTED_MEDIA_TYPE.value: {"model": ErrorResponse},
            HTTPStatus.INTERNAL_SERVER_ERROR.value: {"model": ErrorResponse},
        },
    )(invocations(supported_tasks, model_config))

    #############################################################
    ## Disaggregated Everything

    if "generate" in supported_tasks:
        ### Tokens IN <> Tokens OUT
        from .disagg.api_router import generate

        router.post(
            "/inference/v1/generate",
            dependencies=[Depends(validate_json_request)],
            responses={
                HTTPStatus.OK.value: {"content": {"text/event-stream": {}}},
                HTTPStatus.BAD_REQUEST.value: {"model": ErrorResponse},
                HTTPStatus.NOT_FOUND.value: {"model": ErrorResponse},
                HTTPStatus.INTERNAL_SERVER_ERROR.value: {"model": ErrorResponse},
            },
        )(generate)

        if getattr(app.state.args, "tokens_only", False):
            from .disagg.api_router import abort_requests

            router.post("/abort_requests")(abort_requests)

        ### Renderer APIs
        from .disagg.protocol import GenerateRequest
        from .render.api_router import render_chat_completion, render_completion

        router.post(
            "/v1/completions/render",
            dependencies=[Depends(validate_json_request)],
            response_model=list[GenerateRequest],
            responses={
                HTTPStatus.BAD_REQUEST.value: {"model": ErrorResponse},
                HTTPStatus.NOT_FOUND.value: {"model": ErrorResponse},
                HTTPStatus.INTERNAL_SERVER_ERROR.value: {"model": ErrorResponse},
            },
        )(render_completion)

        router.post(
            "/v1/chat/completions/render",
            dependencies=[Depends(validate_json_request)],
            response_model=GenerateRequest,
            responses={
                HTTPStatus.BAD_REQUEST.value: {"model": ErrorResponse},
                HTTPStatus.NOT_FOUND.value: {"model": ErrorResponse},
                HTTPStatus.NOT_IMPLEMENTED.value: {"model": ErrorResponse},
                HTTPStatus.INTERNAL_SERVER_ERROR.value: {"model": ErrorResponse},
            },
        )(render_chat_completion)

        ### Derenderer APIs
        from vllm.entrypoints.openai.chat_completion.protocol import (
            ChatCompletionResponse,
        )
        from vllm.entrypoints.openai.completion.protocol import CompletionResponse
        from vllm.entrypoints.serve.render.api_router import (
            derender_chat_completion,
            derender_completion,
        )

        router.post(
            "/v1/chat/completions/derender",
            dependencies=[Depends(validate_json_request)],
            response_model=ChatCompletionResponse,
            responses={
                HTTPStatus.BAD_REQUEST.value: {"model": ErrorResponse},
                HTTPStatus.NOT_FOUND.value: {"model": ErrorResponse},
                HTTPStatus.INTERNAL_SERVER_ERROR.value: {"model": ErrorResponse},
            },
        )(derender_chat_completion)

        router.post(
            "/v1/completions/derender",
            dependencies=[Depends(validate_json_request)],
            response_model=CompletionResponse,
            responses={
                HTTPStatus.BAD_REQUEST.value: {"model": ErrorResponse},
                HTTPStatus.NOT_FOUND.value: {"model": ErrorResponse},
                HTTPStatus.INTERNAL_SERVER_ERROR.value: {"model": ErrorResponse},
            },
        )(derender_completion)

    #############################################################
    ## Tokenize APIs

    from .tokenize.api_router import detokenize, get_tokenizer_info, tokenize

    router.post(
        "/tokenize",
        dependencies=[Depends(validate_json_request)],
        responses={
            HTTPStatus.BAD_REQUEST.value: {"model": ErrorResponse},
            HTTPStatus.NOT_FOUND.value: {"model": ErrorResponse},
            HTTPStatus.INTERNAL_SERVER_ERROR.value: {"model": ErrorResponse},
            HTTPStatus.NOT_IMPLEMENTED.value: {"model": ErrorResponse},
        },
    )(tokenize)

    router.post(
        "/detokenize",
        dependencies=[Depends(validate_json_request)],
        responses={
            HTTPStatus.BAD_REQUEST.value: {"model": ErrorResponse},
            HTTPStatus.NOT_FOUND.value: {"model": ErrorResponse},
            HTTPStatus.INTERNAL_SERVER_ERROR.value: {"model": ErrorResponse},
        },
    )(detokenize)

    if getattr(app.state.args, "enable_tokenizer_info_endpoint", False):
        router.get("/tokenizer_info")(get_tokenizer_info)

    #############################################################
    ## Elastic Expert Parallelism (EEP)

    if "generate" in supported_tasks:
        from .elastic_ep.api_router import is_scaling_elastic_ep, scale_elastic_ep

        router.post(
            "/scale_elastic_ep",
            dependencies=[Depends(validate_json_request)],
            responses={
                HTTPStatus.OK.value: {"model": dict},
                HTTPStatus.BAD_REQUEST.value: {"model": ErrorResponse},
                HTTPStatus.REQUEST_TIMEOUT.value: {"model": ErrorResponse},
                HTTPStatus.INTERNAL_SERVER_ERROR.value: {"model": ErrorResponse},
            },
        )(scale_elastic_ep)

        router.post("/is_scaling_elastic_ep")(is_scaling_elastic_ep)

    #############################################################
    ## Server in development mode

    if envs.VLLM_SERVER_DEV_MODE:
        logger.warning(
            "SECURITY WARNING: Development endpoints are enabled! "
            "This should NOT be used in production!"
        )

        ### Cache Management APIs
        from .dev.cache.api_router import (
            reset_encoder_cache,
            reset_mm_cache,
            reset_prefix_cache,
        )

        router.post("/reset_prefix_cache")(reset_prefix_cache)
        router.post("/reset_mm_cache")(reset_mm_cache)
        router.post("/reset_encoder_cache")(reset_encoder_cache)

        ### Weight Transfer APIs (RL Training)
        from .dev.rlhf.api_router import (
            finish_weight_update,
            get_world_size,
            init_weight_transfer_engine,
            is_paused,
            pause_generation,
            resume_generation,
            start_weight_update,
            update_weights,
        )

        router.post("/pause")(pause_generation)
        router.post("/resume")(resume_generation)
        router.get("/is_paused")(is_paused)
        router.post("/init_weight_transfer_engine")(init_weight_transfer_engine)
        router.post("/start_weight_update")(start_weight_update)
        router.post("/update_weights")(update_weights)
        router.post("/finish_weight_update")(finish_weight_update)
        router.get("/get_world_size")(get_world_size)

        ### Collective RPC
        from .dev.rpc.api_router import collective_rpc

        router.post("/collective_rpc")(collective_rpc)

        ### Server info
        from .dev.server_info.api_router import show_server_info

        router.get("/server_info")(show_server_info)

        ### Sleep Mode APIs
        from .dev.sleep.api_router import is_sleeping, sleep, wake_up

        router.post("/sleep")(sleep)
        router.post("/wake_up")(wake_up)
        router.get("/is_sleeping")(is_sleeping)

    return router
