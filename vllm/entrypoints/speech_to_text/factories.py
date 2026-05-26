# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


from typing import TYPE_CHECKING

from fastapi import FastAPI

if TYPE_CHECKING:
    from argparse import Namespace

    from starlette.datastructures import State

    from vllm.engine.protocol import EngineClient
    from vllm.entrypoints.logger import RequestLogger
    from vllm.tasks import SupportedTask
else:
    RequestLogger = object


def register_speech_to_text_api_routers(
    app: FastAPI,
    supported_tasks: tuple["SupportedTask", ...],
):
    if "realtime" in supported_tasks:
        from .realtime.api_router import router as realtime_router

        app.include_router(realtime_router)

    if "transcription" in supported_tasks:
        from .transcription.api_router import router as transcription_router

        app.include_router(transcription_router)

        from .translation.api_router import router as translation_router

        app.include_router(translation_router)


def add_websocket_metrics_middleware(app: FastAPI):
    from .realtime.metrics import WebSocketMetricsMiddleware

    app.add_middleware(WebSocketMetricsMiddleware)


def init_speech_to_text_state(
    engine_client: "EngineClient",
    state: "State",
    args: "Namespace",
    request_logger: RequestLogger | None,
    supported_tasks: tuple["SupportedTask", ...],
):
    from vllm.entrypoints.chat_utils import UsagePolicy

    usage_policy = UsagePolicy(
        include_usage=args.include_usage_policy,
        continuous_usage=args.continuous_usage_policy,
    )

    # Deprecated --enable-force-include-usage → UsagePolicy conversion.
    # STT: flag only forced include_usage=True, continuous still from request.
    def _maybe_force_usage(
        policy: UsagePolicy,
        enable_force: bool,
        *,
        set_continuous: bool,
    ) -> UsagePolicy:
        if enable_force and policy.include_usage is None:
            return UsagePolicy(
                include_usage="always",
                continuous_usage="always" if set_continuous else None,
            )
        return policy

    usage_policy = _maybe_force_usage(
        usage_policy, args.enable_force_include_usage, set_continuous=False
    )

    if "transcription" in supported_tasks:
        from .transcription.serving import OpenAIServingTranscription

        state.openai_serving_transcription = OpenAIServingTranscription(
            engine_client,
            state.openai_serving_models,
            request_logger=request_logger,
            usage_policy=usage_policy,
        )

        from .translation.serving import OpenAIServingTranslation

        state.openai_serving_translation = OpenAIServingTranslation(
            engine_client,
            state.openai_serving_models,
            request_logger=request_logger,
            usage_policy=usage_policy,
        )

    if "realtime" in supported_tasks:
        from .realtime.serving import OpenAIServingRealtime

        state.openai_serving_realtime = OpenAIServingRealtime(
            engine_client,
            state.openai_serving_models,
            request_logger=request_logger,
            usage_policy=usage_policy,
        )
