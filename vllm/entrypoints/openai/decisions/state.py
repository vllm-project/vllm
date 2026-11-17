# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from argparse import Namespace
from typing import Any

from starlette.datastructures import State

from vllm.engine.protocol import EngineClient
from vllm.entrypoints.serve.utils.request_logger import RequestLogger
from vllm.logger import init_logger
from vllm.tasks import SupportedTask

from .serving import OpenAIServingDecisions
from .strategies import (
    ReadContext,
    ReadStrategy,
    has_knowline_marker,
    select_read_strategy,
)

logger = init_logger(__name__)


def init_decisions_state(
    engine_client: EngineClient,
    state: State,
    args: Namespace,
    request_logger: RequestLogger | None,
    chat_template: str | None,
    default_chat_template_kwargs: dict[str, Any],
    supported_tasks: tuple[SupportedTask, ...],
) -> ReadStrategy | None:
    state.openai_serving_decisions = None
    model_config = engine_client.model_config
    try:
        strategy_cls = select_read_strategy(model_config, supported_tasks)
    except ValueError as e:
        # Only a checkpoint that opted in by its marker expects the
        # endpoints, so only its misconfiguration warns: a marker on an
        # architecture the format cannot serve, a task the server does not
        # serve, or a logprobs mode the reads cannot use. A stock model
        # stays quiet whether it maps to no strategy or runs another task
        # or logprobs mode (a pooling Qwen3-Embedding, --logprobs-mode
        # processed_logits): one line per server start is noise.
        log = logger.warning if has_knowline_marker(model_config) else logger.debug
        log("Decisions endpoints disabled: %s", e)
        return None
    try:
        strategy = strategy_cls(
            ReadContext(
                engine_client=engine_client,
                online_renderer=state.online_renderer,
                chat_template=chat_template,
                chat_template_content_format=args.chat_template_content_format,
                default_chat_template_kwargs=default_chat_template_kwargs,
            )
        )
    except ValueError as e:
        # The model has a strategy that cannot start: a misconfiguration.
        logger.warning("Decisions endpoints disabled: %s", e)
        return None

    if strategy.task == "generate":
        state.openai_serving_decisions = OpenAIServingDecisions(
            state.openai_serving_models,
            strategy,
            request_logger=request_logger,
        )
    return strategy
