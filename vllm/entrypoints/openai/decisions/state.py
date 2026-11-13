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
    NoStrategyError,
    ReadContext,
    ReadStrategy,
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
    try:
        strategy_cls = select_read_strategy(engine_client.model_config, supported_tasks)
    except NoStrategyError as e:
        # Most models map to no strategy, so the common case stays out of
        # the default log level: one line per server start is noise.
        logger.debug("Decisions endpoints disabled: %s", e)
        return None
    except ValueError as e:
        # A misconfiguration an operator must see: a marker on an
        # architecture it cannot serve, a task the server does not serve,
        # or a logprobs mode the reads cannot use.
        logger.warning("Decisions endpoints disabled: %s", e)
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
