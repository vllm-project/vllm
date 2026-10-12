# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from argparse import Namespace
from typing import Any

from starlette.datastructures import State

from vllm.engine.protocol import EngineClient
from vllm.entrypoints.serve.utils.request_logger import RequestLogger

from .serving import OpenAIServingDecisions
from .strategies import ReadContext, ReadStrategy, select_read_strategy


def init_decisions_state(
    engine_client: EngineClient,
    state: State,
    args: Namespace,
    request_logger: RequestLogger | None,
    chat_template: str | None,
    default_chat_template_kwargs: dict[str, Any],
) -> ReadStrategy | None:
    try:
        strategy_cls = select_read_strategy(engine_client.model_config)
        strategy = strategy_cls(
            ReadContext(
                engine_client=engine_client,
                online_renderer=state.online_renderer,
                chat_template=chat_template,
                chat_template_content_format=args.chat_template_content_format,
                default_chat_template_kwargs=default_chat_template_kwargs,
            )
        )
    except ValueError:
        state.openai_serving_decisions = None
        return None

    state.openai_serving_decisions = OpenAIServingDecisions(
        state.openai_serving_models,
        strategy,
        request_logger=request_logger,
    )
    return strategy
