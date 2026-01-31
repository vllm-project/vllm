# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""API router for the Generative Scores endpoint.

This module defines the FastAPI routes for the /v1/generative-scores endpoint.
"""

from http import HTTPStatus
from typing import TYPE_CHECKING

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse
from typing_extensions import assert_never

from vllm.entrypoints.openai.engine.protocol import ErrorResponse
from vllm.entrypoints.openai.generative_scores.protocol import (
    GenerativeScoreRequest,
    GenerativeScoreResponse,
)
from vllm.entrypoints.openai.utils import validate_json_request
from vllm.entrypoints.utils import load_aware_call, with_cancellation
from vllm.logger import init_logger

if TYPE_CHECKING:
    from argparse import Namespace

    from starlette.datastructures import State

    from vllm.engine.protocol import EngineClient
    from vllm.entrypoints.logger import RequestLogger
    from vllm.tasks import SupportedTask


router = APIRouter()

logger = init_logger(__name__)


def generative_scores(request: Request):
    """Get the generative scores handler from app state."""
    return request.app.state.openai_serving_generative_scores


@router.post(
    "/v1/generative-scores",
    dependencies=[Depends(validate_json_request)],
    responses={
        HTTPStatus.BAD_REQUEST.value: {"model": ErrorResponse},
        HTTPStatus.INTERNAL_SERVER_ERROR.value: {"model": ErrorResponse},
    },
)
@with_cancellation
@load_aware_call
async def create_generative_score(
    request: GenerativeScoreRequest,
    raw_request: Request,
):
    """Compute generative scores for the given query and items.

    This endpoint scores the probability of specified token IDs appearing after 
    the given query and item are appended together. For example:

        query = "<|user|>Is the following city the capital of France? "
        items = ["Paris <|assistant|>", "London <|assistant|>", "Berlin <|assistant|>"]
        label_token_ids = [2332, 1223]  # Token IDs for "Yes" and "No"
        item_first = False

        This would pass the following prompts to the model:
        - "<|user|>Is the following city the capital of France? Paris <|assistant|>"
        - "<|user|>Is the following city the capital of France? London <|assistant|>"
        - "<|user|>Is the following city the capital of France? Berlin <|assistant|>"

        The API would return the probabilities of the model producing "Yes" and "No" 
        as the next token for each prompt.

    Args:
        request: The GenerativeScoreRequest containing:
            - model: The model to use (optional)
            - query: The query text or pre-tokenized token IDs
            - items: List of item texts or pre-tokenized token IDs
            - label_token_ids: List of token IDs to compute probabilities for
            - apply_softmax: Whether to normalize over only label tokens (default: True)
            - item_first: Whether to prepend items to query (default: False)

    Returns:
        GenerativeScoreResponse containing probabilities for each item.

    Raises:
        400 Bad Request: If label_token_ids are out of vocabulary range.
        500 Internal Server Error: If an internal error occurs.
    """
    handler = generative_scores(raw_request)
    if handler is None:
        base_server = raw_request.app.state.openai_serving_tokenization
        return base_server.create_error_response(
            message="The model does not support Generative Scores API"
        )

    try:
        generator = await handler.create_generative_score(request, raw_request)
    except Exception as e:
        return handler.create_error_response(e)

    if isinstance(generator, ErrorResponse):
        return JSONResponse(
            content=generator.model_dump(), status_code=generator.error.code
        )
    elif isinstance(generator, GenerativeScoreResponse):
        return JSONResponse(content=generator.model_dump())

    assert_never(generator)


def register_generative_scores_api_routers(app):
    """Register the generative scores API router with the app."""
    app.include_router(router)


async def init_generative_scores_state(
    engine_client: "EngineClient",
    state: "State",
    args: "Namespace",
    request_logger: "RequestLogger | None",
    supported_tasks: tuple["SupportedTask", ...],
):
    """Initialize the generative scores serving state.

    Args:
        engine_client: The engine client for model inference.
        state: The application state to store the handler.
        args: Command line arguments.
        request_logger: Logger for request logging.
        supported_tasks: Tuple of supported tasks.
    """
    from vllm.entrypoints.openai.generative_scores.serving import (
        OpenAIServingGenerativeScores,
    )

    # Only initialize for generative models
    if "generate" in supported_tasks:
        state.openai_serving_generative_scores = OpenAIServingGenerativeScores(
            engine_client,
            state.openai_serving_models,
            request_logger=request_logger,
            log_error_stack=args.log_error_stack,
        )
    else:
        state.openai_serving_generative_scores = None
