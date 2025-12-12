# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Warmup module for vLLM OpenAI-compatible server.

This module provides functionality to run warmup requests before the server
reports healthy, ensuring consistent performance from the first production
request.
"""

import asyncio
import json
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from starlette.datastructures import State

from vllm.entrypoints.openai.protocol import (
    ChatCompletionRequest,
    CompletionRequest,
)
from vllm.entrypoints.openai.serving_chat import OpenAIServingChat
from vllm.entrypoints.openai.serving_completion import OpenAIServingCompletion
from vllm.entrypoints.pooling.classify.protocol import (
    ClassificationCompletionRequest,
)
from vllm.entrypoints.pooling.classify.serving import ServingClassification
from vllm.entrypoints.pooling.embed.protocol import EmbeddingCompletionRequest
from vllm.entrypoints.pooling.embed.serving import OpenAIServingEmbedding
from vllm.entrypoints.pooling.pooling.protocol import PoolingCompletionRequest
from vllm.entrypoints.pooling.pooling.serving import OpenAIServingPooling
from vllm.entrypoints.pooling.score.protocol import RerankRequest, ScoreRequest
from vllm.entrypoints.pooling.score.serving import ServingScores
from vllm.logger import init_logger

logger = init_logger(__name__)

# Supported endpoints for warmup
SUPPORTED_ENDPOINTS = {
    "/v1/chat/completions",
    "/v1/completions",
    "/v1/embeddings",
    "/pooling",
    "/classify",
    "/score",
    "/v1/score",
    "/rerank",
    "/v1/rerank",
    "/v2/rerank",
}


@dataclass
class WarmupRequestConfig:
    """Configuration for a single warmup request."""

    endpoint: str
    payload: dict[str, Any]
    count: int = 1


@dataclass
class WarmupConfig:
    """Configuration for server warmup."""

    concurrency: int = 1
    requests: list[WarmupRequestConfig] = field(default_factory=list)

    @classmethod
    def from_file(cls, path: str) -> "WarmupConfig":
        """Load warmup configuration from a JSON file."""
        config_path = Path(path)
        if not config_path.exists():
            raise FileNotFoundError(f"Warmup config file not found: {path}")

        with open(config_path) as f:
            data = json.load(f)

        concurrency = data.get("concurrency", 1)
        requests = []

        for req_data in data.get("requests", []):
            endpoint = req_data.get("endpoint")
            if not endpoint:
                raise ValueError("Each warmup request must have an 'endpoint' field")

            payload = req_data.get("payload")
            if not payload:
                raise ValueError("Each warmup request must have a 'payload' field")

            count = req_data.get("count", 1)
            requests.append(
                WarmupRequestConfig(endpoint=endpoint, payload=payload, count=count)
            )

        return cls(concurrency=concurrency, requests=requests)


async def _run_chat_completion_warmup(
    handler: OpenAIServingChat,
    payload: dict[str, Any],
    model: str,
    request_id: str,
) -> None:
    """Run a single chat completion warmup request."""
    # Copy payload and set defaults
    request_payload = payload.copy()
    request_payload.setdefault("model", model)
    request_payload.setdefault("request_id", request_id)

    request = ChatCompletionRequest(**request_payload)

    result = await handler.create_chat_completion(request, raw_request=None)

    # Consume the generator if streaming
    if isinstance(result, AsyncIterator):
        async for _ in result:
            pass


async def _run_completion_warmup(
    handler: OpenAIServingCompletion,
    payload: dict[str, Any],
    model: str,
    request_id: str,
) -> None:
    """Run a single completion warmup request."""
    # Copy payload and set defaults
    request_payload = payload.copy()
    request_payload.setdefault("model", model)
    request_payload.setdefault("request_id", request_id)

    request = CompletionRequest(**request_payload)

    result = await handler.create_completion(request, raw_request=None)

    # Consume the generator if streaming
    if isinstance(result, AsyncIterator):
        async for _ in result:
            pass


async def _run_embedding_warmup(
    handler: OpenAIServingEmbedding,
    payload: dict[str, Any],
    model: str,
    request_id: str,
) -> None:
    """Run a single embedding warmup request."""
    request_payload = payload.copy()
    request_payload.setdefault("model", model)
    request_payload.setdefault("request_id", request_id)

    request = EmbeddingCompletionRequest(**request_payload)
    result = await handler.create_embedding(request, raw_request=None)

    # Consume if generator
    if isinstance(result, AsyncIterator):
        async for _ in result:
            pass


async def _run_pooling_warmup(
    handler: OpenAIServingPooling,
    payload: dict[str, Any],
    model: str,
    request_id: str,
) -> None:
    """Run a single pooling warmup request."""
    request_payload = payload.copy()
    request_payload.setdefault("model", model)
    request_payload.setdefault("request_id", request_id)

    request = PoolingCompletionRequest(**request_payload)
    result = await handler.create_pooling(request, raw_request=None)

    if isinstance(result, AsyncIterator):
        async for _ in result:
            pass


async def _run_classify_warmup(
    handler: ServingClassification,
    payload: dict[str, Any],
    model: str,
    request_id: str,
) -> None:
    """Run a single classification warmup request."""
    request_payload = payload.copy()
    request_payload.setdefault("model", model)

    request = ClassificationCompletionRequest(**request_payload)
    result = await handler.create_classify(request, raw_request=None)

    if isinstance(result, AsyncIterator):
        async for _ in result:
            pass


async def _run_score_warmup(
    handler: ServingScores,
    payload: dict[str, Any],
    model: str,
    request_id: str,
) -> None:
    """Run a single score warmup request."""
    request_payload = payload.copy()
    request_payload.setdefault("model", model)

    request = ScoreRequest(**request_payload)
    result = await handler.create_score(request, raw_request=None)

    if isinstance(result, AsyncIterator):
        async for _ in result:
            pass


async def _run_rerank_warmup(
    handler: ServingScores,
    payload: dict[str, Any],
    model: str,
    request_id: str,
) -> None:
    """Run a single rerank warmup request."""
    request_payload = payload.copy()
    request_payload.setdefault("model", model)

    request = RerankRequest(**request_payload)
    result = await handler.do_rerank(request, raw_request=None)

    if isinstance(result, AsyncIterator):
        async for _ in result:
            pass


async def _run_single_warmup(
    state: State,
    endpoint: str,
    payload: dict[str, Any],
    model: str,
    request_id: str,
) -> None:
    """Run a single warmup request to the specified endpoint."""
    if endpoint == "/v1/chat/completions":
        handler = state.openai_serving_chat
        if handler is None:
            raise NotImplementedError(
                f"Chat completions endpoint not available for this model. "
                f"Cannot run warmup for {endpoint}"
            )
        await _run_chat_completion_warmup(handler, payload, model, request_id)

    elif endpoint == "/v1/completions":
        handler = state.openai_serving_completion
        if handler is None:
            raise NotImplementedError(
                f"Completions endpoint not available for this model. "
                f"Cannot run warmup for {endpoint}"
            )
        await _run_completion_warmup(handler, payload, model, request_id)

    elif endpoint == "/v1/embeddings":
        handler = state.openai_serving_embedding
        if handler is None:
            raise NotImplementedError(
                f"Embeddings endpoint not available for this model. "
                f"Cannot run warmup for {endpoint}"
            )
        await _run_embedding_warmup(handler, payload, model, request_id)

    elif endpoint == "/pooling":
        handler = state.openai_serving_pooling
        if handler is None:
            raise NotImplementedError(
                f"Pooling endpoint not available for this model. "
                f"Cannot run warmup for {endpoint}"
            )
        await _run_pooling_warmup(handler, payload, model, request_id)

    elif endpoint == "/classify":
        handler = state.openai_serving_classification
        if handler is None:
            raise NotImplementedError(
                f"Classification endpoint not available for this model. "
                f"Cannot run warmup for {endpoint}"
            )
        await _run_classify_warmup(handler, payload, model, request_id)

    elif endpoint in ("/score", "/v1/score"):
        handler = state.openai_serving_scores
        if handler is None:
            raise NotImplementedError(
                f"Score endpoint not available for this model. "
                f"Cannot run warmup for {endpoint}"
            )
        await _run_score_warmup(handler, payload, model, request_id)

    elif endpoint in ("/rerank", "/v1/rerank", "/v2/rerank"):
        handler = state.openai_serving_scores
        if handler is None:
            raise NotImplementedError(
                f"Rerank endpoint not available for this model. "
                f"Cannot run warmup for {endpoint}"
            )
        await _run_rerank_warmup(handler, payload, model, request_id)

    else:
        raise NotImplementedError(
            f"Warmup for endpoint '{endpoint}' is not yet implemented. "
            f"Currently supported endpoints: {', '.join(sorted(SUPPORTED_ENDPOINTS))}"
        )


async def run_warmup(state: State, config_path: str) -> None:
    """
    Run warmup requests based on the configuration file.

    Args:
        state: The FastAPI app state containing serving handlers
        config_path: Path to the warmup configuration JSON file
    """
    logger.info("Loading warmup configuration from %s", config_path)
    config = WarmupConfig.from_file(config_path)

    if not config.requests:
        logger.warning("Warmup config has no requests, skipping warmup")
        return

    # Get the model name from the served models
    model = state.openai_serving_models.base_model_paths[0].name

    # Expand requests based on count
    all_requests: list[tuple[str, dict[str, Any]]] = []
    for req_config in config.requests:
        for _ in range(req_config.count):
            all_requests.append((req_config.endpoint, req_config.payload))

    total_requests = len(all_requests)
    logger.info(
        "Starting warmup: %d requests with concurrency %d",
        total_requests,
        config.concurrency,
    )

    start_time = time.perf_counter()
    completed = 0
    failed = 0

    # Create semaphore for concurrency control
    semaphore = asyncio.Semaphore(config.concurrency)

    async def run_with_semaphore(
        idx: int, endpoint: str, payload: dict[str, Any]
    ) -> bool:
        nonlocal completed, failed
        async with semaphore:
            request_id = f"warmup-{idx}"
            try:
                await _run_single_warmup(state, endpoint, payload, model, request_id)
                completed += 1
                return True
            except Exception as e:
                failed += 1
                logger.warning(
                    "Warmup request %d to %s failed: %s", idx, endpoint, str(e)
                )
                return False

    # Run all warmup requests
    tasks = [
        run_with_semaphore(idx, endpoint, payload)
        for idx, (endpoint, payload) in enumerate(all_requests)
    ]
    await asyncio.gather(*tasks)

    elapsed = time.perf_counter() - start_time
    logger.info(
        "Warmup completed: %d/%d successful in %.2fs (%.1f req/s)",
        completed,
        total_requests,
        elapsed,
        total_requests / elapsed if elapsed > 0 else 0,
    )

    if failed > 0:
        logger.warning("%d warmup requests failed", failed)
