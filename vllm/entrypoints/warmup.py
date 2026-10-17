# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pre-serve warmup utilities for vLLM.

This module provides functionality to warm up the vLLM engine with actual
requests before the server starts accepting traffic. This helps avoid the
~5 minute lag that occurs when kernels compile lazily on first real request.

Supports multiple endpoints:
- /v1/completions (task=generate, prompt field)
- /v1/chat/completions (task=generate, messages field)
- /v1/embeddings (task=embed, input field)
"""

import asyncio
import dataclasses
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, get_args

from vllm.exceptions import VLLMValidationError
from vllm.logger import init_logger
from vllm.pooling_params import PoolingParams
from vllm.sampling_params import SamplingParams
from vllm.utils import random_uuid

if TYPE_CHECKING:
    from vllm.engine.protocol import EngineClient
    from vllm.renderers.online_renderer import OnlineRenderer

logger = init_logger(__name__)

WarmupTask = Literal["generate", "embed"]


@dataclass
class WarmupPrompt:
    """A single warmup request.

    Supports multiple input styles to mirror different OpenAI endpoints:

    - ``prompt`` for ``/v1/completions``
    - ``messages`` for ``/v1/chat/completions``
    - ``input`` for ``/v1/embeddings``

    Exactly one of ``prompt``, ``messages``, or ``input`` must be provided.
    """

    prompt: str | None = None
    messages: list[dict[str, Any]] | None = None
    input: str | list[str] | None = None
    max_tokens: int = 256


@dataclass
class WarmupConfig:
    """Configuration for engine warmup.

    Args:
        prompts: List of warmup requests.
        task: Which engine task to exercise. ``"generate"`` exercises
            ``/v1/completions`` and ``/v1/chat/completions``.
            ``"embed"`` exercises ``/v1/embeddings``.
        concurrency: Concurrency levels to sweep.
        request_params: Extra SamplingParams or PoolingParams kwargs
            merged into every warmup request.

    """

    prompts: list[WarmupPrompt]
    task: WarmupTask = "generate"
    concurrency: list[int] = field(default_factory=lambda: [1])
    request_params: dict[str, Any] = field(default_factory=dict)


def load_warmup_config(
    path_or_json: str | dict[str, Any] | WarmupConfig | None,
) -> WarmupConfig | None:
    """Parse and validate a warmup config from a file path, JSON string,
    or dict. An already-parsed `WarmupConfig` is returned unchanged.

    Raises:
        ValueError: If the configuration is malformed.

    """
    if path_or_json is None or isinstance(path_or_json, WarmupConfig):
        return path_or_json

    if isinstance(path_or_json, dict):
        config_dict = path_or_json
    elif path_or_json.lstrip().startswith("{"):
        config_dict = json.loads(path_or_json)
    else:
        config_dict = json.loads(Path(path_or_json).read_text())

    if not isinstance(config_dict, dict):
        raise ValueError("Warmup config must be a JSON object")
    allowed_keys = {f.name for f in dataclasses.fields(WarmupConfig)}
    if unknown_keys := config_dict.keys() - allowed_keys:
        raise ValueError(
            f"Unknown warmup config key(s) {sorted(unknown_keys)}; "
            f"expected a subset of {sorted(allowed_keys)}"
        )

    task = config_dict.get("task", "generate")
    if task not in get_args(WarmupTask):
        raise ValueError(
            f"Invalid warmup task {task!r}; expected one of {get_args(WarmupTask)}"
        )

    raw_prompts = config_dict.get("prompts")
    if not isinstance(raw_prompts, list) or not raw_prompts:
        raise ValueError("Warmup config 'prompts' must be a non-empty list")
    prompts = [_parse_prompt(p, task) for p in raw_prompts]

    raw_concurrency = config_dict.get("concurrency", 1)
    concurrency = (
        [raw_concurrency] if isinstance(raw_concurrency, int) else raw_concurrency
    )
    if (
        not isinstance(concurrency, list)
        or not concurrency
        or not all(isinstance(c, int) and c > 0 for c in concurrency)
    ):
        raise ValueError(
            "Warmup config 'concurrency' must be a positive int or a "
            f"non-empty list of positive ints, got {raw_concurrency!r}"
        )

    request_params = config_dict.get("request_params", {})
    if not isinstance(request_params, dict):
        raise ValueError("Warmup config 'request_params' must be a JSON object")
    if task == "generate" and "max_tokens" in request_params:
        raise ValueError(
            "Set 'max_tokens' on each warmup prompt, not in 'request_params'"
        )
    # Fail fast on invalid params instead of after the model has loaded.
    try:
        if task == "embed":
            _make_pooling_params(request_params)
        else:
            SamplingParams(**request_params)
    except (TypeError, ValueError, VLLMValidationError) as e:
        raise ValueError(f"Invalid warmup 'request_params': {e}") from e

    return WarmupConfig(
        prompts=prompts,
        task=task,
        concurrency=concurrency,
        request_params=request_params,
    )


def _parse_prompt(raw: Any, task: WarmupTask) -> WarmupPrompt:
    if not isinstance(raw, dict):
        raise ValueError(f"Warmup prompt must be a JSON object, got {raw!r}")
    allowed_keys = {f.name for f in dataclasses.fields(WarmupPrompt)}
    if unknown_keys := raw.keys() - allowed_keys:
        raise ValueError(
            f"Unknown warmup prompt key(s) {sorted(unknown_keys)}; "
            f"expected a subset of {sorted(allowed_keys)}"
        )
    prompt = WarmupPrompt(**raw)

    inputs = [k for k in ("prompt", "messages", "input") if raw.get(k) is not None]
    if len(inputs) != 1:
        raise ValueError(
            "Warmup prompt must set exactly one of 'prompt', 'messages', or "
            f"'input', got {inputs or 'none'}"
        )
    if not raw[inputs[0]]:
        raise ValueError(f"Warmup prompt '{inputs[0]}' must be non-empty")
    if task == "embed" and prompt.messages is not None:
        raise ValueError("Warmup prompt 'messages' is not supported for task 'embed'")
    if task == "generate" and prompt.input is not None:
        raise ValueError(
            "Warmup prompt 'input' requires task 'embed'; use 'prompt' or "
            "'messages' for task 'generate'"
        )
    if prompt.prompt is not None and not isinstance(prompt.prompt, str):
        raise ValueError("Warmup prompt 'prompt' must be a string")
    if prompt.messages is not None and not isinstance(prompt.messages, list):
        raise ValueError("Warmup prompt 'messages' must be a list of messages")
    if prompt.input is not None and not (
        isinstance(prompt.input, str)
        or (
            isinstance(prompt.input, list)
            and all(isinstance(s, str) for s in prompt.input)
        )
    ):
        raise ValueError("Warmup prompt 'input' must be a string or list of strings")
    if not isinstance(prompt.max_tokens, int) or prompt.max_tokens <= 0:
        raise ValueError(
            f"Warmup prompt 'max_tokens' must be a positive int, "
            f"got {prompt.max_tokens!r}"
        )
    return prompt


def _make_pooling_params(request_params: dict[str, Any]) -> PoolingParams:
    # Match /v1/embeddings, which always pools with task="embed".
    return PoolingParams(**{"task": "embed", **request_params})


async def warmup_engine(
    engine_client: "EngineClient",
    online_renderer: "OnlineRenderer",
    config: WarmupConfig,
) -> None:
    """Run every configured warmup item at each concurrency level.

    At each level, ``max(concurrency, num_items)`` requests are issued with at
    most ``concurrency`` in flight, so all items run and the level is fully
    exercised even when there are fewer items than the concurrency.
    """
    items: list[WarmupPrompt | str]
    if config.task == "embed":
        # /v1/embeddings treats each string of a list input as its own prompt.
        items = []
        for p in config.prompts:
            text = p.input if p.input is not None else p.prompt
            assert text is not None
            if isinstance(text, list):
                items.extend(text)
            else:
                items.append(text)
    else:
        items = list(config.prompts)

    logger.info(
        "Starting engine warmup: task=%s, %d item(s), concurrency=%s",
        config.task,
        len(items),
        config.concurrency,
    )

    for concurrency in config.concurrency:
        num_requests = max(concurrency, len(items))
        logger.info(
            "Warming up %s with concurrency=%d, %d request(s)",
            config.task,
            concurrency,
            num_requests,
        )
        semaphore = asyncio.Semaphore(concurrency)
        await asyncio.gather(
            *(
                _warmup_one(
                    engine_client,
                    online_renderer,
                    items[i % len(items)],
                    config,
                    semaphore,
                )
                for i in range(num_requests)
            )
        )

    logger.info("Engine warmup completed")


async def _warmup_one(
    engine_client: "EngineClient",
    online_renderer: "OnlineRenderer",
    item: WarmupPrompt | str,
    config: WarmupConfig,
    semaphore: asyncio.Semaphore,
) -> None:
    async with semaphore:
        await _run_request(engine_client, online_renderer, item, config)


async def _run_request(
    engine_client: "EngineClient",
    online_renderer: "OnlineRenderer",
    item: WarmupPrompt | str,
    config: WarmupConfig,
) -> None:
    request_id = f"warmup-{random_uuid()}"
    if isinstance(item, str):
        stream = engine_client.encode(
            prompt=item,
            pooling_params=_make_pooling_params(config.request_params),
            request_id=request_id,
        )
    else:
        engine_input = await _render(online_renderer, item)
        params = SamplingParams(max_tokens=item.max_tokens, **config.request_params)
        stream = engine_client.generate(  # type: ignore[assignment]
            prompt=engine_input, sampling_params=params, request_id=request_id
        )

    async for _ in stream:
        pass


async def _render(online_renderer: "OnlineRenderer", item: WarmupPrompt) -> Any:
    """Render a warmup prompt exactly as the matching OpenAI endpoint would,
    using the server's chat template, content format and default kwargs."""
    from vllm.entrypoints.openai.chat_completion.protocol import (
        ChatCompletionRequest,
    )
    from vllm.entrypoints.openai.completion.protocol import CompletionRequest
    from vllm.entrypoints.serve.engine.protocol import ErrorResponse

    result: Any
    if item.messages is not None:
        result = await online_renderer.render_chat(
            ChatCompletionRequest(
                messages=item.messages, max_completion_tokens=item.max_tokens
            )
        )
        if not isinstance(result, ErrorResponse):
            result = result[1]
    else:
        result = await online_renderer.render_completion(
            CompletionRequest(prompt=item.prompt, max_tokens=item.max_tokens)
        )
    if isinstance(result, ErrorResponse):
        raise ValueError(f"Failed to render warmup prompt: {result.error.message}")
    return result[0]
