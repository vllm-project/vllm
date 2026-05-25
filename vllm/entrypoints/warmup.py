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
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from vllm.logger import init_logger
from vllm.pooling_params import PoolingParams
from vllm.renderers import ChatParams
from vllm.sampling_params import SamplingParams

if TYPE_CHECKING:
    from vllm.engine.protocol import EngineClient

logger = init_logger(__name__)


@dataclass
class WarmupPrompt:
    """A single warmup request.

    Supports multiple input styles to mirror different OpenAI endpoints:

    - ``prompt`` for ``/v1/completions``
    - ``messages`` for ``/v1/chat/completions``
    - ``input`` for ``/v1/embeddings``

    Only one of ``prompt``, ``messages``, or ``input`` should be provided.
    If none are set, an empty prompt is used.
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
    task: str = "generate"
    concurrency: list[int] = field(default_factory=lambda: [1])
    request_params: dict[str, Any] = field(default_factory=dict)


def load_warmup_config(
    path_or_json: str | dict[str, Any] | None,
) -> WarmupConfig | None:
    if path_or_json is None:
        return None

    if isinstance(path_or_json, dict):
        config_dict = path_or_json
    elif path_or_json.lstrip().startswith("{"):
        config_dict = json.loads(path_or_json)
    else:
        config_dict = json.loads(Path(path_or_json).read_text())

    raw_prompts = config_dict.get("prompts", [])
    prompts = [WarmupPrompt(**p) if isinstance(p, dict) else p for p in raw_prompts]

    raw_concurrency = config_dict.get("concurrency", 1)
    concurrency = (
        [raw_concurrency] if isinstance(raw_concurrency, int) else list(raw_concurrency)
    )

    return WarmupConfig(
        prompts=prompts,
        task=config_dict.get("task", "generate"),
        concurrency=concurrency,
        request_params=config_dict.get("request_params", {}),
    )


async def warmup_engine(
    engine_client: "EngineClient",
    config: WarmupConfig,
) -> None:
    logger.info(
        "Starting engine warmup: task=%s, %d prompt(s)",
        config.task,
        len(config.prompts),
    )

    if not config.prompts:
        return

    is_embed = config.task == "embed"
    for concurrency in config.concurrency:
        logger.info(
            "Warming up %s with concurrency=%d, %d item(s)",
            config.task,
            concurrency,
            len(config.prompts),
        )
        await asyncio.gather(
            *(
                _warmup_one(
                    engine_client,
                    config.prompts[i % len(config.prompts)],
                    config.request_params,
                    is_embed,
                    concurrency,
                    i,
                )
                for i in range(concurrency)
            )
        )

    logger.info("Engine warmup completed")


async def _warmup_one(
    engine_client: "EngineClient",
    prompt_config: WarmupPrompt,
    request_params: dict[str, Any],
    is_embed: bool,
    concurrency: int,
    idx: int,
) -> None:
    if is_embed:
        request_id = f"warmup_embed_{id(prompt_config)}_{concurrency}_{idx}"
        prompt = prompt_config.prompt or prompt_config.input or ""
        pooling_params = PoolingParams(**request_params)
        stream = engine_client.encode(
            prompt=prompt, pooling_params=pooling_params, request_id=request_id
        )
    else:
        request_id = f"warmup_{id(prompt_config)}_{concurrency}_{idx}"
        if prompt_config.prompt is not None:
            prompt = prompt_config.prompt
        elif prompt_config.messages is not None:
            prompt = await _render_messages(engine_client, prompt_config.messages)
        else:
            prompt = ""
        params = SamplingParams(max_tokens=prompt_config.max_tokens, **request_params)
        stream = engine_client.generate(
            prompt=prompt, sampling_params=params, request_id=request_id
        )

    async for _ in stream:
        pass


async def _render_messages(
    engine_client: "EngineClient",
    messages: list[dict[str, Any]],
) -> Any:
    """Convert a list of chat messages to an engine input object."""
    _, engine_inputs = await engine_client.renderer.render_chat_async(
        [messages],
        ChatParams(),
    )
    return engine_inputs[0]
