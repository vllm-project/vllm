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

    def get_input_text(self) -> str | list[str] | None:
        """Return the raw text input for this warmup item."""
        if self.prompt is not None:
            return self.prompt
        if self.input is not None:
            return self.input
        if self.messages is not None:
            # Messages need renderer conversion; caller must handle
            return None
        return ""


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
        config_dict = dict(path_or_json)
    else:
        path = Path(path_or_json)
        if path.exists():
            config_dict = json.loads(path.read_text())
        else:
            config_dict = json.loads(path_or_json)

    raw_prompts = config_dict.get("prompts", [])
    prompts = [WarmupPrompt(**p) if isinstance(p, dict) else p for p in raw_prompts]

    raw_concurrency = config_dict.get("concurrency", 1)
    concurrency: list[int]
    if isinstance(raw_concurrency, int):
        concurrency = [raw_concurrency]
    else:
        concurrency = list(raw_concurrency)

    return WarmupConfig(
        prompts=prompts,
        task=config_dict.get("task", "generate"),
        concurrency=concurrency,
        request_params=config_dict.get("request_params", {}),
    )


def _get_sampling_params(
    prompt: WarmupPrompt,
    request_params: dict[str, Any],
) -> SamplingParams:
    return SamplingParams(
        max_tokens=prompt.max_tokens,
        **request_params,
    )


def _get_pooling_params(
    request_params: dict[str, Any],
) -> PoolingParams:
    return PoolingParams(**request_params)


async def warmup_engine(
    engine_client: "EngineClient",
    config: WarmupConfig,
) -> None:
    logger.info(
        "Starting engine warmup: task=%s, %d prompt(s)",
        config.task,
        len(config.prompts),
    )

    if config.task == "embed":
        await _warmup_embed(engine_client, config)
    else:
        await _warmup_generate(engine_client, config)

    logger.info("Engine warmup completed")


async def _warmup_generate(
    engine_client: "EngineClient",
    config: WarmupConfig,
) -> None:
    if not config.prompts:
        return

    for concurrency in config.concurrency:
        logger.info(
            "Warming up generate with concurrency=%d, %d item(s)",
            concurrency,
            len(config.prompts),
        )

        tasks = [
            _warmup_generate_one(
                engine_client,
                config.prompts[i % len(config.prompts)],
                config.request_params,
                concurrency,
                i,
            )
            for i in range(concurrency)
        ]
        await asyncio.gather(*tasks)


async def _warmup_generate_one(
    engine_client: "EngineClient",
    prompt_config: WarmupPrompt,
    request_params: dict[str, Any],
    concurrency: int,
    idx: int,
) -> None:
    sampling_params = _get_sampling_params(prompt_config, request_params)
    request_id = f"warmup_{id(prompt_config)}_{concurrency}_{idx}"

    prompt: Any = ""

    if prompt_config.prompt is not None:
        prompt = prompt_config.prompt
    elif prompt_config.messages is not None:
        prompt = await _render_messages(
            engine_client,
            prompt_config.messages,
        )

    async for _ in engine_client.generate(
        prompt=prompt,
        sampling_params=sampling_params,
        request_id=request_id,
    ):
        pass


async def _warmup_embed(
    engine_client: "EngineClient",
    config: WarmupConfig,
) -> None:
    if not config.prompts:
        return

    for concurrency in config.concurrency:
        logger.info(
            "Warming up embed with concurrency=%d, %d item(s)",
            concurrency,
            len(config.prompts),
        )

        tasks = [
            _warmup_embed_one(
                engine_client,
                config.prompts[i % len(config.prompts)],
                config.request_params,
                concurrency,
                i,
            )
            for i in range(concurrency)
        ]
        await asyncio.gather(*tasks)


async def _warmup_embed_one(
    engine_client: "EngineClient",
    prompt_config: WarmupPrompt,
    request_params: dict[str, Any],
    concurrency: int,
    idx: int,
) -> None:
    pooling_params = _get_pooling_params(request_params)
    request_id = f"warmup_embed_{id(prompt_config)}_{concurrency}_{idx}"
    prompt = prompt_config.get_input_text() or ""

    async for _ in engine_client.encode(
        prompt=prompt,
        pooling_params=pooling_params,
        request_id=request_id,
    ):
        pass


async def _render_messages(
    engine_client: "EngineClient",
    messages: list[dict[str, Any]],
) -> Any:
    """Convert a list of chat messages to an engine input object."""
    from vllm.renderers import ChatParams

    renderer = engine_client.renderer
    chat_params = ChatParams()

    _, engine_inputs = await renderer.render_chat_async(
        [messages],
        chat_params,
    )

    return engine_inputs[0]
