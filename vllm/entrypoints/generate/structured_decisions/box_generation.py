# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Coordinate generation through a served autoregressive vision model."""

import asyncio
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from vllm.config import ModelConfig
from vllm.entrypoints.openai.decisions.strategies import (
    NextTokenStrategy,
    ReadPromptRequest,
)
from vllm.logger import init_logger
from vllm.lora.request import LoRARequest
from vllm.renderers import TokenizeParams
from vllm.sampling_params import SamplingParams, StructuredOutputsParams

from .box_contract import (
    BOX_MAX_TOKENS,
    BOX_SCHEMA,
    BoxOutputError,
    BoxQuestion,
    parse_box,
)

logger = init_logger(__name__)


class BoxPromptRequest(ReadPromptRequest):
    def build_tok_params(self, model_config: ModelConfig) -> TokenizeParams:
        return TokenizeParams(
            max_total_tokens=model_config.max_model_len,
            max_output_tokens=BOX_MAX_TOKENS,
            add_special_tokens=False,
        )


@dataclass
class BoxRead:
    answer: dict[str, Any]
    input_tokens: int
    output_tokens: int


async def generate_boxes(
    strategy: NextTokenStrategy,
    questions: list[BoxQuestion],
    image: str,
    instructions: str | None,
    state: str,
    *,
    request_id: str,
    chat_template_kwargs: dict[str, Any] | None,
    lora_request: LoRARequest | None,
    priority: int,
    cache_salt: str | None,
    trace_headers: Mapping[str, str] | None,
) -> list[BoxRead]:
    request = BoxPromptRequest(
        **strategy._read_request(chat_template_kwargs, cache_salt).model_dump()
    )
    image_parts = await strategy._image_parts([image])
    rendered = []
    for question in questions:
        messages: list[dict[str, Any]] = [
            {
                "role": "user",
                "content": [
                    *image_parts,
                    {"type": "text", "text": question.prompt(state)},
                ],
            }
        ]
        if instructions:
            messages.insert(0, {"role": "system", "content": instructions})
        rendered.append(await strategy._render(request, messages))

    engine = strategy.context.engine_client
    reads = []
    for n, (engine_input, prompt_ids) in enumerate(rendered):
        rid = f"{request_id}-box-{n}"
        params = SamplingParams(
            max_tokens=BOX_MAX_TOKENS,
            temperature=0.0,
            structured_outputs=StructuredOutputsParams(json=BOX_SCHEMA),
        )
        result = None
        complete = False
        try:
            async for result in engine.generate(
                engine_input,
                params,
                rid,
                lora_request=lora_request,
                priority=priority,
                trace_headers=trace_headers,
            ):
                pass
            complete = result is not None and result.finished
            if not complete or result is None or len(result.outputs) != 1:
                raise BoxOutputError("box generation returned no completed answer")
            output = result.outputs[0]
            if output.finish_reason == "length":
                raise BoxOutputError("box generation exceeded its token budget")
            if output.finish_reason != "stop":
                raise BoxOutputError("box generation did not finish normally")
            reads.append(
                BoxRead(
                    answer=parse_box(output.text),
                    input_tokens=len(prompt_ids),
                    output_tokens=len(output.token_ids),
                )
            )
        finally:
            if not complete:
                try:
                    await asyncio.shield(engine.abort(rid))
                except Exception:
                    logger.warning("Could not abort box request %s", rid, exc_info=True)
    return reads
