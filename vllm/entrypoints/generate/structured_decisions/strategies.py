# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Read strategies: how a model is asked for label probabilities.

The server picks a strategy per model at startup: the first registered strategy
whose ``supports`` accepts the model serves every decision on that server. When
no strategy supports the model, the decision route answers 501.
"""

import math
from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator
from dataclasses import dataclass
from typing import Any

from vllm.config import ModelConfig
from vllm.engine.protocol import EngineClient
from vllm.entrypoints.chat_utils import ChatTemplateContentFormatOption
from vllm.lora.request import LoRARequest
from vllm.outputs import RequestOutput
from vllm.renderers.inputs.preprocess import extract_prompt_components
from vllm.renderers.online_renderer import OnlineRenderer
from vllm.sampling_params import MAX_LOGPROB_TOKEN_IDS, SamplingParams
from vllm.utils.async_utils import merge_async_iterators

from .protocol import ReadPromptRequest
from .question_types import Question, StructuredDecisionError, label_softmax
from .templates import DecisionTemplate


@dataclass
class ReadContext:
    engine_client: EngineClient
    online_renderer: OnlineRenderer
    chat_template: str | None
    chat_template_content_format: ChatTemplateContentFormatOption
    default_chat_template_kwargs: dict[str, Any]


@dataclass(frozen=True)
class DecisionLimits:
    max_questions: int
    max_options: int


@dataclass
class QuestionRead:
    probs: list[float]
    label_mass: float
    argmax_is_label: bool
    input_tokens: int
    output_tokens: int


class ReadStrategy(ABC):
    def __init__(self, context: ReadContext):
        self.context = context

    @classmethod
    @abstractmethod
    def supports(cls, model_config: ModelConfig) -> bool: ...

    @classmethod
    @abstractmethod
    def limits(cls, model_config: ModelConfig) -> DecisionLimits: ...

    @abstractmethod
    async def read(
        self,
        questions: list[Question],
        template: DecisionTemplate,
        instructions: str | None,
        state: str,
        *,
        request_id: str,
        chat_template_kwargs: dict[str, Any] | None,
        lora_request: LoRARequest | None,
        priority: int,
    ) -> list[QuestionRead]:
        """One read per question, in order. Raises StructuredDecisionError for
        a request the model cannot answer."""


READ_STRATEGIES: list[type[ReadStrategy]] = []


def register_read_strategy(cls: type[ReadStrategy]) -> type[ReadStrategy]:
    READ_STRATEGIES.append(cls)
    return cls


def select_read_strategy(model_config: ModelConfig) -> type[ReadStrategy] | None:
    return next((s for s in READ_STRATEGIES if s.supports(model_config)), None)


@register_read_strategy
class NextTokenStrategy(ReadStrategy):
    """Autoregressive models. Each question is one request: the chat prompt with
    the reply prefilled up to the question's label, one generated token, and the
    logprobs of the label tokens. The requests share the system prompt
    and the state, so prefix caching prefills them once."""

    @classmethod
    def supports(cls, model_config: ModelConfig) -> bool:
        return not model_config.is_diffusion

    @classmethod
    def limits(cls, model_config: ModelConfig) -> DecisionLimits:
        return DecisionLimits(max_questions=64, max_options=MAX_LOGPROB_TOKEN_IDS)

    async def read(
        self,
        questions: list[Question],
        template: DecisionTemplate,
        instructions: str | None,
        state: str,
        *,
        request_id: str,
        chat_template_kwargs: dict[str, Any] | None,
        lora_request: LoRARequest | None,
        priority: int,
    ) -> list[QuestionRead]:
        ctx = self.context
        tokenizer = ctx.online_renderer.renderer.get_tokenizer()
        rendered = template.render(instructions, questions)
        read_request = ReadPromptRequest(chat_template_kwargs=chat_template_kwargs)

        slots, engine_inputs = [], []
        for q in questions:
            slot = rendered.slot(tokenizer, q)
            messages = [
                {"role": "system", "content": rendered.system_text},
                {"role": "user", "content": state},
                {"role": "assistant", "content": tokenizer.decode(slot.prefix_ids)},
            ]
            _, (engine_input,) = await ctx.online_renderer.preprocess_chat(
                read_request,
                messages,
                default_template=ctx.chat_template,
                default_template_content_format=ctx.chat_template_content_format,
                default_template_kwargs=ctx.default_chat_template_kwargs,
            )
            prompt_ids = extract_prompt_components(
                ctx.engine_client.model_config, engine_input
            ).token_ids
            n = len(slot.prefix_ids)
            if not prompt_ids or list(prompt_ids[-n:]) != slot.prefix_ids:
                raise StructuredDecisionError(
                    f"question {q.id!r}: the chat template changed the answer "
                    "text before the label"
                )
            slots.append(slot)
            engine_inputs.append(engine_input)

        generators: list[AsyncGenerator[RequestOutput, None]] = [
            ctx.engine_client.generate(
                engine_input,
                SamplingParams(
                    max_tokens=1, temperature=0.0, logprob_token_ids=slot.label_ids
                ),
                f"{request_id}-{i}",
                lora_request=lora_request,
                priority=priority,
            )
            for i, (slot, engine_input) in enumerate(zip(slots, engine_inputs))
        ]
        results: list[RequestOutput | None] = [None] * len(generators)
        async for i, res in merge_async_iterators(*generators):
            results[i] = res

        reads = []
        for q, slot, result in zip(questions, slots, results):
            if result is None or not result.outputs or not result.outputs[0].logprobs:
                raise RuntimeError(f"question {q.id!r}: the read returned no logprobs")
            output = result.outputs[0]
            logprobs = output.logprobs[0]
            label_logprobs = [logprobs[t].logprob for t in slot.label_ids]
            reads.append(
                QuestionRead(
                    probs=label_softmax(label_logprobs),
                    label_mass=sum(math.exp(lp) for lp in label_logprobs),
                    argmax_is_label=bool(output.token_ids)
                    and output.token_ids[0] in slot.label_ids,
                    input_tokens=len(result.prompt_token_ids or ()),
                    output_tokens=len(output.token_ids),
                )
            )
        return reads
