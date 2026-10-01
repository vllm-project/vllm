# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Read strategies: how a model is asked for label probabilities.

Diffusion models have no read strategy yet, so the decision route returns 501
for them.
"""

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from vllm.config import ModelConfig
from vllm.engine.protocol import EngineClient
from vllm.entrypoints.chat_utils import ChatTemplateContentFormatOption
from vllm.entrypoints.generate.label_reads import next_token_label_reads
from vllm.lora.request import LoRARequest
from vllm.renderers.inputs.preprocess import extract_prompt_components
from vllm.renderers.online_renderer import OnlineRenderer
from vllm.sampling_params import MAX_LOGPROB_TOKEN_IDS, SamplingParams

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

    @abstractmethod
    def limits(self) -> DecisionLimits: ...

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
        """One read per question, in order."""


class NextTokenStrategy(ReadStrategy):
    """Autoregressive models. Each question is one request: the state, then the
    template rendered for that question alone, the reply prefilled up to the
    question's label, one generated token, and the logprobs of the label tokens.
    The requests share the state. With prefix caching, it is prefilled once."""

    def limits(self) -> DecisionLimits:
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
        read_request = ReadPromptRequest(chat_template_kwargs=chat_template_kwargs)

        slots, engine_inputs = [], []
        for q in questions:
            # With every question in one prompt, later questions lost accuracy
            # (Qwen3-0.6B: 81% at the first, 60% at the fourth).
            rendered = template.render(instructions, [q])
            slot = rendered.slot(tokenizer, q)
            messages = [
                {"role": "user", "content": f"{state}\n\n{rendered.text}"},
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

        label_reads = await next_token_label_reads(
            ctx.engine_client,
            engine_inputs,
            [
                SamplingParams(
                    max_tokens=1, temperature=0.0, logprob_token_ids=slot.label_ids
                )
                for slot in slots
            ],
            request_id,
            lora_request=lora_request,
            priority=priority,
        )

        reads = []
        for slot, label_read in zip(slots, label_reads):
            output = label_read.result.outputs[0]
            reads.append(
                QuestionRead(
                    probs=label_softmax(label_read.logprobs),
                    label_mass=sum(math.exp(lp) for lp in label_read.logprobs),
                    argmax_is_label=bool(output.token_ids)
                    and output.token_ids[0] in slot.label_ids,
                    input_tokens=len(label_read.result.prompt_token_ids or ()),
                    output_tokens=len(output.token_ids),
                )
            )
        return reads


def select_read_strategy(model_config: ModelConfig) -> type[ReadStrategy] | None:
    return None if model_config.is_diffusion else NextTokenStrategy
