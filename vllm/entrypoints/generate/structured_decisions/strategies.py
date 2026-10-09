# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Read strategies: how a model is asked for label probabilities.

Only models in NEXT_TOKEN_ARCHITECTURES are currently supported.
"""

import math
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from vllm.config import ModelConfig
from vllm.engine.protocol import EngineClient
from vllm.entrypoints.chat_utils import ChatTemplateContentFormatOption
from vllm.entrypoints.generate.label_reads import next_token_label_reads
from vllm.inputs import EngineInput
from vllm.lora.request import LoRARequest
from vllm.renderers.inputs.preprocess import extract_prompt_components
from vllm.renderers.online_renderer import OnlineRenderer
from vllm.sampling_params import SamplingParams
from vllm.tokenizers import TokenizerLike

from .protocol import ReadPromptRequest
from .question_types import (
    LABELS,
    QUESTION_TYPES,
    Question,
    StructuredDecisionError,
    label_softmax,
)


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
        instructions: str | None,
        state: str,
        *,
        request_id: str,
        chat_template_kwargs: dict[str, Any] | None,
        lora_request: LoRARequest | None,
        priority: int,
    ) -> list[QuestionRead]:
        """One read per question, in order."""


def reply_label_ids(
    tokenizer: TokenizerLike,
    prompt_ids: Sequence[int],
    labels: Sequence[str] = LABELS,
) -> tuple[list[int], list[int]]:
    """The prompt's ids after its last added token, and the token each of
    ``labels`` adds after them as the first token of the reply. Raises
    ValueError if a label is not one distinct token there."""
    added = set(tokenizer.get_added_vocab().values())
    start = max((i + 1 for i, t in enumerate(prompt_ids) if t in added), default=0)
    tail = list(prompt_ids[start:])
    text = tokenizer.decode(tail)
    ids: list[int] = []
    for label in labels:
        extended = tokenizer.encode(text + label, add_special_tokens=False)
        if extended[:-1] != tail or extended[-1] in ids:
            raise ValueError(
                f"label {label!r} is not one distinct token after this model's "
                "chat prompt"
            )
        ids.append(extended[-1])
    return tail, ids


class NextTokenStrategy(ReadStrategy):
    """Autoregressive models. Each question is one request: the state, then the
    question with its labeled options, and the logprobs of the label tokens as
    the reply's first token. The requests share the state. With prefix caching,
    it is prefilled once."""

    def __init__(self, context: ReadContext):
        super().__init__(context)
        tokenizer = context.online_renderer.renderer.get_tokenizer()
        probe = tokenizer.apply_chat_template(
            [{"role": "user", "content": "x"}],
            **{
                **context.default_chat_template_kwargs,
                "chat_template": context.chat_template,
                "add_generation_prompt": True,
                "tokenize": True,
                "return_dict": False,
                "enable_thinking": False,
            },
        )
        if isinstance(probe, str):
            probe = tokenizer.encode(probe, add_special_tokens=False)
        # Every prompt ends with the same generation prompt, so a label's token
        # is the same for every question.
        self.tail, _ = reply_label_ids(tokenizer, probe, ())
        self.label_ids: dict[str, int] = {}
        for qtype in QUESTION_TYPES.values():
            _, ids = reply_label_ids(tokenizer, probe, qtype.label_set)
            self.label_ids.update(zip(qtype.label_set, ids))

    def limits(self) -> DecisionLimits:
        return DecisionLimits(max_questions=64, max_options=len(LABELS))

    def _read_request(
        self, chat_template_kwargs: dict[str, Any] | None
    ) -> ReadPromptRequest:
        if (chat_template_kwargs or {}).get("enable_thinking"):
            raise StructuredDecisionError(
                "a read is the reply's first token, so thinking must be off"
            )
        return ReadPromptRequest(chat_template_kwargs=chat_template_kwargs)

    async def _render(
        self, read_request: ReadPromptRequest, messages: list[dict[str, Any]]
    ) -> tuple[EngineInput, list[int]]:
        ctx = self.context
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
        return engine_input, list(prompt_ids or [])

    async def read(
        self,
        questions: list[Question],
        instructions: str | None,
        state: str,
        *,
        request_id: str,
        chat_template_kwargs: dict[str, Any] | None,
        lora_request: LoRARequest | None,
        priority: int,
    ) -> list[QuestionRead]:
        ctx = self.context
        read_request = self._read_request(chat_template_kwargs)

        slots, engine_inputs = [], []
        for q in questions:
            messages = [{"role": "user", "content": f"{state}\n\n{q.type.prompt(q)}"}]
            if instructions:
                messages.insert(0, {"role": "system", "content": instructions})
            engine_input, prompt_ids = await self._render(read_request, messages)
            if prompt_ids[len(prompt_ids) - len(self.tail) :] != self.tail:
                raise StructuredDecisionError(
                    "these chat options end the prompt differently, so the "
                    "labels' tokens are unknown"
                )
            slots.append([self.label_ids[label] for label in q.labels])
            engine_inputs.append(engine_input)

        label_reads = await next_token_label_reads(
            ctx.engine_client,
            engine_inputs,
            [
                SamplingParams(max_tokens=1, temperature=0.0, logprob_token_ids=ids)
                for ids in slots
            ],
            request_id,
            lora_request=lora_request,
            priority=priority,
        )

        reads = []
        for ids, label_read in zip(slots, label_reads):
            output = label_read.result.outputs[0]
            reads.append(
                QuestionRead(
                    probs=label_softmax(label_read.logprobs),
                    label_mass=sum(math.exp(lp) for lp in label_read.logprobs),
                    argmax_is_label=bool(output.token_ids)
                    and output.token_ids[0] in ids,
                    input_tokens=len(label_read.result.prompt_token_ids or ()),
                    output_tokens=len(output.token_ids),
                )
            )
        return reads


NEXT_TOKEN_ARCHITECTURES = frozenset(
    {
        "Qwen3ForCausalLM",
        "Qwen3_5ForConditionalGeneration",
        "Qwen3_5MoeForConditionalGeneration",
    }
)


# label_mass sums the labels' full-vocabulary probabilities. The logprobs modes
# return those probabilities. The logits modes return raw logits instead.
LOGPROBS_MODES = frozenset({"raw_logprobs", "processed_logprobs"})


def select_read_strategy(model_config: ModelConfig) -> type[ReadStrategy]:
    """Raise ValueError when the model cannot serve structured decisions."""
    if model_config.architecture not in NEXT_TOKEN_ARCHITECTURES:
        raise ValueError(
            "The structured decisions API does not support "
            f"{model_config.architecture}. Supported architectures: "
            f"{sorted(NEXT_TOKEN_ARCHITECTURES)}"
        )
    if model_config.logprobs_mode not in LOGPROBS_MODES:
        raise ValueError(
            "Structured decisions need --logprobs-mode raw_logprobs or "
            f"processed_logprobs, not {model_config.logprobs_mode}"
        )
    return NextTokenStrategy
