# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Read strategies: how a model is asked for label probabilities.

Only models in NEXT_TOKEN_ARCHITECTURES are currently supported.
"""

import json
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
from vllm.logger import init_logger
from vllm.lora.request import LoRARequest
from vllm.renderers.inputs.preprocess import extract_prompt_components
from vllm.renderers.online_renderer import OnlineRenderer
from vllm.sampling_params import MAX_LOGPROB_TOKEN_IDS, SamplingParams
from vllm.tokenizers import TokenizerLike

from .protocol import ReadPromptRequest
from .question_types import LABELS, Question, StructuredDecisionError, label_softmax

logger = init_logger(__name__)


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
    async def label_pool(
        self, chat_template_kwargs: dict[str, Any] | None
    ) -> tuple[str, ...]:
        """The labels questions take in order, for these chat options."""

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


def generation_tail(
    tokenizer: TokenizerLike, prompt_ids: Sequence[int]
) -> tuple[list[int], str]:
    """The prompt after its last added token. Labels are checked against it, so
    the check does not grow with the state."""
    added = set(tokenizer.get_added_vocab().values())
    start = max((i + 1 for i, t in enumerate(prompt_ids) if t in added), default=0)
    tail_ids = list(prompt_ids[start:])
    return tail_ids, tokenizer.decode(tail_ids)


def label_token_id(
    tokenizer: TokenizerLike, tail_ids: list[int], tail: str, label: str
) -> int | None:
    """The one token ``label`` adds after the tail, or None."""
    ids = tokenizer.encode(tail + label, add_special_tokens=False)
    return ids[-1] if ids[:-1] == tail_ids else None


def single_token_labels(
    tokenizer: TokenizerLike, prompt_ids: Sequence[int]
) -> tuple[str, ...]:
    """The candidates in LABELS that are one distinct token at the start of the
    reply, in order."""
    tail_ids, tail = generation_tail(tokenizer, prompt_ids)
    pool, seen = [], set()
    for label in LABELS:
        token = label_token_id(tokenizer, tail_ids, tail, label)
        if token is not None and token not in seen:
            pool.append(label)
            seen.add(token)
    return tuple(pool)


def label_token_ids(
    tokenizer: TokenizerLike, prompt_ids: Sequence[int], question: Question
) -> list[int]:
    """The token each label adds as the reply's first token."""
    tail_ids, tail = generation_tail(tokenizer, prompt_ids)
    ids: list[int] = []
    for label in question.labels:
        token = label_token_id(tokenizer, tail_ids, tail, label)
        if token is None or token in ids:
            raise StructuredDecisionError(
                f"question {question.id!r}: label {label!r} is not one distinct "
                "token after this model's chat prompt"
            )
        ids.append(token)
    return ids


class NextTokenStrategy(ReadStrategy):
    """Autoregressive models. Each question is one request: the state, then the
    question with its labeled options, and the logprobs of the label tokens as
    the reply's first token. The requests share the state. With prefix caching,
    it is prefilled once."""

    # Label pools cached by chat options, which requests choose.
    MAX_POOLS = 16

    def __init__(self, context: ReadContext):
        super().__init__(context)
        self._pools: dict[str, tuple[str, ...]] = {}

    def limits(self) -> DecisionLimits:
        return DecisionLimits(max_questions=64, max_options=MAX_LOGPROB_TOKEN_IDS)

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

    async def label_pool(
        self, chat_template_kwargs: dict[str, Any] | None
    ) -> tuple[str, ...]:
        read_request = self._read_request(chat_template_kwargs)
        key = json.dumps(chat_template_kwargs or {}, sort_keys=True, default=str)
        if key not in self._pools:
            _, prompt_ids = await self._render(
                read_request, [{"role": "user", "content": "x"}]
            )
            tokenizer = self.context.online_renderer.renderer.get_tokenizer()
            pool = single_token_labels(tokenizer, prompt_ids)
            if len(self._pools) >= self.MAX_POOLS:
                return pool
            self._pools[key] = pool
        return self._pools[key]

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
        tokenizer = ctx.online_renderer.renderer.get_tokenizer()
        read_request = self._read_request(chat_template_kwargs)

        slots, engine_inputs = [], []
        for q in questions:
            messages = [{"role": "user", "content": f"{state}\n\n{q.type.prompt(q)}"}]
            if instructions:
                messages.insert(0, {"role": "system", "content": instructions})
            engine_input, prompt_ids = await self._render(read_request, messages)
            slots.append(label_token_ids(tokenizer, prompt_ids, q))
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
    {"Qwen3ForCausalLM", "Qwen3_5MoeForConditionalGeneration"}
)


# label_mass sums the labels' full-vocabulary probabilities. The logprobs modes
# return those probabilities. The logits modes return raw logits instead.
LOGPROBS_MODES = frozenset({"raw_logprobs", "processed_logprobs"})


def select_read_strategy(model_config: ModelConfig) -> type[ReadStrategy] | None:
    if model_config.architecture not in NEXT_TOKEN_ARCHITECTURES:
        return None
    if model_config.logprobs_mode not in LOGPROBS_MODES:
        logger.warning(
            "Structured decisions need --logprobs-mode raw_logprobs or "
            "processed_logprobs, not %s. /v1/systemone will return 501.",
            model_config.logprobs_mode,
        )
        return None
    return NextTokenStrategy
