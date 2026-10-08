# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Read strategies: how a model is asked for label probabilities.

Only models in READ_STRATEGIES are currently supported.
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
from vllm.lora.request import LoRARequest
from vllm.renderers.inputs.preprocess import extract_prompt_components
from vllm.renderers.online_renderer import OnlineRenderer
from vllm.sampling_params import SamplingParams
from vllm.tokenizers import TokenizerLike

from .protocol import ReadPromptRequest
from .question_types import LABELS, Question, StructuredDecisionError, label_softmax


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
        state: Any,
        images: list[str],
        *,
        request_id: str,
        chat_template_kwargs: dict[str, Any] | None,
        lora_request: LoRARequest | None,
        priority: int,
    ) -> list[QuestionRead]:
        """One read per question, in order."""


def reply_tail(
    tokenizer: TokenizerLike, prompt_ids: Sequence[int]
) -> tuple[list[int], str]:
    """The prompt's ids after its last added token, and the text the reply's
    first token continues."""
    added = set(tokenizer.get_added_vocab().values())
    start = max((i + 1 for i, t in enumerate(prompt_ids) if t in added), default=0)
    tail = list(prompt_ids[start:])
    return tail, tokenizer.decode(tail)


def label_token_ids(
    tokenizer: TokenizerLike, tail: list[int], tail_text: str, labels: Sequence[str]
) -> list[int]:
    """The token each label adds after ``tail`` as the first token of the
    reply. Raises ValueError if a label is not one distinct token there."""
    ids: list[int] = []
    for label in labels:
        extended = tokenizer.encode(tail_text + label, add_special_tokens=False)
        if extended[:-1] != tail or extended[-1] in ids:
            raise ValueError(
                f"label {label!r} is not one distinct token after this model's "
                "chat prompt"
            )
        ids.append(extended[-1])
    return ids


class NextTokenStrategy(ReadStrategy):
    """Autoregressive models. Each question is one request: the state, then the
    question with its labeled options, and the logprobs of the label tokens as
    the reply's first token. The requests share the state. With prefix caching,
    it is prefilled once."""

    def __init__(self, context: ReadContext):
        super().__init__(context)
        tokenizer = context.online_renderer.renderer.get_tokenizer()
        self._tokenizer = tokenizer
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
        self.tail, self.tail_text = reply_tail(tokenizer, probe)
        # A choice labels its options A to Z, so fail at startup when one of
        # those is not one token here.
        label_token_ids(tokenizer, self.tail, self.tail_text, LABELS)
        self._label_ids: dict[str, int] = {}

    def limits(self) -> DecisionLimits:
        return DecisionLimits(max_questions=64, max_options=len(LABELS))

    def label_ids(self, question: Question) -> list[int]:
        """The token id of each of the question's labels."""
        ids = []
        for label in question.labels:
            token = self._label_ids.get(label)
            if token is None:
                try:
                    token = label_token_ids(
                        self._tokenizer, self.tail, self.tail_text, (label,)
                    )[0]
                except ValueError:
                    raise StructuredDecisionError(
                        f"label {label!r} is not one token after this model's "
                        "chat prompt"
                    ) from None
                self._label_ids[label] = token
            ids.append(token)
        if len(set(ids)) != len(ids):
            raise StructuredDecisionError(
                f"question {question.id!r}: two labels read as the same token"
            )
        return ids

    def label_groups(self, question: Question) -> list[list[int]]:
        """The tokens that count for each label. A label scores its best one."""
        return [[i] for i in self.label_ids(question)]

    def content(self, state: Any, question: Question) -> str:
        """The user turn: the state, then the question."""
        text = (
            state if isinstance(state, str) else json.dumps(state, ensure_ascii=False)
        )
        return f"{text}\n\n{question.type.prompt(question)}"

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
        state: Any,
        images: list[str],
        *,
        request_id: str,
        chat_template_kwargs: dict[str, Any] | None,
        lora_request: LoRARequest | None,
        priority: int,
    ) -> list[QuestionRead]:
        ctx = self.context
        read_request = self._read_request(chat_template_kwargs)
        if images and not ctx.engine_client.model_config.is_multimodal_model:
            raise StructuredDecisionError("images: this model reads text only")
        image_parts = [{"type": "image_url", "image_url": {"url": u}} for u in images]

        slots, engine_inputs = [], []
        for q in questions:
            content: Any = self.content(state, q)
            if image_parts:
                content = [*image_parts, {"type": "text", "text": content}]
            messages = [{"role": "user", "content": content}]
            if instructions:
                messages.insert(0, {"role": "system", "content": instructions})
            engine_input, prompt_ids = await self._render(read_request, messages)
            if prompt_ids[len(prompt_ids) - len(self.tail) :] != self.tail:
                raise StructuredDecisionError(
                    "these chat options end the prompt differently, so the "
                    "labels' tokens are unknown"
                )
            slots.append(self.label_groups(q))
            engine_inputs.append(engine_input)

        label_reads = await next_token_label_reads(
            ctx.engine_client,
            engine_inputs,
            [
                SamplingParams(
                    max_tokens=1,
                    temperature=0.0,
                    logprob_token_ids=[i for ids in groups for i in ids],
                )
                for groups in slots
            ],
            request_id,
            lora_request=lora_request,
            priority=priority,
        )

        reads = []
        for groups, label_read in zip(slots, label_reads):
            output = label_read.result.outputs[0]
            logprobs = iter(label_read.logprobs)
            reads.append(
                QuestionRead(
                    probs=label_softmax(
                        [max(next(logprobs) for _ in g) for g in groups]
                    ),
                    label_mass=sum(math.exp(lp) for lp in label_read.logprobs),
                    argmax_is_label=bool(output.token_ids)
                    and any(output.token_ids[0] in ids for ids in groups),
                    input_tokens=len(label_read.result.prompt_token_ids or ()),
                    output_tokens=len(output.token_ids),
                )
            )
        return reads


class LiquidStrategy(NextTokenStrategy):
    """Liquid decision models (d1). The prompt and the label tokens are the
    ones the checkpoint was trained with, from the `prompt.py` it ships."""

    def _single_tokens(self, texts: Sequence[str]) -> list[int]:
        ids: list[int] = []
        for text in texts:
            encoded = self._tokenizer.encode(text, add_special_tokens=False)
            if len(encoded) == 1 and encoded[0] not in ids:
                ids.append(encoded[0])
        return ids

    @staticmethod
    def _codes(question: Question) -> list[str]:
        """A choice's codes: A to Z, or the option names when those are
        already single letters. A score's are its level indices."""
        if question.type.name == "score":
            if len(question.options) > 10:
                raise StructuredDecisionError(
                    f"question {question.id!r}: at most 10 levels for this model"
                )
            return [str(i) for i in range(len(question.options))]
        names = [o.name.strip() for o in question.options]
        if all(len(n) == 1 and n.isalpha() for n in names):
            return names
        return list(LABELS[: len(names)])

    def label_groups(self, question: Question) -> list[list[int]]:
        if question.type.name == "noul":
            forms: list[Sequence[str]] = [("yes", "Yes", "YES"), ("no", "No", "NO")]
        elif question.type.name == "choice":
            forms = [(code, f" {code}") for code in self._codes(question)]
        else:
            forms = [(code,) for code in self._codes(question)]
        groups = [self._single_tokens(f) for f in forms]
        seen: set[int] = set()
        for (label, *_), ids in zip(forms, groups):
            if self._single_tokens([label]) != ids[:1] or seen & set(ids):
                raise StructuredDecisionError(
                    f"question {question.id!r}: label {label!r} is not one "
                    "distinct token for this model"
                )
            seen.update(ids)
        return groups

    def content(self, state: Any, question: Question) -> str:
        q = question
        if q.type.name == "choice":
            lines = "\n".join(
                f"{code} {o.description or o.name.replace('_', ' ')}"
                for code, o in zip(self._codes(q), q.options)
            )
            ask = (
                f"{q.instructions}\n\nOptions:\n{lines}\n\n"
                "Reply with the option code only."
            )
        elif q.type.name == "noul":
            yes, no = (o.description for o in q.options)
            sides = f"\nYes: {yes}\nNo: {no}" if yes or no else ""
            ask = f"{q.instructions}{sides}\n\nReply with yes or no only."
        elif q.type.name == "score":
            legend = "\n".join(
                f"{code} {o.name}" for code, o in zip(self._codes(q), q.options)
            )
            ask = (
                f"{q.instructions}\n\n{legend}\n\n"
                f"Reply with a single digit 0-{len(q.options) - 1} only."
            )
        else:
            raise StructuredDecisionError(
                f"question {q.id!r}: this model does not answer {q.type.name!r}"
            )
        if state is None:
            return ask
        if not isinstance(state, str):
            state = json.dumps(state, ensure_ascii=False, indent=2)
        return f"{state}\n\n\nQUESTION:\n{ask}"


NEXT_TOKEN_ARCHITECTURES = frozenset(
    {
        "Qwen3ForCausalLM",
        "Qwen3_5ForConditionalGeneration",
        "Qwen3_5MoeForConditionalGeneration",
    }
)


READ_STRATEGIES: dict[str, type[ReadStrategy]] = {
    **dict.fromkeys(NEXT_TOKEN_ARCHITECTURES, NextTokenStrategy),
    "Lfm2VlForConditionalGeneration": LiquidStrategy,
}


# label_mass sums the labels' full-vocabulary probabilities. The logprobs modes
# return those probabilities. The logits modes return raw logits instead.
LOGPROBS_MODES = frozenset({"raw_logprobs", "processed_logprobs"})


def select_read_strategy(model_config: ModelConfig) -> type[ReadStrategy]:
    """Raise ValueError when the model cannot serve structured decisions."""
    strategy = READ_STRATEGIES.get(model_config.architecture)
    if strategy is None:
        raise ValueError(
            "The structured decisions API does not support "
            f"{model_config.architecture}. Supported architectures: "
            f"{sorted(READ_STRATEGIES)}"
        )
    if model_config.logprobs_mode not in LOGPROBS_MODES:
        raise ValueError(
            "Structured decisions need --logprobs-mode raw_logprobs or "
            f"processed_logprobs, not {model_config.logprobs_mode}"
        )
    return strategy
