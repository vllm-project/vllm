# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Read strategies: how a model is asked for label probabilities.

Only models in READ_STRATEGIES are currently supported.
"""

import hashlib
import json
import math
import random
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

from vllm.config import ModelConfig
from vllm.engine.protocol import EngineClient
from vllm.entrypoints.generate.label_reads import next_token_label_reads
from vllm.entrypoints.serve.engine.protocol import OpenAIBaseModel
from vllm.inputs import EngineInput, tokens_input
from vllm.lora.request import LoRARequest
from vllm.renderers import ChatParams, TokenizeParams, merge_kwargs
from vllm.renderers.chat_utils import ChatTemplateContentFormatOption
from vllm.renderers.inputs.preprocess import extract_prompt_components
from vllm.renderers.online_renderer import OnlineRenderer
from vllm.sampling_params import SamplingParams
from vllm.tokenizers import TokenizerLike

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
    cached_tokens: int = 0
    cache_write_tokens: int = 0
    confidence: float | None = None


class ReadPromptRequest(OpenAIBaseModel):
    """Chat options for one read, ending at the generation prompt so the label
    is the reply's first token. Thinking is off unless the request enables it."""

    chat_template_kwargs: dict[str, Any] | None = None
    cache_salt: str | None = None

    def build_chat_params(
        self,
        default_template: str | None,
        default_template_content_format: ChatTemplateContentFormatOption,
    ) -> ChatParams:
        return ChatParams(
            chat_template=default_template,
            chat_template_content_format=default_template_content_format,
            chat_template_kwargs=merge_kwargs(
                merge_kwargs({"enable_thinking": False}, self.chat_template_kwargs),
                dict(add_generation_prompt=True, continue_final_message=False),
            ),
        )

    def build_tok_params(self, model_config: ModelConfig) -> TokenizeParams:
        return TokenizeParams(
            max_total_tokens=model_config.max_model_len,
            max_output_tokens=1,
            add_special_tokens=False,
        )


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
        state: str | list[dict[str, Any]],
        *,
        request_id: str,
        chat_template_kwargs: dict[str, Any] | None,
        lora_request: LoRARequest | None,
        priority: int,
        cache_salt: str | None = None,
        trace_headers: Mapping[str, str] | None = None,
    ) -> list[QuestionRead]:
        """One read per question, in order."""


def prompt_tail(tokenizer: TokenizerLike, prompt_ids: Sequence[int]) -> list[int]:
    """The prompt's ids after its last added token."""
    added = set(tokenizer.get_added_vocab().values())
    start = max((i + 1 for i, t in enumerate(prompt_ids) if t in added), default=0)
    return list(prompt_ids[start:])


def reply_label_ids(
    tokenizer: TokenizerLike,
    prompt_ids: Sequence[int],
    labels: Sequence[str] = LABELS,
) -> tuple[list[int], list[int]]:
    """The prompt's ids after its last added token, and the token each of
    ``labels`` adds after them as the first token of the reply. Raises
    ValueError if a label is not one distinct token there."""
    tail = prompt_tail(tokenizer, prompt_ids)
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


def type_label_ids(
    tokenizer: TokenizerLike, prompt_ids: Sequence[int]
) -> dict[str, int]:
    """The reply's first token for each label of every registered question
    type. Raises ValueError as reply_label_ids does."""
    label_ids: dict[str, int] = {}
    for qtype in QUESTION_TYPES.values():
        _, ids = reply_label_ids(tokenizer, prompt_ids, qtype.label_set)
        label_ids.update(zip(qtype.label_set, ids))
    return label_ids


class NextTokenStrategy(ReadStrategy):
    """Autoregressive models. Each question is one request: the state, then the
    question with its labeled options, and the logprobs of the label tokens as
    the reply's first token. The requests share the state. With prefix caching,
    it is prefilled once."""

    def __init__(self, context: ReadContext):
        super().__init__(context)
        # Every prompt ends with the same generation prompt, so a label's token
        # is the same for every question.
        prompt = self._generation_prompt()
        self.tail = prompt_tail(self._tokenizer(), prompt)
        self.label_ids = type_label_ids(self._tokenizer(), prompt)

    def _tokenizer(self) -> TokenizerLike:
        return self.context.online_renderer.renderer.get_tokenizer()

    def _generation_prompt(self) -> list[int]:
        """A chat prompt's ids up to the generation prompt, with thinking off."""
        context, tokenizer = self.context, self._tokenizer()
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
        return list(probe)

    def _sampling_params(
        self, label_ids: list[int], prompt_ids: list[int]
    ) -> SamplingParams:
        return SamplingParams(
            max_tokens=1, temperature=0.0, logprob_token_ids=label_ids
        )

    def _read_input(
        self, engine_input: EngineInput, prompt_ids: list[int]
    ) -> EngineInput:
        """The engine input for a read of the rendered prompt."""
        return engine_input

    def limits(self) -> DecisionLimits:
        return DecisionLimits(max_questions=64, max_options=len(LABELS))

    def _read_request(
        self, chat_template_kwargs: dict[str, Any] | None, cache_salt: str | None = None
    ) -> ReadPromptRequest:
        if (chat_template_kwargs or {}).get("enable_thinking"):
            raise StructuredDecisionError(
                "a read is the reply's first token, so thinking must be off"
            )
        return ReadPromptRequest(
            chat_template_kwargs=chat_template_kwargs, cache_salt=cache_salt
        )

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
        state: str | list[dict[str, Any]],
        *,
        request_id: str,
        chat_template_kwargs: dict[str, Any] | None,
        lora_request: LoRARequest | None,
        priority: int,
        cache_salt: str | None = None,
        trace_headers: Mapping[str, str] | None = None,
    ) -> list[QuestionRead]:
        ctx = self.context
        read_request = self._read_request(chat_template_kwargs, cache_salt)

        slots, engine_inputs, params = [], [], []
        for q in questions:
            messages: list[dict[str, Any]]
            if isinstance(state, str):
                messages = [
                    {"role": "user", "content": f"{state}\n\n{q.type.prompt(q)}"}
                ]
            else:
                if not ctx.engine_client.model_config.is_multimodal_model:
                    raise StructuredDecisionError(
                        "This model does not support image input"
                    )
                messages = deepcopy(state)
                messages[-1]["content"].append(
                    {"type": "text", "text": "\n\n" + q.type.prompt(q)}
                )
            if instructions:
                messages.insert(0, {"role": "system", "content": instructions})
            engine_input, prompt_ids = await self._render(read_request, messages)
            if prompt_ids[len(prompt_ids) - len(self.tail) :] != self.tail:
                raise StructuredDecisionError(
                    "these chat options end the prompt differently, so the "
                    "labels' tokens are unknown"
                )
            slots.append([self.label_ids[label] for label in q.labels])
            engine_inputs.append(self._read_input(engine_input, prompt_ids))
            params.append(self._sampling_params(slots[-1], prompt_ids))

        label_reads = await next_token_label_reads(
            ctx.engine_client,
            engine_inputs,
            params,
            request_id,
            lora_request=lora_request,
            trace_headers=trace_headers,
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
                    cached_tokens=label_read.result.num_cached_tokens or 0,
                    cache_write_tokens=(
                        label_read.result.num_cache_creation_tokens or 0
                    ),
                )
            )
        return reads


class DiffusionGemmaCanvasStrategy(NextTokenStrategy):
    """DiffusionGemma. Each question is one read-only request with one denoising
    step."""

    THOUGHT = "<|channel>thought\n<channel|>"
    END_OF_TURN = "<turn|>"
    CANVAS_STEP = 16

    def __init__(self, context: ReadContext):
        ReadStrategy.__init__(self, context)
        tokenizer = self._tokenizer()
        self.tail = prompt_tail(tokenizer, self._generation_prompt())
        self.thought = tokenizer.encode(self.THOUGHT, add_special_tokens=False)
        self.label_ids = type_label_ids(tokenizer, self.thought)
        end = tokenizer.encode(self.END_OF_TURN, add_special_tokens=False)
        self.pad = tokenizer.pad_token_id
        if self.pad is None or len(end) != 1:
            raise ValueError(
                "Structured decisions need the model's "
                f"{self.END_OF_TURN!r} and pad tokens"
            )
        self.end = end[0]
        vllm_config = context.engine_client.vllm_config
        served = (
            vllm_config.diffusion_config.canvas_length
            if vllm_config.diffusion_config
            and vllm_config.diffusion_config.canvas_length
            else vllm_config.model_config.hf_config.canvas_length
        )
        self.width = self.CANVAS_STEP
        if self.width > served:
            raise ValueError(
                f"Structured decisions need a canvas of {self.width}, "
                f"and the served canvas is {served}"
            )
        self.vocab_size = vllm_config.model_config.get_vocab_size()
        self.max_model_len = vllm_config.model_config.max_model_len

    def _sampling_params(
        self, label_ids: list[int], prompt_ids: list[int]
    ) -> SamplingParams:
        # The noise at the label slot is drawn from the prompt, so a repeated
        # read gets the same canvas.
        rng = random.Random(hashlib.sha256(json.dumps(prompt_ids).encode()).digest())
        canvas = [rng.randrange(self.vocab_size), self.end]
        canvas += [self.pad] * (self.width - len(canvas))
        return SamplingParams(
            max_tokens=2,
            logprob_token_ids=label_ids,
            extra_args={
                "diffusion_seed_canvas": canvas,
                "diffusion_canvas_length": self.width,
                "diffusion_max_steps": 1,
                "diffusion_read_only": True,
            },
        )

    def _read_input(
        self, engine_input: EngineInput, prompt_ids: list[int]
    ) -> EngineInput:
        extra = len(self.thought) + self.width
        if len(prompt_ids) + extra > self.max_model_len:
            raise StructuredDecisionError(
                f"the prompt has {len(prompt_ids)} tokens and a canvas read adds "
                f"{extra}, more than max_model_len={self.max_model_len}"
            )
        salt = engine_input.get("cache_salt")
        # The thought block _must_ appear outside of the canvas (far worse results
        # otherwise)
        return tokens_input(
            prompt_ids + self.thought,
            cache_salt=salt if isinstance(salt, str) else None,
        )


READ_STRATEGIES: dict[str, type[ReadStrategy]] = {
    "Qwen3ForCausalLM": NextTokenStrategy,
    "Qwen3_5ForConditionalGeneration": NextTokenStrategy,
    "Qwen3_5MoeForConditionalGeneration": NextTokenStrategy,
    "DiffusionGemmaForBlockDiffusion": DiffusionGemmaCanvasStrategy,
}


# label_mass sums the labels' full-vocabulary probabilities. The logprobs modes
# return those probabilities. The logits modes return raw logits instead.
LOGPROBS_MODES = frozenset({"raw_logprobs", "processed_logprobs"})


def select_read_strategy(model_config: ModelConfig) -> type[ReadStrategy]:
    """Raise ValueError when the model cannot serve structured decisions."""
    config = getattr(model_config, "hf_config", None)
    protocol = getattr(config, "decision_read_strategy", None)
    if protocol == "winnow":
        from .winnow import WinnowStrategy

        if model_config.architecture not in {
            "Gemma4ForCausalLM",
            "Gemma4ForConditionalGeneration",
        }:
            raise ValueError("Winnow requires a Gemma4 causal or multimodal model")
        if model_config.logprobs_mode != "raw_logprobs":
            raise ValueError("Winnow requires raw_logprobs")
        return WinnowStrategy
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
