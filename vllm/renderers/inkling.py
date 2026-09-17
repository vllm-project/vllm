# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Native Inkling chat renderer for the Python frontend.

Mirrors the Rust frontend's native Inkling renderer
(``rust/src/chat/src/renderer/inkling/mod.rs``): chat messages are rendered
directly to token ids — Inkling has no Jinja chat template and no faithful
text form.

The encoding logic lives in ``inkling_encoding.py`` behind a narrow
"OpenAI messages + tools -> token ids" call; see the swap-point comment
in :meth:`InklingRenderer._render` for adopting a standalone Inkling
input-processing library (mistral-common style) later.
"""

from vllm.config import VllmConfig
from vllm.entrypoints.chat_utils import (
    ChatCompletionMessageParam,
    ConversationMessage,
    parse_chat_messages,
    parse_chat_messages_async,
)
from vllm.logger import init_logger
from vllm.tokenizers.hf import HfTokenizer
from vllm.utils.async_utils import make_async

from .base import BaseRenderer
from .inkling_encoding import SPECIAL_TOKEN_SPELLINGS, render_inkling_messages
from .inputs import DictPrompt
from .inputs.preprocess import parse_dec_only_prompt
from .params import ChatParams

logger = init_logger(__name__)

_NAMED_REASONING_EFFORT = {
    "none": 0.0,
    "minimal": 0.1,
    "low": 0.2,
    "medium": 0.7,
    "high": 0.9,
    "xhigh": 0.99,
    "max": 0.99,
}

_DEFAULT_REASONING_EFFORT = 0.9


def _resolve_reasoning_effort(value: object) -> float | int | None:
    if value is None:
        return _DEFAULT_REASONING_EFFORT
    if isinstance(value, str):
        return _NAMED_REASONING_EFFORT.get(value)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return value


class _HfBackedTmlTokenizer:
    """Adapts an HF tokenizer to the encoding core's tokenizer protocol.

    Special-token ids are resolved from the tokenizer vocab at
    construction (never hardcoded), trying each known spelling — the Inkling
    HF vocab exposes some semantic slots as ``<|unused_NNNNNN|>`` tokens.
    """

    def __init__(self, tokenizer: HfTokenizer) -> None:
        self._tokenizer = tokenizer

        vocab = tokenizer.get_vocab()
        special_ids: dict[str, int] = {}
        missing: list[str] = []
        for token, spellings in SPECIAL_TOKEN_SPELLINGS.items():
            for spelling in spellings:
                token_id = vocab.get(spelling)
                if token_id is not None:
                    special_ids[token] = token_id
                    break
            else:
                missing.append(token)
        if missing:
            raise ValueError(f"Inkling tokenizer is missing special tokens: {missing}")
        self._special_ids = special_ids

    def encode_text(self, text: str) -> list[int]:
        return self._tokenizer.encode(text, add_special_tokens=False)

    def encode_special(self, token: str) -> int:
        return self._special_ids[token]


class InklingRenderer(BaseRenderer[HfTokenizer]):
    def __init__(
        self,
        config: VllmConfig,
        tokenizer: HfTokenizer | None,
    ) -> None:
        super().__init__(config, tokenizer)

        self._inkling_tokenizer = _HfBackedTmlTokenizer(self.get_tokenizer())
        self._render_async = make_async(self._render, executor=self._executor)

    def _render(
        self,
        messages: list[ChatCompletionMessageParam],
        params: ChatParams,
    ) -> list[int]:
        kwargs = params.chat_template_kwargs or {}

        if kwargs.get("continue_final_message"):
            raise ValueError("Inkling renderer does not support continue_final_message")

        reasoning_effort = _resolve_reasoning_effort(kwargs.get("reasoning_effort"))

        try:
            # Swap point: to adopt a standalone Inkling input-processing
            # library, replace this call (and the _HfBackedTmlTokenizer
            # adapter above) with the library's renderer.
            return render_inkling_messages(
                messages,
                self._inkling_tokenizer,
                add_generation_prompt=kwargs.get("add_generation_prompt", True),
                tools=kwargs.get("tools"),
                reasoning_effort=reasoning_effort,
            )
        except ValueError:
            raise
        except (TypeError, KeyError) as e:
            # Malformed request content; surface as a request error.
            raise ValueError(str(e)) from e
        except Exception as e:
            logger.exception("Error while rendering Inkling chat messages")
            raise ValueError(str(e)) from e

    def render_messages(
        self,
        messages: list[ChatCompletionMessageParam],
        params: ChatParams,
    ) -> tuple[list[ConversationMessage], DictPrompt]:
        conversation, mm_data, mm_uuids = parse_chat_messages(
            messages,
            self.model_config,
            content_format="string",
            media_io_kwargs=params.media_io_kwargs,
            mm_processor_kwargs=params.mm_processor_kwargs,
        )

        token_ids = self._render(messages, params)

        prompt = parse_dec_only_prompt(token_ids)
        if mm_data is not None:
            prompt["multi_modal_data"] = mm_data
        if mm_uuids is not None:
            prompt["multi_modal_uuids"] = mm_uuids

        return conversation, prompt

    async def render_messages_async(
        self,
        messages: list[ChatCompletionMessageParam],
        params: ChatParams,
    ) -> tuple[list[ConversationMessage], DictPrompt]:
        conversation, mm_data, mm_uuids = await parse_chat_messages_async(
            messages,
            self.model_config,
            content_format="string",
            media_io_kwargs=params.media_io_kwargs,
            mm_processor_kwargs=params.mm_processor_kwargs,
        )

        token_ids = await self._render_async(messages, params)

        prompt = parse_dec_only_prompt(token_ids)
        if mm_data is not None:
            prompt["multi_modal_data"] = mm_data
        if mm_uuids is not None:
            prompt["multi_modal_uuids"] = mm_uuids

        return conversation, prompt
