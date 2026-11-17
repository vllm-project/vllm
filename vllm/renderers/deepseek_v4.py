# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from __future__ import annotations

from typing import TYPE_CHECKING

from vllm.logger import init_logger

from vllm.config import VllmConfig
from vllm.renderers.chat_utils import (
    ChatCompletionMessageParam,
    ConversationMessage,
    parse_chat_messages,
    parse_chat_messages_async,
)
from vllm.renderers.chat_utils.item_tracker import _get_embeds_data
from vllm.tokenizers.deepseek_v4 import DeepseekV4Tokenizer
from vllm.utils.async_utils import make_async

from .base import BaseRenderer
from .deepseek_recipe_encoder import DeepSeekRecipeEncoder
from .hf import _encode_prompt_with_recipe
from .inputs import DictPrompt
from .inputs.preprocess import parse_dec_only_prompt
from .params import ChatParams

logger = init_logger(__name__)


def _backend_tokenizer_of(tokenizer):
    """The tokenizers backend behind a vLLM tokenizer wrapper."""
    inner = getattr(tokenizer, "tokenizer", tokenizer)
    return getattr(inner, "backend_tokenizer", None)


if TYPE_CHECKING:
    from vllm.inputs import TextPrompt, TokensPrompt
    from vllm.renderers.params import TokenizeParams

_FIM_BEGIN = "<｜fim▁begin｜>"
_FIM_HOLE = "<｜fim▁hole｜>"
_FIM_END = "<｜fim▁end｜>"


class DeepseekV4Renderer(BaseRenderer[DeepseekV4Tokenizer]):
    # deepseek-recipe spec for the official encoder.
    _recipe_spec = "dsv4"

    def __init__(
        self,
        config: VllmConfig,
        tokenizer: DeepseekV4Tokenizer | None,
    ) -> None:
        super().__init__(config, tokenizer)

        self._apply_chat_template_async = make_async(
            self._apply_chat_template, executor=self._executor
        )

    def _recipe_render(self, messages, conversation, params):
        """Render with the official deepseek-recipe encoder, or None to fall
        back to vLLM's own renderer.

        Text and tool conversations are handed to recipe; images keep vLLM's
        item tracker and processor pipeline. Tool definitions travel on the
        parsed developer message, tool calls in the raw OpenChat messages.
        """
        for message in messages:
            if message.get("role") not in ("system", "user", "assistant", "tool"):
                return None
            content = message.get("content")
            if content is not None and not isinstance(content, str):
                return None
        tools = params.chat_template_kwargs.get("tools")
        if not tools:
            # The serving layer may also pin tools on a developer message.
            for message in conversation:
                if message.get("role") == "developer" and message.get("tools"):
                    tools = message["tools"]
                    break
        tokenizer = self.get_tokenizer()
        backend_tokenizer = getattr(
            getattr(tokenizer, "tokenizer", tokenizer), "backend_tokenizer", None
        )
        if backend_tokenizer is None:
            return None
        encoder = DeepSeekRecipeEncoder.get(backend_tokenizer, self._recipe_spec)
        if encoder is None:
            return None
        kwargs = params.get_apply_chat_template_kwargs()
        thinking = kwargs.get("thinking_mode") == "thinking"
        try:
            return encoder.render(messages, thinking=thinking, tools=tools)
        except Exception as exc:  # pragma: no cover - package drift
            logger.warning_once("deepseek-recipe render failed: %r", exc)
            return None

    def _apply_chat_template(self, *args, **kwargs):
        return self.get_tokenizer().apply_chat_template(*args, **kwargs), None

    def _tokenize_prompt(
        self,
        prompt: TextPrompt,
        params: TokenizeParams,
    ) -> TokensPrompt:
        # Same hook as HfRenderer: route the string prompt through the recipe
        # backend when enabled (add_special_tokens=False default chat path),
        # falling back to the builtin encoder otherwise. DeepseekV41Renderer
        # inherits this.
        if (
            not params.add_special_tokens
            and not self._wants_offsets(prompt, params)
            and self.model_config.tokenize_backend == "recipe"
        ):
            # Official recipe ranges first: when render_messages handed the
            # prompt to the reference encoder, its special ranges replace the
            # regex scan below.
            ranges = prompt.get("_recipe_special_ranges")
            if ranges is not None:
                encoder = DeepSeekRecipeEncoder.get(
                    _backend_tokenizer_of(self.get_tokenizer()),
                    self._recipe_spec,
                )
                if encoder is not None:
                    token_ids = encoder.encode(prompt["prompt"], ranges)
                    if token_ids is not None:
                        return self._build_tokens_prompt(token_ids, prompt)
            token_ids = _encode_prompt_with_recipe(
                prompt["prompt"], self.get_tokenizer()
            )
            if token_ids is not None:
                return self._build_tokens_prompt(token_ids, prompt)
        return super()._tokenize_prompt(prompt, params)

    def _build_prompt(self, rendered, mm_data, mm_uuids) -> DictPrompt:
        prompt_raw, image_order = rendered
        prompt = parse_dec_only_prompt(prompt_raw)
        image_uuids = (mm_uuids or {}).get("image", [])
        if image_order is not None and (
            not image_order or image_order != list(range(len(image_uuids)))
        ):
            for items in (mm_data, mm_uuids):
                if items is not None and "image" in items:
                    if image_order:
                        images = items["image"]
                        if isinstance(images, dict):
                            processor = self.get_mm_processor()
                            parsed = processor.info.parse_mm_data({"image": images})
                            items["image"] = _get_embeds_data(
                                "image",
                                [dict(parsed["image"][i]) for i in image_order],
                                processor,
                            )
                        else:
                            items["image"] = [images[i] for i in image_order]
                    else:
                        del items["image"]
        if mm_data:
            prompt["multi_modal_data"] = mm_data
        if mm_uuids:
            prompt["multi_modal_uuids"] = mm_uuids
        return prompt

    def render_completion_suffix(self, prompt: str, suffix: str) -> str | None:
        return f"{_FIM_BEGIN}{prompt}{_FIM_HOLE}{suffix}{_FIM_END}"

    def render_messages(
        self,
        messages: list[ChatCompletionMessageParam],
        params: ChatParams,
    ) -> tuple[list[ConversationMessage], DictPrompt]:
        conversation, mm_data, mm_uuids = parse_chat_messages(
            messages,
            self.model_config,
            content_format="openai",
            media_io_kwargs=params.media_io_kwargs,
            mm_processor_kwargs=params.mm_processor_kwargs,
        )

        rendered_recipe = (
            self._recipe_render(messages, conversation, params)
            if self.model_config.tokenize_backend == "recipe"
            else None
        )
        if rendered_recipe is not None:
            prompt_raw, ranges = rendered_recipe
            prompt = self._build_prompt((prompt_raw, None), mm_data, mm_uuids)
            prompt["_recipe_special_ranges"] = ranges
            return conversation, prompt

        rendered = self._apply_chat_template(
            conversation=conversation,
            messages=messages,
            **params.get_apply_chat_template_kwargs(),
        )

        return conversation, self._build_prompt(rendered, mm_data, mm_uuids)

    async def render_messages_async(
        self,
        messages: list[ChatCompletionMessageParam],
        params: ChatParams,
    ) -> tuple[list[ConversationMessage], DictPrompt]:
        conversation, mm_data, mm_uuids = await parse_chat_messages_async(
            messages,
            self.model_config,
            content_format="openai",
            media_io_kwargs=params.media_io_kwargs,
            mm_processor_kwargs=params.mm_processor_kwargs,
        )

        rendered_recipe = (
            self._recipe_render(messages, conversation, params)
            if self.model_config.tokenize_backend == "recipe"
            else None
        )
        if rendered_recipe is not None:
            prompt_raw, ranges = rendered_recipe
            prompt = self._build_prompt((prompt_raw, None), mm_data, mm_uuids)
            prompt["_recipe_special_ranges"] = ranges
            return conversation, prompt

        rendered = await self._apply_chat_template_async(
            conversation=conversation,
            messages=messages,
            **params.get_apply_chat_template_kwargs(),
        )

        return conversation, self._build_prompt(rendered, mm_data, mm_uuids)


class DeepseekV41Renderer(DeepseekV4Renderer):
    _recipe_spec = "dsv41"

    def _apply_chat_template(self, *args, **kwargs):
        image_order: list[int] = []
        kwargs["image_order"] = image_order
        prompt_raw = self.get_tokenizer().apply_chat_template(*args, **kwargs)
        return prompt_raw, image_order
