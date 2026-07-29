# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import copy
from typing import Any

from vllm.exceptions import VLLMValidationError

from .encoding_k3 import build_chat_segments, is_batched_conversation
from .hf import CachedHfTokenizer, HfTokenizer
from .protocol import TokenizerLike

_K3_THINKING_EFFORTS = ("low", "high", "max")


def get_kimi_k3_tokenizer(tokenizer: HfTokenizer) -> HfTokenizer:
    """Wrap a Kimi K3 tokenizer with vLLM's local XTML chat encoding."""
    kimi_tokenizer = copy.copy(tokenizer)

    class _KimiK3Tokenizer(tokenizer.__class__):  # type: ignore
        def apply_chat_template(
            self,
            conversation: Any,
            tools: list[dict[str, Any]] | None = None,
            *,
            tokenize: bool = False,
            add_generation_prompt: bool = True,
            thinking: bool | None = None,
            padding: bool | str = False,
            truncation: bool = False,
            max_length: int | None = None,
            return_tensors: str | None = None,
            return_dict: bool = False,
            **kwargs: Any,
        ) -> Any:
            is_batched = is_batched_conversation(conversation)
            conversations = conversation if is_batched else [conversation]
            image_prompts = kwargs.pop("image_prompts", None)
            if is_batched and image_prompts is not None:
                raise ValueError("image_prompts is only supported for one chat.")

            enable_thinking = kwargs.pop("enable_thinking", None)
            if thinking is None:
                thinking = True if enable_thinking is None else enable_thinking
            reasoning_effort = kwargs.pop("reasoning_effort", None)
            if reasoning_effort == "none":
                thinking = False
            elif reasoning_effort is not None:
                kwargs.setdefault("thinking_effort", reasoning_effort)
            kwargs.setdefault("thinking_effort", "max")
            thinking_effort = kwargs["thinking_effort"]
            if thinking_effort not in _K3_THINKING_EFFORTS:
                supported = ", ".join(_K3_THINKING_EFFORTS)
                raise VLLMValidationError(
                    f"Kimi K3 supports thinking_effort values: {supported}",
                    parameter="thinking_effort",
                    value=thinking_effort,
                )

            segment_batches = [
                build_chat_segments(
                    messages,
                    tools=tools,
                    add_generation_prompt=add_generation_prompt,
                    thinking=thinking,
                    image_prompts=image_prompts,
                    **kwargs,
                )
                for messages in conversations
            ]
            if not tokenize:
                rendered = [
                    "".join(segment.text for segment in segments)
                    for segments in segment_batches
                ]
                return rendered if is_batched else rendered[0]

            encoded_inputs = [
                self._encode_chat_segments(segments) for segments in segment_batches
            ]
            return self._format_chat_token_output(
                encoded_inputs,
                is_batched=is_batched,
                padding=padding,
                truncation=truncation,
                max_length=max_length,
                return_tensors=return_tensors,
                return_dict=return_dict,
            )

        def __reduce__(self):
            return get_kimi_k3_tokenizer, (tokenizer,)

    _KimiK3Tokenizer.__name__ = f"KimiK3{tokenizer.__class__.__name__}"
    kimi_tokenizer.__class__ = _KimiK3Tokenizer
    return kimi_tokenizer


class KimiK3Tokenizer(TokenizerLike):
    @classmethod
    def from_pretrained(cls, *args, **kwargs) -> HfTokenizer:
        tokenizer = CachedHfTokenizer.from_pretrained(*args, **kwargs)
        return get_kimi_k3_tokenizer(tokenizer)
