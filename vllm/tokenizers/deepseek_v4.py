# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import copy
from typing import Any

from transformers import PreTrainedTokenizerFast

from vllm.entrypoints.chat_utils import ChatCompletionMessageParam

from .deepseek_v4_encoding import encode_messages
from .hf import HfTokenizer, get_cached_tokenizer
from .protocol import TokenizerLike


def get_deepseek_v4_tokenizer(tokenizer: HfTokenizer) -> HfTokenizer:
    """
    Wraps a tokenizer to use the custom DeepSeek V4 chat template encoding.
    """
    dsv4_tokenizer = copy.copy(tokenizer)

    added_vocab = tokenizer.get_added_vocab()
    added_vocab_size = len(added_vocab)
    tokenizer_vocab_size = tokenizer.vocab_size

    class _DeepseekV4Tokenizer(tokenizer.__class__):  # type: ignore
        def apply_chat_template(
            self,
            messages: list["ChatCompletionMessageParam"],
            tools: list[dict[str, Any]] | None = None,
            **kwargs,
        ) -> str | list[int]:
            thinking = kwargs.get("thinking", False)
            enable_thinking = kwargs.get("enable_thinking", False)
            thinking = thinking or enable_thinking
            thinking_mode = "thinking" if thinking else "chat"

            conversation = kwargs.get("conversation", messages)
            messages = conversation.copy()
            if tools is not None and len(tools) > 0:
                messages.insert(0, {"role": "system"})
                messages[0]["tools"] = tools  # type: ignore[typeddict-unknown-key]

            # When `continue_final_message=True`, the final assistant message
            # must be rendered without the trailing EOS so the model can
            # continue generating from it. Mark that message with `wo_eos`
            # so the encoder honours it. Mirrors HuggingFace's behaviour for
            # `apply_chat_template(continue_final_message=True)`.
            continue_final_message = kwargs.get("continue_final_message", False)
            if continue_final_message:
                if not messages or messages[-1].get("role") != "assistant":
                    raise ValueError(
                        "Cannot set `continue_final_message` to True when "
                        "the last message is not from the assistant."
                    )
                # Copy to avoid mutating the caller's message dict.
                last = dict(messages[-1])
                last["wo_eos"] = True
                messages[-1] = last  # type: ignore[assignment]

            # The V4 reference currently accepts only "max", "high", or None.
            reasoning_effort = kwargs.get("reasoning_effort")
            if not isinstance(reasoning_effort, str):
                reasoning_effort = None
            elif reasoning_effort == "none":
                thinking_mode = "chat"
                reasoning_effort = None
            elif reasoning_effort in ("max", "xhigh"):
                reasoning_effort = "max"
            else:
                reasoning_effort = "high"

            encode_config = dict(
                thinking_mode=thinking_mode,
                drop_thinking=kwargs.get("drop_thinking", True),
                reasoning_effort=reasoning_effort,
            )

            prompt_str = encode_messages(messages, **encode_config)  # type: ignore

            if kwargs.get("tokenize", True):
                tokenizer_kwargs = {
                    k: kwargs[k] for k in ("truncation", "max_length") if k in kwargs
                }
                return self.encode(
                    prompt_str,
                    add_special_tokens=False,
                    **tokenizer_kwargs,
                )

            return prompt_str

        def num_special_tokens_to_add(self) -> int:
            return len(self.encode(""))

        def __len__(self) -> int:
            return tokenizer_vocab_size + added_vocab_size

        def get_added_vocab(self) -> dict[str, int]:
            return added_vocab.copy()

        def __reduce__(self):
            return get_deepseek_v4_tokenizer, (tokenizer,)

    _DeepseekV4Tokenizer.__name__ = f"DSV4{tokenizer.__class__.__name__}"

    dsv4_tokenizer.__class__ = _DeepseekV4Tokenizer
    return dsv4_tokenizer


class DeepseekV4Tokenizer(TokenizerLike):
    @classmethod
    def from_pretrained(cls, *args, **kwargs) -> HfTokenizer:
        tokenizer = PreTrainedTokenizerFast.from_pretrained(*args, **kwargs)
        return get_cached_tokenizer(get_deepseek_v4_tokenizer(tokenizer))
