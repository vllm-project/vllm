# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast, overload

from mistral_common.protocol.instruct.request import (
    ChatCompletionRequest as MistralChatCompletionRequest,
)
from mistral_common.protocol.instruct.tool_calls import Function, Tool
from mistral_common.protocol.instruct.validator import ValidationMode
from mistral_common.tokens.tokenizers.base import (
    SpecialTokenPolicy,
    SpecialTokens,
)
from mistral_common.tokens.tokenizers.instruct import InstructTokenizerV13
from mistral_common.tokens.tokenizers.sentencepiece import (
    SentencePieceTokenizer,
)
from mistral_common.tokens.tokenizers.tekken import Tekkenizer

from vllm.entrypoints.chat_utils import ChatCompletionMessageParam
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.logger import init_logger

from .protocol import TokenizerLike

if TYPE_CHECKING:
    from transformers import BatchEncoding

    try:
        # Transformers v5
        from transformers.tokenization_mistral_common import MistralCommonBackend
    except ImportError:
        # Transformers v4
        from transformers.tokenization_mistral_common import (
            MistralCommonTokenizer as MistralCommonBackend,
        )

logger = init_logger(__name__)


def maybe_serialize_tool_calls(request: "MistralChatCompletionRequest"):
    # SEE: https://github.com/vllm-project/vllm/pull/9951
    # Credits go to: @gcalmettes
    # NOTE: There is currently a bug in pydantic where attributes
    # declared as iterables are replaced in in the instances by
    # pydantic-core ValidatorIterator instance. In particular, this
    # affects tool_calls defined in ChatCompletionAssistantMessageParam
    # model:
    # see:
    #   - https://github.com/pydantic/pydantic/issues/9467
    # As a result, tool_calls from assistant messages are never
    # deserialized in the request object if the tool_calls iterator is
    # not consumed. This affect messages passed to the MistralTokenizer
    # since no chat template is applied and therefore the tools_calls
    # iterator is not directly consumed.
    # Issue is tracked on Pydantic side, with resolution planned for
    # v2.11 release. In the meantime, the official workaround is to
    # consume the iterator so the tool_calls are correctly deserialized
    # in the OpenAI ChatCompletionAssistantMessageParam object
    # https://github.com/pydantic/pydantic/issues/9467#issuecomment-2442097291 # noqa: E501
    # Official Pydantic Issues:
    #   - https://github.com/pydantic/pydantic/issues/9541
    # TODO: remove when pydantic v2.11 is released
    for i, message in enumerate(request.messages):
        if message.get("role") == "assistant":
            tool_calls_validator = message.get("tool_calls", ().__iter__())
            validated_tool_calls = []
            while True:
                try:
                    tool_call = next(tool_calls_validator)  # type: ignore
                    validated_tool_calls.append(tool_call)
                except StopIteration:
                    break

            request.messages[i]["tool_calls"] = validated_tool_calls


def truncate_tool_call_ids(request: "MistralChatCompletionRequest"):
    """Truncates tool call IDs for Mistral's ID requirements."""
    for i, message in enumerate(request.messages):
        if message.get("role") == "assistant":
            tool_calls = message.get("tool_calls", [])
            for tool_call in tool_calls:
                if len(tool_call["id"]) > 9:
                    logger.warning(
                        "Truncating tool call ID: %s to %s",
                        tool_call["id"],
                        tool_call["id"][-9:],
                    )
                    tool_call["id"] = tool_call["id"][-9:]

            request.messages[i]["tool_calls"] = tool_calls

        elif message.get("role") in {"tool_results", "tool"}:
            if "tool_call_id" in message:
                tool_call_id = message["tool_call_id"]

                if len(tool_call_id) > 9:
                    logger.warning(
                        "Truncating tool_call_id: %s to %s",
                        tool_call_id,
                        tool_call_id[-9:],
                    )
                    tool_call_id = tool_call_id[-9:]
                request.messages[i]["tool_call_id"] = tool_call_id


def _prepare_apply_chat_template_tools_and_messages(
    messages: list["ChatCompletionMessageParam"],
    tools: list[dict[str, Any]] | None = None,
    continue_final_message: bool = False,
    add_generation_prompt: bool = False,
) -> tuple[list["ChatCompletionMessageParam"], list[dict[str, Any]] | None]:
    if add_generation_prompt and continue_final_message:
        raise ValueError(
            "Cannot set both `add_generation_prompt` and "
            "`continue_final_message` to True."
        )

    last_message = cast(dict[str, Any], messages[-1])
    # add_generation_prompt is directly handled by the tokenizer but we
    # check if the user is trying to use it with a final assistant message
    # which is probably not what they want.
    # If add_generation_prompt is False, we don't need to check anything.
    if add_generation_prompt and last_message["role"] == "assistant":
        raise ValueError(
            "Cannot set `add_generation_prompt` to True when "
            "the last message is from the assistant. Consider "
            "using `continue_final_message` instead."
        )
    if continue_final_message and last_message["role"] != "assistant":
        raise ValueError(
            "Cannot set `continue_final_message` to True when "
            "the last message is not from the assistant."
        )

    # mistral-common requires AssistantMessage content to be string [1].
    #
    # [1]: https://github.com/mistralai/mistral-common/blob/f4a06998b75ed78bbf5aaf569590b772ea26c9f6/src/mistral_common/protocol/instruct/messages.py#L80
    for message in messages:
        # Remove reasoning as unsupported by Mistral
        _ = message.pop("reasoning", None)  # type: ignore

    # The Mistral client, in comparison to the OpenAI client, requires the
    # "parameters" dict and the "description" string to be present
    # even if they are empty.
    if tools:
        for function in [
            tool["function"] for tool in tools if tool["type"] == "function"
        ]:
            if function.get("parameters") is None:
                function["parameters"] = {}
            if function.get("description") is None:
                function["description"] = ""

        # We filter not supported arguments to avoid throwing an error.
        # TODO(juliendenize): remove this once OpenAI API is better supported by
        # `mistral-common`.
        tools_fields = set(Tool.model_fields.keys())
        function_fields = set(Function.model_fields.keys())
        for tool in tools:
            tool_keys = list(tool.keys())
            for tool_key in tool_keys:
                if tool_key not in tools_fields:
                    tool.pop(tool_key)
                    logger.warning_once(
                        f"'{tool_key}' is not supported by mistral-common for tools. "
                        "It has been poped from the tool definition."
                    )
                if tool["type"] == "function":
                    function_keys = list(tool["function"].keys())
                    for function_key in function_keys:
                        if function_key not in function_fields:
                            tool["function"].pop(function_key)
                            logger.warning_once(
                                f"'{function_key}' is not supported by mistral-common "
                                "for function tools. It has been poped from the "
                                "function definition."
                            )
                else:
                    raise ValueError("mistral-common only supports function tools.")

    return messages, tools


def validate_request_params(request: "ChatCompletionRequest"):
    if request.chat_template is not None or request.chat_template_kwargs is not None:
        raise ValueError("chat_template is not supported for Mistral tokenizers.")


def _tekken_token_to_id(tokenizer: "Tekkenizer", t: str | bytes) -> int:
    assert isinstance(tokenizer, Tekkenizer), type(tokenizer)

    t_bytes = t.encode("utf-8") if not isinstance(t, bytes) else t
    shift = tokenizer.num_special_tokens
    try:
        return shift + tokenizer._tekken_token2id_nospecial[t_bytes]
    except KeyError:
        t_str = t_bytes.decode("utf-8")
        if t_str in tokenizer._special_tokens_reverse_vocab:
            return tokenizer._special_tokens_reverse_vocab[t_str]
        logger.warning(
            "Failed to convert token %s to id, replacing with <unk>", t_bytes
        )
        return tokenizer.unk_id


class MistralTokenizer(TokenizerLike):
    @classmethod
    def from_pretrained(
        cls,
        path_or_repo_id: str | Path,
        *args,
        trust_remote_code: bool = False,
        revision: str | None = None,
        download_dir: str | None = None,
        **kwargs,
    ) -> "MistralTokenizer":
        try:
            # Transformers v5
            from transformers.tokenization_mistral_common import MistralCommonBackend
        except ImportError:
            # Transformers v4
            from transformers.tokenization_mistral_common import (
                MistralCommonTokenizer as MistralCommonBackend,
            )

        tokenizer = MistralCommonBackend.from_pretrained(
            path_or_repo_id,
            *args,
            mode=ValidationMode.test,
            cache_dir=download_dir,
            revision="main" if revision is None else revision,
            **kwargs,
        )

        return cls(tokenizer)

    def __init__(self, tokenizer: "MistralCommonBackend") -> None:
        super().__init__()

        self.transformers_tokenizer = tokenizer
        self.mistral = tokenizer.tokenizer
        self.instruct = self.mistral.instruct_tokenizer
        self.tokenizer = self.instruct.tokenizer

        mode = self.mistral._chat_completion_request_validator._mode
        if mode != ValidationMode.test:
            raise ValueError(
                "Mistral tokenizer must be in test mode. Make sure to "
                "set `mode='ValidationMode.test'` when creating the "
                "Mistral tokenizer."
            )

        _mistral_version_str = str(self.tokenizer.version.value)
        self.version: int = int(_mistral_version_str.split("v")[-1])

        self.is_tekken = isinstance(self.tokenizer, Tekkenizer)
        self.is_spm = isinstance(self.tokenizer, SentencePieceTokenizer)
        if not (self.is_tekken or self.is_spm):
            raise TypeError(f"Unsupported tokenizer: {type(self.tokenizer)}")

        # Reverse order to ensure that the lowest token id is kept.
        self._vocab_dict = {
            self.convert_ids_to_tokens([i], skip_special_tokens=False)[0]: i
            for i in range(self.vocab_size - 1, -1, -1)
        }
        # Sort the dict for convenience
        self._vocab_dict = dict(sorted(self._vocab_dict.items(), key=lambda x: x[1]))

        # Vocab sorted by token id.
        self._vocab = self.tokenizer.vocab()
        self._max_token_id = self.vocab_size - 1

        # Cache special tokens for faster access.
        self._special_token_ids = self._get_special_token_ids()
        self._special_token_ids_set = set(self._special_token_ids)
        self._special_tokens = self._get_special_tokens(self._special_token_ids)
        self._special_tokens_set = set(self._special_tokens)

    def _get_special_token_ids(self) -> list[int]:
        return [i for i in range(len(self._vocab)) if self.tokenizer.is_special(i)]

    def _get_special_tokens(self, all_special_ids: list[int]) -> list[str]:
        return [
            self.tokenizer.decode([i], special_token_policy=SpecialTokenPolicy.KEEP)
            for i in all_special_ids
        ]

    def num_special_tokens_to_add(self) -> int:
        return len(self.encode(""))

    # the following attributes are set to fit vLLM's design and are used
    # by the structured output backends.
    @property
    def all_special_tokens(self) -> list[str]:
        return self._special_tokens

    @property
    def all_special_ids(self) -> list[int]:
        return self._special_token_ids

    @property
    def bos_token_id(self) -> int:
        return self.tokenizer.bos_id

    @property
    def eos_token_id(self) -> int:
        return self.tokenizer.eos_id

    @property
    def pad_token_id(self) -> int:
        return self.tokenizer.pad_id

    @property
    def is_fast(self) -> bool:
        return True

    @property
    def vocab_size(self) -> int:
        return self.transformers_tokenizer.vocab_size

    @property
    def max_token_id(self) -> int:
        return self._max_token_id

    @property
    def truncation_side(self) -> str:
        return self.transformers_tokenizer.truncation_side

    def _is_special_token_id(self, token_id: int) -> bool:
        return token_id in self._special_token_ids_set

    def __hash__(self) -> int:
        return hash(id(self))

    def __len__(self) -> int:
        return self.vocab_size

    def __call__(
        self,
        text: str | list[str],
        text_pair: str | None = None,
        add_special_tokens: bool = True,
        truncation: bool = False,
        max_length: int | None = None,
    ) -> "BatchEncoding":
        if text_pair is not None:
            raise ValueError(
                "`text_pair` is not supported by `MistralTokenizer.__call__`."
            )

        encoded = self.transformers_tokenizer(
            text=text,
            text_pair=text_pair,
            add_special_tokens=add_special_tokens,
            truncation=truncation,
            max_length=max_length,
        )
        # TODO(juliendenize): once https://github.com/huggingface/transformers/pull/41962
        # is in, revert to only call self.transformers_tokenizer(...).
        # Hack to fix wrongly added eos token, when fix will be supported the condition
        # below will be False even before the revert is done.
        if encoded["input_ids"] and encoded["input_ids"][-1] == self.eos_token_id:
            encoded["input_ids"].pop(-1)
            if attention_mask := encoded.get("attention_mask"):
                attention_mask.pop(-1)
        return encoded

    @property
    def vocab(self) -> list[str]:
        return self._vocab

    def get_vocab(self) -> dict[str, int]:
        return self._vocab_dict

    def get_added_vocab(self) -> dict[str, int]:
        # Mistral tokenizers have no added vocabulary
        return {}

    def encode(
        self,
        text: str,
        truncation: bool | None = None,
        max_length: int | None = None,
        add_special_tokens: bool = True,
    ) -> list[int]:
        # TODO(juliendenize): once https://github.com/huggingface/transformers/pull/41962
        # is in, directly call self.transformers_tokenizer.encode(...).
        encoded = self.tokenizer.encode(text, bos=add_special_tokens, eos=False)

        if truncation is not False and max_length is not None:
            return encoded[:max_length]
        else:
            return encoded

    def apply_chat_template(
        self,
        messages: list["ChatCompletionMessageParam"],
        tools: list[dict[str, Any]] | None = None,
        **kwargs,
    ) -> list[int]:
        add_generation_prompt = kwargs.pop("add_generation_prompt", False)
        continue_final_message = kwargs.get("continue_final_message", False)
        tokenize = kwargs.get("tokenize", True)
        padding = kwargs.get("padding", False)
        truncation = kwargs.get("truncation", False)
        max_length = kwargs.get("max_length")

        messages, tools = _prepare_apply_chat_template_tools_and_messages(
            messages, tools, continue_final_message, add_generation_prompt
        )

        return self.transformers_tokenizer.apply_chat_template(
            conversation=messages,
            tools=tools,
            continue_final_message=continue_final_message,
            tokenize=tokenize,
            padding=padding,
            truncation=truncation,
            max_length=max_length,
            return_tensors=None,
            return_dict=False,
        )

    def decode(self, ids: list[int] | int, skip_special_tokens: bool = False) -> str:
        # TODO(juliendenize): once https://github.com/huggingface/transformers/pull/41962
        # is in, directly call self.transformers_tokenizer.decode(...).
        if isinstance(ids, int):
            ids = [ids]

        return self.transformers_tokenizer.decode(
            ids, skip_special_tokens=skip_special_tokens
        )

    def batch_decode(
        self, ids: list[list[int]] | list[int], skip_special_tokens: bool = False
    ) -> str:
        return self.transformers_tokenizer.batch_decode(
            ids, skip_special_tokens=skip_special_tokens
        )

    @overload
    def convert_tokens_to_ids(self, tokens: str) -> int: ...

    @overload
    def convert_tokens_to_ids(self, tokens: list[str]) -> list[int]: ...

    def convert_tokens_to_ids(self, tokens: str | list[str]) -> int | list[int]:
        return self.transformers_tokenizer.convert_tokens_to_ids(tokens)

    def convert_tokens_to_string(self, tokens: list[str]) -> str:
        to_decode_special_tokens = {SpecialTokens.tool_calls}
        if self.is_tekken:
            assert isinstance(self.tokenizer, Tekkenizer), type(self.tokenizer)
            tokens = [
                t
                for t in tokens
                if (t in to_decode_special_tokens or t not in self._special_tokens_set)
            ]

            if any(isinstance(t, bytes) for t in tokens):
                # we need to encode and decode all tokens again
                ids = [_tekken_token_to_id(self.tokenizer, t) for t in tokens]
                # We filtered unwanted special tokens before
                # so we can decode the rest.
                decoded = self.tokenizer.decode(ids, SpecialTokenPolicy.KEEP)
            else:
                decoded = "".join(tokens)
        else:
            # make sure certain special tokens like Tool calls are
            # not decoded
            assert isinstance(self.tokenizer, SentencePieceTokenizer), type(
                self.tokenizer
            )

            regular_tokens: list[str] = []
            decoded_list: list[str] = []
            decoded = ""

            for token in tokens:
                if token in to_decode_special_tokens:
                    if regular_tokens:
                        decoded_list.append(
                            self.tokenizer.decode(
                                regular_tokens, SpecialTokenPolicy.IGNORE
                            )
                        )
                        regular_tokens = []
                    decoded_list.append(token)
                else:
                    regular_tokens.append(token)

            if regular_tokens:
                decoded_list.append(
                    self.tokenizer.decode(regular_tokens, SpecialTokenPolicy.IGNORE)
                )
            decoded = "".join(decoded_list)

        return decoded

    def convert_ids_to_tokens(
        self,
        ids: list[int],
        skip_special_tokens: bool = False,
    ) -> list[str]:
        if not skip_special_tokens:
            return [self.tokenizer.id_to_piece(token_id) for token_id in ids]

        non_skip_special_tokens_ids = {
            self.tokenizer.get_control_token(SpecialTokens.tool_calls),
        }
        if isinstance(self.instruct, InstructTokenizerV13):
            if self.instruct.BEGIN_THINK:
                non_skip_special_tokens_ids.add(self.instruct.BEGIN_THINK)
            if self.instruct.END_THINK:
                non_skip_special_tokens_ids.add(self.instruct.END_THINK)

        ids_kept = [
            i
            for i in ids
            if i in non_skip_special_tokens_ids or not self._is_special_token_id(i)
        ]

        # We filtered unwanted special tokens so we can decode the rest.
        tokens = [self.tokenizer.id_to_piece(token_id) for token_id in ids_kept]

        if any("�" in t for t in tokens) and self.is_tekken:
            # if a decoded token contains the replacement character, then the
            # token has an incomplete UTF-8 character so we must use bytes
            # See: https://github.com/vllm-project/vllm/pull/8640
            #      https://github.com/vllm-project/vllm/pull/9625
            # if underlying tokenizer is sentencepiece, we just add "�".
            # We filtered unwanted special tokens so we can decode the rest.
            tokens = [
                self.tokenizer.id_to_byte_piece(token_id, SpecialTokenPolicy.KEEP)
                if token_id not in self._special_token_ids_set
                else self.tokenizer.decode([token_id], SpecialTokenPolicy.KEEP)
                for token_id in ids_kept
            ]

        return tokens
