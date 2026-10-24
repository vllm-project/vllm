# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from the HyperCLOVA X vLLM plugin (NAVER Cloud, Apache-2.0).

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from typing import TYPE_CHECKING

from vllm.entrypoints.openai.engine.protocol import (
    DeltaFunctionCall,
    DeltaMessage,
    DeltaToolCall,
)
from vllm.logger import init_logger
from vllm.reasoning import ReasoningParser

if TYPE_CHECKING:
    from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
    from vllm.entrypoints.openai.responses.protocol import ResponsesRequest
    from vllm.tokenizers import TokenizerLike

logger = init_logger(__name__)


class HyperCLOVAXSeedThink14BReasoningParser(ReasoningParser):
    """Reasoning parser for ``HyperCLOVAX-SEED-Think-14B``.

    The 14B model wraps reasoning after a ``/think`` prompt and switches to the
    assistant content/tool channel at the ``<|im_end|>\\n<|im_start|>assistant``
    delimiter. Correctly locating that boundary lets vLLM's structured-output
    grammar engage at the right token for ``tool_choice="required"``.
    """

    def __init__(
        self,
        tokenizer: TokenizerLike,
        *args,
        **kwargs,
    ):
        super().__init__(tokenizer, *args, **kwargs)
        self.chat_template_kwargs = kwargs.get("chat_template_kwargs", {}) or {}

        self.think_start_token = "/think\n"
        self.think_end_string_base = "<|im_end|>\n<|im_start|>assistant"
        self.non_reasoning_prompt_end_string = "<|im_start|>assistant\n"

        # for streaming
        self.non_reasoning_mode_start_token = tokenizer.encode("\n")[0]
        self.function_call_role = " -> tool/function_call\n"
        self.no_reasoning_content = False

        # Do not treat bare <|im_end|> as reasoning end. The content turn starts
        # only after the full assistant delimiter has been generated.
        self.exact_think_end_strings = [
            self.think_end_string_base + "\n",
            self.think_end_string_base + self.function_call_role + "[{",
            self.think_end_string_base + self.function_call_role,
            self.think_end_string_base,
        ]
        self.think_end_tokens = [
            tokenizer.encode(think_end_string)
            for think_end_string in self.exact_think_end_strings
        ]
        self._think_end_specs = list(
            zip(self.exact_think_end_strings, self.think_end_tokens, strict=False)
        )
        self.non_reasoning_prompt_end_tokens = tokenizer.encode(
            self.non_reasoning_prompt_end_string
        )

        # attributes for streaming parser mixin
        self.buffer_string = ""
        self.special_strings = [
            self.think_start_token,
            self.think_end_string_base,
            self.function_call_role,
        ]
        self.escaped_special_strings = [re.escape(ss) for ss in self.special_strings]

        # Set to True when the tool-call role is complete after reasoning ends.
        # Causes the reasoning parser to yield None so the downstream tool
        # parser/required structured-output path handles subsequent JSON tokens.
        self.is_tool_call_mode = False
        self.is_delaying_assistant_handoff = False
        self.is_reasoning_ended = False
        self.has_emitted_reasoning_text = False

        # Rolling token history for is_reasoning_end() in streaming paths where
        # input_ids may be a delta (1 token) rather than the full cumulative sequence.
        self._max_think_end_len = max(
            (len(t) for t in self.think_end_tokens), default=0
        )
        self._token_history: list[int] = []

    def _delta(
        self,
        *,
        reasoning: str | None = None,
        content: str | None = None,
    ) -> DeltaMessage:
        kwargs = {}
        if reasoning is not None:
            kwargs["reasoning"] = reasoning
        if content is not None:
            kwargs["content"] = content
        return DeltaMessage(**kwargs)

    def _is_skip_reasoning(
        self, request: ChatCompletionRequest | ResponsesRequest
    ) -> bool:
        chat_template_kwargs = request.chat_template_kwargs or {}
        return bool(chat_template_kwargs.get("skip_reasoning", False)) and not bool(
            chat_template_kwargs.get("force_reasoning", False)
        )

    def _is_streaming_skip_reasoning(self) -> bool:
        return bool(
            self.chat_template_kwargs.get("skip_reasoning", False)
        ) and not bool(self.chat_template_kwargs.get("force_reasoning", False))

    def _strip_think_start(self, model_output: str) -> tuple[bool, str]:
        if model_output.startswith(self.think_start_token):
            return True, model_output.partition(self.think_start_token)[2]
        return False, model_output

    def _split_first_special(self) -> tuple[str, str, str] | None:
        matches = []
        for special_string in self.special_strings:
            index = self.buffer_string.find(special_string)
            if index >= 0:
                matches.append((index, special_string))
        if not matches:
            return None

        index, special_string = min(matches, key=lambda item: item[0])
        before = self.buffer_string[:index]
        after = self.buffer_string[index + len(special_string) :]
        return before, special_string, after

    def _has_partial_think_end_suffix(self) -> bool:
        if not self.buffer_string:
            return False
        if self.think_end_string_base in self.buffer_string:
            return False

        max_len = min(len(self.buffer_string), len(self.think_end_string_base) - 1)
        for ln in range(max_len, 0, -1):
            if self.buffer_string[-ln:] == self.think_end_string_base[:ln]:
                return True
        return False

    def check_is_part_of_special_string(self) -> bool:
        for special_string in self.special_strings:
            min_len = min(len(self.buffer_string), len(special_string))
            for length in range(min_len, 0, -1):
                if self.buffer_string[-length:] == special_string[:length]:
                    return True
        return False

    def _update_token_history(self, delta_token_ids: Sequence[int]) -> None:
        if not self._max_think_end_len or not delta_token_ids:
            return

        self._token_history.extend(delta_token_ids)
        if len(self._token_history) > self._max_think_end_len:
            self._token_history = self._token_history[-self._max_think_end_len :]

    def _normalize_tool_content(self, tool_content: str) -> str:
        stripped = tool_content.lstrip()
        if stripped.startswith("-"):
            after_dash = stripped[1:].lstrip()
            if after_dash.startswith("["):
                tool_content = after_dash
        return re.sub(
            r'"arguments"(?=\s*:)',
            '"parameters"',
            tool_content,
            count=1,
        )

    def _strip_tool_end_tokens(self, tool_content: str) -> str:
        # If the content is a JSON array (the tool-call payload), find its end by
        # JSON structure (raw_decode respects string escaping) so a literal
        # <|im_end|> inside a string argument is not mistaken for the terminator.
        # Fall back to literal search for non-JSON content (e.g. reasoning text).
        stripped = tool_content.lstrip()
        if stripped.startswith("["):
            try:
                _, end = json.JSONDecoder().raw_decode(stripped)
                return stripped[:end]
            except json.JSONDecodeError:
                pass
        for end_token in ("<|im_end|>", "<|stop|>", "<|endofturn|>"):
            idx = tool_content.find(end_token)
            if idx >= 0:
                return tool_content[:idx]
        return tool_content

    def _normalize_required_tool_content(
        self,
        content: str | None,
        request: ChatCompletionRequest | ResponsesRequest,
    ) -> str | None:
        if not content or request.tool_choice != "required":
            return content
        return self._normalize_tool_content(content)

    def _is_partial_tool_role_fragment(self, text: str) -> bool:
        normalized = text.lstrip()
        target = self.function_call_role.lstrip()
        return (
            bool(normalized) and normalized != target and target.startswith(normalized)
        )

    def _decode_token_ids(self, token_ids: Sequence[int]) -> str:
        if not token_ids:
            return ""
        return self.model_tokenizer.decode(list(token_ids), skip_special_tokens=False)

    def _clean_buffered_reasoning_text(self) -> str | None:
        cleaned_reasoning = self.buffer_string
        for special_string in self.special_strings:
            cleaned_reasoning = cleaned_reasoning.replace(special_string, "")
        cleaned_reasoning = self._strip_tool_end_tokens(cleaned_reasoning)
        return cleaned_reasoning or None

    def _build_tool_delta_from_text(
        self,
        tool_text: str,
        reasoning: str | None = None,
    ) -> DeltaMessage | None:
        stripped = tool_text.lstrip("\n")
        if not stripped.startswith(self.function_call_role):
            return None

        tool_content = self._strip_tool_end_tokens(
            stripped[len(self.function_call_role) :]
        )
        tool_content = self._normalize_tool_content(tool_content.strip())

        try:
            raw_tool_calls = json.loads(tool_content)
        except json.JSONDecodeError:
            return None

        if not isinstance(raw_tool_calls, list) or not raw_tool_calls:
            return None

        tool_calls = []
        for index, raw_tool_call in enumerate(raw_tool_calls):
            if not isinstance(raw_tool_call, dict):
                return None
            name = raw_tool_call.get("name")
            arguments = raw_tool_call.get(
                "parameters", raw_tool_call.get("arguments", {})
            )
            if not isinstance(name, str) or not name:
                return None
            if isinstance(arguments, str):
                arguments_json = arguments
            else:
                arguments_json = json.dumps(arguments, ensure_ascii=False)
            tool_calls.append(
                DeltaToolCall(
                    index=index,
                    type="function",
                    function=DeltaFunctionCall(
                        name=name,
                        arguments=arguments_json,
                    ).model_dump(exclude_none=True),
                )
            )

        payload = {"tool_calls": tool_calls}
        if reasoning:
            payload["reasoning"] = reasoning
        return DeltaMessage(**payload)

    def _extract_token_based_transition(
        self,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
    ) -> DeltaMessage | None:
        decoded_current = self._decode_token_ids(current_token_ids)
        decoded_previous = self._decode_token_ids(previous_token_ids)

        if (
            self.think_end_string_base not in decoded_current
            or self.think_end_string_base in decoded_previous
        ):
            return None

        _, _, tool_or_content = decoded_current.partition(self.think_end_string_base)
        tool_delta = self._build_tool_delta_from_text(
            tool_or_content,
            reasoning=self._clean_buffered_reasoning_text(),
        )
        if tool_delta is None:
            return None

        self.buffer_string = ""
        self.is_delaying_assistant_handoff = False
        self.is_tool_call_mode = True
        self.is_reasoning_ended = True
        return tool_delta

    def extract_reasoning(
        self,
        model_output: str,
        request: ChatCompletionRequest | ResponsesRequest,
    ) -> tuple[str | None, str | None]:
        chat_template_kwargs = request.chat_template_kwargs or {}

        if self._is_skip_reasoning(request):
            return None, model_output.lstrip("\n") or None

        is_reasoning = bool(chat_template_kwargs.get("force_reasoning", False))

        has_think_start, model_output = self._strip_think_start(model_output)
        if has_think_start:
            is_reasoning = True

        if self.think_end_string_base not in model_output:
            if is_reasoning:
                # Special tokens (<|im_end|>, <|im_start|>) may be stripped from
                # output.text. Attempt to detect a trailing JSON tool-call array.
                last_bracket = model_output.rfind("[{")
                if last_bracket >= 0 and getattr(request, "tools", None):
                    candidate = model_output[last_bracket:]
                    try:
                        # Find the array end by JSON structure so a literal
                        # <|im_end|> inside a string argument (or a trailing
                        # terminator) does not break parsing.
                        _, end = json.JSONDecoder().raw_decode(candidate)
                        reasoning_part = model_output[:last_bracket].strip("\n")
                        return reasoning_part or None, candidate[:end]
                    except json.JSONDecodeError:
                        pass
                return model_output.strip("\n") or None, None
            if request.tool_choice == "auto" or request.tool_choice is None:
                return None, model_output.lstrip("\n") or None

            content = (
                model_output.replace(self.function_call_role, "").lstrip("\n") or None
            )
            normalized = self._normalize_required_tool_content(content, request)
            return None, normalized

        reasoning_content, _, content = model_output.partition(
            self.think_end_string_base
        )
        content = content.lstrip("\n")
        # Strip " -> tool/function_call\n" prefix and end tokens so that
        # downstream parsers (TypeAdapter for tool_choice="required", or our
        # extract_tool_calls for tool_choice="auto") receive bare JSON.
        if content.startswith(self.function_call_role):
            content = content[len(self.function_call_role) :]
            content = self._strip_tool_end_tokens(content)
            content = content.strip()
        content = self._normalize_required_tool_content(content or None, request)
        return reasoning_content.strip("\n") or None, content

    def extract_reasoning_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
    ) -> DeltaMessage | None:
        if self._is_streaming_skip_reasoning():
            self.no_reasoning_content = True

        force_reasoning = bool(self.chat_template_kwargs.get("force_reasoning", False))
        if (
            not force_reasoning
            and current_token_ids
            and current_token_ids[0] == self.non_reasoning_mode_start_token
        ):
            self.no_reasoning_content = True

        if len(current_text) == 0:
            return None

        self._update_token_history(delta_token_ids)

        if self.is_tool_call_mode:
            return None

        if self.no_reasoning_content:
            return self._delta(content=delta_text)

        self.buffer_string += delta_text

        split_special = self._split_first_special()
        if split_special:
            before, special_string, after = split_special
            self.buffer_string = ""

            if special_string == self.think_start_token:
                if after.strip():
                    self.has_emitted_reasoning_text = True
                return self._delta(reasoning=after)

            if special_string == self.think_end_string_base:
                is_partial_tool_role = (
                    bool(after)
                    and self.function_call_role.startswith(after)
                    and after != self.function_call_role
                )
                if after == "" or is_partial_tool_role:
                    self.is_delaying_assistant_handoff = True
                    self.buffer_string = self.think_end_string_base + after
                    cleaned_before = before if before and before.strip() else None
                    if cleaned_before:
                        self.has_emitted_reasoning_text = True
                    return (
                        self._delta(reasoning=cleaned_before)
                        if cleaned_before
                        else None
                    )

                self.is_delaying_assistant_handoff = False
                if after.startswith(self.function_call_role):
                    tool_content = after[len(self.function_call_role) :]
                    self.is_tool_call_mode = True
                    self.is_reasoning_ended = True
                    return self._delta(
                        content=self._normalize_tool_content(tool_content),
                    )
                self.is_reasoning_ended = True
                return self._delta(reasoning=before, content=after)

            if special_string == self.function_call_role:
                self.is_tool_call_mode = True
                self.is_reasoning_ended = True
                return self._delta(
                    content=self._normalize_tool_content(after),
                )

        if self._is_partial_tool_role_fragment(self.buffer_string):
            return None

        if self.check_is_part_of_special_string():
            token_based_transition = self._extract_token_based_transition(
                previous_token_ids,
                current_token_ids,
            )
            if token_based_transition is not None:
                return token_based_transition
            return None

        delta_text = self.buffer_string
        self.buffer_string = ""

        if self.is_tool_call_mode:
            return None
        if self.think_end_string_base in current_text:
            self.is_delaying_assistant_handoff = False
            self.is_reasoning_ended = True
            return self._delta(content=delta_text)
        if not self.has_emitted_reasoning_text and delta_text.strip() == "":
            self.buffer_string = delta_text
            return None
        if delta_text.strip():
            self.has_emitted_reasoning_text = True
        return self._delta(reasoning=delta_text)

    def is_reasoning_end(self, input_ids: list[int]) -> bool:
        if self.is_delaying_assistant_handoff:
            return False

        if self._has_partial_think_end_suffix():
            return False

        if (
            self.no_reasoning_content
            or self.is_tool_call_mode
            or self.is_reasoning_ended
        ):
            return True

        # With skip_reasoning=True the generation prompt already ends at the
        # assistant content channel. Tell vLLM's streaming/structured-output
        # path that tool parsing may start from the first generated token.
        prompt_end_len = len(self.non_reasoning_prompt_end_tokens)
        if (
            self._is_streaming_skip_reasoning()
            and prompt_end_len > 0
            and len(input_ids) >= prompt_end_len
            and input_ids[-prompt_end_len:] == self.non_reasoning_prompt_end_tokens
        ):
            return True

        # Check using the passed input_ids (cumulative path used by the engine).
        for think_end_string, think_end_tokens in self._think_end_specs:
            think_end_len = len(think_end_tokens)
            if (
                think_end_len > 0
                and len(input_ids) >= think_end_len
                and input_ids[-think_end_len:] == think_end_tokens
            ):
                return True

        for think_end_string, think_end_tokens in self._think_end_specs:
            think_end_len = len(think_end_tokens)
            if (
                think_end_len > 0
                and len(self._token_history) >= think_end_len
                and self._token_history[-think_end_len:] == think_end_tokens
            ):
                return True

        return False

    def extract_content_ids(self, input_ids: list[int]) -> list[int]:
        prompt_end_len = len(self.non_reasoning_prompt_end_tokens)
        if (
            self._is_streaming_skip_reasoning()
            and prompt_end_len > 0
            and len(input_ids) >= prompt_end_len
        ):
            for index in range(len(input_ids) - prompt_end_len, -1, -1):
                if (
                    input_ids[index : index + prompt_end_len]
                    == self.non_reasoning_prompt_end_tokens
                ):
                    return input_ids[index + prompt_end_len :]

        for think_end_tokens in self.think_end_tokens:
            think_end_len = len(think_end_tokens)
            if think_end_len == 0 or len(input_ids) < think_end_len:
                continue

            for index in range(len(input_ids) - think_end_len, -1, -1):
                if input_ids[index : index + think_end_len] == think_end_tokens:
                    return input_ids[index + think_end_len :]

        return []
