# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import ast
from collections.abc import Sequence

import regex as re

import vllm.envs as envs
from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionRequest,
)
from vllm.entrypoints.openai.engine.protocol import (
    DeltaMessage,
    ExtractedToolCallInformation,
)
from vllm.logger import init_logger
from vllm.tokenizers import TokenizerLike
from vllm.tool_parsers.abstract_tool_parser import (
    Tool,
    ToolParser,
)
from vllm.tool_parsers.utils import (
    UnexpectedAstError,
    compute_tool_delta,
    handle_single_tool,
    make_valid_python,
)

logger = init_logger(__name__)

TOOL_CALL_START = "<|tool_call_start|>"
TOOL_CALL_END = "<|tool_call_end|>"


class Lfm2ToolParser(ToolParser):
    """
    Tool call parser for LiquidAI LFM2/LFM2.5 models that produce pythonic
    tool calls wrapped in <|tool_call_start|> and <|tool_call_end|> tokens.

    Example model output:
        <|tool_call_start|>[get_weather(location="Paris")]<|tool_call_end|>
        The weather in Paris is sunny.

    Used when --enable-auto-tool-choice --tool-call-parser lfm2 are all set.
    """

    TOOL_CALL_REGEX = re.compile(
        r"\[([a-zA-Z]+\w*\(([a-zA-Z]+\w*=.*,\s*)*([a-zA-Z]+\w*=.*\s)?\)"
        r",\s*)*([a-zA-Z]+\w*\(([a-zA-Z]+\w*=.*,\s*)*"
        r"([a-zA-Z]+\w*=.*\s*)?\)\s*)+\]",
        re.DOTALL,
    )

    def __init__(
        self,
        tokenizer: TokenizerLike,
        tools: list[Tool] | None = None,
    ):
        super().__init__(tokenizer, tools)

        self.tool_call_start_token_id = self.vocab.get(TOOL_CALL_START)
        self.tool_call_end_token_id = self.vocab.get(TOOL_CALL_END)

        if self.tool_call_start_token_id is None or self.tool_call_end_token_id is None:
            raise RuntimeError(
                "LFM2 tool parser could not locate "
                "<|tool_call_start|>/<|tool_call_end|> tokens in the "
                "tokenizer!"
            )

    # Rename for readability. This is NOT a tool id.
    @property
    def current_tool_index(self) -> int:
        return self.current_tool_id

    @current_tool_index.setter
    def current_tool_index(self, value: int) -> None:
        self.current_tool_id = value

    @staticmethod
    def _extract_tool_call_text(model_output: str) -> tuple[str | None, str | None]:
        """Extract the pythonic call text and surrounding content.

        Returns (tool_text, content) where tool_text is the text between
        the sentinel tokens and content is everything outside them.
        """
        start_idx = model_output.find(TOOL_CALL_START)
        if start_idx == -1:
            return None, model_output

        end_idx = model_output.find(TOOL_CALL_END, start_idx)
        if end_idx == -1:
            # Incomplete — treat entire text after start as tool call
            tool_text = model_output[start_idx + len(TOOL_CALL_START) :]
            content_before = model_output[:start_idx].strip()
            content = content_before or None
            return tool_text, content

        tool_text = model_output[start_idx + len(TOOL_CALL_START) : end_idx]
        content_before = model_output[:start_idx].strip()
        content_after = model_output[end_idx + len(TOOL_CALL_END) :].strip()

        content_parts = []
        if content_before:
            content_parts.append(content_before)
        if content_after:
            content_parts.append(content_after)
        content = "\n".join(content_parts) if content_parts else None

        return tool_text, content

    def extract_tool_calls(
        self, model_output: str, request: ChatCompletionRequest
    ) -> ExtractedToolCallInformation:
        tool_text, content = self._extract_tool_call_text(model_output)

        if tool_text is None:
            return ExtractedToolCallInformation(
                tools_called=False, tool_calls=[], content=model_output
            )

        tool_text = tool_text.strip()

        is_tool_call_pattern = False
        try:
            is_tool_call_pattern = (
                self.TOOL_CALL_REGEX.match(
                    tool_text,
                    timeout=envs.VLLM_TOOL_PARSE_REGEX_TIMEOUT_SECONDS,
                )
                is not None
            )
        except TimeoutError:
            logger.warning("Regex timeout occurred when matching tool call pattern.")

        if not is_tool_call_pattern:
            return ExtractedToolCallInformation(
                tools_called=False, tool_calls=[], content=model_output
            )

        try:
            module = ast.parse(tool_text)
            parsed = getattr(module.body[0], "value", None)
            if isinstance(parsed, ast.List) and all(
                isinstance(e, ast.Call) for e in parsed.elts
            ):
                return ExtractedToolCallInformation(
                    tools_called=True,
                    tool_calls=[
                        handle_single_tool(e)  # type: ignore
                        for e in parsed.elts
                    ],
                    content=content,
                )
            else:
                raise UnexpectedAstError("Tool output must be a list of function calls")
        except Exception:
            logger.exception("Error in extracting tool call from response.")
            return ExtractedToolCallInformation(
                tools_called=False, tool_calls=[], content=model_output
            )

    def extract_tool_calls_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
        request: ChatCompletionRequest,
    ) -> DeltaMessage | None:
        # If the tool call start token hasn't appeared yet, stream as content.
        if TOOL_CALL_START not in current_text:
            return DeltaMessage(content=delta_text)

        # If the tool call end token appeared and tools were already parsed,
        # stream any remaining content after the end token.
        if TOOL_CALL_END in current_text and self.prev_tool_call_arr:
            after_end = current_text.split(TOOL_CALL_END, 1)[1]
            prev_after_end = (
                previous_text.split(TOOL_CALL_END, 1)[1]
                if TOOL_CALL_END in previous_text
                else ""
            )
            new_content = after_end[len(prev_after_end) :]
            if new_content:
                return DeltaMessage(content=new_content)
            return DeltaMessage(content="")

        # Extract the pythonic text between start and end tokens.
        tool_text = current_text.split(TOOL_CALL_START, 1)[1]
        # Strip the end token if present (entire call arrived at once).
        if TOOL_CALL_END in tool_text:
            tool_text = tool_text.split(TOOL_CALL_END, 1)[0]

        try:
            valid_and_added_text = make_valid_python(tool_text)
            if valid_and_added_text is None:
                return None
            valid_text, added_text = valid_and_added_text

            module = ast.parse(valid_text)
            parsed = getattr(module.body[0], "value", None)
            if not isinstance(parsed, ast.List) or not all(
                isinstance(e, ast.Call) for e in parsed.elts
            ):
                raise UnexpectedAstError("Tool output must be a list of function calls")
            tool_calls = [
                handle_single_tool(e)  # type: ignore
                for e in parsed.elts
            ]

            tool_deltas = []
            for index, new_call in enumerate(tool_calls):
                if index < self.current_tool_index:
                    continue

                self.current_tool_index = index
                if len(self.streamed_args_for_tool) == index:
                    self.streamed_args_for_tool.append("")

                new_call_complete = (
                    index < len(tool_calls) - 1 or ")]" not in added_text
                )
                if new_call_complete:
                    self.current_tool_index += 1

                withheld_suffix = added_text[:-2] if not new_call_complete else ""
                if not new_call_complete and added_text[-2] == ")":
                    withheld_suffix = withheld_suffix + "}"
                withheld_suffix = withheld_suffix.replace("'", '"')
                delta = compute_tool_delta(
                    self.streamed_args_for_tool[index],
                    new_call,
                    index,
                    withheld_suffix,
                )

                if delta is not None:
                    tool_deltas.append(delta)
                    if (
                        delta.function is not None
                        and delta.function.arguments is not None
                    ):
                        self.streamed_args_for_tool[index] += delta.function.arguments

            if tool_deltas and not self.prev_tool_call_arr:
                self.prev_tool_call_arr = [{"arguments": {}}]

            if tool_deltas:
                return DeltaMessage(tool_calls=tool_deltas)
            elif not added_text and self.current_tool_id > 0:
                return DeltaMessage(content="")
            else:
                return None
        except Exception:
            logger.exception("Error trying to handle streaming tool call.")
            logger.debug(
                "Skipping chunk as a result of tool streaming extraction error"
            )
            return None
