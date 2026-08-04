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
from vllm.entrypoints.openai.responses.protocol import ResponsesRequest
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

    TOOL_CALL_REGEX = re.compile(r"\[.*\]$", re.DOTALL)

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

        # Trailing content already emitted to the client. Used by the
        # streaming path to suppress LFM2's frequent echo of the tool
        # call body after the first <|tool_call_end|> while still
        # allowing legitimate post-call prose through.
        self._trailing_emitted: str = ""

    def adjust_request(
        self, request: ChatCompletionRequest | ResponsesRequest
    ) -> ChatCompletionRequest | ResponsesRequest:
        request = super().adjust_request(request)
        if request.tools and request.tool_choice != "none":
            # The <|tool_call_start|>/<|tool_call_end|> sentinels are
            # registered as special tokens in the LFM2/LFM2.5 tokenizer.
            # With the default ``skip_special_tokens=True`` they are
            # stripped from the decoded text before reaching this parser,
            # so the tool block becomes invisible. Force the engine to
            # preserve them when tool calling is enabled.
            request.skip_special_tokens = False
        return request

    # Rename for readability. This is NOT a tool id.
    @property
    def current_tool_index(self) -> int:
        return self.current_tool_id

    @current_tool_index.setter
    def current_tool_index(self, value: int) -> None:
        self.current_tool_id = value

    @staticmethod
    def _strip_echo(raw_after: str) -> str:
        """Drop any orphan <|tool_call_end|> (and the preceding text) from
        trailing content. LFM2 occasionally echoes the call body after the
        first end token and caps it with a second end token; everything
        through the last such orphan is model garbage, not user content."""
        last_orphan = raw_after.rfind(TOOL_CALL_END)
        if last_orphan != -1:
            return raw_after[last_orphan + len(TOOL_CALL_END) :]
        return raw_after

    @classmethod
    def _extract_tool_call_text(
        cls, model_output: str
    ) -> tuple[str | None, str | None]:
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
        content_after = cls._strip_echo(
            model_output[end_idx + len(TOOL_CALL_END) :]
        ).strip()

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

        # Compute leading content (before <|tool_call_start|>) that arrived
        # in this delta and hasn't been streamed yet. Without this, when the
        # prefix and the start token land in the same delta the prefix is
        # silently dropped — token-by-token streaming masked the bug because
        # the prefix tokens always arrived in earlier deltas.
        leading_content = ""
        if TOOL_CALL_START not in previous_text:
            start_idx = current_text.find(TOOL_CALL_START)
            # previous_text contained no start token, so it has already been
            # streamed via the no-start-token branch above.
            leading_content = current_text[len(previous_text) : start_idx]

        has_end_in_current = TOOL_CALL_END in current_text
        has_end_in_previous = TOOL_CALL_END in previous_text

        # Compute trailing content (after <|tool_call_end|>) not yet
        # streamed. LFM2 frequently echoes the tool call body again
        # after the first end token, capped with a second end token.
        # Suppress that echo:
        #   - If a second <|tool_call_end|> has appeared, treat
        #     everything through the last one as garbage.
        #   - If the trailing starts with `[` or `<` (potential echo
        #     body or another sentinel) and no second end token has
        #     arrived yet, buffer it instead of emitting.
        trailing_content = ""
        if has_end_in_current:
            end_idx = current_text.find(TOOL_CALL_END) + len(TOOL_CALL_END)
            full_trailing = current_text[end_idx:]
            stripped_trailing = self._strip_echo(full_trailing)
            if stripped_trailing == full_trailing:
                # No second end token yet — possibly mid-echo.
                lstripped = full_trailing.lstrip()
                if lstripped.startswith("[") or lstripped.startswith("<"):
                    # Suspect echo; hold off until resolved.
                    final_trailing = self._trailing_emitted
                else:
                    final_trailing = full_trailing
            else:
                final_trailing = stripped_trailing
            if final_trailing.startswith(self._trailing_emitted):
                trailing_content = final_trailing[len(self._trailing_emitted) :]
            self._trailing_emitted = final_trailing

        # If tools were already parsed in a prior delta, just stream any
        # newly arrived trailing content.
        if has_end_in_current and self.prev_tool_call_arr and has_end_in_previous:
            if trailing_content:
                return DeltaMessage(content=trailing_content)
            return DeltaMessage(content="")

        # Extract the pythonic text between start and end tokens.
        tool_text = current_text.split(TOOL_CALL_START, 1)[1]
        # Strip the end token if present (entire call arrived at once).
        if TOOL_CALL_END in tool_text:
            tool_text = tool_text.split(TOOL_CALL_END, 1)[0]

        def _content_only_or_none() -> DeltaMessage | None:
            """Return a content-only delta if any content arrived in this
            chunk, otherwise None. Used on incremental-parse failure paths
            so leading/trailing content is never silently dropped.
            """
            combined = leading_content + trailing_content
            return DeltaMessage(content=combined) if combined else None

        try:
            valid_and_added_text = make_valid_python(tool_text)
            if valid_and_added_text is None:
                return _content_only_or_none()
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

            combined_content = leading_content + trailing_content

            if tool_deltas or combined_content:
                return DeltaMessage(
                    content=combined_content if combined_content else None,
                    tool_calls=tool_deltas,
                )
            elif not added_text and self.current_tool_id > 0:
                return DeltaMessage(content="")
            else:
                return None
        except Exception:
            logger.exception("Error trying to handle streaming tool call.")
            logger.debug(
                "Skipping chunk as a result of tool streaming extraction error"
            )
            return _content_only_or_none()
