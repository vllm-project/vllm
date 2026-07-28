# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Sequence

from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionRequest,
    ChatCompletionToolsParam,
)
from vllm.entrypoints.openai.engine.protocol import (
    DeltaMessage,
    ExtractedToolCallInformation,
)
from vllm.logger import init_logger
from vllm.tool_parsers.qwen3coder_tool_parser import Qwen3CoderToolParser
from vllm.tool_parsers.utils import iter_response_function_tool_info

logger = init_logger(__name__)


class IquestCoderToolParser(Qwen3CoderToolParser):
    """Tool parser for the iQuest coder model family.

    The iQuest coder models share the qwen3_coder XML tool-call format
    (``<tool_call><function=...><parameter=...>...``), so this parser reuses
    :class:`Qwen3CoderToolParser` for all parsing logic and only fixes one
    whitespace edge case, in both the streaming and non-streaming paths.

    Root cause: the iQuest chat template separates tool calls with formatting
    whitespace, e.g. a ``"\\n"`` before ``<tool_call>`` and between
    ``</tool_call>`` and the next ``<tool_call>``. The base parser treats the
    text preceding a ``<tool_call>`` as content, so this pure scaffolding
    whitespace surfaces as stray, whitespace-only text blocks around the
    ``tool_use`` blocks in the response.

    Non-streaming fix (``extract_tool_calls``): when tool calls were parsed,
    drop the leading content if it is whitespace-only.

    Streaming fix (``extract_tool_calls_streaming``): depending on
    tokenization, the inter-call whitespace can be fused with the following
    ``<tool_call>`` start token into a single delta (``"\\n<tool_call>"``),
    which the base parser returns as a content delta. Drop such
    whitespace-only content whenever we are between tool calls
    (``current_tool_index > 0``). Genuine text is preserved in both paths.
    """

    def _reset_streaming_state(self):
        # The base parser owns ``streamed_args_for_tool`` but never resets it
        # (it doesn't populate it either). Since we mirror argument fragments
        # into it, clear it here so a reused parser instance starts each stream
        # with a clean slate and stays aligned with ``prev_tool_call_arr``.
        super()._reset_streaming_state()
        self.streamed_args_for_tool = []

    def _get_arguments_config(
        self, func_name: str, tools: list[ChatCompletionToolsParam] | None
    ) -> dict:
        """Extract argument configuration for a function.

        The base parser only understands the Chat Completions tool shape
        (name/parameters nested under ``.function``). On the ``/v1/responses``
        path ``request.tools`` are Responses API tools (``FunctionTool`` /
        ``NamespaceTool`` with ``name``/``parameters`` on the tool itself, and
        namespace children exposed as ``namespace__name``). Resolve those here
        so typed parameters (e.g. an integer ``timeout_ms``) are converted from
        their string form instead of leaking through as strings; fall back to
        the base (chat-shape) lookup otherwise.
        """
        for tool in tools or []:
            # Chat Completions tools have a nested ``.function``; leave those to
            # the base implementation.
            if getattr(tool, "function", None) is not None:
                continue
            if getattr(tool, "type", None) not in ("function", "namespace"):
                continue
            for name, params in iter_response_function_tool_info(tool):
                if name != func_name:
                    continue
                if isinstance(params, dict) and "properties" in params:
                    return params["properties"]
                if isinstance(params, dict):
                    return params
                return {}
        return super()._get_arguments_config(func_name, tools)

    def extract_tool_calls(
        self,
        model_output: str,
        request: ChatCompletionRequest,
    ) -> ExtractedToolCallInformation:
        info = super().extract_tool_calls(model_output, request)

        # Drop whitespace-only leading content when tool calls were found. The
        # base parser returns everything before the first ``<tool_call>`` as
        # content; for the iQuest template that leading text is just the
        # newline scaffolding separating prompt from the tool calls, and should
        # not appear as a stray text block. Genuine content is preserved.
        if (
            info.tools_called
            and info.content is not None
            and info.content.strip() == ""
        ):
            info.content = None

        return info

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
        delta_message = super().extract_tool_calls_streaming(
            previous_text,
            current_text,
            delta_text,
            previous_token_ids,
            current_token_ids,
            delta_token_ids,
            request,
        )

        # Mirror every streamed argument fragment into
        # ``streamed_args_for_tool``. The base Qwen3Coder parser emits argument
        # deltas (``{``, per-parameter JSON, string chunks, the closing ``}``)
        # but never records them here, leaving the list empty. The serving
        # layer indexes ``streamed_args_for_tool[len(prev_tool_call_arr) - 1]``
        # when flushing any unstreamed trailing arguments on the final chunk;
        # against an empty list that raises ``IndexError``. This surfaces
        # whenever a tool call is truncated mid-arguments (e.g. hitting
        # max_tokens before the closing ``</function>``), because the last
        # delta still carries a non-``None`` ``arguments`` and thus triggers
        # that flush. Keeping the two lists in lockstep here fixes it.
        if delta_message is not None and delta_message.tool_calls:
            for tool_call in delta_message.tool_calls:
                if tool_call.function is None:
                    continue
                args = tool_call.function.arguments
                if args is None:
                    continue
                index = tool_call.index
                while len(self.streamed_args_for_tool) <= index:
                    self.streamed_args_for_tool.append("")
                self.streamed_args_for_tool[index] += args

        # Drop whitespace-only content emitted between tool calls. The base
        # class has already advanced its streaming state (e.g. set
        # is_tool_call_started), so returning None here only suppresses the
        # stray content block without losing any tool-call progress.
        if (
            delta_message is not None
            and self.current_tool_index > 0
            and delta_message.content is not None
            and not delta_message.tool_calls
            and delta_message.content.strip() == ""
        ):
            return None

        return delta_message
