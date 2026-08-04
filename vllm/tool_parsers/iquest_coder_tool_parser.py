# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import json
from collections.abc import Sequence

from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionRequest,
    ChatCompletionToolsParam,
)
from vllm.entrypoints.openai.engine.protocol import (
    DeltaFunctionCall,
    DeltaMessage,
    DeltaToolCall,
    ExtractedToolCallInformation,
)
from vllm.logger import init_logger
from vllm.tokenizers import TokenizerLike
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

    def __init__(self, tokenizer: TokenizerLike):
        super().__init__(tokenizer)
        self._headers_sent: list[bool] = []
        self._content_sent: int = 0

    def adjust_request(self, request: ChatCompletionRequest) -> ChatCompletionRequest:
        """Keep special tokens in the decoded text while tools are in play.

        ``<tool_call>`` / ``</tool_call>`` are *special* tokens on the iQuest
        tokenizers (ids 14/15 on M1-A15B-SFT-256K), so with the default
        ``skip_special_tokens=True`` they are stripped from the text this parser
        is handed and no tool call is ever recognised -- the whole
        ``<function=...>`` body streams out as plain assistant content. Same
        guard hermes / glm4_moe / deepseekv32 / iquest_coder_v2 apply.
        """
        request = super().adjust_request(request)
        if request.tools and request.tool_choice != "none":
            request.skip_special_tokens = False
        return request

    def _reset_streaming_state(self):
        super()._reset_streaming_state()
        # ``streamed_args_for_tool`` must start empty for every stream so it
        # stays aligned with ``prev_tool_call_arr`` -- the serving layer indexes
        # one by the length of the other when flushing trailing arguments.
        self.streamed_args_for_tool = []
        # Called from the base __init__ before our attributes exist, hence
        # plain assignment rather than mutating in place.
        self._headers_sent = []
        self._content_sent = 0

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

    def _calls_so_far(
        self, text: str, request: ChatCompletionRequest
    ) -> list[tuple[str | None, dict, bool]]:
        """Re-parse ``text`` from scratch into (name, arguments, complete) tuples.

        Only *finished* parameters of an unfinished call are reported, so the
        serialized arguments of a call grow monotonically as more text arrives.
        """
        out: list[tuple[str | None, dict, bool]] = []
        tools = getattr(request, "tools", None)
        idx = 0
        while True:
            start = text.find(self.tool_call_start_token, idx)
            if start == -1:
                break
            body_start = start + len(self.tool_call_start_token)
            end = text.find(self.tool_call_end_token, body_start)
            body = text[body_start : end if end != -1 else len(text)]

            fn_at = body.find(self.tool_call_prefix)  # "<function="
            if fn_at == -1:
                break  # name not emitted yet -- nothing to report for this call
            fn_from = fn_at + len(self.tool_call_prefix)
            fn_end = body.find(self.function_end_token, fn_from)
            complete = end != -1 and fn_end != -1
            func_content = body[fn_from : fn_end if fn_end != -1 else len(body)]

            if not complete:
                # Keep only through the last closed </parameter> so a half
                # written value is never emitted.
                cut = func_content.rfind(self.parameter_end_token)
                if cut == -1:
                    name_end = func_content.find(">")
                    func_content = (
                        func_content[: name_end + 1] if name_end != -1 else ""
                    )
                else:
                    func_content = func_content[: cut + len(self.parameter_end_token)]

            name, args = None, {}
            if func_content:
                try:
                    parsed = self._parse_xml_function_call(func_content, tools)
                    if parsed:
                        name = parsed.function.name
                        args = json.loads(parsed.function.arguments or "{}")
                        if not isinstance(args, dict):
                            args = {}
                except Exception:  # partial text -- report what we have
                    pass
            if name is None:
                nm = func_content.split(">", 1)[0].strip()
                name = nm or None
            if name is not None:
                out.append((name, args, complete))

            if end == -1:
                break
            idx = end + len(self.tool_call_end_token)
        return out

    @staticmethod
    def _serialize(args: dict, complete: bool) -> str:
        """Serialize arguments so shorter states are prefixes of longer ones.

        ``json.dumps`` closes the object, and ``'{"a": 1}'`` is *not* a prefix of
        ``'{"a": 1, "b": 2}'``. Withholding the final ``}`` until the call is
        complete makes each successive state a true prefix, which is what lets
        the diff below emit only new bytes.
        """
        full = json.dumps(args, ensure_ascii=False)
        return full if complete else full[:-1]

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
        """Chunk-size-invariant streaming.

        The inherited Qwen3Coder implementation reacts to ``delta_text`` and
        tracks hand-rolled flags (json_started / json_closed / param_count /
        in_function), which makes its output depend on how the tokens happen to
        be chunked. With speculative decoding (MTP) a single delta can carry an
        entire tool call, and that produced truncated arguments -- even a
        one-parameter call arrived as ``{"file_path": "/x/main.py"`` with no
        closing brace, which clients reject as ``__unparsedToolInput`` /
        ``InputValidationError``. Measured: 97 of 194 tool calls malformed at
        num_speculative_tokens=2.

        This instead re-parses the whole accumulated text every call and emits
        the diff against what was already sent, so the result depends only on
        the text, never on delta boundaries -- the "safe-prefix extraction"
        approach vLLM's own streaming parser engine adopts upstream
        (vllm-project/vllm#44873, PR #45413), which this fork predates.
        """
        calls = self._calls_so_far(current_text, request)

        deltas: list[DeltaToolCall] = []
        for i, (name, args, complete) in enumerate(calls):
            while len(self.streamed_args_for_tool) <= i:
                self.streamed_args_for_tool.append("")
            while len(self._headers_sent) <= i:
                self._headers_sent.append(False)
            while len(self.prev_tool_call_arr) <= i:
                self.prev_tool_call_arr.append({"name": None, "arguments": {}})

            # Serving layer needs these: non-empty prev_tool_call_arr drives
            # finish_reason="tool_calls", and it json.dumps() "arguments" when
            # flushing any tail we did not stream (a no-op here, since by the
            # final delta we have emitted everything).
            self.prev_tool_call_arr[i]["name"] = name
            self.prev_tool_call_arr[i]["arguments"] = args

            new_header = not self._headers_sent[i]
            if new_header:
                self._headers_sent[i] = True

            fragment = ""
            target = self._serialize(args, complete)
            sent = self.streamed_args_for_tool[i]
            if target != sent:
                if target.startswith(sent):
                    fragment = target[len(sent) :]
                    self.streamed_args_for_tool[i] = target
                else:
                    # Should not happen (states are prefix-ordered); prefer
                    # emitting nothing over corrupt JSON and let the serving
                    # layer flush the tail from prev_tool_call_arr.
                    logger.warning(
                        "iquest_coder: non-monotonic arguments for tool %d (%s)",
                        i,
                        name,
                    )

            if not new_header and not fragment:
                continue

            # EXACTLY ONE entry per index per chunk. Emitting the header and the
            # first argument fragment as two DeltaToolCalls sharing an index made
            # clients keep only one of them, so the opening "{" vanished and
            # arguments arrived as '"file_path": "..."}' -- 227 of 234 tool calls
            # malformed. name and arguments may travel in the same delta.
            deltas.append(
                DeltaToolCall(
                    index=i,
                    id=self._generate_tool_call_id() if new_header else None,
                    type="function" if new_header else None,
                    function=DeltaFunctionCall(
                        name=name if new_header else None,
                        arguments=fragment or None,
                    ),
                )
            )

        self.current_tool_index = max(len(calls) - 1, 0)
        if calls:
            self.is_tool_call_started = True

        # Content is only whatever precedes the first tool call; the whitespace
        # the iQuest template puts between calls is scaffolding, not content.
        first = current_text.find(self.tool_call_start_token)
        region = current_text if first == -1 else current_text[:first]
        if first == -1:
            # Hold back a tail that could still turn out to be the start token,
            # or a partial "<tool_call" leaks out as content when the tokens
            # arrive in small chunks. (Upstream's lexer calls this prefix
            # buffering.)
            start_tok = self.tool_call_start_token
            for keep in range(len(start_tok) - 1, 0, -1):
                if region.endswith(start_tok[:keep]):
                    region = region[:-keep]
                    break
        new_content = region[self._content_sent :]
        self._content_sent = max(self._content_sent, len(region))
        if first != -1 and new_content.strip() == "":
            new_content = ""

        if not deltas and not new_content:
            return None
        return DeltaMessage(content=new_content or None, tool_calls=deltas or [])
