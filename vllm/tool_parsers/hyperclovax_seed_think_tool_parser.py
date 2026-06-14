"""HyperCLOVAX-SEED-Think tool call parser for vLLM.

Tool call output format (as specified in the chat_template's system prompt):

    <tool_call>{function-name}
    <arg_key>{k1}</arg_key>
    <arg_value>{v1}</arg_value>
    ...
    </tool_call>

Multiple ``<tool_call>`` blocks may appear consecutively, separated by ``\\n``.

Full output with thinking=true:
    [reasoning]</think>\\n\\n[content?]<tool_call>...</tool_call>...<|im_end|>
With thinking=false the prompt embeds ``</think>`` so the output omits it:
    [content?]<tool_call>...</tool_call>...<|im_end|>

For ``tool_choice="required"`` and named tool_choice, vLLM injects a JSON
list schema via ``StructuredOutputsParams``. When the schema actually
engages (thinking=true case, after ``</think>``) the model produces
``[{"name": ..., "parameters": {...}}]`` instead of the XML form. We accept
both formats; see ``_emit_json_tool_calls`` for the streaming JSON path.

Streaming policy:
  vLLM's ``parse_delta`` already routes the reasoning phase through the
  reasoning parser, so we only receive post-``</think>`` deltas. Text
  outside ``<tool_call>`` blocks is emitted as ``content`` and complete
  ``<tool_call>...</tool_call>`` blocks as ``tool_calls``. Multiple
  complete blocks observed in a single delta are bundled into one
  ``DeltaMessage`` to avoid losing them when the stream ends.
"""

import json
import re
from collections.abc import Sequence

from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.entrypoints.openai.engine.protocol import (
    DeltaFunctionCall,
    DeltaMessage,
    DeltaToolCall,
    ExtractedToolCallInformation,
    FunctionCall,
    ToolCall,
)
from vllm.logger import init_logger
from vllm.tool_parsers.abstract_tool_parser import ToolParser

logger = init_logger(__name__)

TOOL_CALL_OPEN = "<tool_call>"
TOOL_CALL_CLOSE = "</tool_call>"
THINK_END = "</think>"
IM_END = "<|im_end|>"

_TOOL_CALL_RE = re.compile(
    re.escape(TOOL_CALL_OPEN) + r"(.*?)" + re.escape(TOOL_CALL_CLOSE),
    re.DOTALL,
)
_ARG_PAIR_RE = re.compile(
    r"<arg_key>(.*?)</arg_key>\s*<arg_value>(.*?)</arg_value>",
    re.DOTALL,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _partial_prefix_len(text: str, target: str) -> int:
    """Length of the longest prefix of ``target`` matching the suffix of ``text``.

    Used to detect protocol tokens split across delta boundaries. The exact-
    match case (full ``target`` as a suffix) is excluded; callers should check
    that separately with ``in`` or ``find``.
    """
    max_len = min(len(text), len(target) - 1)
    for ln in range(max_len, 0, -1):
        if text[-ln:] == target[:ln]:
            return ln
    return 0


def _partial_prefix_len_any(text: str, targets: list[str]) -> int:
    """Maximum partial-prefix length of ``text`` against any of ``targets``."""
    return max((_partial_prefix_len(text, t) for t in targets), default=0)


def _parse_tool_call_block(block: str) -> ToolCall | None:
    """Parse the body of a ``<tool_call>...</tool_call>`` block into a ToolCall.

    The chat_template's system prompt teaches the model to emit arguments as
    ``<arg_key>{k}</arg_key>\\n<arg_value>{v}</arg_value>`` pairs. Each value
    is JSON-loaded when possible (numbers/bool/null/list/dict) and falls back
    to the raw string otherwise.
    """
    name, _, rest = block.strip().partition("\n")
    name = name.strip()
    if not name:
        return None

    arguments: dict = {}
    for k, v in _ARG_PAIR_RE.findall(rest):
        k = k.strip()
        v = v.strip()
        try:
            arguments[k] = json.loads(v)
        except ValueError:
            arguments[k] = v

    return ToolCall(
        type="function",
        function=FunctionCall(
            name=name,
            arguments=json.dumps(arguments, ensure_ascii=False),
        ),
    )


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------


class HyperCLOVAXSeedThinkToolParser(ToolParser):
    """Tool call parser for the HyperCLOVAX-SEED-Think model.

    Parses ``<tool_call>...</tool_call>`` blocks (or the alternate
    ``<arguments>`` body) from the model output. Streaming uses a cursor over
    the accumulated buffer to handle multi-block and split-delta cases.
    """

    # HyperCLOVAX-SEED-Think emits tool calls in the XML
    # ``<tool_call>...</tool_call>`` format defined in its chat_template. For
    # ``tool_choice="required"`` or named tool_choice, vLLM normally enforces
    # a JSON list schema via ``StructuredOutputsParams`` and uses its built-in
    # JSON extractor. With ``thinking=true`` the schema kicks in after
    # ``</think>`` and the model emits JSON; with ``thinking=false`` the
    # ``</think>`` is already in the prompt so the schema never engages and
    # the model produces natural XML. To handle both consistently, we set
    # ``supports_required_and_named = False`` so that all tool_choice modes
    # route through this parser, which then accepts either the XML format or
    # the JSON list fallback.
    supports_required_and_named: bool = False

    def __init__(self, tokenizer, tools=None):
        super().__init__(tokenizer, tools)

        # Streaming state
        self.buffer_string: str = ""
        self.cursor: int = 0
        self.reasoning_ended: bool = False
        self.json_tool_emitted: bool = False

    # ------------------------------------------------------------------
    # Non-streaming
    # ------------------------------------------------------------------

    def extract_tool_calls(
        self,
        model_output: str,
        request: ChatCompletionRequest,
    ) -> ExtractedToolCallInformation:
        # Primary: XML ``<tool_call>...</tool_call>`` format from the chat_template.
        if TOOL_CALL_OPEN in model_output:
            try:
                tool_calls = [
                    tc
                    for block in _TOOL_CALL_RE.findall(model_output)
                    if (tc := _parse_tool_call_block(block)) is not None
                ]
                if not tool_calls:
                    raise ValueError("no valid tool calls parsed")

                # Text before the first <tool_call> → content. Use ``</think>``
                # presence directly; more robust than relying on the
                # chat_template_kwargs flag.
                before_tool, _, _ = model_output.partition(TOOL_CALL_OPEN)
                _, sep, after_think = before_tool.partition(THINK_END)
                content = (after_think if sep else before_tool).strip() or None

                return ExtractedToolCallInformation(
                    tools_called=True,
                    tool_calls=tool_calls,
                    content=content,
                )
            except Exception:
                logger.exception("Error extracting XML tool call.")
                return ExtractedToolCallInformation(
                    tools_called=False, tool_calls=[], content=model_output
                )

        # Fallback: JSON list ``[{"name": ..., "parameters": {...}}]`` —
        # produced when vLLM's ``required``/named-tool path forces the model
        # into a JSON schema via ``StructuredOutputsParams``. Strip any
        # leading reasoning so we can locate the JSON payload.
        _, sep, post_think = model_output.partition(THINK_END)
        candidate = (post_think if sep else model_output).strip()
        if candidate.startswith("[") or candidate.startswith("{"):
            try:
                parsed = json.loads(candidate)
                if isinstance(parsed, dict):
                    parsed = [parsed]
                if isinstance(parsed, list) and parsed:
                    tool_calls = []
                    for item in parsed:
                        if not isinstance(item, dict):
                            continue
                        name = item.get("name") or item.get("function", {}).get("name")
                        if not name:
                            continue
                        # ``parameters`` (Function schema) or ``arguments``
                        args_obj = item.get("parameters")
                        if args_obj is None:
                            args_obj = item.get("arguments", {})
                        if isinstance(args_obj, str):
                            args_str = args_obj
                        else:
                            args_str = json.dumps(args_obj, ensure_ascii=False)
                        tool_calls.append(
                            ToolCall(
                                type="function",
                                function=FunctionCall(name=name, arguments=args_str),
                            )
                        )
                    if tool_calls:
                        return ExtractedToolCallInformation(
                            tools_called=True,
                            tool_calls=tool_calls,
                            content=None,
                        )
            except (ValueError, TypeError):
                pass

        return ExtractedToolCallInformation(
            tools_called=False, tool_calls=[], content=model_output
        )

    # ------------------------------------------------------------------
    # Streaming
    # ------------------------------------------------------------------

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
        self.buffer_string += delta_text

        # vLLM's parse_delta only routes here after the reasoning phase is
        # over (it strips ``</think>`` and earlier tokens before invoking us).
        # If our buffer still happens to contain ``</think>`` — e.g. when this
        # parser is exercised standalone — advance past it; otherwise this is
        # a no-op.
        if not self.reasoning_ended:
            idx = self.buffer_string.find(THINK_END)
            if idx != -1:
                self.cursor = idx + len(THINK_END)
                while (
                    self.cursor < len(self.buffer_string)
                    and self.buffer_string[self.cursor] == "\n"
                ):
                    self.cursor += 1
                self.reasoning_ended = True
            elif _partial_prefix_len(self.buffer_string, THINK_END) > 0:
                # ``</think>`` may straddle deltas; hold the buffer until the
                # tag completes so we don't leak its tail as content.
                return None
            else:
                # No </think> in the buffer and no partial match — production
                # path (reasoning parser already stripped it) or thinking=false.
                self.reasoning_ended = True

        # JSON list fallback for ``tool_choice="required"`` / named when the
        # structured-output schema forces the model into a JSON payload
        # (``[{"name": ..., "parameters": {...}}]``). We only need to handle
        # this path here because, with ``supports_required_and_named=False``,
        # all tool_choice modes route through this parser.
        unprocessed = self.buffer_string[self.cursor :].lstrip()
        if not self.json_tool_emitted and unprocessed.startswith(("[", "{")):
            return self._emit_json_tool_calls()

        return self._emit_next()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _emit_next(self) -> DeltaMessage | None:
        """Drain processable content / tool_calls into one DeltaMessage."""
        content_parts: list[str] = []
        tool_call_deltas: list[DeltaToolCall] = []

        while self.cursor < len(self.buffer_string):
            unprocessed = self.buffer_string[self.cursor :]
            open_idx = unprocessed.find(TOOL_CALL_OPEN)

            # Case A: no <tool_call> in remaining → stream as content.
            if open_idx == -1:
                # Hold any tail that is a partial of <tool_call> or <|im_end|>
                # to avoid leaking a split protocol token across delta boundaries.
                partial_len = _partial_prefix_len_any(
                    unprocessed, [TOOL_CALL_OPEN, IM_END]
                )
                safe_end = len(unprocessed) - partial_len
                if safe_end <= 0:
                    break
                chunk = unprocessed[:safe_end].replace(IM_END, "")
                self.cursor += safe_end
                if chunk:
                    content_parts.append(chunk)
                if partial_len > 0:
                    break
                continue

            # Case B: text precedes the next <tool_call> → emit as content first.
            if open_idx > 0:
                pre = unprocessed[:open_idx].replace(IM_END, "")
                self.cursor += open_idx
                # Whitespace-only segments between tool_calls are separators; drop them.
                if pre.strip():
                    content_parts.append(pre)
                continue

            # Case C: cursor sits at <tool_call> — wait for </tool_call>.
            close_idx = unprocessed.find(TOOL_CALL_CLOSE, len(TOOL_CALL_OPEN))
            if close_idx == -1:
                break

            block = unprocessed[len(TOOL_CALL_OPEN) : close_idx]
            self.cursor += close_idx + len(TOOL_CALL_CLOSE)

            tc = _parse_tool_call_block(block)
            if tc is None:
                continue

            self.current_tool_id += 1
            self.prev_tool_call_arr.append(
                {
                    "name": tc.function.name,
                    "arguments": tc.function.arguments,
                }
            )
            self.streamed_args_for_tool.append(tc.function.arguments)

            tool_call_deltas.append(
                DeltaToolCall(
                    index=self.current_tool_id,
                    type="function",
                    id=f"hyperclovax_seed_think_tool_{self.current_tool_id}",
                    function=DeltaFunctionCall(
                        name=tc.function.name,
                        arguments=tc.function.arguments,
                    ).model_dump(exclude_none=True),
                )
            )

        if not content_parts and not tool_call_deltas:
            return None

        # ``DeltaMessage.tool_calls`` rejects ``None`` (must be a list), so only
        # set the field when we actually have tool_call deltas to emit.
        kwargs: dict = {}
        if content_parts:
            kwargs["content"] = "".join(content_parts)
        if tool_call_deltas:
            kwargs["tool_calls"] = tool_call_deltas
        return DeltaMessage(**kwargs)

    def _emit_json_tool_calls(self) -> DeltaMessage | None:
        """Emit tool_calls parsed from a JSON list payload.

        Called when the buffered output looks like a JSON object/array rather
        than the XML ``<tool_call>`` format. Waits for a fully parseable
        payload and then emits all tool_calls in one ``DeltaMessage``.
        """
        payload = self.buffer_string[self.cursor :].lstrip()
        try:
            parsed = json.loads(payload)
        except ValueError:
            return None  # not yet complete; wait for more

        if isinstance(parsed, dict):
            parsed = [parsed]
        if not isinstance(parsed, list) or not parsed:
            return None

        tool_call_deltas: list[DeltaToolCall] = []
        for item in parsed:
            if not isinstance(item, dict):
                continue
            name = item.get("name") or (item.get("function") or {}).get("name")
            if not name:
                continue
            args_obj = item.get("parameters")
            if args_obj is None:
                args_obj = item.get("arguments", {})
            args_str = (
                args_obj
                if isinstance(args_obj, str)
                else json.dumps(args_obj, ensure_ascii=False)
            )

            self.current_tool_id += 1
            self.prev_tool_call_arr.append({"name": name, "arguments": args_str})
            self.streamed_args_for_tool.append(args_str)

            tool_call_deltas.append(
                DeltaToolCall(
                    index=self.current_tool_id,
                    type="function",
                    id=f"hyperclovax_seed_think_tool_{self.current_tool_id}",
                    function=DeltaFunctionCall(
                        name=name,
                        arguments=args_str,
                    ).model_dump(exclude_none=True),
                )
            )

        if not tool_call_deltas:
            return None

        self.json_tool_emitted = True
        self.cursor = len(self.buffer_string)
        return DeltaMessage(tool_calls=tool_call_deltas)
