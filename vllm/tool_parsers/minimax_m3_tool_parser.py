# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import re
from collections.abc import Sequence
from typing import Any

from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.logger import init_logger
from vllm.tool_parsers.rust_tool_parser import RustToolParser

logger = init_logger(__name__)

# The MiniMax M3 namespace prefix that must precede every structural tag.
_NS = "]<]minimax[>["
_TOOL_CALL_OPEN = _NS + "<tool_call>"
_NS_ESC = re.escape(_NS)

# Matches any XML-like tag that is NOT already preceded by the namespace
# prefix.  Used to normalise model output that drops the prefix on inner tags.
_MISSING_NS_RE = re.compile(
    r"(?<!" + _NS_ESC + r")(</?[A-Za-z_][A-Za-z0-9_.:-]*(?:\s[^>]*)?>)"
)

# Used by _repair_unclosed_params to track open/close param tags inside <invoke>.
_NS_OPEN_TAG_RE = re.compile(_NS_ESC + r"<([A-Za-z_][A-Za-z0-9_-]*)>")
_NS_CLOSE_TAG_RE = re.compile(_NS_ESC + r"</([A-Za-z_][A-Za-z0-9_-]*)>")
_NS_INVOKE_OPEN_RE = re.compile(_NS_ESC + r"<invoke\b[^>]*>")
_NS_INVOKE_CLOSE = _NS + "</invoke>"


def _normalise_ns(text: str) -> str:
    """Re-add the MiniMax namespace prefix to any tag inside the tool_call
    block that is missing it.

    The model occasionally emits inner tags (``<invoke>``, ``</invoke>``,
    ``<param-name>``, ``</param-name>``) without the ``]<]minimax[>[`` prefix.
    The Rust parser requires every structural tag to carry it.  This function
    scopes the fix to the content *after* the opening ``]<]minimax[>[<tool_call>``
    token to avoid accidentally prefixing unrelated text.
    """
    start = text.find(_TOOL_CALL_OPEN)
    if start == -1:
        return text
    before = text[: start + len(_TOOL_CALL_OPEN)]
    inside = text[start + len(_TOOL_CALL_OPEN) :]
    return before + _MISSING_NS_RE.sub(_NS + r"\1", inside)


def _repair_unclosed_params(text: str) -> str:
    """Insert missing NS</param> closing tags before each NS</invoke>.

    The model occasionally opens a parameter tag (e.g. ``]<]minimax[>[<ranges>``)
    inside an ``<invoke>`` block but omits the matching closing tag, going
    directly to ``]<]minimax[>[</invoke>``.  The Rust parser rejects such output
    with a blank ``ValueError``.  This function detects unclosed parameter tags
    inside each ``<invoke>…</invoke>`` pair and injects the missing closers.
    """
    if _NS_INVOKE_CLOSE not in text:
        return text
    start = text.find(_TOOL_CALL_OPEN)
    if start == -1:
        return text

    result = text[: start + len(_TOOL_CALL_OPEN)]
    rest = text[start + len(_TOOL_CALL_OPEN) :]
    pos = 0
    while pos < len(rest):
        m = _NS_INVOKE_OPEN_RE.search(rest, pos)
        if m is None:
            result += rest[pos:]
            break
        result += rest[pos : m.end()]
        close_pos = rest.find(_NS_INVOKE_CLOSE, m.end())
        if close_pos == -1:
            result += rest[m.end() :]
            break

        invoke_body = rest[m.end() : close_pos]

        # Walk the body tracking open/close param tags.
        open_tags: list[str] = []
        scan = 0
        while scan < len(invoke_body):
            cm = _NS_CLOSE_TAG_RE.search(invoke_body, scan)
            om = _NS_OPEN_TAG_RE.search(invoke_body, scan)
            if om is None and cm is None:
                break
            if om is not None and (cm is None or om.start() < cm.start()):
                open_tags.append(om.group(1))
                scan = om.end()
            else:
                assert cm is not None
                tag = cm.group(1)
                if open_tags and open_tags[-1] == tag:
                    open_tags.pop()
                scan = cm.end()

        if open_tags:
            # Close unclosed tags in LIFO order before </invoke>.
            result += invoke_body + "".join(
                _NS + "</" + t + ">" for t in reversed(open_tags)
            )
        else:
            result += invoke_body

        result += _NS_INVOKE_CLOSE
        pos = close_pos + len(_NS_INVOKE_CLOSE)

    return result


class MinimaxM3ToolParser(RustToolParser):
    """Adapter from the Rust MiniMax M3 parser to vLLM ToolParser.

    The real M3 grammar lives in the Rust tool-parser crate. This class only
    configures the generic Rust bridge with the MiniMax M3 parser name.

    M3 is not M2 with renamed tags: it prefixes each structural tag with the
    MiniMax namespace marker, allows multiple ``<invoke>`` tags in one wrapper,
    and represents nested arguments with parameter-name XML tags.

    Fallback — applied in order, retry after each transformation:
    1. Missing namespace prefix on inner tags: ``_normalise_ns`` re-adds it.
    2. Unclosed parameter tag before ``</invoke>``: ``_repair_unclosed_params``
       injects the missing close tag.
    """

    rust_parser_name = "MinimaxM3ToolParser"
    tool_call_start_token = _TOOL_CALL_OPEN

    # ------------------------------------------------------------------
    # _parse_complete: repair-and-retry on Rust parse failure
    # ------------------------------------------------------------------

    def _parse_complete(
        self, model_output: str
    ) -> tuple[Any, dict[int, str]] | None:
        result = super()._parse_complete(model_output)
        if result is not None:
            return result

        if _TOOL_CALL_OPEN not in model_output:
            return None

        # Apply both repair passes and retry once.
        fixed = _normalise_ns(model_output)
        fixed = _repair_unclosed_params(fixed)
        if fixed == model_output:
            return None

        logger.debug(
            "MinimaxM3ToolParser: Rust parse failed; retrying after "
            "namespace/structure repair"
        )
        return super()._parse_complete(fixed)

    # ------------------------------------------------------------------
    # extract_tool_calls_streaming: normalise delta before Rust parse
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
    ):
        # If we are inside a tool_call block, normalise the delta to add any
        # missing namespace prefixes before the Rust incremental parser sees it.
        if _TOOL_CALL_OPEN in current_text:
            delta_text = _MISSING_NS_RE.sub(_NS + r"\1", delta_text)

        return super().extract_tool_calls_streaming(
            previous_text,
            current_text,
            delta_text,
            previous_token_ids,
            current_token_ids,
            delta_token_ids,
            request,
        )
