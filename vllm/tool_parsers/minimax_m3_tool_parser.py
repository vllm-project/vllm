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

# Matches any XML-like tag that is NOT already preceded by the namespace
# prefix.  Used to normalise model output that drops the prefix on inner tags.
_MISSING_NS_RE = re.compile(
    r"(?<!" + re.escape(_NS) + r")(</?[A-Za-z_][A-Za-z0-9_.:-]*(?:\s[^>]*)?>)"
)


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


class MinimaxM3ToolParser(RustToolParser):
    """Adapter from the Rust MiniMax M3 parser to vLLM ToolParser.

    The real M3 grammar lives in the Rust tool-parser crate. This class only
    configures the generic Rust bridge with the MiniMax M3 parser name.

    M3 is not M2 with renamed tags: it prefixes each structural tag with the
    MiniMax namespace marker, allows multiple ``<invoke>`` tags in one wrapper,
    and represents nested arguments with parameter-name XML tags.

    Fallback: the model occasionally emits inner tags (``<invoke>``,
    ``</invoke>``, ``<param-name>``, ``</param-name>``) without the namespace
    prefix.  When the Rust parser raises on such output we normalise the
    missing prefixes and retry once before giving up.
    """

    rust_parser_name = "MinimaxM3ToolParser"
    tool_call_start_token = _TOOL_CALL_OPEN

    # ------------------------------------------------------------------
    # _parse_complete: normalise-and-retry on Rust parse failure
    # ------------------------------------------------------------------

    def _parse_complete(
        self, model_output: str
    ) -> tuple[Any, dict[int, str]] | None:
        result = super()._parse_complete(model_output)
        if result is not None:
            return result

        # Rust parser failed.  If the output contains the tool_call start
        # marker, attempt to normalise any inner tags missing the namespace
        # prefix and retry once.
        if _TOOL_CALL_OPEN not in model_output:
            return None

        normalised = _normalise_ns(model_output)
        if normalised == model_output:
            return None

        logger.debug(
            "MinimaxM3ToolParser: Rust parse failed; retrying after "
            "namespace normalisation"
        )
        return super()._parse_complete(normalised)

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
