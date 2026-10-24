# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from sgl-project/sglang PR #36957 (d75f648a8).
"""GLM tool-result encoding: linearize the quadratic template ordering.

The GLM-5.3-Flash chat template validates and reorders contiguous tool-result
blocks in O(n^2) Jinja. Preparation returns a matched conversation/template
pair with the validation and ordering performed in O(n) Python.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from functools import lru_cache
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from vllm.entrypoints.chat_utils import ConversationMessage

logger = logging.getLogger(__name__)

_GLM_TOOL_RESULT_SORT_START = "    {%- set ns_a = namespace(tool_calls=none) -%}"
_GLM_TOOL_RESULT_SORT_END = "\n{% endif -%}\n{%- elif m.role == 'system' -%}"
_GLM_LINEAR_TOOL_RESULT_RENDER = (
    "    {%- for k in range(block_start, ns_blk.end + 1) -%}\n"
    "        {{- render_tool_response(messages[k]) -}}\n"
    "    {%- endfor -%}"
)


def _tool_call_id(item: Any) -> str | None:
    if not isinstance(item, Mapping):
        return None
    value = item.get("tool_call_id") or item.get("id")
    return str(value) if value else None


def _canonical_tool_output(item: Any) -> Any:
    # The template's sorted path renders only entry.output; other keys (e.g. a
    # "type") would send the split-off entry down a different template branch.
    if not isinstance(item, Mapping):
        return item
    return {"output": item["output"]} if "output" in item else {}


def _is_list_of_tool_outputs(message: Mapping[str, Any]) -> bool:
    content = message.get("content")
    return bool(
        isinstance(content, list)
        and content
        and isinstance(content[0], Mapping)
        and "output" in content[0]
    )


def _order_tool_result_block(
    tool_calls: list[dict[str, Any]],
    tool_results: list[dict[str, Any]],
) -> list[dict[str, Any]] | None:
    call_ids = []
    seen_call_ids = set()
    for tool_call in tool_calls:
        call_id = _tool_call_id(tool_call)
        if call_id is None or call_id in seen_call_ids:
            return None
        call_ids.append(call_id)
        seen_call_ids.add(call_id)

    results_by_id = {}
    for message in tool_results:
        if _is_list_of_tool_outputs(message):
            units = [
                ({"role": "tool", "content": [_canonical_tool_output(item)]}, item)
                for item in message["content"]
            ]
        else:
            units = [(message, message)]

        for unit, item in units:
            result_id = _tool_call_id(item)
            if result_id in results_by_id or result_id not in seen_call_ids:
                return None
            results_by_id[result_id] = unit

    return [results_by_id[call_id] for call_id in call_ids if call_id in results_by_id]


def _order_glm_tool_results(
    messages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Order valid GLM tool-result blocks by their declared tool calls."""
    ordered_messages = []
    index = 0
    while index < len(messages):
        message = messages[index]
        ordered_messages.append(message)
        tool_calls = message.get("tool_calls")
        if message.get("role") != "assistant" or not isinstance(tool_calls, list):
            index += 1
            continue

        block_start = index + 1
        block_end = block_start
        while block_end < len(messages) and messages[block_end].get("role") == "tool":
            block_end += 1
        if block_end == block_start:
            index += 1
            continue

        ordered_block = _order_tool_result_block(
            tool_calls, messages[block_start:block_end]
        )
        ordered_messages.extend(
            ordered_block
            if ordered_block is not None
            else messages[block_start:block_end]
        )
        index = block_end

    return ordered_messages


@lru_cache(maxsize=16)
def _linear_template(template: str) -> str | None:
    """Cache only immutable template text, never request messages."""
    if (
        not isinstance(template, str)
        or template.count(_GLM_TOOL_RESULT_SORT_START) != 1
        or template.count(_GLM_TOOL_RESULT_SORT_END) != 1
    ):
        logger.info(
            "GLM tool-result optimization skipped: missing template or ambiguous "
            "ordering anchors; keeping the stock template."
        )
        return None

    start = template.index(_GLM_TOOL_RESULT_SORT_START)
    end = template.find(_GLM_TOOL_RESULT_SORT_END, start)
    if end < 0:
        logger.info(
            "GLM tool-result optimization skipped: ordering end anchor precedes "
            "start; keeping the stock template."
        )
        return None
    logger.info("Replacing quadratic GLM tool-result ordering with a linear render.")
    return template[:start] + _GLM_LINEAR_TOOL_RESULT_RENDER + template[end:]


def prepare_glm_tool_results(
    conversation: list[ConversationMessage],
    template: Any,
    *,
    architectures: list[str] | None = None,
    template_override: bool = False,
    continue_final_message: bool = False,
) -> tuple[list[ConversationMessage], Any]:
    """Return a paired conversation/template without mutating caller state.

    Explicit templates and continuation paths remain unchanged.
    No model-revision hashes are used; callers upgrading GLM template semantics
    must retain output-equivalence coverage.
    """
    if (
        template_override
        or continue_final_message
        or not isinstance(template, str)
        or not any(
            arch in ("Glm5NextForConditionalGeneration", "GlmMoeDsaForCausalLM")
            for arch in architectures or []
        )
    ):
        return conversation, template
    optimized = _linear_template(template)
    if optimized is None:
        return conversation, template
    ordered = _order_glm_tool_results(cast(list[dict[str, Any]], conversation))
    return cast("list[ConversationMessage]", ordered), optimized
