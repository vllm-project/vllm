# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Reasoning parser for Kolibri 1.

Qwen3 grammar, but the starting state follows the Kolibri 1 chat template:
``reasoning_effort`` takes precedence over ``enable_thinking``, while
``Qwen3Parser`` reads ``enable_thinking`` alone. ``continue_final_message``
always renders a closed think block, which is not accounted for.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from vllm.parser.qwen3 import Qwen3Parser

if TYPE_CHECKING:
    from vllm.tokenizers import TokenizerLike
    from vllm.tool_parsers.abstract_tool_parser import Tool


def thinking_enabled(chat_template_kwargs: Mapping[str, Any] | None) -> bool:
    """Mirror the Kolibri 1 chat template's thinking switch.

    Args:
        chat_template_kwargs: Effective chat template kwargs of the request.

    Returns:
        False if ``reasoning_effort`` is ``"none"``, or if no
        ``reasoning_effort`` is given and ``enable_thinking`` is ``False``.
        True otherwise.

    """
    kwargs = chat_template_kwargs or {}
    effort = kwargs.get("reasoning_effort")
    if effort is not None:
        return effort != "none"
    return kwargs.get("enable_thinking") is not False


class Kolibri1Parser(Qwen3Parser):
    """Qwen3 grammar with the starting state chosen like the Kolibri 1 template."""

    CONFIG_NAME = "kolibri1"

    def __init__(
        self,
        tokenizer: TokenizerLike,
        tools: list[Tool] | None = None,
        **kwargs,
    ) -> None:
        chat_kwargs = dict(kwargs.get("chat_template_kwargs") or {})
        chat_kwargs["enable_thinking"] = thinking_enabled(chat_kwargs)
        kwargs["chat_template_kwargs"] = chat_kwargs
        super().__init__(tokenizer, tools, **kwargs)
