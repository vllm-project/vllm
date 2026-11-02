# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Granite 4.2 tool-call parser.

Granite 4.2 emits tool calls in an XML format::

    <tool_call>
    <function=func_name>
    <parameter=key>value</parameter>
    </function>
    </tool_call>

alongside ``<think>``/``</think>`` reasoning. :class:`GraniteThinkingParser`
already drives both reasoning and this tool-call format in a single engine, so
this parser is a thin subclass of it: it inherits the exact same grammar,
argument converter and streaming logic, and is exposed as a standalone
``--tool-call-parser`` separate from the reasoning parser.

Because the tool-call extraction is inherited unchanged, future fixes to the
shared grammar flow in automatically. The parser works on its own: it extracts
tool calls correctly whether or not ``--reasoning-parser granite_thinking_parser``
is also configured, in both thinking modes.
"""

from __future__ import annotations

from vllm.parser.granite_thinking import GraniteThinkingParser


class GraniteToolCallParser(GraniteThinkingParser):
    """Granite 4.2 tool-call parser.

    Subclass of :class:`GraniteThinkingParser` that reuses its engine (Granite
    reasoning + Granite XML tool calls) and exposes it as a standalone tool
    parser. The reasoning-side overrides inherited from the parent only affect
    the reasoning adapter; the tool adapter generated from this class extracts
    tool calls through the shared grammar.
    """
