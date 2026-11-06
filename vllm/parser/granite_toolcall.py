# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Granite 4.2 tool-call parser.

Granite 4.2 emits tool calls in an XML format::

    <tool_call>
    <function=func_name>
    <parameter=key>value</parameter>
    </function>
    </tool_call>

Granite 4.2 shares the Nemotron V3 tool-call and reasoning format, so this
parser subclasses :class:`NemotronV3Parser` to reuse its grammar, argument
converter and streaming logic, exposed as a standalone ``--tool-call-parser``.

It deliberately does NOT subclass :class:`GraniteThinkingParser`: that parser
strips a leading newline from content (to clean up the chat template's
``\\n</think>\\n`` markup), which is a reasoning-side concern. A tool parser
running on its own (no reasoning parser) must leave content untouched, so the
leading-newline handling stays in the reasoning parser where it belongs.

Because the tool-call extraction is inherited unchanged, fixes to the shared
grammar flow in automatically. The parser works on its own, with any reasoning
parser, or with none, in both thinking modes.
"""

from __future__ import annotations

from vllm.parser.nemotron_v3 import NemotronV3Parser


class GraniteToolCallParser(NemotronV3Parser):
    """Granite 4.2 tool-call parser.

    Subclass of :class:`NemotronV3Parser` (whose tool-call and reasoning format
    Granite 4.2 shares) exposed as a standalone tool parser. The tool adapter
    generated from this class extracts tool calls through the shared grammar;
    reasoning/content splitting, when enabled, is handled by the separately
    configured reasoning parser.
    """
