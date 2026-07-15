# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""poolside (Laguna) tool parser, backed by the declarative parser engine.

Replaces the previous standalone regex implementation. Laguna emits the same
``<tool_call>name<arg_key>k</arg_key><arg_value>v</arg_value></tool_call>``
surface format as GLM-4.7 (function name inline, no newline separator), so it is
backed by ``PoolsideParser`` (see ``vllm/parser/poolside.py``) and inherits the
engine's incremental string streaming and schema-aware argument coercion.

Only the tool parser is migrated; reasoning stays on
``PoolsideV1ReasoningParser`` (vllm/reasoning/poolside_v1_reasoning_parser.py),
which scopes the backward ``</think>`` search to the current assistant turn —
Laguna-specific behaviour the generic engine reasoning parser does not provide.
"""

from __future__ import annotations

from vllm.parser.engine.registered_adapters import PoolsideParserToolAdapter


class PoolsideV1ToolParser(PoolsideParserToolAdapter):  # type: ignore[valid-type, misc]
    supports_required_and_named = False
    # Laguna shares GLM-4.7's structural-tag (guided-decoding) format.
    structural_tag_model = "glm_4_7"
