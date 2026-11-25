# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.parser.engine.registered_adapters import MinimaxM3ParserToolAdapter


class MinimaxM3ToolParser(MinimaxM3ParserToolAdapter):  # type: ignore[valid-type, misc]
    # Required and named tool choices parse the native M3 syntax instead of
    # forcing JSON output.
    supports_required_and_named = False
