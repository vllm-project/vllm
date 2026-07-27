# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.tool_parsers.rust_tool_parser import RustToolParser


class MinimaxM3ToolParser(RustToolParser):
    """Adapter from the Rust MiniMax M3 parser to vLLM ToolParser.

    The real M3 grammar lives in the Rust tool-parser crate. This class only
    configures the generic Rust bridge with the MiniMax M3 parser name.

    M3 is not M2 with renamed tags: it prefixes each structural tag with the
    MiniMax namespace marker, allows multiple ``<invoke>`` tags in one wrapper,
    and represents nested arguments with parameter-name XML tags.
    """

    rust_parser_name = "MinimaxM3ToolParser"
    tool_call_start_token = "]<]minimax[>[<tool_call>"
