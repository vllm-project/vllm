# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.parser.abstract_parser import (
    DelegatingParser,
    Parser,
    _WrappedParser,
)
from vllm.parser.parser_manager import ParserManager

__all__ = [
    "Parser",
    "DelegatingParser",
    "ParserManager",
    "_WrappedParser",
]

# Register lazy parsers
ParserManager.register_lazy_module(
    name="minimax_m2",
    module_path="vllm.parser.minimax_m2_parser",
    class_name="MiniMaxM2Parser",
)
