# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.parser.engine.registered_adapters import LlamaJsonParserToolAdapter


class Llama3JsonToolParser(LlamaJsonParserToolAdapter):  # type: ignore[valid-type, misc]
    """Llama 3.x/4 JSON tool parser backed by the declarative parser
    engine (see vllm/parser/llama_json.py).

    Used when --enable-auto-tool-choice --tool-call-parser llama3_json or
    llama4_json are set.
    """

    structural_tag_model = "llama"
