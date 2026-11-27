# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from dataclasses import replace

from grammar import bound_reply_whitespace, gemma_tools, resolve_tool_token_ids

from vllm.tool_parsers import ToolParserManager
from vllm.tool_parsers.gemma4_engine_tool_parser import Gemma4EngineToolParser
from vllm.tool_parsers.structural_tag_registry import register_vllm_structural_tag


class GemmaLabParser(Gemma4EngineToolParser):
    structural_tag_model = "gemma4_lab"

    def adjust_request(self, request):
        request = super().adjust_request(request)
        opts = getattr(request, "structured_outputs", None)
        if opts and opts.structural_tag:
            tag = json.loads(opts.structural_tag)
            bound_reply_whitespace(tag)
            resolve_tool_token_ids(tag.get("format"), self.vocab)
            request.structured_outputs = replace(opts, structural_tag=json.dumps(tag))
        return request


register_vllm_structural_tag("gemma4_lab")(gemma_tools)
ToolParserManager.register_module(name="gemma4_lab", module=GemmaLabParser)
