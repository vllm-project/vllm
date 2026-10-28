# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.parser.engine.registered_adapters import Raon1MiniParserToolAdapter


class Raon1MiniToolParser(Raon1MiniParserToolAdapter):  # type: ignore[valid-type, misc]
    structural_tag_model = "qwen_3_coder"
