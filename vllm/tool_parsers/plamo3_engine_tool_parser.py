# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.parser.engine.registered_adapters import Plamo3ParserToolAdapter


class Plamo3EngineToolParser(Plamo3ParserToolAdapter):  # type: ignore[valid-type, misc]
    structural_tag_model = None
    supports_required_and_named = False
