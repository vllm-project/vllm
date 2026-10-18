# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.parser.engine.registered_adapters import NemotronV3CompactParserToolAdapter


class NemotronV3CompactToolParser(NemotronV3CompactParserToolAdapter):  # type: ignore[valid-type, misc]
    # Same tag framing as Qwen3-Coder; string values are raw text in the
    # grammar, so it accepts the compact value framing as well.
    structural_tag_model = "qwen_3_coder"
