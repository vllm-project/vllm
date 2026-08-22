# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.parser.engine.registered_adapters import LongcatParserToolAdapter


class LongcatFlashToolParser(LongcatParserToolAdapter):  # type: ignore[valid-type, misc]
    # Inherited from Hermes before the migration; kept for guided decoding.
    structural_tag_model = "hermes"
