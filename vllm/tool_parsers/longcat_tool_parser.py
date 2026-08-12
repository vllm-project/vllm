# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.parser.engine.registered_adapters import LongcatParserToolAdapter


class LongcatFlashToolParser(LongcatParserToolAdapter):  # type: ignore[valid-type, misc]
    # Inherited from Hermes2ProToolParser before the parser-engine migration;
    # set explicitly here to keep guided decoding (and the
    # supports_required_and_named=False it implies) unchanged.
    structural_tag_model = "hermes"
