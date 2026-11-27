# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.parser.engine.registered_adapters import MuseParserToolAdapter


class MuseEngineToolParser(MuseParserToolAdapter):  # type: ignore[valid-type, misc]
    supports_required_and_named = False
