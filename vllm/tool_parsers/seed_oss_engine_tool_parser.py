# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.parser.engine.registered_adapters import SeedOssParserToolAdapter


class SeedOssEngineToolParser(SeedOssParserToolAdapter):  # type: ignore[valid-type, misc]
    structural_tag_model = None
