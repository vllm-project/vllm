# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import ClassVar

from vllm.tool_parsers import ToolParser


class BaseCohereCommandToolParser(ToolParser):
    melody_preset: ClassVar[str]


class CohereCommand3ToolParser(BaseCohereCommandToolParser):
    melody_preset = "cmd3"


class CohereCommand4ToolParser(BaseCohereCommandToolParser):
    melody_preset = "cmd4"
