# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import ClassVar

import regex as re

from vllm.tool_parsers.hermes_tool_parser import Hermes2ProToolParser


class LongcatFlashToolParser(Hermes2ProToolParser):
    tool_call_start_token: ClassVar[str] = "<longcat_tool_call>"
    tool_call_end_token: ClassVar[str] = "</longcat_tool_call>"
    tool_call_regex: ClassVar[re.Pattern] = re.compile(
        r"<longcat_tool_call>(.*?)</longcat_tool_call>"
        r"|<longcat_tool_call>(.*)",
        re.DOTALL,
    )
