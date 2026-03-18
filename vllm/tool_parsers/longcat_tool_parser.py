# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import regex as re

from vllm.tokenizers import TokenizerLike
from vllm.tool_parsers.hermes_tool_parser import Hermes2ProToolParser


class LongcatFlashToolParser(Hermes2ProToolParser):
    def __init__(self, tokenizer: TokenizerLike):
        super().__init__(tokenizer)

        self.tool_call_start_token: str = "<longcat_tool_call>"
        self.tool_call_end_token: str = "</longcat_tool_call>"

        self.tool_call_regex = re.compile(
            r"<longcat_tool_call>(.*?)</longcat_tool_call>|<longcat_tool_call>(.*)",
            re.DOTALL,
        )

        start_ids, start_arr = self._get_cached_token_data(
            self.model_tokenizer, self.tool_call_start_token
        )
        end_ids, end_arr = self._get_cached_token_data(
            self.model_tokenizer, self.tool_call_end_token
        )
        self.tool_call_start_token_ids = start_ids
        self.tool_call_end_token_ids = end_ids
        self.tool_call_start_token_array = start_arr
        self.tool_call_end_token_array = end_arr
