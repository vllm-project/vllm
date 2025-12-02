# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import regex as re

from vllm.entrypoints.openai.protocol import ChatCompletionRequest
from vllm.entrypoints.openai.tool_parsers.minimax_m2_tool_parser import (
    MinimaxM2ToolParser,
)
from vllm.logger import init_logger
from vllm.tokenizers import TokenizerLike

logger = init_logger(__name__)

class DeepSeekV32ToolParser(MinimaxM2ToolParser):
    def __init__(self, tokenizer: TokenizerLike):
        self.prev_tool_call_arr: list[dict] = []
        # the index of the tool call that is currently being parsed
        self.current_tool_id: int = -1
        self.current_tool_name_sent: bool = False
        self.streamed_args_for_tool: list[str] = []

        self.model_tokenizer = tokenizer

        # Sentinel tokens
        self.tool_call_start_token: str = "<｜DSML｜function_calls>"
        self.tool_call_end_token: str = "</｜DSML｜function_calls>"
        self.invoke_start_prefix: str = "<｜DSML｜invoke name="
        self.invoke_end_token: str = "</｜DSML｜invoke>"
        self.parameter_prefix: str = "<｜DSML｜parameter name="
        self.parameter_end_token: str = "</｜DSML｜parameter>"

        # Streaming state variables
        self.current_tool_name_sent: bool = False
        # Override base class type - we use string IDs for tool calls
        self.current_tool_id: str | None = None  # type: ignore
        self.streamed_args_for_tool: list[str] = []
        self.is_tool_call_started: bool = False
        self.failed_count: int = 0

        # Initialize streaming state variables
        self.current_tool_index: int = 0
        self.invoke_index: int = 0
        self.header_sent: bool = False
        self.current_function_name: str | None = None
        self.current_param_name: str | None = None
        self.current_param_value: str = ""
        self.param_count: int = 0
        self.in_param: bool = False
        self.in_function: bool = False
        self.accumulated_text: str = ""
        self.json_started: bool = False
        self.json_closed: bool = False
        self.accumulated_params: dict = {}
        self.streaming_request: ChatCompletionRequest | None = None

        # Enhanced streaming state - reset for each new message
        self._reset_streaming_state()

        # Regex patterns for complete parsing
        self.tool_call_complete_regex = re.compile(
            r"<｜DSML｜function_calls>(.*?)</｜DSML｜function_calls>", re.DOTALL
        )
        self.invoke_complete_regex = re.compile(
            r"<｜DSML｜invoke name=(.*?)</｜DSML｜invoke>", re.DOTALL
        )
        self.parameter_complete_regex = re.compile(
            r'<｜DSML｜parameter\s+name="([^"]+)"\s+string=".*?"\s*>(.*?)</｜DSML｜parameter>'
        )

        self.tool_call_start_token_id = None
        self.tool_call_end_token_id = None
