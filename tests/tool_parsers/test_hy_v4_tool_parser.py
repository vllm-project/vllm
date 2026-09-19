# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest.mock import MagicMock, patch

import vllm.envs as envs
from vllm.tool_parsers.hy_v4_tool_parser import HYV4ToolExtractor


def _hy_v4_extractor() -> HYV4ToolExtractor:
    vocab = {
        "<tool_calls>": 1,
        "</tool_calls>": 2,
        "<tool_call>": 3,
        "</tool_call>": 4,
        "<arg_key>": 5,
        "</arg_key>": 6,
        "<arg_value>": 7,
        "</arg_value>": 8,
    }
    return HYV4ToolExtractor(vocab, token_suffix="", strict=False)


def test_regex_timeout_treated_as_no_tool_call():
    extractor = _hy_v4_extractor()
    model_output = "<tool_calls><tool_call>" + "c" * 100 + "</tool_calls>"
    mock_regex = MagicMock()
    mock_regex.findall.side_effect = TimeoutError("Regex timeout")

    with patch.object(extractor, "tool_call_regex", mock_regex):
        result = extractor.extract_tool_calls(model_output, None)

    assert result["tools_called"] is False
    assert result["tool_calls"] == []
    assert result["content"] == model_output
    mock_regex.findall.assert_called_once()
    assert (
        mock_regex.findall.call_args.kwargs["timeout"]
        == envs.VLLM_TOOL_PARSE_REGEX_TIMEOUT_SECONDS
    )


def test_streaming_args_regex_timeout_skips_later_deltas():
    extractor = _hy_v4_extractor()
    extractor._streaming_tool_name = "get_weather"
    extractor.current_tool_id = 0
    extractor._buffer = "<arg_key>" + "c" * 100
    mock_regex = MagicMock()
    mock_regex.findall.side_effect = TimeoutError("Regex timeout")

    with patch.object(extractor, "func_args_regex", mock_regex):
        first = extractor._extract_streaming_incremental(False, None, None)
        second = extractor.extract_tool_calls_streaming(
            "prev",
            "prevx",
            "x",
            [1],
            [1, 3],
            [3],
            None,
        )

    assert first is None
    assert second is None
    mock_regex.findall.assert_called_once()
    assert (
        mock_regex.findall.call_args.kwargs["timeout"]
        == envs.VLLM_TOOL_PARSE_REGEX_TIMEOUT_SECONDS
    )
