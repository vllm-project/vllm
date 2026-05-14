# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest.mock import MagicMock, patch

import pytest
from transformers import AutoTokenizer

from tests.tool_parsers.utils import (
    run_tool_extraction,
    run_tool_extraction_streaming,
)
from vllm.entrypoints.openai.engine.protocol import FunctionCall
from vllm.tokenizers import TokenizerLike
from vllm.tool_parsers import ToolParser, ToolParserManager

TOOL_CALL_START = "<|tool_call_start|>"
TOOL_CALL_END = "<|tool_call_end|>"

SIMPLE_FUNCTION_OUTPUT = "get_candidate_status(candidate_id='12345')"
SIMPLE_FUNCTION_CALL = FunctionCall(
    name="get_candidate_status",
    arguments='{"candidate_id": "12345"}',
)
MORE_TYPES_FUNCTION_OUTPUT = (
    "register_user(name='John Doe', "
    "age=37, "
    "address={'city': 'San Francisco', 'state': 'CA'}, "
    "role=None, "
    "passed_test=True, "
    "aliases=['John', 'Johnny'])"
)
MORE_TYPES_FUNCTION_CALL = FunctionCall(
    name="register_user",
    arguments='{"name": "John Doe", '
    '"age": 37, '
    '"address": {"city": "San Francisco", "state": "CA"}, '
    '"role": null, '
    '"passed_test": true, '
    '"aliases": ["John", "Johnny"]}',
)
PARAMETERLESS_FUNCTION_OUTPUT = "get_weather()"
PARAMETERLESS_FUNCTION_CALL = FunctionCall(
    name="get_weather",
    arguments="{}",
)
EMPTY_DICT_FUNCTION_OUTPUT = "do_something_cool(additional_data={})"
EMPTY_DICT_FUNCTION_CALL = FunctionCall(
    name="do_something_cool",
    arguments='{"additional_data": {}}',
)
EMPTY_LIST_FUNCTION_OUTPUT = "do_something_cool(steps=[])"
EMPTY_LIST_FUNCTION_CALL = FunctionCall(
    name="do_something_cool",
    arguments='{"steps": []}',
)
ESCAPED_STRING_FUNCTION_OUTPUT = (
    r"get_weather(city='Martha\'s Vineyard', metric='\"cool units\"')"
)
ESCAPED_STRING_FUNCTION_CALL = FunctionCall(
    name="get_weather",
    arguments='{"city": "Martha\'s Vineyard", "metric": "\\"cool units\\""}',
)
DOTTED_NAME_FUNCTION_OUTPUT = (
    "grocery.orderIngredients("
    "ingredientList=[{'name': 'Lasagna noodles', 'amount': 250, 'unit': 'g'}], "
    "deliveryAddress='845 Willow Lane, Springfield, IL 62704')"
)
DOTTED_NAME_FUNCTION_CALL = FunctionCall(
    name="grocery.orderIngredients",
    arguments=(
        '{"ingredientList": ['
        '{"name": "Lasagna noodles", "amount": 250, "unit": "g"}], '
        '"deliveryAddress": "845 Willow Lane, Springfield, IL 62704"}'
    ),
)


@pytest.fixture(scope="module")
def lfm2_tokenizer() -> TokenizerLike:
    return AutoTokenizer.from_pretrained("LiquidAI/LFM2.5-1.2B-Instruct")


def _wrap(tool_text: str, content_after: str = "") -> str:
    """Wrap pythonic tool call in LFM2.5 sentinel tokens."""
    result = f"{TOOL_CALL_START}[{tool_text}]{TOOL_CALL_END}"
    if content_after:
        result += f"\n{content_after}"
    return result


@pytest.mark.parametrize("streaming", [True, False])
def test_no_tool_call(streaming: bool, lfm2_tokenizer: TokenizerLike):
    tool_parser: ToolParser = ToolParserManager.get_tool_parser("lfm2")(lfm2_tokenizer)
    model_output = "How can I help you today?"

    content, tool_calls = run_tool_extraction(
        tool_parser, model_output, streaming=streaming
    )

    assert content == model_output
    assert len(tool_calls) == 0


TEST_CASES = [
    pytest.param(
        True,
        _wrap(SIMPLE_FUNCTION_OUTPUT),
        [SIMPLE_FUNCTION_CALL],
        None,
        id="simple_streaming",
    ),
    pytest.param(
        False,
        _wrap(SIMPLE_FUNCTION_OUTPUT),
        [SIMPLE_FUNCTION_CALL],
        None,
        id="simple_nonstreaming",
    ),
    pytest.param(
        True,
        _wrap(MORE_TYPES_FUNCTION_OUTPUT),
        [MORE_TYPES_FUNCTION_CALL],
        None,
        id="more_types_streaming",
    ),
    pytest.param(
        False,
        _wrap(MORE_TYPES_FUNCTION_OUTPUT),
        [MORE_TYPES_FUNCTION_CALL],
        None,
        id="more_types_nonstreaming",
    ),
    pytest.param(
        True,
        _wrap(PARAMETERLESS_FUNCTION_OUTPUT),
        [PARAMETERLESS_FUNCTION_CALL],
        None,
        id="parameterless_streaming",
    ),
    pytest.param(
        False,
        _wrap(PARAMETERLESS_FUNCTION_OUTPUT),
        [PARAMETERLESS_FUNCTION_CALL],
        None,
        id="parameterless_nonstreaming",
    ),
    pytest.param(
        True,
        _wrap(EMPTY_DICT_FUNCTION_OUTPUT),
        [EMPTY_DICT_FUNCTION_CALL],
        None,
        id="empty_dict_streaming",
    ),
    pytest.param(
        False,
        _wrap(EMPTY_DICT_FUNCTION_OUTPUT),
        [EMPTY_DICT_FUNCTION_CALL],
        None,
        id="empty_dict_nonstreaming",
    ),
    pytest.param(
        True,
        _wrap(EMPTY_LIST_FUNCTION_OUTPUT),
        [EMPTY_LIST_FUNCTION_CALL],
        None,
        id="empty_list_streaming",
    ),
    pytest.param(
        False,
        _wrap(EMPTY_LIST_FUNCTION_OUTPUT),
        [EMPTY_LIST_FUNCTION_CALL],
        None,
        id="empty_list_nonstreaming",
    ),
    pytest.param(
        True,
        _wrap(ESCAPED_STRING_FUNCTION_OUTPUT),
        [ESCAPED_STRING_FUNCTION_CALL],
        None,
        id="escaped_string_streaming",
    ),
    pytest.param(
        False,
        _wrap(ESCAPED_STRING_FUNCTION_OUTPUT),
        [ESCAPED_STRING_FUNCTION_CALL],
        None,
        id="escaped_string_nonstreaming",
    ),
    pytest.param(
        True,
        _wrap(f"{SIMPLE_FUNCTION_OUTPUT}, {MORE_TYPES_FUNCTION_OUTPUT}"),
        [SIMPLE_FUNCTION_CALL, MORE_TYPES_FUNCTION_CALL],
        None,
        id="parallel_calls_streaming",
    ),
    pytest.param(
        False,
        _wrap(f"{SIMPLE_FUNCTION_OUTPUT}, {MORE_TYPES_FUNCTION_OUTPUT}"),
        [SIMPLE_FUNCTION_CALL, MORE_TYPES_FUNCTION_CALL],
        None,
        id="parallel_calls_nonstreaming",
    ),
    # LFM2.5 specific: content AFTER tool call
    pytest.param(
        False,
        _wrap(
            SIMPLE_FUNCTION_OUTPUT,
            content_after="Checking the current status of candidate ID 12345.",
        ),
        [SIMPLE_FUNCTION_CALL],
        "Checking the current status of candidate ID 12345.",
        id="content_after_tool_call_nonstreaming",
    ),
    # Dotted / class-method function names: grocery.orderIngredients(...)
    pytest.param(
        True,
        _wrap(DOTTED_NAME_FUNCTION_OUTPUT),
        [DOTTED_NAME_FUNCTION_CALL],
        None,
        id="dotted_name_streaming",
    ),
    pytest.param(
        False,
        _wrap(DOTTED_NAME_FUNCTION_OUTPUT),
        [DOTTED_NAME_FUNCTION_CALL],
        None,
        id="dotted_name_nonstreaming",
    ),
]


@pytest.mark.parametrize(
    "streaming, model_output, expected_tool_calls, expected_content",
    TEST_CASES,
)
def test_tool_call(
    streaming: bool,
    model_output: str,
    expected_tool_calls: list[FunctionCall],
    expected_content: str | None,
    lfm2_tokenizer: TokenizerLike,
):
    tool_parser: ToolParser = ToolParserManager.get_tool_parser("lfm2")(lfm2_tokenizer)

    content, tool_calls = run_tool_extraction(
        tool_parser, model_output, streaming=streaming
    )

    if expected_content and not streaming:
        assert content == expected_content
    assert len(tool_calls) == len(expected_tool_calls)
    for actual, expected in zip(tool_calls, expected_tool_calls):
        assert actual.type == "function"
        assert actual.function == expected


def test_streaming_tool_call_with_large_steps(lfm2_tokenizer: TokenizerLike):
    tool_parser: ToolParser = ToolParserManager.get_tool_parser("lfm2")(lfm2_tokenizer)
    model_output_deltas = [
        f"{TOOL_CALL_START}[get_candidate_status(candidate_id='12345'), "
        f"{PARAMETERLESS_FUNCTION_OUTPUT}, "
        f"{EMPTY_LIST_FUNCTION_OUTPUT}]{TOOL_CALL_END}",
    ]

    reconstructor = run_tool_extraction_streaming(
        tool_parser, model_output_deltas, assert_one_tool_per_delta=False
    )

    assert len(reconstructor.tool_calls) == 3
    assert reconstructor.tool_calls[0].function == SIMPLE_FUNCTION_CALL
    assert reconstructor.tool_calls[1].function == PARAMETERLESS_FUNCTION_CALL
    assert reconstructor.tool_calls[2].function == EMPTY_LIST_FUNCTION_CALL


def test_streaming_full_block_and_trailing_in_single_delta(
    lfm2_tokenizer: TokenizerLike,
):
    """The entire <|tool_call_start|>[...]<|tool_call_end|> block plus
    trailing assistant text arrive in one delta. Trailing content must
    still be emitted — not silently dropped."""
    tool_parser: ToolParser = ToolParserManager.get_tool_parser("lfm2")(lfm2_tokenizer)
    full_text = f"{TOOL_CALL_START}[{SIMPLE_FUNCTION_OUTPUT}]{TOOL_CALL_END}\nDone."

    reconstructor = run_tool_extraction_streaming(tool_parser, [full_text])

    assert len(reconstructor.tool_calls) == 1
    assert reconstructor.tool_calls[0].function == SIMPLE_FUNCTION_CALL
    assert "Done." in reconstructor.other_content


def test_streaming_leading_content_and_full_block_in_single_delta(
    lfm2_tokenizer: TokenizerLike,
):
    """Leading assistant text plus the entire tool block arrive in one
    delta. Leading content must be emitted — not silently dropped."""
    tool_parser: ToolParser = ToolParserManager.get_tool_parser("lfm2")(lfm2_tokenizer)
    full_text = (
        f"Let me check. {TOOL_CALL_START}[{SIMPLE_FUNCTION_OUTPUT}]{TOOL_CALL_END}"
    )

    reconstructor = run_tool_extraction_streaming(tool_parser, [full_text])

    assert len(reconstructor.tool_calls) == 1
    assert reconstructor.tool_calls[0].function == SIMPLE_FUNCTION_CALL
    assert "Let me check." in reconstructor.other_content


def test_streaming_leading_block_and_trailing_in_single_delta(
    lfm2_tokenizer: TokenizerLike,
):
    """Leading text + complete tool block + trailing text in one delta.
    Both leading and trailing content must be preserved."""
    tool_parser: ToolParser = ToolParserManager.get_tool_parser("lfm2")(lfm2_tokenizer)
    full_text = (
        "Let me check. "
        f"{TOOL_CALL_START}[{SIMPLE_FUNCTION_OUTPUT}]{TOOL_CALL_END}\nDone."
    )

    reconstructor = run_tool_extraction_streaming(tool_parser, [full_text])

    assert len(reconstructor.tool_calls) == 1
    assert reconstructor.tool_calls[0].function == SIMPLE_FUNCTION_CALL
    assert "Let me check." in reconstructor.other_content
    assert "Done." in reconstructor.other_content


def test_echoed_tool_call_body_not_leaked_to_content(
    lfm2_tokenizer: TokenizerLike,
):
    """LFM2 sometimes emits the tool call body again after the first
    <|tool_call_end|>, capped with a second <|tool_call_end|>. The
    echoed body must not surface as assistant content — neither in
    streaming nor non-streaming paths."""
    tool_parser: ToolParser = ToolParserManager.get_tool_parser("lfm2")(lfm2_tokenizer)
    body = (
        "[grocery.orderIngredients("
        "ingredientList=[{'name': 'apple', 'quantity': '2'}], "
        "deliveryAddress='123 Main St')]"
    )
    model_output = f"{TOOL_CALL_START}{body}{TOOL_CALL_END}{body}{TOOL_CALL_END}"

    # Non-streaming
    content_ns, tool_calls_ns = run_tool_extraction(
        tool_parser, model_output, streaming=False
    )
    assert len(tool_calls_ns) == 1
    assert tool_calls_ns[0].function.name == "grocery.orderIngredients"
    assert content_ns in (None, "")

    # Streaming: re-fetch a fresh parser since state was mutated above.
    tool_parser2: ToolParser = ToolParserManager.get_tool_parser("lfm2")(lfm2_tokenizer)
    content_s, tool_calls_s = run_tool_extraction(
        tool_parser2, model_output, streaming=True
    )
    assert len(tool_calls_s) == 1
    assert tool_calls_s[0].function.name == "grocery.orderIngredients"
    # Echoed body must not leak as content.
    assert content_s in (None, "")
    assert "grocery.orderIngredients" not in (content_s or "")
    assert TOOL_CALL_END not in (content_s or "")


def test_streaming_char_by_char_multi_dict_list(lfm2_tokenizer: TokenizerLike):
    """Stream a tool call containing a list of multiple dicts one
    character at a time. Every prefix lands in some partial-parse state
    (mid-key, mid-value, open quote inside dict, empty dict, etc.). The
    parser must not raise — incomplete prefixes should silently wait for
    more text instead of logging exceptions."""
    tool_parser: ToolParser = ToolParserManager.get_tool_parser("lfm2")(lfm2_tokenizer)
    full_text = (
        f"{TOOL_CALL_START}[grocery.orderIngredients("
        "ingredientList=["
        '{"name": "apple", "quantity": "2"}, '
        '{"name": "bread", "quantity": "1"}'
        f"])]{TOOL_CALL_END}"
    )
    deltas = [c for c in full_text]

    reconstructor = run_tool_extraction_streaming(
        tool_parser, deltas, assert_one_tool_per_delta=False
    )

    assert len(reconstructor.tool_calls) == 1
    assert reconstructor.tool_calls[0].function.name == "grocery.orderIngredients"
    import json

    args = json.loads(reconstructor.tool_calls[0].function.arguments)
    assert args == {
        "ingredientList": [
            {"name": "apple", "quantity": "2"},
            {"name": "bread", "quantity": "1"},
        ]
    }


def test_streaming_dotted_name_in_single_delta(lfm2_tokenizer: TokenizerLike):
    """A pythonic call with a dotted/attribute function name (e.g.
    ``domain.method(arg=...)``) must be parsed correctly in streaming mode
    just as in non-streaming mode."""
    tool_parser: ToolParser = ToolParserManager.get_tool_parser("lfm2")(lfm2_tokenizer)
    full_text = f"{TOOL_CALL_START}[{DOTTED_NAME_FUNCTION_OUTPUT}]{TOOL_CALL_END}"

    reconstructor = run_tool_extraction_streaming(tool_parser, [full_text])

    assert len(reconstructor.tool_calls) == 1
    assert reconstructor.tool_calls[0].function == DOTTED_NAME_FUNCTION_CALL


def test_adjust_request_disables_skip_special_tokens(
    lfm2_tokenizer: TokenizerLike,
):
    """When tools are present, the parser must force
    ``skip_special_tokens=False`` so the engine does not strip the
    <|tool_call_start|>/<|tool_call_end|> sentinels before they reach the
    parser."""
    from vllm.entrypoints.openai.chat_completion.protocol import (
        ChatCompletionRequest,
    )

    tool_parser: ToolParser = ToolParserManager.get_tool_parser("lfm2")(lfm2_tokenizer)

    request_with_tools = ChatCompletionRequest(
        messages=[],
        model="test-model",
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "parameters": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                    },
                },
            }
        ],
    )
    assert request_with_tools.skip_special_tokens is True
    adjusted = tool_parser.adjust_request(request_with_tools)
    assert adjusted.skip_special_tokens is False

    # No tools → no override; default behaviour preserved.
    request_no_tools = ChatCompletionRequest(messages=[], model="test-model")
    assert request_no_tools.skip_special_tokens is True
    adjusted_no_tools = tool_parser.adjust_request(request_no_tools)
    assert adjusted_no_tools.skip_special_tokens is True


@pytest.mark.parametrize("streaming", [False])
def test_regex_timeout_handling(streaming: bool, lfm2_tokenizer: TokenizerLike):
    """Test regex timeout is handled gracefully."""
    tool_parser: ToolParser = ToolParserManager.get_tool_parser("lfm2")(lfm2_tokenizer)

    fake_input = f"{TOOL_CALL_START}[A(A=" + "\t)A(A=,\t" * 2
    fake_input += f"]{TOOL_CALL_END}"

    mock_regex = MagicMock()
    mock_regex.match.side_effect = TimeoutError("Regex timeout")

    with patch.object(tool_parser, "TOOL_CALL_REGEX", mock_regex):
        content, tool_calls = run_tool_extraction(
            tool_parser, fake_input, streaming=streaming
        )

        assert content == fake_input
        assert len(tool_calls) == 0
        mock_regex.match.assert_called_once()
