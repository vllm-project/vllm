import pytest
from transformers import AutoTokenizer

from tests.entrypoints.openai.reasoning_parsers.utils import (
    run_reasoning_extraction)
from vllm.entrypoints.openai.reasoning_parsers import (ReasoningParser,
                                                       ReasoningParserManager)

parser_name = "deepseek_r1"
start_token = "<think>"
end_token = "</think>"

SIMPLE_REASONING = {
    "output": "<think>This is a reasoning section</think>This is the rest",
    "reasoning_content": "This is a reasoning section",
    "content": "This is the rest",
}
COMPLETE_REASONING = {
    "output": "<think>This is a reasoning section</think>",
    "reasoning_content": "This is a reasoning section",
    "content": None,
}
NO_REASONING = {
    "output": "This is a reasoning section",
    "reasoning_content": None,
    "content": "This is a reasoning section",
}
MULTIPLE_LINES = {
    "output": "<think>This\nThat</think>This is the rest\nThat",
    "reasoning_content": "This\nThat",
    "content": "This is the rest\nThat",
}
SHORTEST_REASONING_NO_STREAMING = {
    "output": "<think></think>This is the rest",
    "reasoning_content": "",
    "content": "This is the rest",
}
SHORTEST_REASONING = {
    "output": "<think></think>This is the rest",
    "reasoning_content": None,
    "content": "This is the rest",
}

TEST_CASES = [
    pytest.param(
        False,
        SIMPLE_REASONING["output"],
        SIMPLE_REASONING["reasoning_content"],
        SIMPLE_REASONING["content"],
        id="simple_streaming",
    ),
    pytest.param(
        True,
        SIMPLE_REASONING["output"],
        SIMPLE_REASONING["reasoning_content"],
        SIMPLE_REASONING["content"],
        id="simple_streaming",
    ),
    pytest.param(
        False,
        COMPLETE_REASONING["output"],
        COMPLETE_REASONING["reasoning_content"],
        COMPLETE_REASONING["content"],
        id="complete_streaming",
    ),
    pytest.param(
        True,
        COMPLETE_REASONING["output"],
        COMPLETE_REASONING["reasoning_content"],
        COMPLETE_REASONING["content"],
        id="complete_streaming",
    ),
    pytest.param(
        False,
        NO_REASONING["output"],
        NO_REASONING["reasoning_content"],
        NO_REASONING["content"],
        id="no_streaming",
    ),
    pytest.param(
        True,
        NO_REASONING["output"],
        NO_REASONING["reasoning_content"],
        NO_REASONING["content"],
        id="no_streaming",
    ),
    pytest.param(
        False,
        MULTIPLE_LINES["output"],
        MULTIPLE_LINES["reasoning_content"],
        MULTIPLE_LINES["content"],
        id="multiple_lines_streaming",
    ),
    pytest.param(
        True,
        MULTIPLE_LINES["output"],
        MULTIPLE_LINES["reasoning_content"],
        MULTIPLE_LINES["content"],
        id="multiple_lines_streaming",
    ),
    pytest.param(
        True,
        SHORTEST_REASONING["output"],
        SHORTEST_REASONING["reasoning_content"],
        SHORTEST_REASONING["content"],
        id="shortest_streaming",
    ),
    pytest.param(
        False,
        SHORTEST_REASONING_NO_STREAMING["output"],
        SHORTEST_REASONING_NO_STREAMING["reasoning_content"],
        SHORTEST_REASONING_NO_STREAMING["content"],
        id="shortest_streaming",
    ),
]


@pytest.mark.parametrize(
    "streaming, model_output, expected_reasoning, expected_content",
    TEST_CASES)
def test_reasoning(
    streaming: bool,
    model_output: str,
    expected_reasoning: str,
    expected_content: str,
):
    tokenizer = AutoTokenizer.from_pretrained("facebook/opt-125m")
    tokenizer.add_tokens([start_token, end_token])
    output = tokenizer.tokenize(model_output)
    # decode everything to tokens
    output_tokens = [
        tokenizer.convert_tokens_to_string([token]) for token in output
    ]
    parser: ReasoningParser = ReasoningParserManager.get_reasoning_parser(
        parser_name)(tokenizer)

    reasoning, content = run_reasoning_extraction(parser,
                                                  output_tokens,
                                                  streaming=streaming)

    assert reasoning == expected_reasoning
    assert content == expected_content
