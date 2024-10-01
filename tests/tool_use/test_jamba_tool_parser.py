import json
from typing import Dict

import pytest

from vllm.entrypoints.openai.tool_parsers import JambaToolParser
from vllm.transformers_utils.tokenizer import get_tokenizer

MODEL = "ai21labs/Jamba-tiny-dev"

@pytest.fixture
def jamba_tool_parser():
    tokenizer = get_tokenizer(tokenizer_name=MODEL)
    return JambaToolParser(tokenizer)


def test_extract_tool_calls_no_tools(jamba_tool_parser):
    model_output = "This is a test"
    extracted_tool_calls = jamba_tool_parser.extract_tool_calls(model_output)
    assert extracted_tool_calls.tools_called == False
    assert extracted_tool_calls.tool_calls == []
    assert extracted_tool_calls.content == model_output


def test_extract_tool_calls_single_tool(jamba_tool_parser):
    model_output = '''<tool_calls> [\n{"name": "get_current_weather", "arguments": {"city": "Dallas", "state": "TX", "unit": "fahrenheit"}}] </tool_calls>'''
    extracted_tool_calls = jamba_tool_parser.extract_tool_calls(model_output)
    assert extracted_tool_calls.tools_called == True
    assert len(extracted_tool_calls.tool_calls) == 1
    tool_call = extracted_tool_calls.tool_calls[0]
    assert isinstance(tool_call.id, str)
    assert len(tool_call.id) > 16

    assert tool_call.type == "function"
    assert tool_call.function is not None
    assert tool_call.function.name == "get_current_weather"
    assert isinstance(tool_call.function.arguments, str)

    parsed_arguments = json.loads(tool_call.function.arguments)
    assert isinstance(parsed_arguments, Dict)
    assert parsed_arguments["city"] == "Dallas"
    assert parsed_arguments["state"] == "TX"
    assert parsed_arguments["unit"] == "fahrenheit"

    assert extracted_tool_calls.content is None


def test_extract_tool_calls_single_tool_with_content(jamba_tool_parser):
    model_output = '''Sure! let me call the tool for you. <tool_calls> [\n{"name": "get_current_weather", "arguments": {"city": "Dallas", "state": "TX", "unit": "fahrenheit"}}] </tool_calls>'''
    extracted_tool_calls = jamba_tool_parser.extract_tool_calls(model_output)
    assert extracted_tool_calls.tools_called == True
    assert len(extracted_tool_calls.tool_calls) == 1
    tool_call = extracted_tool_calls.tool_calls[0]
    assert isinstance(tool_call.id, str)
    assert len(tool_call.id) > 16

    assert tool_call.type == "function"
    assert tool_call.function is not None
    assert tool_call.function.name == "get_current_weather"
    assert isinstance(tool_call.function.arguments, str)

    parsed_arguments = json.loads(tool_call.function.arguments)
    assert isinstance(parsed_arguments, Dict)
    assert parsed_arguments["city"] == "Dallas"
    assert parsed_arguments["state"] == "TX"
    assert parsed_arguments["unit"] == "fahrenheit"

    assert extracted_tool_calls.content == "Sure! let me call the tool for you. "

def test_extract_tool_calls_parallel_tools(jamba_tool_parser):
    model_output = '''<tool_calls> [\n{"name": "get_current_weather", "arguments": {"city": "Dallas", "state": "TX", "unit": "fahrenheit"}}, \n{"name": "get_current_weather", "arguments": {"city": "Orlando", "state": "FL", "unit": "fahrenheit"}}] </tool_calls>'''
    extracted_tool_calls = jamba_tool_parser.extract_tool_calls(model_output)
    assert extracted_tool_calls.tools_called == True
    assert len(extracted_tool_calls.tool_calls) == 2
    tool_call = extracted_tool_calls.tool_calls[0]
    assert isinstance(tool_call.id, str)
    assert len(tool_call.id) > 16

    assert tool_call.type == "function"
    assert tool_call.function is not None
    assert tool_call.function.name == "get_current_weather"
    assert isinstance(tool_call.function.arguments, str)

    parsed_arguments = json.loads(tool_call.function.arguments)
    assert isinstance(parsed_arguments, Dict)
    assert parsed_arguments["city"] == "Dallas"
    assert parsed_arguments["state"] == "TX"
    assert parsed_arguments["unit"] == "fahrenheit"

    tool_call = extracted_tool_calls.tool_calls[1]
    assert isinstance(tool_call.id, str)
    assert len(tool_call.id) > 16

    assert tool_call.type == "function"
    assert tool_call.function is not None
    assert tool_call.function.name == "get_current_weather"
    assert isinstance(tool_call.function.arguments, str)

    parsed_arguments = json.loads(tool_call.function.arguments)
    assert isinstance(parsed_arguments, Dict)
    assert parsed_arguments["city"] == "Orlando"
    assert parsed_arguments["state"] == "FL"
    assert parsed_arguments["unit"] == "fahrenheit"

    assert extracted_tool_calls.content is None