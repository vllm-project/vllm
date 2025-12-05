# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Integration tests for GPT-OSS structural tags functionality (PR #25515)."""

import json
from unittest.mock import Mock

import pytest
import pytest_asyncio
from openai import OpenAI

from vllm.entrypoints.openai.protocol import (
    StructuredOutputsParams,
)
from vllm.reasoning.gptoss_reasoning_parser import (
    GptOssReasoningParser,
)

from ...utils import RemoteOpenAIServer

MODEL_NAME = "openai/gpt-oss-20b"

GET_WEATHER_SCHEMA = {
    "type": "function",
    "name": "get_weather",
    "description": "Get current temperature for provided coordinates in celsius.",
    "parameters": {
        "type": "object",
        "properties": {
            "latitude": {"type": "number"},
            "longitude": {"type": "number"},
        },
        "required": ["latitude", "longitude"],
        "additionalProperties": False,
    },
    "strict": True,
}


class TestGptOssStructuralTagsIntegration:
    """Integration tests for structural tags in GPT-OSS tool calls."""

    @pytest.fixture
    def mock_tokenizer(self):
        """Create a mock tokenizer."""
        tokenizer = Mock()
        tokenizer.encode = Mock(return_value=[1, 2, 3, 4, 5])
        return tokenizer

    @pytest.fixture
    def gptoss_parser(self, mock_tokenizer):
        """Create a real GptOssReasoningParser instance."""
        return GptOssReasoningParser(mock_tokenizer)

    def test_end_to_end_no_tools(self, gptoss_parser):
        """Test end-to-end flow when no tools are available."""
        # Test the parser directly with no tools
        result = gptoss_parser.prepare_structured_tag(None, [])
        parsed_result = json.loads(result)

        # Verify basic structure
        assert parsed_result["type"] == "structural_tag"
        assert parsed_result["format"]["type"] == "triggered_tags"

        # With no tools, should have 5 BASE_TAGS (commentary excluded)
        # (analysis, final, and variants - no commentary)
        assert len(parsed_result["format"]["tags"]) == 5

        # Verify BASE_TAGS are present (check for key channel types)
        tag_begins = [tag["begin"] for tag in parsed_result["format"]["tags"]]
        assert "<|channel|>analysis" in tag_begins
        assert "<|channel|>commentary" not in tag_begins  # Excluded without tools
        assert "<|channel|>final" in tag_begins

        # Verify all tags use regex content type and <|message|> end tag
        for tag in parsed_result["format"]["tags"]:
            assert tag["content"]["type"] == "regex"
            assert tag["end"] == "<|message|>"

        # Verify triggers include both channel and assistant start
        assert "<|channel|>" in parsed_result["format"]["triggers"]
        assert "<|start|>assistant" in parsed_result["format"]["triggers"]
        assert parsed_result["format"]["stop_after_first"] is False

    def test_structural_tag_with_python_tool(self, gptoss_parser):
        """Test structural tag generation with Python tool enabled."""
        # Python is an analysis tool
        result = gptoss_parser.prepare_structured_tag(None, ["python"])
        parsed_result = json.loads(result)

        # Should have 5 BASE_TAGS (no commentary) + 2 python tags
        assert len(parsed_result["format"]["tags"]) == 7

        # Verify python tags are present in analysis channel
        tag_begins = [tag["begin"] for tag in parsed_result["format"]["tags"]]
        assert "<|channel|>analysis to=python" in tag_begins
        assert "<|start|>assistant<|channel|>analysis to=python" in tag_begins
        # Commentary BASE_TAG should not be present (no commentary tools)
        assert "<|channel|>commentary" not in tag_begins

        # Verify triggers are set
        assert "<|channel|>" in parsed_result["format"]["triggers"]
        assert "<|start|>assistant" in parsed_result["format"]["triggers"]

    def test_structured_outputs_params_integration(self, gptoss_parser):
        """Test integration with StructuredOutputsParams."""
        # Generate structural tag with python tool
        structural_tag = gptoss_parser.prepare_structured_tag(None, ["python"])

        # Create StructuredOutputsParams
        params = StructuredOutputsParams(structural_tag=structural_tag)

        # Verify the tag is properly stored and accessible
        assert params.structural_tag == structural_tag

        # Verify the tag is valid JSON
        parsed_tag = json.loads(params.structural_tag)
        assert parsed_tag["type"] == "structural_tag"

    @pytest.mark.parametrize(
        "tool_names, expected_tags",
        [
            # No tools: 5 BASE_TAGS (no commentary)
            ([], 5),
            # Single analysis tool: 5 BASE_TAGS (no commentary) + 2 tags
            (["python"], 7),
            # Multiple analysis tools: 5 BASE_TAGS (no commentary) + 4 tags
            (["python", "browser.search"], 9),
            # All builtin tool types: 5 BASE_TAGS (no commentary) + 6 tags
            (["python", "browser.search", "container.exec"], 11),
        ],
    )
    def test_tool_server_interaction_flow(
        self, gptoss_parser, tool_names, expected_tags
    ):
        """Test the complete tool interaction flow with different tool combinations."""

        # Run the parser and verify results
        result = gptoss_parser.prepare_structured_tag(None, tool_names)
        parsed_result = json.loads(result)

        # Validate number of tags (BASE_TAGS + 2 per tool)
        assert len(parsed_result["format"]["tags"]) == expected_tags

        # Verify tool-specific tags exist for enabled tools
        tag_begins = [tag["begin"] for tag in parsed_result["format"]["tags"]]
        for tool_name in tool_names:
            # Each tool should have both first and last message variants
            assert f"<|channel|>analysis to={tool_name}" in tag_begins
            assert f"<|start|>assistant<|channel|>analysis to={tool_name}" in tag_begins

    def test_original_tag_preservation(self, gptoss_parser):
        """Test that original tags are preserved when provided."""
        original_tag = '{"type": "custom_tag", "data": "preserved"}'

        result = gptoss_parser.prepare_structured_tag(original_tag, ["python"])

        # Should return original tag unchanged
        assert result == original_tag

    @pytest.mark.parametrize(
        "tool_names",
        [
            [],
            ["python"],
            ["browser.search"],
            ["container.exec"],
            ["python", "browser.search"],
            ["browser.search", "container.exec"],
            ["python", "container.exec"],
            ["python", "browser.search", "container.exec"],
            ["functions.get_weather"],  # Function tool
            ["python", "functions.get_weather"],  # Mixed builtin and function
        ],
    )
    def test_json_validity_comprehensive(self, gptoss_parser, tool_names):
        """Test JSON validity across all possible tool combinations."""

        result = gptoss_parser.prepare_structured_tag(None, tool_names)

        # Should be valid JSON
        parsed_result = json.loads(result)

        # Should have correct structure
        assert parsed_result["type"] == "structural_tag"
        assert "format" in parsed_result
        assert "tags" in parsed_result["format"]
        assert "triggers" in parsed_result["format"]

        # Tag count depends on whether there are commentary tools
        num_analysis_tools = sum(
            1 for tool in tool_names if not tool.startswith("functions")
        )
        num_commentary_tools = sum(
            1 for tool in tool_names if tool.startswith("functions")
        )

        if num_commentary_tools > 0:
            # With commentary: 6 BASE_TAGS + 4 per analysis + 2 per commentary
            expected_tag_count = (
                6 + (4 * num_analysis_tools) + (2 * num_commentary_tools)
            )
        else:
            # Without commentary: 5 BASE_TAGS + 2 per analysis
            expected_tag_count = 5 + (2 * num_analysis_tools)

        assert len(parsed_result["format"]["tags"]) == expected_tag_count

    def test_error_handling_empty_tool_names(self, gptoss_parser):
        """Test handling of empty list for tool names."""
        # Empty list should be handled gracefully
        result_empty = gptoss_parser.prepare_structured_tag(None, [])
        parsed_empty = json.loads(result_empty)

        # Should have only BASE_TAGS (5 without commentary)
        assert len(parsed_empty["format"]["tags"]) == 5

        # Verify it's valid JSON with correct structure
        assert parsed_empty["type"] == "structural_tag"
        assert "format" in parsed_empty
        assert "tags" in parsed_empty["format"]

    def test_concurrent_requests_isolation(self, gptoss_parser):
        """Test that concurrent requests don't interfere with each other."""
        # Simulate concurrent requests with different tool configurations
        tool_names_1 = ["python"]
        tool_names_2 = ["browser.search"]

        # Generate tags concurrently
        result_1 = gptoss_parser.prepare_structured_tag(None, tool_names_1)
        result_2 = gptoss_parser.prepare_structured_tag(None, tool_names_2)

        # Parse results
        parsed_1 = json.loads(result_1)
        parsed_2 = json.loads(result_2)

        # Verify they have different tool configurations
        tags_1 = [tag["begin"] for tag in parsed_1["format"]["tags"]]
        tags_2 = [tag["begin"] for tag in parsed_2["format"]["tags"]]

        # Result 1 should have python tags but not browser tags
        assert "<|channel|>analysis to=python" in tags_1
        assert "<|channel|>analysis to=browser.search" not in tags_1

        # Result 2 should have browser tags but not python tags
        assert "<|channel|>analysis to=browser.search" in tags_2
        assert "<|channel|>analysis to=python" not in tags_2

    def test_tag_format_consistency(self, gptoss_parser):
        """Test that all generated tags follow consistent format."""
        tool_names = ["python", "browser.search"]

        result = gptoss_parser.prepare_structured_tag(None, tool_names)
        parsed_result = json.loads(result)

        # Verify all tags have required fields
        for tag in parsed_result["format"]["tags"]:
            assert "begin" in tag
            assert "content" in tag
            assert "end" in tag
            assert tag["content"]["type"] == "regex"

            # End tag should always end with <|message|>
            # BASE_TAGS have just "<|message|>"
            # Tool-specific tags have content type prefix like:
            # " code<|message|>" or " <|constrain|>json<|message|>"
            assert tag["end"].endswith("<|message|>")

            # Verify begin format starts with channel or assistant start
            assert tag["begin"].startswith("<|channel|>") or tag["begin"].startswith(
                "<|start|>assistant"
            )

    def test_trigger_configuration(self, gptoss_parser):
        """Test trigger configuration for different tool setups."""
        # Test with no tools
        result_no_tools = gptoss_parser.prepare_structured_tag(None, [])
        parsed_no_tools = json.loads(result_no_tools)

        # Triggers should include channel and assistant start
        expected_triggers = ["<|channel|>", "<|start|>assistant"]
        assert set(parsed_no_tools["format"]["triggers"]) == set(expected_triggers)

        # Test with tools - should have same triggers
        result_with_tools = gptoss_parser.prepare_structured_tag(None, ["python"])
        parsed_with_tools = json.loads(result_with_tools)

        # Triggers remain the same regardless of tools
        assert set(parsed_with_tools["format"]["triggers"]) == set(expected_triggers)

    def test_tool_names_with_namespaces(self, gptoss_parser):
        """Test that tool names with namespace prefixes work correctly."""
        # Test with all namespace formats
        tool_names = [
            "python",  # Special case - no namespace
            "browser.search",  # Browser namespace
            "browser.find",
            "browser.open",
            "container.exec",  # Container namespace
            "functions.get_weather",  # Functions namespace
            "functions.calculate",
        ]

        result = gptoss_parser.prepare_structured_tag(None, tool_names)
        parsed_result = json.loads(result)

        # 5 analysis tools, 2 commentary tools
        # 6 BASE_TAGS + (5 × 4) + (2 × 2) = 6 + 20 + 4 = 30 tags
        assert len(parsed_result["format"]["tags"]) == 30

        # Verify all tool names appear in tags
        tag_begins = [tag["begin"] for tag in parsed_result["format"]["tags"]]
        for tool_name in tool_names:
            # Functions go to commentary channel only
            # Analysis tools go to BOTH channels (when commentary exists)
            if tool_name.startswith("functions"):
                assert f"<|channel|>commentary to={tool_name}" in tag_begins
                # Should NOT be on analysis
                assert f"<|channel|>analysis to={tool_name}" not in tag_begins
            else:
                # Analysis tools on both channels when commentary exists
                assert f"<|channel|>analysis to={tool_name}" in tag_begins
                assert f"<|channel|>commentary to={tool_name}" in tag_begins

    def test_analysis_vs_commentary_tools(self, gptoss_parser):
        """Test tools categorization into analysis vs commentary channels."""
        # Analysis tools: builtin tools (python, browser.*, container.*)
        analysis_tools = ["python", "browser.search", "container.exec"]

        # Commentary tools: custom functions
        commentary_tools = ["functions.get_weather", "functions.calculate"]

        # Test with both types
        all_tools = analysis_tools + commentary_tools
        result = gptoss_parser.prepare_structured_tag(None, all_tools)
        parsed_result = json.loads(result)

        tag_begins = [tag["begin"] for tag in parsed_result["format"]["tags"]]
        all_tags = parsed_result["format"]["tags"]

        # When commentary tools exist, analysis tools get BOTH channels
        for tool in analysis_tools:
            # Analysis tools always on analysis channel
            assert f"<|channel|>analysis to={tool}" in tag_begins
            assert f"<|start|>assistant<|channel|>analysis to={tool}" in tag_begins
            # Also on commentary (to handle model flipping)
            assert f"<|channel|>commentary to={tool}" in tag_begins
            assert f"<|start|>assistant<|channel|>commentary to={tool}" in tag_begins

            # Verify content_type: analysis tools use "code" on BOTH channels
            for tag in all_tags:
                if f"to={tool}" in tag["begin"]:
                    assert tag["end"] == " code<|message|>", (
                        f"Analysis tool {tool} should use 'code' format on "
                        f"{tag['begin'].split('|')[1]} channel"
                    )

        # Verify commentary tools go to commentary channel only
        for tool in commentary_tools:
            assert f"<|channel|>commentary to={tool}" in tag_begins
            assert f"<|start|>assistant<|channel|>commentary to={tool}" in tag_begins
            # Should NOT be in analysis
            assert f"<|channel|>analysis to={tool}" not in tag_begins

            # Verify content_type: function tools use "json" format
            for tag in all_tags:
                if f"to={tool}" in tag["begin"]:
                    assert tag["end"] == " <|constrain|>json<|message|>", (
                        f"Function tool {tool} should use 'json' format"
                    )

    def test_mixed_function_and_builtin_tools(self, gptoss_parser):
        """Test structural tag generation with mixed function and builtin tools."""
        # Test with both function and builtin tools
        tool_names = [
            "python",  # Code interpreter (builtin)
            "functions.get_weather",  # Custom function
            "functions.search_database",  # Another custom function
        ]

        result = gptoss_parser.prepare_structured_tag(None, tool_names)
        parsed_result = json.loads(result)

        # Verify valid JSON structure
        assert parsed_result["type"] == "structural_tag"
        assert parsed_result["format"]["type"] == "triggered_tags"

        # 1 analysis tool, 2 commentary tools
        # 6 BASE_TAGS + (1 × 4) + (2 × 2) = 6 + 4 + 4 = 14 tags
        assert len(parsed_result["format"]["tags"]) == 14

        tag_begins = [tag["begin"] for tag in parsed_result["format"]["tags"]]

        # Verify python (builtin) is on BOTH channels (commentary exists)
        assert "<|channel|>analysis to=python" in tag_begins
        assert "<|start|>assistant<|channel|>analysis to=python" in tag_begins
        assert "<|channel|>commentary to=python" in tag_begins
        assert "<|start|>assistant<|channel|>commentary to=python" in tag_begins

        # Verify functions are in commentary channel only
        assert "<|channel|>commentary to=functions.get_weather" in tag_begins
        assert (
            "<|start|>assistant<|channel|>commentary to=functions.get_weather"
            in tag_begins
        )
        assert "<|channel|>commentary to=functions.search_database" in tag_begins
        assert (
            "<|start|>assistant<|channel|>commentary to=functions.search_database"
            in tag_begins
        )
        # Functions should NOT be on analysis
        assert "<|channel|>analysis to=functions.get_weather" not in tag_begins
        assert "<|channel|>analysis to=functions.search_database" not in tag_begins

        # Verify triggers are correct
        assert "<|channel|>" in parsed_result["format"]["triggers"]
        assert "<|start|>assistant" in parsed_result["format"]["triggers"]

        # Verify at_least_one and stop_after_first settings
        assert parsed_result["format"]["at_least_one"] is True
        assert parsed_result["format"]["stop_after_first"] is False


# E2E tests with real server and model inference
@pytest.fixture(scope="module")
def monkeypatch_module():
    from _pytest.monkeypatch import MonkeyPatch

    mpatch = MonkeyPatch()
    yield mpatch
    mpatch.undo()


@pytest.fixture(scope="module")
def server(monkeypatch_module: pytest.MonkeyPatch):
    """Start vLLM server with gpt-oss model for e2e tests."""
    args = [
        "--enforce-eager",
        "--tool-server",
        "demo",
        '--structured_outputs_config={"enable_in_reasoning": true}',
    ]

    with monkeypatch_module.context() as m:
        m.setenv("VLLM_ENABLE_RESPONSES_API_STORE", "1")
        m.setenv("PYTHON_EXECUTION_BACKEND", "dangerously_use_uv")
        # Helps the model follow instructions better
        m.setenv("VLLM_GPT_OSS_HARMONY_SYSTEM_INSTRUCTIONS", "1")
        with RemoteOpenAIServer(MODEL_NAME, args) as remote_server:
            yield remote_server


@pytest_asyncio.fixture
async def client(server):
    """Get async OpenAI client for e2e tests."""
    async with server.get_async_client() as async_client:
        yield async_client


@pytest.mark.asyncio
@pytest.mark.parametrize("model_name", [MODEL_NAME])
@pytest.mark.skip(reason="gpt-oss 20b messes up to much, works on 120b though")
async def test_e2e_mixed_function_and_builtin_tools(client: OpenAI, model_name: str):
    """E2E test with both function tools and code_interpreter (builtin tool).

    This test validates that the structural tags work correctly when both
    function tools and builtin tools are available, ensuring proper channel
    categorization (functions -> commentary, builtin -> analysis).
    """
    tools = [
        GET_WEATHER_SCHEMA,  # Function tool
        {"type": "code_interpreter", "container": {"type": "auto"}},
    ]

    # Make a request that could potentially use either tool
    response = await client.responses.create(
        model=model_name,
        input=(
            "Execute the following code using the python tool: "
            "import random; print(random.randint(1, 50), random.randint(1, 50))"
            "\n Then tell me what the weather "
            "is like at coordinates 48.8566, 2.3522"
        ),
        tools=tools,
        instructions=(
            "You must use the Python tool to execute code. Never simulate execution."
        ),
        extra_body={"enable_response_messages": True},
    )

    assert response is not None
    assert response.status == "completed"

    # Validate using output_messages - check recipients and channels
    python_tool_call_found = False
    python_tool_response_found = False
    function_tool_call_found = False

    for message in response.output_messages:
        recipient = message.get("recipient")
        channel = message.get("channel")
        author = message.get("author", {})

        # Check for python tool call
        if recipient and recipient.startswith("python"):
            python_tool_call_found = True

        # Check for python tool response
        if (
            author.get("role") == "tool"
            and author.get("name")
            and author.get("name").startswith("python")
        ):
            python_tool_response_found = True

        # Check for function tool call
        if recipient and recipient.startswith("functions.get_weather"):
            function_tool_call_found = True
            assert channel == "commentary", (
                "Function tool call should be on commentary channel"
            )

    # Verify both tool types were used
    assert python_tool_call_found, "Should have found python tool call"
    assert python_tool_response_found, "Should have found python tool response"
    assert function_tool_call_found, (
        "Should have found function tool call on commentary channel"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("model_name", [MODEL_NAME])
async def test_e2e_function_tool_edge_case_names(client: OpenAI, model_name: str):
    """E2E test with function tool having edge case name (100 alternating L's).

    Without guided decoding via structural tags, the model would struggle
    to generate valid tool calls with unusual names like 100 alternating L's.
    This test validates that structural tags enable correct generation.
    """
    # Function tool with name that's alternating uppercase/lowercase L's
    # 100 characters total - edge case that would typically cause issues
    # without proper guided decoding
    edge_case_name = "".join("Ll" for _ in range(50))  # "LlLlLl..." 100 chars

    edge_case_tool = {
        "type": "function",
        "name": edge_case_name,
        "description": "A test function with alternating L's in the name",
        "parameters": {
            "type": "object",
            "properties": {
                "value": {"type": "number"},
            },
            "required": ["value"],
            "additionalProperties": False,
        },
        "strict": True,
    }

    response = await client.responses.create(
        model=model_name,
        input="Use the LlLl tool with value 42",
        instructions=(
            "You must call the function tool available to you. "
            "Just jump straight into calling it."
        ),
        tools=[edge_case_tool],
        temperature=0.0,
        extra_body={"enable_response_messages": True},
        reasoning={"effort": "low"},
    )

    assert response is not None
    assert response.status == "completed"

    # Validate using output_messages like in test_response_api_mcp_tools.py
    tool_call_found = False
    tool_call_on_commentary = False

    for message in response.output_messages:
        recipient = message.get("recipient")
        channel = message.get("channel")

        # Check for function tool call with edge case name
        expected_recipient = f"functions.{edge_case_name}"
        if recipient and recipient == expected_recipient:
            tool_call_found = True
            # Function tools should go to commentary channel
            assert channel == "commentary", (
                "Function tool call should be on commentary channel"
            )
            tool_call_on_commentary = True

    # Verify the structural tags enabled successful tool call generation
    assert tool_call_found, (
        f"Should have found a function tool call with edge case name "
        f"(100 char alternating L's). "
        f"Structural tags should enable this despite unusual name. "
        f"Expected recipient: 'functions.{edge_case_name[:20]}...'"
    )
    assert tool_call_on_commentary, "Function tool should be on commentary channel"

    # Verify input messages have developer message with function tool
    developer_message_found = False
    for message in response.input_messages:
        author = message.get("author", {})
        if author.get("role") == "developer":
            developer_message_found = True
            break

    assert developer_message_found, (
        "Developer message with function tool should be present in input_messages"
    )
