# ruff: noqa: E501
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from typing import Any

import jsonschema
import pytest
import regex as re

from tests.entrypoints.llm.structured_output._structured_output import (
    PARAMS_MODELS_BACKENDS_TOKENIZER_MODE,
    SAMPLE_JSON_SCHEMA,
    SAMPLE_REGEX,
    SAMPLE_SQL_EBNF,
    SAMPLE_SQL_LARK,
    SAMPLE_STRUCTURED_OUTPUTS_CHOICES,
    UNSUPPORTED_JSON_SCHEMA,
    CarDescription,
    platform_args,
)
from vllm.exceptions import VLLMValidationError
from vllm.outputs import RequestOutput
from vllm.platforms import current_platform
from vllm.sampling_params import SamplingParams, StructuredOutputsParams


@pytest.mark.parametrize(
    "model_name, backend, tokenizer_mode, speculative_config",
    PARAMS_MODELS_BACKENDS_TOKENIZER_MODE,
)
def test_structured_output(
    backend: str,
    tokenizer_mode: str,
    model_name: str,
    speculative_config: dict[str, Any],
    vllm_runner,
):
    sample_json_schema = SAMPLE_JSON_SCHEMA
    unsupported_json_schema = UNSUPPORTED_JSON_SCHEMA
    sample_sql_ebnf = SAMPLE_SQL_EBNF
    sample_sql_lark = SAMPLE_SQL_LARK
    sample_regex = SAMPLE_REGEX
    sample_structured_outputs_choices = SAMPLE_STRUCTURED_OUTPUTS_CHOICES
    if current_platform.is_tpu() and speculative_config:
        pytest.skip("TPU does not support speculative decoding")

    # Use a single LLM instance for several scenarios to
    # speed up the test suite.
    with vllm_runner(
        model_name,
        enforce_eager=True,
        max_model_len=1024,
        structured_outputs_config=dict(
            backend=backend, disable_any_whitespace=backend in {"xgrammar", "guidance"}
        ),
        seed=120,
        tokenizer_mode=tokenizer_mode,
        load_format="auto" if not model_name.startswith("mistralai/") else "hf",
        config_format="auto" if not model_name.startswith("mistralai/") else "hf",
        speculative_config=speculative_config,
        **platform_args,
    ) as runner:
        #
        # Test 1: Generate JSON output based on a provided schema
        #
        sampling_params = SamplingParams(
            temperature=1.0,
            max_tokens=4096,
            structured_outputs=StructuredOutputsParams(json=sample_json_schema),
        )

        prompt = (
            "Give an example JSON for an employee profile that fits this "
            "schema. Make the response as short as possible. Schema: "
            f"{sample_json_schema}"
        )
        outputs = runner.llm.generate(
            [prompt] * 2,
            sampling_params=sampling_params,
            use_tqdm=True,
        )

        assert outputs is not None

        for output in outputs:
            assert output is not None
            assert isinstance(output, RequestOutput)
            prompt = output.prompt

            generated_text = output.outputs[0].text
            assert generated_text is not None
            if backend != "lm-format-enforcer":
                assert "\n" not in generated_text
            print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
            try:
                output_json = json.loads(generated_text)
            except json.JSONDecodeError as e:
                pytest.fail(
                    f"Invalid JSON from backend={backend}: {generated_text!r}\n"
                    f"Schema: {sample_json_schema}\nError: {e}"
                )
            jsonschema.validate(instance=output_json, schema=sample_json_schema)

        #
        # Test 2: Generate JSON object without a schema
        #
        if backend != "outlines":
            sampling_params = SamplingParams(
                temperature=1.0,
                max_tokens=4096,
                n=2,
                structured_outputs=StructuredOutputsParams(json_object=True),
            )

            outputs = runner.llm.generate(
                prompts=(
                    "Generate a JSON object with curly braces for a person with "
                    "name and age fields for John Smith who is 31 years old. "
                    "Make the response as short as possible."
                ),
                sampling_params=sampling_params,
                use_tqdm=True,
            )

            assert outputs is not None
            for output in outputs:
                assert output is not None
                assert isinstance(output, RequestOutput)

                for i in range(2):
                    generated_text = output.outputs[i].text
                    print(generated_text)
                    assert generated_text is not None

                    # Parse to verify it is a valid JSON object
                    parsed_json = json.loads(generated_text)
                    assert isinstance(parsed_json, dict)

        #
        # Test 3: test a jsonschema incompatible with xgrammar
        #
        sampling_params = SamplingParams(
            temperature=1.0,
            max_tokens=4096,
            structured_outputs=StructuredOutputsParams(json=unsupported_json_schema),
        )
        if backend.startswith("xgrammar"):
            with pytest.raises(
                VLLMValidationError,
                match="The provided JSON schema contains features "
                "not supported by xgrammar.",
            ):
                prompt = (
                    f"Give an example JSON for an employee profile that "
                    f"fits this schema: {unsupported_json_schema}. "
                    f"Make the response as short as possible."
                )
                runner.llm.generate(
                    [prompt] * 2,
                    sampling_params=sampling_params,
                    use_tqdm=True,
                )
        else:
            prompt = (
                f"Give an example JSON object for a grade that "
                f"fits this schema: {unsupported_json_schema}. "
                f"Make the response as short as possible."
            )
            outputs = runner.llm.generate(
                prompt,
                sampling_params=sampling_params,
                use_tqdm=True,
            )
            assert outputs is not None
            for output in outputs:
                assert output is not None
                assert isinstance(output, RequestOutput)
                generated_text = output.outputs[0].text
                assert generated_text is not None
                print(generated_text)

                # Parse to verify it is valid JSON
                parsed_json = json.loads(generated_text)
                assert isinstance(parsed_json, dict)

        if backend not in ["outlines", "lm-format-enforcer"]:
            #
            # Test 4: Generate SQL statement using EBNF grammar
            #
            sampling_params = SamplingParams(
                temperature=0.8,
                top_p=0.95,
                max_tokens=1000,
                structured_outputs=StructuredOutputsParams(grammar=sample_sql_ebnf),
            )
            outputs = runner.llm.generate(
                (
                    "Generate a sql statement that selects col_1 from "
                    "table_1 where it is equal to 1. Make the response as short as "
                    "possible."
                ),
                sampling_params=sampling_params,
                use_tqdm=True,
            )

            assert outputs is not None
            for output in outputs:
                assert output is not None
                assert isinstance(output, RequestOutput)
                prompt = output.prompt

                generated_text = output.outputs[0].text
                assert generated_text is not None

                # remove spaces for comparison b/c we removed them in the grammar
                ground_truth = "SELECT col_1 from table_1 where col_1 = 1".replace(
                    " ", ""
                )

                assert generated_text.strip() == ground_truth

                print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")

            #
            # Test 5: Generate SQL statement using Lark grammar
            #
            sampling_params = SamplingParams(
                temperature=0.8,
                top_p=0.95,
                max_tokens=1000,
                structured_outputs=StructuredOutputsParams(grammar=sample_sql_lark),
            )
            outputs = runner.llm.generate(
                (
                    "Generate a sql statement that selects col_1 from "
                    "table_1 where it is equal to 1. Make the response as short as "
                    "possible."
                ),
                sampling_params=sampling_params,
                use_tqdm=True,
            )

            assert outputs is not None
            for output in outputs:
                assert output is not None
                assert isinstance(output, RequestOutput)
                prompt = output.prompt

                generated_text = output.outputs[0].text
                assert generated_text is not None

                # use Lark to parse the output, and make sure it's a valid parse tree
                from lark import Lark

                parser = Lark(sample_sql_lark)
                parser.parse(generated_text)

                # remove spaces for comparison b/c we removed them in the grammar
                ground_truth = "SELECT col_1 from table_1 where col_1 = 1".replace(
                    " ", ""
                )

                assert generated_text.strip() == ground_truth

                print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")

            #
            # Test 6: Test invalid grammar input
            #
            sampling_params = SamplingParams(
                temperature=0.8,
                top_p=0.95,
                max_tokens=1000,
                structured_outputs=StructuredOutputsParams(grammar="not a grammar"),
            )
            with pytest.raises(
                VLLMValidationError, match="Invalid grammar specification"
            ):
                runner.llm.generate(
                    (
                        "Generate a sql statement that selects col_1 from "
                        "table_1 where it is equal to 1. Make the response as short "
                        "as possible."
                    ),
                    sampling_params=sampling_params,
                    use_tqdm=True,
                )

        #
        # Test 7: Generate text based on a regex pattern
        #
        sampling_params = SamplingParams(
            temperature=0.8,
            top_p=0.95,
            structured_outputs=StructuredOutputsParams(regex=sample_regex),
        )

        prompt = (
            f"Give an example IPv4 address with this regex: {sample_regex}. "
            f"Make the response as short as possible."
        )
        outputs = runner.llm.generate(
            [prompt] * 2,
            sampling_params=sampling_params,
            use_tqdm=True,
        )

        assert outputs is not None
        for output in outputs:
            assert output is not None
            assert isinstance(output, RequestOutput)
            prompt = output.prompt
            generated_text = output.outputs[0].text
            print(generated_text)
            assert generated_text is not None
            assert re.fullmatch(sample_regex, generated_text) is not None
            print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")

        #
        # Test 8: Generate text based on a choices
        #
        sampling_params = SamplingParams(
            temperature=0.8,
            top_p=0.95,
            structured_outputs=StructuredOutputsParams(
                choice=sample_structured_outputs_choices
            ),
        )

        outputs = runner.llm.generate(
            (
                "The best language for type-safe systems programming is "
                "(Make the response as short as possible.) "
            ),
            sampling_params=sampling_params,
            use_tqdm=True,
        )
        assert outputs is not None
        for output in outputs:
            assert output is not None
            assert isinstance(output, RequestOutput)
            prompt = output.prompt
            generated_text = output.outputs[0].text
            print(generated_text)
            assert generated_text is not None
            assert generated_text in sample_structured_outputs_choices
            print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")

        #
        # Test 9: Generate structured output using a Pydantic model with an enum
        #
        json_schema = CarDescription.model_json_schema()
        sampling_params = SamplingParams(
            temperature=1.0,
            max_tokens=1000,
            structured_outputs=StructuredOutputsParams(json=json_schema),
        )

        outputs = runner.llm.generate(
            (
                "Generate a JSON with the brand, model and car_type of the most "
                "iconic car from the 90's. Make the response as short as "
                "possible."
            ),
            sampling_params=sampling_params,
            use_tqdm=True,
        )

        assert outputs is not None

        for output in outputs:
            assert output is not None
            assert isinstance(output, RequestOutput)
            prompt = output.prompt

            generated_text = output.outputs[0].text
            assert generated_text is not None
            print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
            try:
                output_json = json.loads(generated_text)
            except json.JSONDecodeError as e:
                pytest.fail(
                    f"Invalid JSON from backend={backend}: {generated_text!r}\n"
                    f"Schema: {json_schema}\nError: {e}"
                )
            jsonschema.validate(instance=output_json, schema=json_schema)

        #
        # Test 10: Generate structured with minLength and maxLength
        #
        min_length = 50
        max_length = 50
        json_schema = {
            "type": "object",
            "properties": {
                "description": {
                    "type": "string",
                    "maxLength": max_length,
                    "minLength": min_length,
                }
            },
            "required": ["description"],
            "additionalProperties": False,
        }

        sampling_params = SamplingParams(
            temperature=1.0,
            max_tokens=4096,
            structured_outputs=StructuredOutputsParams(json=json_schema),
        )

        outputs = runner.llm.generate(
            (
                "Generate a description of a frog using 50 characters. "
                "Make the response as short as possible."
            ),
            sampling_params=sampling_params,
            use_tqdm=True,
        )

        assert outputs is not None

        for output in outputs:
            assert output is not None
            assert isinstance(output, RequestOutput)
            prompt = output.prompt

            generated_text = output.outputs[0].text
            assert generated_text is not None
            print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
            try:
                output_json = json.loads(generated_text)
            except json.JSONDecodeError as e:
                pytest.fail(
                    f"Invalid JSON from backend={backend}: {generated_text!r}\n"
                    f"Schema: {json_schema}\nError: {e}"
                )
            jsonschema.validate(instance=output_json, schema=json_schema)

        if backend not in ["outlines", "lm-format-enforcer"]:
            #
            # Test 11: Generate structured output using structural_tag format
            #
            structural_tag_config = {
                "type": "structural_tag",
                "structures": [
                    {
                        "begin": "<function=get_weather>",
                        "schema": {
                            "type": "object",
                            "properties": {"city": {"type": "string"}},
                            "additionalProperties": False,
                        },
                        "end": "</function>",
                    }
                ],
                "triggers": ["<function="],
            }

            sampling_params = SamplingParams(
                temperature=0.0,
                max_tokens=4096,
                structured_outputs=StructuredOutputsParams(
                    structural_tag=json.dumps(structural_tag_config)
                ),
            )

            prompt = """
You have access to the following function to retrieve the weather in a city:

    {
        "name": "get_weather",
        "parameters": {
            "city": {
                "param_type": "string",
                "description": "The city to get the weather for",
                "required": True
            }
        }
    }

If a you choose to call a function ONLY reply in the following format:
<{start_tag}={function_name}>{parameters}{end_tag}
where

start_tag => `<function`
parameters => a JSON dict with the function argument name
            as key and function argument value as value.
end_tag => `</function>`

Here is an example,
<function=example_function_name>{"example_name": "example_value"}</function>

Reminder:
- Function calls MUST follow the specified format
- Required parameters MUST be specified
- Only call one function at a time
- Put the entire function call reply on one line
- Always add your sources when using search results to answer the user query

You are a helpful assistant.

Given the previous instructions, what is the weather in New York City? \
Make the response as short as possible.
"""

            # Change this once other backends support structural_tag
            outputs = runner.llm.generate(
                prompt, sampling_params=sampling_params, use_tqdm=True
            )
            assert outputs is not None

            for output in outputs:
                assert output is not None
                assert isinstance(output, RequestOutput)
                generated_text = output.outputs[0].text
                assert generated_text is not None

                # Search for function call pattern in the response
                function_call_pattern = r"<function=get_weather>(.*?)</function>"
                matches = re.findall(function_call_pattern, generated_text)

                if not matches:
                    print(
                        f"Warning: No function calls found in response: {generated_text!r}"
                    )
                    continue

                # Take the first function call if multiple are found
                json_str = matches[0]
                try:
                    json_content = json.loads(json_str)
                    assert "city" in json_content
                    assert isinstance(json_content["city"], str)
                    print(f"Found valid function call: {generated_text!r}")
                except (json.JSONDecodeError, AssertionError) as e:
                    pytest.fail(
                        f"Invalid function call format: {generated_text!r}\nError: {str(e)}"
                    )
