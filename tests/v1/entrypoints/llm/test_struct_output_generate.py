# ruff: noqa: E501
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from enum import Enum
from typing import Any

import jsonschema
import pytest
import regex as re
import torch
from pydantic import BaseModel

from tests.reasoning.utils import run_reasoning_extraction
from vllm.config import StructuredOutputsConfig
from vllm.distributed import cleanup_dist_env_and_memory
from vllm.entrypoints.llm import LLM
from vllm.outputs import RequestOutput
from vllm.platforms import current_platform
from vllm.reasoning.abs_reasoning_parsers import ReasoningParserManager
from vllm.sampling_params import (
    SamplingParams,
    StructuredOutputsParams,
)

NGRAM_SPEC_CONFIG = {
    "model": "[ngram]",
    "num_speculative_tokens": 5,
    "prompt_lookup_max": 5,
    "prompt_lookup_min": 1,
}

EAGLE_SPEC_CONFIG = {
    "method": "eagle",
    "model": "yuhuili/EAGLE-LLaMA3.1-Instruct-8B",
    "num_speculative_tokens": 5,
}

PARAMS_MODELS_BACKENDS_TOKENIZER_MODE = [
    ("mistralai/Ministral-8B-Instruct-2410", "xgrammar", "auto", None),
    # FIXME: Since "auto" will use Mistral tokenizer and these backends do not support
    # it, we skip these tests for now.
    # ("mistralai/Ministral-8B-Instruct-2410", "guidance", "auto", None),
    # ("mistralai/Ministral-8B-Instruct-2410", "lm-format-enforcer", "auto", None),
    ("mistralai/Ministral-8B-Instruct-2410", "guidance", "hf", None),
    pytest.param(
        "mistralai/Ministral-8B-Instruct-2410",
        "lm-format-enforcer",
        "hf",
        None,
        marks=pytest.mark.skip(
            reason=(
                "Flaky: lm-format-enforcer intermittently returns"
                "incomplete JSON."
                "See https://github.com/noamgat/lm-format-enforcer/issues/169"
            )
        ),
    ),
    ("mistralai/Ministral-8B-Instruct-2410", "xgrammar", "mistral", None),
    ("Qwen/Qwen2.5-1.5B-Instruct", "xgrammar", "auto", None),
    pytest.param(
        "Qwen/Qwen2.5-1.5B-Instruct",
        "lm-format-enforcer",
        "auto",
        None,
        marks=pytest.mark.skip(
            reason=(
                "Flaky: lm-format-enforcer intermittently returns"
                "incomplete JSON."
                "See https://github.com/noamgat/lm-format-enforcer/issues/169"
            )
        ),
    ),
    # FIXME: This tests are flaky on CI thus disabled. Tracking in Issue #24402
    # ("mistralai/Ministral-8B-Instruct-2410", "outlines", "auto", None),
    # ("mistralai/Ministral-8B-Instruct-2410", "outlines", "mistral", None),
    # ("Qwen/Qwen2.5-1.5B-Instruct", "guidance", "auto"),
    ("mistralai/Ministral-8B-Instruct-2410", "outlines", "auto", NGRAM_SPEC_CONFIG),
    ("mistralai/Ministral-8B-Instruct-2410", "guidance", "hf", NGRAM_SPEC_CONFIG),
    ("Qwen/Qwen2.5-1.5B-Instruct", "xgrammar", "auto", NGRAM_SPEC_CONFIG),
    ("meta-llama/Meta-Llama-3.1-8B-Instruct", "xgrammar", "auto", EAGLE_SPEC_CONFIG),
]

PARAMS_MODELS_TOKENIZER_MODE = [
    ("mistralai/Ministral-8B-Instruct-2410", "auto"),
    ("Qwen/Qwen2.5-1.5B-Instruct", "auto"),
]


class CarType(str, Enum):
    sedan = "sedan"
    suv = "SUV"
    truck = "Truck"
    coupe = "Coupe"


class CarDescription(BaseModel):
    brand: str
    model: str
    car_type: CarType


@pytest.mark.parametrize(
    "model_name, backend, tokenizer_mode, speculative_config",
    PARAMS_MODELS_BACKENDS_TOKENIZER_MODE,
)
def test_structured_output(
    sample_json_schema: dict[str, Any],
    unsupported_json_schema: dict[str, Any],
    sample_sql_ebnf: str,
    sample_sql_lark: str,
    sample_regex: str,
    sample_structured_outputs_choices: str,
    backend: str,
    tokenizer_mode: str,
    model_name: str,
    speculative_config: dict[str, Any],
):
    if current_platform.is_tpu() and speculative_config:
        pytest.skip("TPU does not support speculative decoding")

    # Use a single LLM instance for several scenarios to
    # speed up the test suite.
    llm = LLM(
        model=model_name,
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
    )

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
    outputs = llm.generate(
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

        outputs = llm.generate(
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
            ValueError,
            match="The provided JSON schema contains features "
            "not supported by xgrammar.",
        ):
            prompt = (
                f"Give an example JSON for an employee profile that "
                f"fits this schema: {unsupported_json_schema}. "
                f"Make the response as short as possible."
            )
            llm.generate(
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
        outputs = llm.generate(
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
        outputs = llm.generate(
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
            ground_truth = "SELECT col_1 from table_1 where col_1 = 1".replace(" ", "")

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
        outputs = llm.generate(
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
            ground_truth = "SELECT col_1 from table_1 where col_1 = 1".replace(" ", "")

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
        with pytest.raises(ValueError, match="Failed to convert the grammar "):
            llm.generate(
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
    outputs = llm.generate(
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

    outputs = llm.generate(
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

    outputs = llm.generate(
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

    outputs = llm.generate(
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
        outputs = llm.generate(prompt, sampling_params=sampling_params, use_tqdm=True)
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


@pytest.mark.parametrize(
    "model_name, backend, tokenizer_mode, reasoning_parser, speculative_config, async_scheduling",  # noqa: E501
    [
        (
            "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
            "xgrammar",
            "auto",
            "deepseek_r1",
            NGRAM_SPEC_CONFIG,
            False,
        ),
        ("Qwen/Qwen3-1.7B", "xgrammar", "auto", "deepseek_r1", None, False),
        ("Qwen/Qwen3-1.7B", "xgrammar", "auto", "deepseek_r1", None, True),
    ],
)
def test_structured_output_with_reasoning_matrices(
    backend: str,
    tokenizer_mode: str,
    reasoning_parser: str,
    model_name: str,
    speculative_config: dict[str, Any] | None,
    async_scheduling: bool,
):
    if current_platform.is_tpu() and speculative_config:
        pytest.skip("TPU does not support speculative decoding")

    # Use a single LLM instance for several scenarios to
    # speed up the test suite.
    llm = LLM(
        model=model_name,
        # Don't use eager execution on TPUs because we want to test for no
        # recompilation at runtime
        enforce_eager=bool(not current_platform.is_tpu()),
        max_model_len=1024,
        max_num_seqs=16,
        structured_outputs_config=dict(
            backend=backend,
            disable_any_whitespace=backend in {"xgrammar", "guidance"},
            reasoning_parser=reasoning_parser,
        ),
        tokenizer_mode=tokenizer_mode,
        speculative_config=speculative_config,
        async_scheduling=async_scheduling,
    )
    tokenizer = llm.get_tokenizer()
    reasoner = ReasoningParserManager.get_reasoning_parser(reasoning_parser)(
        tokenizer=tokenizer
    )

    reasoning_prompt = "Solve the following math problem step-by-step, then provide the final answer as JSON object with a single key 'result'. Make sure to correct your reasoning if there are any issue should it arise.\nProblem: What is 5 * 8 + 2?"  # noqa: E501
    reasoning_schema = {
        "type": "object",
        "properties": {"result": {"type": "integer"}},
        "required": ["result"],
        "additionalProperties": False,
    }
    if "Qwen3" in model_name:
        reasoning_prompt += "<think>\n"

    sampling_params = SamplingParams(
        temperature=0.1,
        max_tokens=8192,
        structured_outputs=StructuredOutputsParams(json=reasoning_schema),
    )
    outputs = llm.generate(
        [reasoning_prompt],
        sampling_params=sampling_params,
        use_tqdm=True,
    )

    assert outputs is not None
    output = outputs[0]
    assert output is not None and isinstance(output, RequestOutput)
    prompt = output.prompt
    generated_text = output.outputs[0].text
    reasoning, content = run_reasoning_extraction(reasoner, [generated_text])
    print(f"Prompt: {prompt!r}\nReasoning: {reasoning!r}\nContent: {content!r}")

    if "Qwen3" in model_name:
        assert content is not None

    assert reasoning is not None

    if content is not None:
        output_json = json.loads(content)
        jsonschema.validate(instance=output_json, schema=reasoning_schema)


@pytest.mark.parametrize("model_name, tokenizer_mode", PARAMS_MODELS_TOKENIZER_MODE)
def test_structured_output_auto_mode(
    unsupported_json_schema: dict[str, Any],
    model_name: str,
    tokenizer_mode: str,
):
    llm = LLM(
        model=model_name,
        max_model_len=1024,
        structured_outputs_config=dict(backend="auto"),
        tokenizer_mode=tokenizer_mode,
        load_format="auto",
        config_format="auto",
    )

    sampling_params = SamplingParams(
        temperature=1.0,
        max_tokens=1000,
        structured_outputs=StructuredOutputsParams(json=unsupported_json_schema),
    )

    prompts = (
        "Give an example JSON object for a grade "
        "that fits this schema: "
        f"{unsupported_json_schema}. Make the response as short as possible."
    )
    # This would fail with the default of "xgrammar", but in "auto"
    # we will handle fallback automatically.
    outputs = llm.generate(prompts, sampling_params=sampling_params, use_tqdm=True)
    # Make sure `auto` backend handling doesn't mess up sampling_params
    # and that we can reuse it without error.
    outputs.extend(
        llm.generate(prompts, sampling_params=sampling_params, use_tqdm=True)
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


def test_guidance_no_additional_properties():
    llm = LLM(
        model="Qwen/Qwen2.5-1.5B-Instruct",
        max_model_len=1024,
        structured_outputs_config=dict(
            backend="guidance",
            disable_any_whitespace=True,
            disable_additional_properties=True,
        ),
    )

    schema = {
        "type": "object",
        "properties": {
            "a1": {"type": "string"},
            "a2": {"type": "string"},
            "a3": {"type": "string"},
        },
        "required": ["a1", "a2", "a3"],
    }

    prompt = (
        "<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a "
        "helpful assistant.<|im_end|>\n<|im_start|>user\nPlease generate a "
        "large JSON object with key-value pairs a1=b1, a2=b2, ..., a20=b20. "
        "Make the response as short as possible."
        "<|im_end|>\n<|im_start|>assistant\n"
    )

    def generate_with_backend(backend):
        structured_outputs_params = StructuredOutputsParams(
            json=schema,
            backend=backend,
            disable_any_whitespace=True,
            disable_additional_properties=True,
        )
        sampling_params = SamplingParams(
            temperature=0, max_tokens=256, structured_outputs=structured_outputs_params
        )

        outputs = llm.generate(prompt, sampling_params=sampling_params)
        assert outputs is not None
        generated_text = outputs[0].outputs[0].text
        assert generated_text is not None
        parsed_json = json.loads(generated_text)
        assert isinstance(parsed_json, dict)
        jsonschema.validate(instance=parsed_json, schema=schema)
        return parsed_json

    generated = generate_with_backend("guidance")
    assert "a1" in generated
    assert "a2" in generated
    assert "a3" in generated
    assert "a4" not in generated
    assert "a5" not in generated
    assert "a6" not in generated


@pytest.mark.parametrize("backend", ["guidance", "xgrammar", "outlines"])
def test_structured_output_batched_with_non_structured_outputs_requests(
    sample_json_schema: dict[str, Any],
    backend: str,
):
    # Don't use eager execution on TPUs because we want to test for no
    # recompilation at runtime
    enforce_eager = bool(not current_platform.is_tpu())

    llm = LLM(
        model="meta-llama/Meta-Llama-3.1-8B-Instruct",
        enforce_eager=enforce_eager,
        max_model_len=1024,
        structured_outputs_config=StructuredOutputsConfig(
            backend=backend,
            disable_any_whitespace=backend in {"xgrammar", "guidance"},
        ),
    )

    structured_outputs_prompt = (
        "Give an example JSON for an employee profile that fits this "
        "schema. Make the response as short as possible. Schema: "
        f"{sample_json_schema}"
    )

    non_structured_outputs_prompt = "The diameter of the Earth in kilometers is "

    prompts = [structured_outputs_prompt, non_structured_outputs_prompt]
    sampling_params = [
        SamplingParams(
            temperature=1.0,
            max_tokens=400,
            structured_outputs=StructuredOutputsParams(json=sample_json_schema),
        ),
        # No max tokens, temp=0 to assert on contents
        SamplingParams(
            seed=42,
            temperature=0,
            top_p=1.0,
        ),
    ]

    outputs = llm.generate(
        prompts=prompts, sampling_params=sampling_params, use_tqdm=True
    )

    assert outputs is not None

    # Free memory as soon as possible as failed assertions
    # will short circuit and not free up memory
    del llm
    torch.cuda.empty_cache()
    cleanup_dist_env_and_memory()

    for index, output in enumerate(outputs):
        assert output is not None
        assert isinstance(output, RequestOutput)
        prompt = output.prompt

        generated_text = output.outputs[0].text
        assert generated_text is not None
        print(f"Prompt:\n{prompt!r}\nGenerated text:\n{generated_text!r}")

        if index == 0:
            # First prompt is structured outputs, expect valid JSON
            assert "\n" not in generated_text
            output_json = json.loads(generated_text)
            jsonschema.validate(instance=output_json, schema=sample_json_schema)
        else:
            # Second prompt is not structured outputs, expect valid output
            # Cannot assert on exact output, but we can expect it to be factual
            assert "12,742" in generated_text

            # non-structured outputs requests should not return a valid JSON here
            with pytest.raises(ValueError):
                output_json = json.loads(generated_text)


@pytest.mark.parametrize("backend", ["xgrammar"])
def test_structured_output_with_structural_tag(backend: str):
    llm = LLM(
        model="Qwen/Qwen2.5-1.5B-Instruct",
        structured_outputs_config=StructuredOutputsConfig(backend=backend),
    )

    structural_tag_config = {
        "type": "structural_tag",
        "format": {
            "type": "triggered_tags",
            "tags": [
                {"begin": "hello_flag", "content": {"type": "any_text"}, "end": "hello"}
            ],
            "triggers": ["hello"],
            "stop_after_first": False,
        },
    }

    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=500,
        structured_outputs=StructuredOutputsParams(
            structural_tag=json.dumps(structural_tag_config)
        ),
    )

    prompt = "Hello and repete hello 10 times, do not say anything else. Only say hello hello hello, now start"
    outputs = llm.generate(prompt, sampling_params=sampling_params, use_tqdm=True)
    assert outputs is not None
    for output in outputs:
        assert output is not None
        assert isinstance(output, RequestOutput)
        prompt = output.prompt
        generated_text = output.outputs[0].text
        assert generated_text is not None
        assert "hello_flag" in generated_text, (
            f"Expected 'hello_flag' to be in generated text, but got: {generated_text}"
        )
