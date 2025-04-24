# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import re
from enum import Enum
from typing import Any

import jsonschema
import pytest
from pydantic import BaseModel

from vllm.entrypoints.llm import LLM
from vllm.outputs import RequestOutput
from vllm.platforms import current_platform
from vllm.sampling_params import GuidedDecodingParams, SamplingParams

PARAMS_MODELS_BACKENDS_TOKENIZER_MODE = [
    ("mistralai/Ministral-8B-Instruct-2410", "xgrammar:disable-any-whitespace",
     "auto"),
    ("mistralai/Ministral-8B-Instruct-2410", "guidance:disable-any-whitespace",
     "auto"),
    ("mistralai/Ministral-8B-Instruct-2410", "xgrammar:disable-any-whitespace",
     "mistral"),
    ("Qwen/Qwen2.5-1.5B-Instruct", "xgrammar:disable-any-whitespace", "auto"),
    #FIXME: This test is flaky on CI thus disabled
    #("Qwen/Qwen2.5-1.5B-Instruct", "guidance:disable-any-whitespace", "auto"),
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


@pytest.mark.skip_global_cleanup
@pytest.mark.parametrize("model_name, guided_decoding_backend, tokenizer_mode",
                         PARAMS_MODELS_BACKENDS_TOKENIZER_MODE)
def test_structured_output(
    monkeypatch: pytest.MonkeyPatch,
    sample_json_schema: dict[str, Any],
    unsupported_json_schema: dict[str, Any],
    sample_sql_ebnf: str,
    sample_sql_lark: str,
    sample_regex: str,
    sample_guided_choice: str,
    guided_decoding_backend: str,
    tokenizer_mode: str,
    model_name: str,
):
    monkeypatch.setenv("VLLM_USE_V1", "1")

    # Don't use eager execution on TPUs because we want to test for no
    # recompilation at runtime
    enforce_eager = bool(not current_platform.is_tpu())
    # Use a single LLM instance for several scenarios to
    # speed up the test suite.
    llm = LLM(model=model_name,
              enforce_eager=enforce_eager,
              max_model_len=1024,
              guided_decoding_backend=guided_decoding_backend,
              tokenizer_mode=tokenizer_mode)

    #
    # Test 1: Generate JSON output based on a provided schema
    #
    sampling_params = SamplingParams(
        temperature=1.0,
        max_tokens=1000,
        guided_decoding=GuidedDecodingParams(json=sample_json_schema))
    outputs = llm.generate(prompts=[
        f"Give an example JSON for an employee profile "
        f"that fits this schema: {sample_json_schema}"
    ] * 2,
                           sampling_params=sampling_params,
                           use_tqdm=True)

    assert outputs is not None

    for output in outputs:
        assert output is not None
        assert isinstance(output, RequestOutput)
        prompt = output.prompt

        generated_text = output.outputs[0].text
        assert generated_text is not None
        if 'disable-any-whitespace' in guided_decoding_backend:
            assert "\n" not in generated_text
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
        output_json = json.loads(generated_text)
        jsonschema.validate(instance=output_json, schema=sample_json_schema)

    #
    # Test 2: Generate JSON object without a schema
    #
    sampling_params = SamplingParams(
        temperature=1.0,
        max_tokens=100,
        n=2,
        guided_decoding=GuidedDecodingParams(json_object=True))

    outputs = llm.generate(
        prompts=("Generate a JSON object with curly braces for a person with "
                 "name and age fields for John Smith who is 31 years old."),
        sampling_params=sampling_params,
        use_tqdm=True)

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
        max_tokens=1000,
        guided_decoding=GuidedDecodingParams(json=unsupported_json_schema))
    if guided_decoding_backend.startswith("xgrammar"):
        with pytest.raises(ValueError,
                           match="The provided JSON schema contains features "
                           "not supported by xgrammar."):
            llm.generate(prompts=[
                f"Give an example JSON for an employee profile "
                f"that fits this schema: {unsupported_json_schema}"
            ] * 2,
                         sampling_params=sampling_params,
                         use_tqdm=True)
    else:
        outputs = llm.generate(
            prompts=("Give an example JSON object for a grade "
                     "that fits this schema: "
                     f"{unsupported_json_schema}"),
            sampling_params=sampling_params,
            use_tqdm=True)
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

    #
    # Test 4: Generate SQL statement using EBNF grammar
    #
    sampling_params = SamplingParams(
        temperature=0.8,
        top_p=0.95,
        max_tokens=1000,
        guided_decoding=GuidedDecodingParams(grammar=sample_sql_ebnf))
    outputs = llm.generate(
        prompts=("Generate a sql statement that selects col_1 from "
                 "table_1 where it is equal to 1"),
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
            " ", "")

        assert generated_text.strip() == ground_truth

        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")

    #
    # Test 5: Generate SQL statement using Lark grammar
    #
    sampling_params = SamplingParams(
        temperature=0.8,
        top_p=0.95,
        max_tokens=1000,
        guided_decoding=GuidedDecodingParams(grammar=sample_sql_lark))
    outputs = llm.generate(
        prompts=("Generate a sql statement that selects col_1 from "
                 "table_1 where it is equal to 1"),
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
            " ", "")

        assert generated_text.strip() == ground_truth

        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")

    #
    # Test 6: Test invalid grammar input
    #
    sampling_params = SamplingParams(
        temperature=0.8,
        top_p=0.95,
        max_tokens=1000,
        guided_decoding=GuidedDecodingParams(grammar="not a grammar"))
    with pytest.raises(ValueError, match="Failed to convert the grammar "):
        llm.generate(
            prompts=("Generate a sql statement that selects col_1 from "
                     "table_1 where it is equal to 1"),
            sampling_params=sampling_params,
            use_tqdm=True,
        )

    #
    # Test 7: Generate text based on a regex pattern
    #
    sampling_params = SamplingParams(
        temperature=0.8,
        top_p=0.95,
        guided_decoding=GuidedDecodingParams(regex=sample_regex))
    outputs = llm.generate(
        prompts=[
            f"Give an example IPv4 address with this regex: {sample_regex}"
        ] * 2,
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
        guided_decoding=GuidedDecodingParams(choice=sample_guided_choice))
    outputs = llm.generate(
        prompts="The best language for type-safe systems programming is ",
        sampling_params=sampling_params,
        use_tqdm=True)
    assert outputs is not None
    for output in outputs:
        assert output is not None
        assert isinstance(output, RequestOutput)
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(generated_text)
        assert generated_text is not None
        assert generated_text in sample_guided_choice
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")

    #
    # Test 9: Generate structured output using a Pydantic model with an enum
    #
    json_schema = CarDescription.model_json_schema()
    sampling_params = SamplingParams(
        temperature=1.0,
        max_tokens=1000,
        guided_decoding=GuidedDecodingParams(json=json_schema))
    outputs = llm.generate(
        prompts="Generate a JSON with the brand, model and car_type of"
        "the most iconic car from the 90's",
        sampling_params=sampling_params,
        use_tqdm=True)

    assert outputs is not None

    for output in outputs:
        assert output is not None
        assert isinstance(output, RequestOutput)
        prompt = output.prompt

        generated_text = output.outputs[0].text
        assert generated_text is not None
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
        output_json = json.loads(generated_text)
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
                "minLength": min_length
            }
        },
        "required": ["description"]
    }

    sampling_params = SamplingParams(
        temperature=1.0,
        max_tokens=1000,
        guided_decoding=GuidedDecodingParams(json=json_schema))
    outputs = llm.generate(
        prompts="Generate a description of a frog using 50 characters.",
        sampling_params=sampling_params,
        use_tqdm=True)

    assert outputs is not None

    for output in outputs:
        assert output is not None
        assert isinstance(output, RequestOutput)
        prompt = output.prompt

        generated_text = output.outputs[0].text
        assert generated_text is not None
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
        output_json = json.loads(generated_text)
        jsonschema.validate(instance=output_json, schema=json_schema)


@pytest.mark.skip_global_cleanup
@pytest.mark.parametrize("model_name, tokenizer_mode",
                         PARAMS_MODELS_TOKENIZER_MODE)
def test_structured_output_auto_mode(
    monkeypatch: pytest.MonkeyPatch,
    unsupported_json_schema: dict[str, Any],
    model_name: str,
    tokenizer_mode: str,
):
    monkeypatch.setenv("VLLM_USE_V1", "1")

    llm = LLM(model=model_name,
              max_model_len=1024,
              guided_decoding_backend="auto",
              tokenizer_mode=tokenizer_mode)

    sampling_params = SamplingParams(
        temperature=1.0,
        max_tokens=1000,
        guided_decoding=GuidedDecodingParams(json=unsupported_json_schema))

    prompts = ("Give an example JSON object for a grade "
               "that fits this schema: "
               f"{unsupported_json_schema}")
    # This would fail with the default of "xgrammar", but in "auto"
    # we will handle fallback automatically.
    outputs = llm.generate(prompts=prompts,
                           sampling_params=sampling_params,
                           use_tqdm=True)
    # Make sure `auto` backend handling doesn't mess up sampling_params
    # and that we can reuse it without error.
    outputs.extend(
        llm.generate(prompts=prompts,
                     sampling_params=sampling_params,
                     use_tqdm=True))

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


@pytest.mark.skip_global_cleanup
def test_guidance_no_additional_properties(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("VLLM_USE_V1", "1")

    backend = 'guidance:no-additional-properties,disable-any-whitespace'
    llm = LLM(model="Qwen/Qwen2.5-1.5B-Instruct",
              max_model_len=1024,
              guided_decoding_backend=backend)

    schema = {
        'type': 'object',
        'properties': {
            'a1': {
                'type': 'string'
            },
            'a2': {
                'type': 'string'
            },
            'a3': {
                'type': 'string'
            }
        },
        'required': ['a1', 'a2', 'a3'],
    }

    prompt = (
        "<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a "
        "helpful assistant.<|im_end|>\n<|im_start|>user\nPlease generate a "
        "large JSON object with key-value pairs a1=b1, a2=b2, ..., a20=b20"
        "<|im_end|>\n<|im_start|>assistant\n")

    def generate_with_backend(backend):
        guided_params = GuidedDecodingParams(json=schema, backend=backend)
        sampling_params = SamplingParams(temperature=0,
                                         max_tokens=256,
                                         guided_decoding=guided_params)

        outputs = llm.generate(prompts=prompt, sampling_params=sampling_params)
        assert outputs is not None
        generated_text = outputs[0].outputs[0].text
        assert generated_text is not None
        parsed_json = json.loads(generated_text)
        assert isinstance(parsed_json, dict)
        jsonschema.validate(instance=parsed_json, schema=schema)
        return parsed_json

    generated = generate_with_backend(
        'guidance:no-additional-properties,disable-any-whitespace')
    assert "a1" in generated
    assert "a2" in generated
    assert "a3" in generated
    assert "a4" not in generated
    assert "a5" not in generated
    assert "a6" not in generated
