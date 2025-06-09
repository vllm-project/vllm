# ruff: noqa: E501
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import json
from enum import Enum
from typing import TYPE_CHECKING, Any

import jsonschema
import pytest
import regex as re
from pydantic import BaseModel

from tests.reasoning.utils import run_reasoning_extraction
from vllm.entrypoints.llm import LLM
from vllm.outputs import RequestOutput
from vllm.platforms import current_platform
from vllm.reasoning.abs_reasoning_parsers import ReasoningParserManager
from vllm.sampling_params import GuidedDecodingParams, SamplingParams

if TYPE_CHECKING:
    from vllm.config import TokenizerMode

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
    ("mistralai/Ministral-8B-Instruct-2410", "guidance", "auto", None),
    ("mistralai/Ministral-8B-Instruct-2410", "xgrammar", "mistral", None),
    ("Qwen/Qwen2.5-1.5B-Instruct", "xgrammar", "auto", None),
    # FIXME: This test is flaky on CI thus disabled
    # ("Qwen/Qwen2.5-1.5B-Instruct", "guidance", "auto"),
    ("mistralai/Ministral-8B-Instruct-2410", "guidance", "auto",
     NGRAM_SPEC_CONFIG),
    ("Qwen/Qwen2.5-1.5B-Instruct", "xgrammar", "auto", NGRAM_SPEC_CONFIG),
    ("meta-llama/Meta-Llama-3.1-8B-Instruct", "xgrammar", "auto",
     EAGLE_SPEC_CONFIG)
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


def check_platform(speculative_config: dict[str, Any]):
    enforce_eager = True
    if current_platform.is_tpu():
        # NOTE: Don't use eager execution on TPUs because we want to test for no
        # recompilation at runtime.
        enforce_eager = False
        if speculative_config:
            pytest.skip("TPU does not support speculative decoding.")
    return enforce_eager


def get_llm_output(outputs: list[RequestOutput]):
    generated_text_list = []
    assert outputs is not None
    for output in outputs:
        assert output is not None
        assert isinstance(output, RequestOutput)
        prompt = output.prompt
        generated_text = output.outputs[0].text
        assert generated_text is not None
        print(f"Prompt: {prompt!r}")
        print(f"Generated text: {generated_text!r}")
        generated_text_list.append(generated_text)
    return generated_text_list


@pytest.mark.skip_global_cleanup
@pytest.mark.parametrize(
    "model_name, guided_decoding_backend, tokenizer_mode, speculative_config",
    PARAMS_MODELS_BACKENDS_TOKENIZER_MODE)
def test_structured_output(
    monkeypatch: pytest.MonkeyPatch,
    sample_prompts_for_structured_output: list[str],
    sample_json_schema: dict[str, Any],
    unsupported_json_schema: dict[str, Any],
    sample_sql_ebnf: str,
    sample_sql_lark: str,
    sample_regex: str,
    sample_guided_choice: str,
    guided_decoding_backend: str,
    tokenizer_mode: str,
    model_name: str,
    speculative_config: dict[str, Any],
):
    monkeypatch.setenv("VLLM_USE_V1", "1")
    enforce_eager = check_platform(speculative_config=speculative_config)

    # Use a single LLM instance for several scenarios to
    # speed up the test suite.
    llm = LLM(model=model_name,
              enforce_eager=enforce_eager,
              max_model_len=1024,
              guided_decoding_backend=guided_decoding_backend,
              guided_decoding_disable_any_whitespace=True,
              tokenizer_mode=tokenizer_mode,
              speculative_config=speculative_config)

    # ---------------------------------------------------------
    #  Test 1: Generate JSON output based on a provided schema
    # ---------------------------------------------------------
    sampling_params = SamplingParams(
        temperature=1.0,
        max_tokens=4096,
        guided_decoding=GuidedDecodingParams(json=sample_json_schema))
    prompts = f"{sample_prompts_for_structured_output[0]} {sample_json_schema}"
    outputs = llm.generate(prompts,
                           sampling_params=sampling_params,
                           use_tqdm=True)
    generated_text_list = get_llm_output(outputs)
    for generated_text in generated_text_list:
        output_json = json.loads(generated_text)
        jsonschema.validate(instance=output_json, schema=sample_json_schema)

    # -----------------------------------------------
    #  Test 2: Generate JSON object without a schema
    # -----------------------------------------------
    sampling_params = SamplingParams(
        temperature=1.0,
        max_tokens=4096,
        n=2,
        guided_decoding=GuidedDecodingParams(json_object=True))
    prompts = sample_prompts_for_structured_output[1]
    outputs = llm.generate(prompts,
                           sampling_params=sampling_params,
                           use_tqdm=True)
    generated_text_list = get_llm_output(outputs)
    for generated_text in generated_text_list:
        # Parse to verify it is a valid JSON object
        parsed_json = json.loads(generated_text)
        assert isinstance(parsed_json, dict)

    # ------------------------------------------------------
    #  Test 3: test a jsonschema incompatible with xgrammar
    # ------------------------------------------------------
    sampling_params = SamplingParams(
        temperature=1.0,
        max_tokens=4096,
        guided_decoding=GuidedDecodingParams(json=unsupported_json_schema))
    prompts = (f"{sample_prompts_for_structured_output[2]}"
               f"{unsupported_json_schema}. "
               f"Make the response as short as possible.")
    if guided_decoding_backend.startswith("xgrammar"):
        with pytest.raises(ValueError,
                           match="The provided JSON schema contains features "
                           "not supported by xgrammar."):
            llm.generate(prompts,
                         sampling_params=sampling_params,
                         use_tqdm=True)
    else:
        outputs = llm.generate(prompts,
                               sampling_params=sampling_params,
                               use_tqdm=True)
        generated_text_list = get_llm_output(outputs)
        for generated_text in generated_text_list:
            # Parse to verify it is a valid JSON object
            parsed_json = json.loads(generated_text)
            assert isinstance(parsed_json, dict)

    # ---------------------------------------------------
    #  Test 4: Generate SQL statement using EBNF grammar
    # ---------------------------------------------------
    sampling_params = SamplingParams(
        temperature=0.8,
        top_p=0.95,
        max_tokens=1000,
        guided_decoding=GuidedDecodingParams(grammar=sample_sql_ebnf))
    prompts = sample_prompts_for_structured_output[3]
    outputs = llm.generate(prompts,
                           sampling_params=sampling_params,
                           use_tqdm=True)
    generated_text_list = get_llm_output(outputs)
    for generated_text in generated_text_list:
        # Remove spaces for comparison b/c we removed them in the grammar
        ground_truth = "SELECT col_1 from table_1 where col_1 = 1".replace(
            " ", "")
        assert generated_text.strip() == ground_truth

    # ---------------------------------------------------
    #  Test 5: Generate SQL statement using Lark grammar
    # ---------------------------------------------------
    sampling_params = SamplingParams(
        temperature=0.8,
        top_p=0.95,
        max_tokens=1000,
        guided_decoding=GuidedDecodingParams(grammar=sample_sql_lark))
    prompts = sample_prompts_for_structured_output[3]
    outputs = llm.generate(prompts,
                           sampling_params=sampling_params,
                           use_tqdm=True)
    generated_text_list = get_llm_output(outputs)
    for generated_text in generated_text_list:
        # use Lark to parse the output, and make sure it's a valid parse tree
        from lark import Lark
        parser = Lark(sample_sql_lark)
        parser.parse(generated_text)
        # remove spaces for comparison b/c we removed them in the grammar
        ground_truth = "SELECT col_1 from table_1 where col_1 = 1".replace(
            " ", "")
        assert generated_text.strip() == ground_truth

    # ------------------------------------
    #  Test 6: Test invalid grammar input
    # ------------------------------------
    sampling_params = SamplingParams(
        temperature=0.8,
        top_p=0.95,
        max_tokens=1000,
        guided_decoding=GuidedDecodingParams(grammar="not a grammar"))
    prompts = sample_prompts_for_structured_output[3]
    with pytest.raises(ValueError, match="Failed to convert the grammar "):
        llm.generate(prompts, sampling_params=sampling_params, use_tqdm=True)

    # ------------------------------------------------
    #  Test 7: Generate text based on a regex pattern
    # ------------------------------------------------
    sampling_params = SamplingParams(
        temperature=0.8,
        top_p=0.95,
        guided_decoding=GuidedDecodingParams(regex=sample_regex))
    prompts = (f"{sample_prompts_for_structured_output[4]}"
               f"{sample_regex}. "
               f"Make the response as short as possible.")
    outputs = llm.generate(prompts,
                           sampling_params=sampling_params,
                           use_tqdm=True)
    generated_text_list = get_llm_output(outputs)
    for generated_text in generated_text_list:
        assert re.fullmatch(sample_regex, generated_text) is not None

    # ------------------------------------------
    #  Test 8: Generate text based on a choices
    # ------------------------------------------
    sampling_params = SamplingParams(
        temperature=0.8,
        top_p=0.95,
        guided_decoding=GuidedDecodingParams(choice=sample_guided_choice))
    prompts = sample_prompts_for_structured_output[5]
    outputs = llm.generate(prompts, use_tqdm=True)
    generated_text_list = get_llm_output(outputs)
    for generated_text in generated_text_list:
        assert generated_text in sample_guided_choice

    # ------------------------------------------------------------------------
    #  Test 9: Generate structured output using a Pydantic model with an enum
    # ------------------------------------------------------------------------
    json_schema = CarDescription.model_json_schema()
    sampling_params = SamplingParams(
        temperature=1.0,
        max_tokens=1000,
        guided_decoding=GuidedDecodingParams(json=json_schema))
    prompts = sample_prompts_for_structured_output[6]
    outputs = llm.generate(prompts,
                           sampling_params=sampling_params,
                           use_tqdm=True)
    generated_text_list = get_llm_output(outputs)
    for generated_text in generated_text_list:
        output_json = json.loads(generated_text)
        jsonschema.validate(instance=output_json, schema=json_schema)

    # -----------------------------------------------------------
    #  Test 10: Generate structured with minLength and maxLength
    # -----------------------------------------------------------
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
        "required": ["description"],
        "additionalProperties": False
    }

    sampling_params = SamplingParams(
        temperature=1.0,
        max_tokens=4096,
        guided_decoding=GuidedDecodingParams(json=json_schema))
    prompts = sample_prompts_for_structured_output[7]
    outputs = llm.generate(prompts,
                           sampling_params=sampling_params,
                           use_tqdm=True)
    generated_text_list = get_llm_output(outputs)
    for generated_text in generated_text_list:
        output_json = json.loads(generated_text)
        jsonschema.validate(instance=output_json, schema=json_schema)

    # -----------------------------------------------------------------
    #  Test 11: Generate structured output using structural_tag format
    # -----------------------------------------------------------------
    structural_tag_config = {
        "type":
        "structural_tag",
        "structures": [{
            "begin": "<function=get_weather>",
            "schema": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string"
                    }
                },
                "additionalProperties": False
            },
            "end": "</function>"
        }],
        "triggers": ["<function="]
    }

    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=4096,
        guided_decoding=GuidedDecodingParams(
            structural_tag=json.dumps(structural_tag_config)))
    prompts = sample_prompts_for_structured_output[8]
    # Change this once other backends support structural_tag
    outputs = llm.generate(prompts,
                           sampling_params=sampling_params,
                           use_tqdm=True)
    generated_text_list = get_llm_output(outputs)
    for generated_text in generated_text_list:
        # Search for function call pattern in the response
        function_call_pattern = r'<function=get_weather>(.*?)</function>'
        matches = re.findall(function_call_pattern, generated_text)
        if not matches:
            print(f"Warning: No function calls found in response: "
                  f"{generated_text!r}")
            continue
        # Take the first function call if multiple are found
        json_str = matches[0]
        try:
            json_content = json.loads(json_str)
            assert "city" in json_content
            assert isinstance(json_content["city"], str)
            print(f"Found valid function call: {generated_text!r}")
        except (json.JSONDecodeError, AssertionError) as e:
            pytest.fail("Invalid function call format: "
                        f"{generated_text!r}\nError: {str(e)}")


@pytest.mark.skip_global_cleanup
@pytest.mark.parametrize(
    "model_name, guided_decoding_backend, tokenizer_mode, reasoning_parser, speculative_config",  # noqa: E501
    [
        ("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", "xgrammar", "auto",
         "deepseek_r1", NGRAM_SPEC_CONFIG),
        ("Qwen/Qwen3-1.7B", "xgrammar", "auto", "deepseek_r1", None),
    ],
)
def test_structured_output_with_reasoning_matrices(
    monkeypatch: pytest.MonkeyPatch,
    guided_decoding_backend: str,
    tokenizer_mode: TokenizerMode,
    reasoning_parser: str,
    model_name: str,
    speculative_config: dict[str, Any] | None,
):
    monkeypatch.setenv("VLLM_USE_V1", "1")
    enforce_eager = check_platform(speculative_config=speculative_config)

    # Use a single LLM instance for several scenarios to
    # speed up the test suite.
    llm = LLM(
        model=model_name,
        enforce_eager=enforce_eager,
        max_model_len=1024,
        max_num_seqs=16,
        guided_decoding_backend=guided_decoding_backend,
        guided_decoding_disable_any_whitespace=True,
        tokenizer_mode=tokenizer_mode,
        reasoning_parser=reasoning_parser,
        speculative_config=speculative_config,
    )
    tokenizer = llm.get_tokenizer(None)
    reasoner = ReasoningParserManager.get_reasoning_parser(reasoning_parser)(
        tokenizer=tokenizer)

    reasoning_prompt = "Solve the following math problem step-by-step, then provide the final answer as JSON object with a single key 'result'. Make sure to correct your reasoning if there are any issue should it arise.\nProblem: What is 5 * 8 + 2?"  # noqa: E501
    reasoning_schema = {
        "type": "object",
        "properties": {
            "result": {
                "type": "integer"
            }
        },
        "required": ["result"],
        "additionalProperties": False
    }
    if "Qwen3" in model_name:
        reasoning_prompt += "<think>\n"

    sampling_params = SamplingParams(
        temperature=0.1,
        max_tokens=8192,
        guided_decoding=GuidedDecodingParams(json=reasoning_schema),
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
    reasoning_content, content = run_reasoning_extraction(
        reasoner, [generated_text])
    print(
        f"Prompt: {prompt!r}\nReasoning: {reasoning_content!r}\nContent: {content!r}"
    )

    assert content is not None and reasoning_content is not None
    output_json = json.loads(content)
    jsonschema.validate(instance=output_json, schema=reasoning_schema)


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

    prompts = (
        "Give an example JSON object for a grade "
        "that fits this schema: "
        f"{unsupported_json_schema}. Make the response as short as possible.")
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

    llm = LLM(model="Qwen/Qwen2.5-1.5B-Instruct",
              max_model_len=1024,
              guided_decoding_backend="guidance",
              guided_decoding_disable_any_whitespace=True,
              guided_decoding_disable_additional_properties=True)

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
        "large JSON object with key-value pairs a1=b1, a2=b2, ..., a20=b20. "
        "Make the response as short as possible."
        "<|im_end|>\n<|im_start|>assistant\n")

    def generate_with_backend(backend):
        guided_params = GuidedDecodingParams(
            json=schema,
            backend=backend,
            disable_any_whitespace=True,
            disable_additional_properties=True)
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

    generated = generate_with_backend("guidance")
    assert "a1" in generated
    assert "a2" in generated
    assert "a3" in generated
    assert "a4" not in generated
    assert "a5" not in generated
    assert "a6" not in generated
