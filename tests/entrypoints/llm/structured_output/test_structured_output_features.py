# ruff: noqa: E501
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from typing import Any

import jsonschema
import pytest

from tests.entrypoints.llm.structured_output._structured_output import (
    NGRAM_SPEC_CONFIG,
    PARAMS_MODELS_TOKENIZER_MODE,
    SAMPLE_JSON_SCHEMA,
    UNSUPPORTED_JSON_SCHEMA,
)
from tests.reasoning.utils import run_reasoning_extraction
from vllm.config import StructuredOutputsConfig
from vllm.outputs import RequestOutput
from vllm.platforms import current_platform
from vllm.reasoning.abs_reasoning_parsers import ReasoningParserManager
from vllm.sampling_params import SamplingParams, StructuredOutputsParams


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
    vllm_runner,
):
    if current_platform.is_tpu() and speculative_config:
        pytest.skip("TPU does not support speculative decoding")

    # Use a single LLM instance for several scenarios to
    # speed up the test suite.
    with vllm_runner(
        model_name,
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
    ) as runner:
        tokenizer = runner.llm.get_tokenizer()
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
        outputs = runner.llm.generate(
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
    model_name: str,
    tokenizer_mode: str,
    vllm_runner,
):
    unsupported_json_schema = UNSUPPORTED_JSON_SCHEMA
    with vllm_runner(
        model_name,
        max_model_len=1024,
        structured_outputs_config=dict(backend="auto"),
        tokenizer_mode=tokenizer_mode,
        load_format="auto",
        config_format="auto",
    ) as runner:
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
        outputs = runner.llm.generate(
            prompts, sampling_params=sampling_params, use_tqdm=True
        )
        # Make sure `auto` backend handling doesn't mess up sampling_params
        # and that we can reuse it without error.
        outputs.extend(
            runner.llm.generate(prompts, sampling_params=sampling_params, use_tqdm=True)
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


def test_guidance_no_additional_properties(vllm_runner):
    with vllm_runner(
        "Qwen/Qwen2.5-1.5B-Instruct",
        max_model_len=1024,
        structured_outputs_config=dict(
            backend="guidance",
            disable_any_whitespace=True,
            disable_additional_properties=True,
        ),
    ) as runner:
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
                temperature=0,
                max_tokens=256,
                structured_outputs=structured_outputs_params,
            )

            outputs = runner.llm.generate(prompt, sampling_params=sampling_params)
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
    backend: str,
    vllm_runner,
):
    sample_json_schema = SAMPLE_JSON_SCHEMA
    # Don't use eager execution on TPUs because we want to test for no
    # recompilation at runtime
    enforce_eager = bool(not current_platform.is_tpu())

    with vllm_runner(
        "meta-llama/Meta-Llama-3.1-8B-Instruct",
        enforce_eager=enforce_eager,
        max_model_len=1024,
        structured_outputs_config=StructuredOutputsConfig(
            backend=backend,
            disable_any_whitespace=backend in {"xgrammar", "guidance"},
        ),
    ) as runner:
        structured_outputs_prompt = (
            "Give an example JSON for an employee profile that fits this "
            "schema. Make the response as short as possible. Schema: "
            f"{sample_json_schema}"
        )

        non_structured_outputs_prompt = "The diameter of the Earth in kilometers is "

        prompts = [structured_outputs_prompt, non_structured_outputs_prompt]
        sampling_params = [
            SamplingParams(
                temperature=0,
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

        outputs = runner.llm.generate(
            prompts=prompts, sampling_params=sampling_params, use_tqdm=True
        )

        assert outputs is not None

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
def test_structured_output_with_structural_tag(backend: str, vllm_runner):
    with vllm_runner(
        "Qwen/Qwen2.5-1.5B-Instruct",
        structured_outputs_config=StructuredOutputsConfig(backend=backend),
    ) as runner:
        structural_tag_config = {
            "type": "structural_tag",
            "format": {
                "type": "triggered_tags",
                "tags": [
                    {
                        "begin": "hello_flag",
                        "content": {"type": "any_text"},
                        "end": "hello",
                    }
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

        prompt = "Hello and repeat hello 10 times, do not say anything else. Only say hello hello hello, now start"
        outputs = runner.llm.generate(
            prompt, sampling_params=sampling_params, use_tqdm=True
        )
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
