import json
import re
import weakref
from typing import List

import jsonschema
import pytest

from vllm.entrypoints.llm import LLM
from vllm.outputs import RequestOutput
from vllm.sampling_params import SamplingParams

from ..conftest import cleanup

MODEL_NAME = "HuggingFaceH4/zephyr-7b-beta"

PROMPTS = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]
TOKEN_IDS = [
    [0],
    [0, 1],
    [0, 2, 1],
    [0, 3, 1, 2],
]

pytestmark = pytest.mark.llm


@pytest.fixture(scope="module")
def llm():
    # pytest caches the fixture so we use weakref.proxy to
    # enable garbage collection
    llm = LLM(model=MODEL_NAME, max_model_len=1024)

    with llm.deprecate_legacy_api():
        yield weakref.proxy(llm)

        del llm

    cleanup()


def assert_outputs_equal(o1: List[RequestOutput], o2: List[RequestOutput]):
    assert [o.outputs for o in o1] == [o.outputs for o in o2]


# @pytest.mark.skip_global_cleanup
# @pytest.mark.parametrize('prompt', PROMPTS)
# def test_v1_v2_api_consistency_single_prompt_string(llm: LLM, prompt):
#     sampling_params = SamplingParams(temperature=0.0, top_p=1.0)

#     with pytest.warns(DeprecationWarning, match="'prompts'"):
#         v1_output = llm.generate(prompts=prompt,
#                                  sampling_params=sampling_params)

#     v2_output = llm.generate(prompt, sampling_params=sampling_params)
#     assert_outputs_equal(v1_output, v2_output)

#     v2_output = llm.generate({"prompt": prompt},
#                              sampling_params=sampling_params)
#     assert_outputs_equal(v1_output, v2_output)


# @pytest.mark.skip_global_cleanup
# @pytest.mark.parametrize('prompt_token_ids', TOKEN_IDS)
# def test_v1_v2_api_consistency_single_prompt_tokens(llm: LLM,
#                                                     prompt_token_ids):
#     sampling_params = SamplingParams(temperature=0.0, top_p=1.0)

#     with pytest.warns(DeprecationWarning, match="'prompt_token_ids'"):
#         v1_output = llm.generate(prompt_token_ids=prompt_token_ids,
#                                  sampling_params=sampling_params)

#     v2_output = llm.generate({"prompt_token_ids": prompt_token_ids},
#                              sampling_params=sampling_params)
#     assert_outputs_equal(v1_output, v2_output)


# @pytest.mark.skip_global_cleanup
# def test_v1_v2_api_consistency_multi_prompt_string(llm: LLM):
#     sampling_params = SamplingParams(temperature=0.0, top_p=1.0)

#     with pytest.warns(DeprecationWarning, match="'prompts'"):
#         v1_output = llm.generate(prompts=PROMPTS,
#                                  sampling_params=sampling_params)

#     v2_output = llm.generate(PROMPTS, sampling_params=sampling_params)
#     assert_outputs_equal(v1_output, v2_output)

#     v2_output = llm.generate(
#         [{
#             "prompt": p
#         } for p in PROMPTS],
#         sampling_params=sampling_params,
#     )
#     assert_outputs_equal(v1_output, v2_output)


# @pytest.mark.skip_global_cleanup
# def test_v1_v2_api_consistency_multi_prompt_tokens(llm: LLM):
#     sampling_params = SamplingParams(temperature=0.0, top_p=1.0)

#     with pytest.warns(DeprecationWarning, match="'prompt_token_ids'"):
#         v1_output = llm.generate(prompt_token_ids=TOKEN_IDS,
#                                  sampling_params=sampling_params)

#     v2_output = llm.generate(
#         [{
#             "prompt_token_ids": p
#         } for p in TOKEN_IDS],
#         sampling_params=sampling_params,
#     )
#     assert_outputs_equal(v1_output, v2_output)


@pytest.mark.skip_global_cleanup
def test_multiple_sampling_params(llm: LLM):
    prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
    ]

    sampling_params = [
        SamplingParams(temperature=0.01, top_p=0.95),
        SamplingParams(temperature=0.3, top_p=0.95),
        SamplingParams(temperature=0.7, top_p=0.95),
        SamplingParams(temperature=0.99, top_p=0.95),
    ]

    # Multiple SamplingParams should be matched with each prompt
    outputs = llm.generate(PROMPTS, sampling_params=sampling_params)
    assert len(PROMPTS) == len(outputs)

    # Exception raised, if the size of params does not match the size of prompts
    with pytest.raises(ValueError):
        outputs = llm.generate(PROMPTS, sampling_params=sampling_params[:3])

    # Single SamplingParams should be applied to every prompt
    single_sampling_params = SamplingParams(temperature=0.3, top_p=0.95)
    outputs = llm.generate(PROMPTS, sampling_params=single_sampling_params)
    assert len(PROMPTS) == len(outputs)

    # sampling_params is None, default params should be applied
    outputs = llm.generate(prompts, sampling_params=None)
    assert len(prompts) == len(outputs)


@pytest.mark.skip_global_cleanup
def test_guided_regex(sample_regex, llm):
    sampling_params = SamplingParams(
        temperature=0.8,
        top_p=0.95,
        )
    outputs = llm.generate(
        prompts=[
            f"Give an example IPv4 address with this regex: {sample_regex}"
        ] * 2,
        sampling_params=sampling_params,
        use_tqdm=True,
        guided_options=dict(guided_regex=sample_regex)
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


@pytest.mark.skip_global_cleanup
def test_guided_json_completion(sample_json_schema, llm):
    sampling_params = SamplingParams(
        temperature=1.0,
        max_tokens=1000,
        )
    outputs = llm.generate(
        prompts=[
            f"Give an example JSON for an employee profile "
            f"that fits this schema: {sample_json_schema}"
        ] * 2,
        sampling_params=sampling_params,
        use_tqdm=True,
        guided_options=dict(guided_json=sample_json_schema)
    )

    assert outputs is not None

    for output in outputs:
        assert output is not None
        assert isinstance(output, RequestOutput)
        prompt = output.prompt

        generated_text = output.outputs[0].text
        assert generated_text is not None
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
        output_json = json.loads(generated_text)
        jsonschema.validate(instance=output_json, schema=sample_json_schema)


@pytest.mark.skip_global_cleanup
def test_guided_choice_completion(sample_guided_choice, llm):
    sampling_params = SamplingParams(
        temperature=0.8,
        top_p=0.95,
        )
    outputs = llm.generate(
        prompts="The best language for type-safe systems programming is ",
        sampling_params=sampling_params,
        use_tqdm=True,
        guided_options=dict(guided_choice=sample_guided_choice)
    )

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


@pytest.mark.skip_global_cleanup
def test_guided_grammar(sample_sql_statements, llm):

    sampling_params = SamplingParams(
        temperature=0.8,
        top_p=0.95,
        )
    outputs = llm.generate(
        prompts=("Generate a sql state that select col_1 from "
                 "table_1 where it is equals to 1"),
        sampling_params=sampling_params,
        use_tqdm=True,
        guided_options=dict(guided_grammar=sample_sql_statements)
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
        parser = Lark(sample_sql_statements)
        parser.parse(generated_text)

        # remove spaces for comparison b/c we removed them in the grammar
        ground_truth = "SELECT col_1 from table_1 where col_1 = 1".replace(
            " ", "")

        assert generated_text.strip() == ground_truth

        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
