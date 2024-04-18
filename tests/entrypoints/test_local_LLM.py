# imports for guided decoding tests
import json
import os
import re

import jsonschema
import pytest


from vllm.entrypoints.llm import LLM
from vllm.sampling_params import SamplingParams

from vllm.transformers_utils.tokenizer import get_tokenizer
from vllm.outputs import RequestOutput

MODEL_NAME = "HuggingFaceH4/zephyr-7b-beta"



prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]


@pytest.fixture(scope="session")
def llm():
    return LLM(model=MODEL_NAME, max_model_len=15000)


@pytest.mark.skip_global_cleanup
def test_simple_prompts(llm):
    sampling_params = SamplingParams(temperature=0.8, top_p=0.95)
    outputs = llm.generate(
        prompts=prompts,
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


@pytest.mark.skip_global_cleanup
def test_guided_regex_(sample_regex, llm):
    sampling_params = SamplingParams(temperature=0.8,
                                     top_p=0.95,
                                     guided_options=dict(guided_regex=sample_regex))
    outputs = llm.generate(
        prompts=[
            f"Give an example IPv4 address with this regex: {sample_regex}"
        ],
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
        assert re.fullmatch(sample_regex, generated_text) is not None
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")


@pytest.mark.skip_global_cleanup
def test_guided_json_completion(sample_json_schema, llm):
    sampling_params = SamplingParams(temperature=0.8,
                                     top_p=0.95,
                                     guided_options=dict(guided_json=sample_json_schema),
                                     max_tokens=1000)
    outputs = llm.generate(
        prompts=[
            f"Give an example JSON for an employee profile "
            f"that fits this schema: {sample_json_schema}"
        ],
        sampling_params=sampling_params,
        use_tqdm=True,
    )

    assert outputs is not None
    print(outputs)
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
        guided_options=dict(guided_choice=sample_guided_choice))
    outputs = llm.generate(
        prompts="The best language for type-safe systems programming is ",
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
        assert generated_text in sample_guided_choice
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")


@pytest.mark.skip_global_cleanup
def test_guided_grammar(llm):
    simple_sql_grammar = """
start: select_statement

select_statement: "SELECT" column "from" table "where" condition

column: "col_1" | "col_2"
table: "table_1" | "table_2"
condition: column "=" number

number: "1" | "2"
"""

    sampling_params = SamplingParams(
        temperature=0.8,
        top_p=0.95,
        guided_options=dict(guided_grammar=simple_sql_grammar))
    outputs = llm.generate(
        prompts=("Generate a sql state that select col_1 from "
                 "table_1 where it is equals to 1"),
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
        parser = Lark(simple_sql_grammar)
        parser.parse(generated_text)

        # remove spaces for comparison b/c we removed them in the grammar
        ground_truth = "SELECT col_1 from table_1 where col_1 = 1".replace(
            " ", "")

        assert generated_text.strip() == ground_truth

        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")


if __name__ == "__main__":
    pytest.main([__file__])
