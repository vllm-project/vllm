# imports for guided decoding tests
import json
import os
import re


import jsonschema
import pytest

# downloading lora to test lora requests
from huggingface_hub import snapshot_download

from vllm.entrypoints.llm import LLM
from vllm.sampling_params import SamplingParams

from vllm.transformers_utils.tokenizer import get_tokenizer
from vllm.outputs import RequestOutput

MAX_SERVER_START_WAIT_S = 600  # wait for server to start for 60 seconds
# any model with a chat template should work here
MODEL_NAME = "HuggingFaceH4/zephyr-7b-beta"
# technically this needs Mistral-7B-v0.1 as base, but we're not testing
# generation quality here
LORA_NAME = "typeof/zephyr-7b-beta-lora"

TEST_SCHEMA = {
    "type": "object",
    "properties": {
        "name": {
            "type": "string"
        },
        "age": {
            "type": "integer"
        },
        "skills": {
            "type": "array",
            "items": {
                "type": "string",
                "maxLength": 10
            },
            "minItems": 3
        },
        "work history": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "company": {
                        "type": "string"
                    },
                    "duration": {
                        "type": "string"
                    },
                    "position": {
                        "type": "string"
                    }
                },
                "required": ["company", "position"]
            }
        }
    },
    "required": ["name", "age", "skills", "work history"]
}

TEST_REGEX = (r"((25[0-5]|(2[0-4]|1\d|[1-9]|)\d)\.){3}"
              r"(25[0-5]|(2[0-4]|1\d|[1-9]|)\d)")

TEST_CHOICE = [
    "Python", "Java", "JavaScript", "C++", "C#", "PHP", "TypeScript", "Ruby",
    "Swift", "Kotlin"
]




@pytest.fixture(scope="session")
def zephyr_lora_files():
    return snapshot_download(repo_id=LORA_NAME)


prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]


llm = LLM(model="facebook/opt-125m")

def test_simple_prompts():
    sampling_params = SamplingParams(temperature=0.8, top_p=0.95)
    outputs = llm.generate(
        prompts=prompts,
        sampling_params= sampling_params,
        use_tqdm = True,
        )


    assert outputs is not None
    for output in outputs:
        assert output is not None
        assert isinstance(output, RequestOutput)
        prompt = output.prompt
        generated_text = output.outputs[0].text
        assert generated_text is not None
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")


def test_guided_regex_():
    sampling_params = SamplingParams(temperature=0.8, top_p=0.95, extra_body=dict(guided_regex=TEST_REGEX))
    outputs = llm.generate(
        prompts=[f"Give an example IPv4 address with this regex: {TEST_REGEX}"],
        sampling_params= sampling_params,
        use_tqdm = True,
        )


    assert outputs is not None
    for output in outputs:
        assert output is not None
        assert isinstance(output, RequestOutput)
        prompt = output.prompt
        generated_text = output.outputs[0].text
        assert generated_text is not None
        assert re.fullmatch(TEST_REGEX, generated_text) is not None
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")

def test_guided_json_completion():
    sampling_params = SamplingParams(temperature=0.8, top_p=0.95, extra_body=dict(guided_json=TEST_SCHEMA))
    outputs = llm.generate(
                prompts=[f"Give an example JSON for an employee profile "
        f"that fits this schema: {TEST_SCHEMA}"],
        sampling_params= sampling_params,
        use_tqdm = True,
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
        jsonschema.validate(instance=output_json, schema=TEST_SCHEMA)
        

       

def test_guided_choice_completion():
    sampling_params = SamplingParams(temperature=0.8, top_p=0.95, extra_body=dict(guided_choice=TEST_CHOICE))
    outputs = llm.generate(
        prompts="The best language for type-safe systems programming is ",
        sampling_params= sampling_params,
        use_tqdm = True,
        )

    assert outputs is not None
    for output in outputs:
        assert output is not None
        assert isinstance(output, RequestOutput)
        prompt = output.prompt
        generated_text = output.outputs[0].text
        assert generated_text is not None 
        assert generated_text in TEST_CHOICE
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")

        

def test_guided_grammar():
    simple_sql_grammar = """
start: select_statement

select_statement: "SELECT" column "from" table "where" condition

column: "col_1" | "col_2"
table: "table_1" | "table_2"
condition: column "=" number

number: "1" | "2"
"""


    sampling_params = SamplingParams(temperature=0.8, top_p=0.95, extra_body=dict(guided_grammar=simple_sql_grammar))
    outputs = llm.generate(
        prompts=("Generate a sql state that select col_1 from "
                "table_1 where it is equals to 1"),
        sampling_params= sampling_params,
        use_tqdm = True,
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
        ground_truth = "SELECT col_1 from table_1 where col_1 = 1".replace(" ", "")

        assert generated_text.strip() == ground_truth
        
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")


    

        
        


if __name__ == "__main__":
    pytest.main([__file__])
