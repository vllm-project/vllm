import json
import re
import weakref

import jsonschema
import pytest

from vllm.distributed import cleanup_dist_env_and_memory
from vllm.entrypoints.llm import LLM
from vllm.outputs import RequestOutput
from vllm.sampling_params import GuidedDecodingParams, SamplingParams

MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
GUIDED_DECODING_BACKENDS = ["outlines", "lm-format-enforcer", "xgrammar"]


@pytest.fixture(scope="module")
def llm():
    # pytest caches the fixture so we use weakref.proxy to
    # enable garbage collection
    llm = LLM(model=MODEL_NAME, max_model_len=1024)

    with llm.deprecate_legacy_api():
        yield weakref.proxy(llm)
        del llm
    cleanup_dist_env_and_memory()


@pytest.mark.skip_global_cleanup
@pytest.mark.parametrize("guided_decoding_backend", GUIDED_DECODING_BACKENDS)
def test_guided_regex(sample_regex, llm, guided_decoding_backend: str):
    sampling_params = SamplingParams(temperature=0.8,
                                     top_p=0.95,
                                     guided_decoding=GuidedDecodingParams(
                                         regex=sample_regex,
                                         backend=guided_decoding_backend))
    outputs = llm.generate(prompts=[
        f"Give an example IPv4 address with this regex: {sample_regex}"
    ] * 2,
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
        assert re.fullmatch(sample_regex, generated_text) is not None
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")


@pytest.mark.skip_global_cleanup
@pytest.mark.parametrize("guided_decoding_backend", GUIDED_DECODING_BACKENDS)
def test_guided_json_completion(sample_json_schema, llm,
                                guided_decoding_backend: str):
    sampling_params = SamplingParams(temperature=1.0,
                                     max_tokens=1000,
                                     guided_decoding=GuidedDecodingParams(
                                         json=sample_json_schema,
                                         backend=guided_decoding_backend))
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
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
        output_json = json.loads(generated_text)
        jsonschema.validate(instance=output_json, schema=sample_json_schema)


@pytest.mark.skip_global_cleanup
@pytest.mark.parametrize("guided_decoding_backend", GUIDED_DECODING_BACKENDS)
def test_guided_complex_json_completion(sample_complex_json_schema, llm,
                                        guided_decoding_backend: str):
    sampling_params = SamplingParams(temperature=1.0,
                                     max_tokens=1000,
                                     guided_decoding=GuidedDecodingParams(
                                         json=sample_complex_json_schema,
                                         backend=guided_decoding_backend))
    outputs = llm.generate(prompts=[
        f"Give an example JSON for an assignment grade "
        f"that fits this schema: {sample_complex_json_schema}"
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
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
        output_json = json.loads(generated_text)
        jsonschema.validate(instance=output_json,
                            schema=sample_complex_json_schema)


@pytest.mark.skip_global_cleanup
@pytest.mark.parametrize("guided_decoding_backend", GUIDED_DECODING_BACKENDS)
def test_guided_definition_json_completion(sample_definition_json_schema, llm,
                                           guided_decoding_backend: str):
    sampling_params = SamplingParams(temperature=1.0,
                                     max_tokens=1000,
                                     guided_decoding=GuidedDecodingParams(
                                         json=sample_definition_json_schema,
                                         backend=guided_decoding_backend))
    outputs = llm.generate(prompts=[
        f"Give an example JSON for solving 8x + 7 = -23 "
        f"that fits this schema: {sample_definition_json_schema}"
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
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
        output_json = json.loads(generated_text)
        jsonschema.validate(instance=output_json,
                            schema=sample_definition_json_schema)


@pytest.mark.skip_global_cleanup
@pytest.mark.parametrize("guided_decoding_backend", GUIDED_DECODING_BACKENDS)
def test_guided_choice_completion(sample_guided_choice, llm,
                                  guided_decoding_backend: str):
    sampling_params = SamplingParams(temperature=0.8,
                                     top_p=0.95,
                                     guided_decoding=GuidedDecodingParams(
                                         choice=sample_guided_choice,
                                         backend=guided_decoding_backend))
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


@pytest.mark.skip_global_cleanup
@pytest.mark.parametrize("guided_decoding_backend", GUIDED_DECODING_BACKENDS)
def test_guided_grammar(sample_sql_statements, llm,
                        guided_decoding_backend: str):
    sampling_params = SamplingParams(temperature=0.8,
                                     top_p=0.95,
                                     max_tokens=1000,
                                     guided_decoding=GuidedDecodingParams(
                                         grammar=sample_sql_statements,
                                         backend=guided_decoding_backend))
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
        parser = Lark(sample_sql_statements)
        parser.parse(generated_text)

        # remove spaces for comparison b/c we removed them in the grammar
        ground_truth = "SELECT col_1 from table_1 where col_1 = 1".replace(
            " ", "")

        assert generated_text.strip() == ground_truth

        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")


@pytest.mark.skip_global_cleanup
def test_guided_options_request_deprecation_warning(sample_regex, llm):
    sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

    with pytest.warns(DeprecationWarning, match="guided_options_request"):
        llm.generate(prompts="This should fail",
                     sampling_params=sampling_params,
                     use_tqdm=True,
                     guided_options_request=dict(guided_regex=sample_regex))


@pytest.mark.skip_global_cleanup
def test_validation_against_both_guided_decoding_options(sample_regex, llm):
    sampling_params = SamplingParams(
        temperature=0.8,
        top_p=0.95,
        guided_decoding=GuidedDecodingParams(regex=sample_regex))

    with pytest.raises(ValueError, match="Cannot set both"):
        llm.generate(prompts="This should fail",
                     sampling_params=sampling_params,
                     use_tqdm=True,
                     guided_options_request=dict(guided_regex=sample_regex))


@pytest.mark.skip_global_cleanup
@pytest.mark.parametrize("guided_decoding_backend", GUIDED_DECODING_BACKENDS)
def test_guided_json_object(llm, guided_decoding_backend: str):
    sampling_params = SamplingParams(temperature=1.0,
                                     max_tokens=100,
                                     n=2,
                                     guided_decoding=GuidedDecodingParams(
                                         json_object=True,
                                         backend=guided_decoding_backend))

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

            # Parse to verify it is valid JSON
            parsed_json = json.loads(generated_text)
            assert isinstance(parsed_json, dict)
