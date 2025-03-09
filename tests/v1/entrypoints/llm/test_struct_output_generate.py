# SPDX-License-Identifier: Apache-2.0

import json

import jsonschema
import pytest

from vllm.entrypoints.llm import LLM
from vllm.outputs import RequestOutput
from vllm.sampling_params import GuidedDecodingParams, SamplingParams

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
GUIDED_DECODING_BACKENDS_V1 = ["xgrammar"]


@pytest.mark.skip_global_cleanup
@pytest.mark.parametrize("guided_decoding_backend",
                         GUIDED_DECODING_BACKENDS_V1)
def test_guided_json_completion(monkeypatch, sample_json_schema,
                                guided_decoding_backend: str):
    monkeypatch.setenv("VLLM_USE_V1", "1")
    llm = LLM(model=MODEL_NAME, max_model_len=1024)
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
@pytest.mark.parametrize("guided_decoding_backend",
                         GUIDED_DECODING_BACKENDS_V1)
def test_guided_json_object(monkeypatch, guided_decoding_backend: str):
    monkeypatch.setenv("VLLM_USE_V1", "1")
    llm = LLM(model=MODEL_NAME, max_model_len=1024)
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


@pytest.mark.skip_global_cleanup
@pytest.mark.parametrize("guided_decoding_backend",
                         GUIDED_DECODING_BACKENDS_V1)
def test_guided_json_unsupported_schema(monkeypatch, unsupported_json_schema,
                                        guided_decoding_backend: str):
    monkeypatch.setenv("VLLM_USE_V1", "1")
    llm = LLM(model=MODEL_NAME, max_model_len=1024)
    sampling_params = SamplingParams(temperature=1.0,
                                     max_tokens=1000,
                                     guided_decoding=GuidedDecodingParams(
                                         json=unsupported_json_schema,
                                         backend=guided_decoding_backend))
    with pytest.raises(ValueError,
                       match="The provided JSON schema contains features "
                       "not supported by xgrammar."):
        llm.generate(prompts=[
            f"Give an example JSON for an employee profile "
            f"that fits this schema: {unsupported_json_schema}"
        ] * 2,
                     sampling_params=sampling_params,
                     use_tqdm=True)


@pytest.mark.skip_global_cleanup
@pytest.mark.parametrize("guided_decoding_backend",
                         GUIDED_DECODING_BACKENDS_V1)
def test_guided_grammar_ebnf(monkeypatch, sample_sql_ebnf,
                             guided_decoding_backend: str):
    monkeypatch.setenv("VLLM_USE_V1", "1")
    llm = LLM(model=MODEL_NAME, max_model_len=1024)
    sampling_params = SamplingParams(temperature=0.8,
                                     top_p=0.95,
                                     max_tokens=1000,
                                     guided_decoding=GuidedDecodingParams(
                                         grammar=sample_sql_ebnf,
                                         backend=guided_decoding_backend))
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


@pytest.mark.skip_global_cleanup
@pytest.mark.parametrize("guided_decoding_backend",
                         GUIDED_DECODING_BACKENDS_V1)
def test_guided_grammar_lark(monkeypatch, sample_sql_lark,
                             guided_decoding_backend: str):
    monkeypatch.setenv("VLLM_USE_V1", "1")
    llm = LLM(model=MODEL_NAME, max_model_len=1024)
    sampling_params = SamplingParams(temperature=0.8,
                                     top_p=0.95,
                                     max_tokens=1000,
                                     guided_decoding=GuidedDecodingParams(
                                         grammar=sample_sql_lark,
                                         backend=guided_decoding_backend))
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


@pytest.mark.skip_global_cleanup
@pytest.mark.parametrize("guided_decoding_backend",
                         GUIDED_DECODING_BACKENDS_V1)
def test_guided_grammar_ebnf_invalid(monkeypatch,
                                     guided_decoding_backend: str):
    monkeypatch.setenv("VLLM_USE_V1", "1")
    llm = LLM(model=MODEL_NAME, max_model_len=1024)
    sampling_params = SamplingParams(temperature=0.8,
                                     top_p=0.95,
                                     max_tokens=1000,
                                     guided_decoding=GuidedDecodingParams(
                                         grammar="not a grammar",
                                         backend=guided_decoding_backend))
    with pytest.raises(ValueError,
                       match="Failed to convert the grammar "
                       "from Lark to EBNF."):
        llm.generate(
            prompts=("Generate a sql statement that selects col_1 from "
                     "table_1 where it is equal to 1"),
            sampling_params=sampling_params,
            use_tqdm=True,
        )


@pytest.mark.skip_global_cleanup
@pytest.mark.parametrize("guided_decoding_backend",
                         GUIDED_DECODING_BACKENDS_V1)
def test_guided_regex(monkeypatch, sample_regex, guided_decoding_backend: str):
    monkeypatch.setenv("VLLM_USE_V1", "1")
    llm = LLM(model=MODEL_NAME, max_model_len=1024)
    sampling_params = SamplingParams(temperature=0.8,
                                     top_p=0.95,
                                     guided_decoding=GuidedDecodingParams(
                                         regex=sample_regex,
                                         backend=guided_decoding_backend))
    with pytest.raises(ValueError,
                       match="Regex guided decoding is not supported."):
        llm.generate(prompts=[
            f"Give an example IPv4 address with this regex: {sample_regex}"
        ] * 2,
                     sampling_params=sampling_params,
                     use_tqdm=True)

    # Once regex is supported --
    #assert outputs is not None
    #for output in outputs:
    #    assert output is not None
    #    assert isinstance(output, RequestOutput)
    #    prompt = output.prompt
    #    generated_text = output.outputs[0].text
    #    print(generated_text)
    #    assert generated_text is not None
    #    assert re.fullmatch(sample_regex, generated_text) is not None
    #    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")


@pytest.mark.skip_global_cleanup
@pytest.mark.parametrize("guided_decoding_backend",
                         GUIDED_DECODING_BACKENDS_V1)
def test_guided_choice_completion(monkeypatch, sample_guided_choice,
                                  guided_decoding_backend: str):
    monkeypatch.setenv("VLLM_USE_V1", "1")
    llm = LLM(model=MODEL_NAME, max_model_len=1024)
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

