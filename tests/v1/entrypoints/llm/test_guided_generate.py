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
