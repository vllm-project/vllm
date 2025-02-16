# SPDX-License-Identifier: Apache-2.0
import pytest

from vllm import LLM, SamplingParams


@pytest.fixture
def test_prompts():
    return [
        "Can you repeat the sentence ten times, this is a sentence.",
        "Can you repeat the sentence ten times, this is a test.",
    ]


@pytest.fixture
def sampling_config():
    # Only support greedy for now
    return SamplingParams(temperature=0, max_tokens=30, ignore_eos=False)


@pytest.fixture
def model_name():
    return "meta-llama/Meta-Llama-3-8B-Instruct"


def test_ngram_correctness(monkeypatch, test_prompts, sampling_config,
                           model_name):
    '''
    Compare the outputs of a original LLM and a speculative LLM
    should be the same when using ngram speculative decoding.
    '''
    with monkeypatch.context() as m:
        m.setenv("VLLM_USE_V1", "1")

        ref_llm = LLM(model=model_name)
        ref_outputs = ref_llm.generate(test_prompts, sampling_config)
        del ref_llm

        spec_llm = LLM(model=model_name,
                       speculative_model='[ngram]',
                       ngram_prompt_lookup_max=5,
                       ngram_prompt_lookup_min=3,
                       num_speculative_tokens=3)
        spec_outputs = spec_llm.generate(test_prompts, sampling_config)
        for ref_output, spec_output in zip(ref_outputs, spec_outputs):
            assert ref_output.outputs[0].text == spec_output.outputs[0].text, \
                (f"ref_output: {ref_output.outputs[0].text},"
                 f"spec_output: {spec_output.outputs[0].text}")
        del spec_llm
