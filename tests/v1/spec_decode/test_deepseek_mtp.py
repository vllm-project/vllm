# SPDX-License-Identifier: Apache-2.0
import pytest

from vllm import LLM, SamplingParams
import torch._dynamo

#torch._dynamo.config.suppress_errors = True


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
    #return "meta-llama/Meta-Llama-3-8B-Instruct"
    return "luccafong/deepseek_mtp_main_random"

@pytest.fixture
def speculative_model_name():
    pass
    #return "abhigoyal/vllm-eagle-llama-68m-random"
    #return "luccafong/deepseek_mtp_main_random"


def test_ngram_correctness(monkeypatch, test_prompts, sampling_config,
                           model_name):
    '''
    Compare the outputs of a original LLM and a speculative LLM
    should be the same when using ngram speculative decoding.
    '''
    with monkeypatch.context() as m:
        m.setenv("VLLM_USE_V1", "1")

        #ref_llm = LLM(model=model_name, trust_remote_code=True)
        #ref_outputs = ref_llm.generate(test_prompts, sampling_config)
        #del ref_llm

        spec_llm = LLM(model=model_name,
                       max_num_batched_tokens=64,
                       max_num_seqs=2,
                       #speculative_model="abhigoyal/vllm-eagle-llama-68m-random",
                       #speculative_model='[ngram]',
                       trust_remote_code=True,
                       num_speculative_tokens=1)
        spec_outputs = spec_llm.generate(test_prompts, sampling_config)
        for ref_output, spec_output in zip(ref_outputs, spec_outputs):
            assert ref_output.outputs[0].text == spec_output.outputs[0].text, \
                (f"ref_output: {ref_output.outputs[0].text},"
                 f"spec_output: {spec_output.outputs[0].text}")
        del spec_llm
