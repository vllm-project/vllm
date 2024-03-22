"""Compare the outputs of a AQLM model between vLLM and HF Transformers

Run `pytest tests/models/test_aqlm.py --forked`.
"""

import pytest
import torch
from vllm.model_executor.layers.quantization import (
    _QUANTIZATION_CONFIG_REGISTRY)

capability = torch.cuda.get_device_capability()
capability = capability[0] * 10 + capability[1]
aqlm_not_supported = (
    capability < _QUANTIZATION_CONFIG_REGISTRY["aqlm"].get_min_capability())

# In this test we hardcode prompts and generations for the model so we don't
# need to require the AQLM package as a dependency
example_prompts = [
    'vLLM is a high-throughput and memory-efficient inference and serving '
    'engine for LLMs.\n',
    'Briefly describe the major milestones in the development of artificial '
    'intelligence from 1950 to 2020.\n',
    'Compare and contrast artificial intelligence with human intelligence in '
    'terms of processing information.\n',
    'Describe the basic components of a neural network and how it can be '
    'trained.\n',
    'Write a short story about a robot that dreams for the first time.\n',
    'Analyze the impact of the COVID-19 pandemic on global economic structures '
    'and future business models.\n',
    'Explain the cultural significance of the Mona Lisa painting, and how its '
    'perception might vary in Western versus Eastern societies.\n',
    "Translate the following English sentence into Japanese, French, and "
    "Swahili: 'The early bird catches the worm.'\n"
]

# These ground truth generations were generated using `transformers==4.38.1
# aqlm==1.1.0 torch==2.2.0`
# and the below code:
# ```python
# from transformers import AutoTokenizer, AutoModelForCausalLM
# model_id = "ISTA-DASLab/Llama-2-7b-AQLM-2Bit-1x16-hf"
# quantized_model = AutoModelForCausalLM.from_pretrained(model_id,
# torch_dtype="auto", device_map="cuda").cuda()
# tokenizer = AutoTokenizer.from_pretrained(model_id)
# outputs = []
# for prompt in example_prompts:
#     input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to("cuda")
#     hf_outputs = quantized_model.generate(input_ids, max_new_tokens=32)
#     outputs.append(tokenizer.decode(hf_outputs[0][input_ids.shape[1]:]))
# ```
ground_truth_generations = [
    '\n### Features\n\n- **High-throughput**: vLLM is designed to be '
    'memory-efficient and high-throughput. It',
    'The major milestones in the development of artificial intelligence from '
    '1950 to 2020 are as follows:\n1950',
    'Compare and contrast artificial intelligence with human intelligence in '
    'terms of processing information. The processing of information is a key '
    'component of artificial intelligence. The processing of information is',
    'Explain the difference between supervised and unsupervised '
    'learning.\nExplain the difference between a feedforward neural network '
    'and a recurrent neural network.\n',
    'Write a short story about a robot that dreams for the first time. The '
    'story should be about 1000 words.\nThe story should be',
    'Analyze the impact of the COVID-19 pandemic on global economic structures '
    'and future business models. The COVID-19 pandemic has had a',
    'The Mona Lisa is a painting by Leonardo da Vinci, and it is considered to '
    'be one of the most famous paintings in the world. The',
    "Translate the following English sentence into Japanese, French, and "
    "Swahili: 'The early bird catches the worm.'\nThe early bird catches"
]


@pytest.mark.skipif(aqlm_not_supported,
                    reason="AQLM is not supported on this GPU type.")
@pytest.mark.parametrize("model", ["ISTA-DASLab/Llama-2-7b-AQLM-2Bit-1x16-hf"])
@pytest.mark.parametrize("dtype", ["half"])
@pytest.mark.parametrize("max_tokens", [32])
@pytest.mark.parametrize("num_logprobs", [3])
def test_models(
    vllm_runner,
    example_prompts,
    model: str,
    dtype: str,
    max_tokens: int,
    num_logprobs: int,
) -> None:

    vllm_model = vllm_runner(model, dtype=dtype)
    vllm_outputs = vllm_model.generate_greedy_logprobs(example_prompts,
                                                       max_tokens,
                                                       num_logprobs)

    # loop through the prompts to compare against the ground truth generations
    for prompt_idx in range(len(example_prompts)):
        vllm_output_ids, vllm_output_str, vllm_logprobs = vllm_outputs[
            prompt_idx]

        assert vllm_output_str == ground_truth_generations[prompt_idx]
