# SPDX-License-Identifier: Apache-2.0
import pytest

from tests.quantization.utils import is_quant_method_supported

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
# print(outputs)
# ```
ground_truth_generations = [
    '\n### Features\n\n- **High-throughput**: v',
    'The major milestones in the development of artificial intelligence from '
    '195',
    'Compare and contrast artificial intelligence with human intelligence in '
    'terms of processing information. The',
    'Explain the difference between supervised and unsupervised learning.'
    '\nExplain',
    'Write a short story about a robot that dreams for the first time. The',
    'Analyze the impact of the COVID-19 pandemic on global economic',
    'The Mona Lisa is a painting by Leonardo da Vinci, and it',
    'The early bird catches the worm.\nThe early bird catches the'
]


@pytest.mark.skipif(not is_quant_method_supported("aqlm"),
                    reason="AQLM is not supported on this GPU type.")
@pytest.mark.parametrize("model", ["ISTA-DASLab/Llama-2-7b-AQLM-2Bit-1x16-hf"])
@pytest.mark.parametrize("dtype", ["half"])
@pytest.mark.parametrize("max_tokens", [16])
@pytest.mark.parametrize("num_logprobs", [1])
def test_models(
    vllm_runner,
    example_prompts,
    model: str,
    dtype: str,
    max_tokens: int,
    num_logprobs: int,
) -> None:

    with vllm_runner(model, dtype=dtype) as vllm_model:
        vllm_outputs = vllm_model.generate_greedy_logprobs(
            example_prompts, max_tokens, num_logprobs)

    # loop through the prompts to compare against the ground truth generations
    for prompt_idx in range(len(example_prompts)):
        vllm_output_ids, vllm_output_str, vllm_logprobs = vllm_outputs[
            prompt_idx]

        print("Prompt:          ", repr(example_prompts[prompt_idx]))
        print("Reference output:", repr(ground_truth_generations[prompt_idx]))
        print("Output output:   ", repr(vllm_output_str))
        assert vllm_output_str == ground_truth_generations[prompt_idx]
