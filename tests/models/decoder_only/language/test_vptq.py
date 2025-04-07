"""Compare the outputs of a VPTQ model between vLLM and HF Transformers

Run `pytest tests/models/test_vptq.py`.
"""

import pytest

from tests.quantization.utils import is_quant_method_supported

# These ground truth generations were generated using `transformers==4.48.0
# vptq==0.0.5 torch==2.4.0`
# and the below code:
#```python
#from transformers import AutoTokenizer, AutoModelForCausalLM
#model_id = "VPTQ-community/Meta-Llama-3.1-8B-Instruct-v12-k65536-4096-woft"
#quantized_model = AutoModelForCausalLM.from_pretrained(model_id,
#torch_dtype="auto", device_map="cuda").cuda()
#tokenizer = AutoTokenizer.from_pretrained(model_id)
#outputs = []
#for prompt in example_prompts:
#    input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to("cuda")
#    hf_outputs = quantized_model.generate(input_ids,
#  max_new_tokens=16, do_sample=False)
#    outputs.append(tokenizer.decode(hf_outputs[0][input_ids.shape[1]:]))
#print(outputs)
#```

ground_truth_generations = [
    'vLLM is designed to be a high-throughput and '
    'memory-efficient inference and', '1950: The Dartmouth Summer Project, '
    'a pioneering AI project, was initiated',
    'Artificial intelligence (AI) and human intelligence are two '
    'distinct forms of intelligence that',
    'A neural network is a type of machine learning model that is composed '
    'of multiple layers',
    'The robot, named Zeta, had been programmed to perform tasks with '
    'precision and',
    'The COVID-19 pandemic has had a profound impact on global economic'
    ' structures and future',
    'The Mona Lisa painting, created by Leonardo da Vinci in the early 16th',
    'English: The early bird catches the worm.\nJapanese: 早い鳥は'
]


@pytest.mark.quant_model
@pytest.mark.skipif(not is_quant_method_supported("vptq"),
                    reason="VPTQ is not supported on this GPU type.")
@pytest.mark.parametrize(
    "model",
    ["VPTQ-community/Meta-Llama-3.1-8B-Instruct-v12-k65536-4096-woft"])
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
