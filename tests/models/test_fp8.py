"""Compares the outputs of gptq vs gptq_marlin 
"""
import os

import pytest
import torch
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.model_executor.layers.quantization import QUANTIZATION_METHODS

os.environ["TOKENIZERS_PARALLELISM"] = "true"

MAX_MODEL_LEN = 1024

MODELS = [
    "nm-testing/mistral-fp8-static",
    "nm-testing/mistral-fp8-dynamic",
    "mistralai/Mistral-7B-Instruct-v0.2",
]

EXPECTED_STRS_MAP = {
    "nm-testing/mistral-fp8-static" : [' VLLM (Vulcan Learning Machine) is a high-performance and memory-efficient', ' 1. 1950s: The Concept of AI is Born: The term', ' Artificial Intelligence (AI) and Human Intelligence (HI) are two distinct ways of processing information.', " A neural network is a type of artificial intelligence model inspired by the human brain's structure and function", ' In the heart of a sprawling industrial city, nestled among the hum of machinery and the rhythm', ' The COVID-19 pandemic has had a profound impact on global economic structures and has forced businesses to', ' The Mona Lisa painting, created by the Italian artist Leonardo da Vinci between 15', ' Japanese: 早く起きる'], # noqa: E501
    "nm-testing/mistral-fp8-dynamic": [' VLLM (Vulcan Learning Machine) is a high-performance and memory-efficient', ' 1. 1950s: The Concept of AI is Born: The term', ' Artificial Intelligence (AI) and Human Intelligence (HI) are two distinct ways of processing information.', " A neural network is a type of artificial intelligence model inspired by the human brain's structure and function", ' In the heart of the bustling city of Neo-Tokyo, nestled among the tow', ' The COVID-19 pandemic has had a profound impact on global economic structures and has forced businesses to', ' The Mona Lisa painting, created by the Italian artist Leonardo da Vinci between 15', ' Japanese: 早く起きる鳥は虫を取る (S'], # noqa: E501
    "mistralai/Mistral-7B-Instruct-v0.2": [' VLLM (Vulcan Learning Machine) is a high-performance and memory-efficient', ' 1. 1950s: The Concept of AI is Born: The term', ' Artificial Intelligence (AI) and Human Intelligence (HI) are two distinct ways of processing information.', " A neural network is a type of machine learning model inspired by the human brain's structure and function", ' In the heart of the bustling city of Neo-Tokyo, nestled among the tow', ' The COVID-19 pandemic has had a profound impact on global economic structures and has forced businesses to', ' The Mona Lisa painting, created by the Italian artist Leonardo da Vinci between 15', ' Japanese: 早く起きる鳥は虫を取る (S'], # noqa: E501
}

capability = torch.cuda.get_device_capability()
capability = capability[0] * 10 + capability[1]
fp8_not_supported = (
    capability < QUANTIZATION_METHODS["fp8"].get_min_capability())

@pytest.mark.skipif(fp8_not_supported,
                    reason="fp8 is not supported on this GPU type.")
@pytest.mark.parametrize("model_name", MODELS)
def test_models(
    example_prompts,
    model_name,
) -> None:
    model = LLM(
        model=model_name,
        max_model_len=MAX_MODEL_LEN,
        enforce_eager=True,
        quantization="fp8")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    formatted_prompts = [
        tokenizer.apply_chat_template(
            [{ "role": "user", "content": prompt }],
            tokenize=False, add_generation_prompt=True
        ) for prompt in example_prompts ]        

    params = SamplingParams(max_tokens=20, temperature=0)
    generations = []
    # Note: these need to be run 1 at a time due to numerical precision,
    # since the expected strs were generated this way.
    for prompt in formatted_prompts:
        outputs = model.generate(prompt, params)
        generations.append(outputs[0].outputs[0].text)
    del model
    
    expected_strs = EXPECTED_STRS_MAP[model_name]
    for i in range(len(example_prompts)):
        generated_str = generations[i]
        expected_str = expected_strs[i]
        assert expected_str == generated_str, (
            f"Test{i}:\nExpected: {expected_str!r}\nvLLM: {generated_str!r}")