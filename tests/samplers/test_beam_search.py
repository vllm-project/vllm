import pytest
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import numpy as np

from vllm import LLM, SamplingParams

# FIXME(zhuohan): The test can not pass if we:
#   1. Increase max_new_tokens to 256.
#   2. Increase beam_width to 8.
#   3. Use the model "huggyllama/llama-7b".
MAX_NEW_TOKENS = [128]
BEAM_WIDTHS = [4]
MODELS = ["facebook/opt-125m"]
PROMPTS = [
    "Hello, my name is",
    "A robot may not injure a human being",
    "To be or not to be,",
    "What is the meaning of life?",
    "It is only with the heart that one can see rightly",
    "The quick brown fox jumps over the lazy dog",
]


def decode_hf(model_name, prompts, beam_width=1, max_new_tokens=16):
    model = AutoModelForCausalLM.from_pretrained(model_name,
                                                 use_safetensors=False)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    token_ids = []
    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt")
        all_outputs = model.generate(inputs.input_ids,
                                     do_sample=False,
                                     num_beams=beam_width,
                                     max_new_tokens=max_new_tokens,
                                     num_return_sequences=beam_width,
                                     output_scores=True,
                                     return_dict_in_generate=True)
        all_generate_ids = all_outputs.sequences
        all_generate_ids = all_generate_ids.tolist()
        all_generate_ids = [[
            x for x in generate_ids if x != tokenizer.pad_token_id
        ] for generate_ids in all_generate_ids]

        token_ids.append(all_generate_ids)

    return token_ids


def decode_vllm(model_name, prompts, beam_width=1, max_new_tokens=16):
    llm = LLM(model=model_name, tokenizer=model_name)
    sampling_params = SamplingParams(n=beam_width,
                                     use_beam_search=True,
                                     temperature=0.0,
                                     max_tokens=max_new_tokens)
    all_results = llm.generate(prompts, sampling_params)
    token_ids = []
    for result in all_results:
        request_token_ids = []
        prompt_token_ids = result.prompt_token_ids
        for output in result.outputs:
            output_token_ids = output.token_ids
            request_token_ids.append(prompt_token_ids + output_token_ids)
        token_ids.append(request_token_ids)
    return token_ids


@pytest.mark.parametrize("model_name", MODELS)
@pytest.mark.parametrize("max_new_tokens", MAX_NEW_TOKENS)
@pytest.mark.parametrize("beam_width", BEAM_WIDTHS)
@pytest.mark.parametrize("prompt", PROMPTS)
def test_beam_search_single_input(model_name, max_new_tokens, beam_width,
                                  prompt):
    hf_token_ids = decode_hf(model_name, [prompt], beam_width, max_new_tokens)
    vllm_token_ids = decode_vllm(model_name, prompt, beam_width,
                                 max_new_tokens)
    assert hf_token_ids == vllm_token_ids
