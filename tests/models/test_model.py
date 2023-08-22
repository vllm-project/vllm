from typing import List, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM
from vllm import LLM, SamplingParams
from vllm.transformers_utils.tokenizer import get_tokenizer

TEST_PROMPTS = [
    "vLLM is a high-throughput and memory-efficient inference and serving engine for LLMs.",
    "Briefly describe the major milestones in the development of artificial intelligence from 1950 to 2020.",
    "Compare and contrast artificial intelligence with human intelligence in terms of processing information.",
    "Describe the basic components of a neural network and how it can be trained.",
    # "Write a short story about a robot that dreams for the first time.",
    # "Analyze the impact of the COVID-19 pandemic on global economic structures and future business models.",
    # "Explain the cultural significance of the Mona Lisa painting, and how its perception might vary in Western versus Eastern societies.",
    # "Translate the following English sentence into Japanese, French, and Swahili: 'The early bird catches the worm.'",
]


def run_hf(
    model_name: str,
    tokenizer_name: str,
    prompts: List[str],
    max_tokens: int,
) -> List[Tuple[List[int], str]]:
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        trust_remote_code=True,
    ).cuda()
    tokenizer = get_tokenizer(tokenizer_name, trust_remote_code=True)

    outputs = []
    for prompt in prompts:
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids
        output_ids = model.generate(
            input_ids.cuda(),
            do_sample=False,
            use_cache=True,
            max_new_tokens=max_tokens,
        )
        output_str = tokenizer.batch_decode(
            output_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        output_ids = output_ids[0].cpu().tolist()
        outputs.append((output_ids, output_str))
    torch.cuda.memory.empty_cache()
    return outputs


def run_vllm(
    model_name: str,
    tokenizer_name: str,
    prompts: List[str],
    max_tokens: int,
) -> List[Tuple[List[int], str]]:
    model = LLM(
        model=model_name,
        tokenizer=tokenizer_name,
        trust_remote_code=True,
        dtype="half",
        swap_space=0,
    )
    sampling_params = SamplingParams(temperature=0.0, max_tokens=max_tokens)
    req_outputs = model.generate(prompts, sampling_params=sampling_params)

    outputs = []
    for req_output in req_outputs:
        prompt_str = req_output.prompt
        prompt_ids = req_output.prompt_token_ids
        output_str = req_output.outputs[0].text
        output_ids = req_output.outputs[0].token_ids
        outputs.append((prompt_ids + output_ids, prompt_str + output_str))
    torch.cuda.memory.empty_cache()
    return outputs


def _test_model(
    model: str,
    tokenizer: Optional[str] = None,
    max_tokens: int = 32,
) -> None:
    if tokenizer is None:
        tokenizer = model
    hf_outputs = run_hf(model, tokenizer, TEST_PROMPTS, max_tokens)
    vllm_outputs = run_vllm(model, tokenizer, TEST_PROMPTS, max_tokens)
    for i in range(len(TEST_PROMPTS)):
        hf_output_ids, hf_output_str = hf_outputs[i]
        vllm_output_ids, vllm_output_str = vllm_outputs[i]
        assert hf_output_str == vllm_output_str, (
            f"Test{i}:\nHF: {hf_output_str!r}\nvLLM: {vllm_output_str!r}")
        assert hf_output_ids == vllm_output_ids, (
            f"Test{i}:\nHF: {hf_output_ids}\nvLLM: {vllm_output_ids}")


# def test_opt() -> None:
#     _test_model(model="facebook/opt-125m")


def test_gpt2() -> None:
    _test_model(model="gpt2")


# def test_llama() -> None:
#     _test_model(model="huggyllama/llama-7b")
