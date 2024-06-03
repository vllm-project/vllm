"""
Common functions used in all benchmarking scripts
"""
import asyncio
import json
import random
from pathlib import Path
from typing import List, Tuple

from transformers import PreTrainedTokenizerBase

from vllm import LLM, SamplingParams
from vllm import __version__ as __vllm_version__
from vllm.outputs import RequestOutput
from vllm.transformers_utils.tokenizer import get_tokenizer

from ...tools.call_cmd import call_cmd
from .backend_request_func import (AsyncRequestVLLM, RequestFuncInput,
                                   RequestFuncOutput)
from .datasets_registry import SHAREGPT_DOWNLOAD_STR, SHAREGPT_PATH


def num_available_gpus() -> int:
    import torch
    return torch.cuda.device_count()


def get_benchmarking_context() -> dict:
    """
    Return the current python, pytorch and CUDA version as a dict
    """
    import sys

    import torch

    cuda_devices = [
        torch.cuda.get_device_properties(dev_idx)
        for dev_idx in range(torch.cuda.device_count())
    ]

    cuda_device_names = [cuda_device.name for cuda_device in cuda_devices]

    return {
        "vllm_version": __vllm_version__,
        "python_version": f"{sys.version}",
        "torch_version": f"{torch.__version__}",
        "torch_cuda_version": f"{torch.version.cuda}",
        "cuda_devices": f"{cuda_devices}",
        "cuda_device_names": cuda_device_names
    }


def generate_synthetic_requests(
        num_input_tokens: int, num_output_tokens: int, num_requests: int,
        tokenizer: PreTrainedTokenizerBase) -> List[Tuple[str, int, int]]:

    share_gpt_path = Path(SHAREGPT_PATH)
    if not share_gpt_path.exists():
        share_gpt_download_list = SHAREGPT_DOWNLOAD_STR.split(" ")
        call_cmd(share_gpt_download_list, stdout=None, stderr=None)
    assert share_gpt_path.exists()

    dataset = None
    with open(share_gpt_path) as f:
        dataset = json.load(f)
    assert dataset

    sampled_requests = []
    while len(sampled_requests) != num_requests:
        # get a random sample.
        convo = random.choice(dataset)

        # build prompt until we fill as many words as num_input_tokens.
        # We would be over-sampling, but that is fine as we truncate below.
        prompt = ""
        for turn in convo["conversations"]:
            prompt = prompt + " " + turn["value"]
            if len(prompt) >= num_input_tokens:
                break

        prompt_ids = tokenizer(prompt).input_ids

        if len(prompt_ids) < num_input_tokens:
            continue

        prompt_ids = prompt_ids[:num_input_tokens]
        prompt = tokenizer.decode(prompt_ids, skip_special_tokens=True)

        sampled_requests.append((prompt, num_input_tokens, num_output_tokens))

    assert len(sampled_requests) == num_requests
    return sampled_requests


def warmup_requests(tokenizer: PreTrainedTokenizerBase,
                    num_requests: int = 1000,
                    num_input_tokens: int = 128,
                    num_output_tokens: int = 1) -> List[Tuple[str, int, int]]:
    """
    Given a tokenizer, generate `num_requests` requests used for warmup
    """
    all_words = list(tokenizer.get_vocab().keys())
    # Remove special tokens like <s>, </s>, <pad> etc. from all_words
    words = list(filter(lambda word: not word.startswith('<'), all_words))
    requests = []
    for _ in range(num_requests):
        # We make up random prompts for warmups in order to avoid the effects of
        # prefix caching during actual benchmarking.
        prompt = " ".join(random.choices(words, k=num_input_tokens))
        prompt_ids = tokenizer(prompt).input_ids
        prompt_ids = prompt_ids[:num_input_tokens]
        prompt = tokenizer.decode(prompt_ids, skip_special_tokens=True)
        requests.append((prompt, num_input_tokens, num_output_tokens))
    return requests


def warmup_vllm_engine(engine: LLM,
                       model: str,
                       num_input_tokens: int = 128,
                       num_output_tokens: int = 1,
                       num_prompts: int = 1000) -> None:

    print(f"Doing warmup : {locals()}")

    tokenizer = get_tokenizer(model)
    requests = warmup_requests(tokenizer,
                               num_requests=num_prompts,
                               num_input_tokens=num_input_tokens,
                               num_output_tokens=num_output_tokens)

    # Add the requests to the engine.
    for prompt, _, output_len in requests:
        sampling_params = SamplingParams(
            n=1,
            temperature=0.0,
            top_p=1.0,
            use_beam_search=False,
            ignore_eos=True,
            max_tokens=output_len,
        )
        engine._add_request(
            prompt=prompt,
            prompt_token_ids=None,
            params=sampling_params,
        )

    engine._run_engine(use_tqdm=False)


def warmup_server(server_host: int,
                  server_port: int,
                  model: str,
                  num_input_tokens: int = 128,
                  num_output_tokens: int = 1,
                  num_prompts: int = 1000) -> None:

    print(f"Doing warmup : {locals()}")

    api_url = f"http://{server_host}:{server_port}/generate"

    async def process_requests(input_requests):
        tasks = []
        for request in input_requests:
            prompt, prompt_len, output_len = request
            request_func_input = RequestFuncInput(
                model=model,
                prompt=prompt,
                api_url=api_url,
                prompt_len=prompt_len,
                output_len=output_len,
                best_of=1,
                use_beam_search=False,
            )
            tasks.append(
                asyncio.create_task(
                    AsyncRequestVLLM.async_request_vllm(
                        request_func_input=request_func_input)))
        _ = await asyncio.gather(*tasks)

    tokenizer = get_tokenizer(model)
    requests = warmup_requests(tokenizer,
                               num_requests=num_prompts,
                               num_input_tokens=num_input_tokens,
                               num_output_tokens=num_output_tokens)
    asyncio.run(process_requests(requests))


def format_io_log(prompt: str, output_text: str, n_prompt_tokens: int,
                  n_output_tokens: int) -> str:
    return f"\n=== Prompt ({n_prompt_tokens}) ==\n{prompt}\n==== output({n_output_tokens}) ==\n{output_text}\n"  # noqa: E501


def print_request_outputs(results: List[RequestOutput]) -> None:
    for result in results:
        output = result.outputs[0]
        io_log = format_io_log(result.prompt, output.text,
                               len(result.prompt_token_ids),
                               len(output.token_ids))
        print(f"\n{io_log}")


def print_serving_request_io(inputs: List[Tuple[str, int, int]],
                             outputs: List[RequestFuncOutput]) -> None:
    """
        inputs: list of tuples of form [prompt, prompt_length, output_length],
        outputs: list of RequestFuncOutput output from benchmark_serving.py
        Format and print the inputs and outputs.
    """
    for i, o in zip(inputs, outputs):
        io_log = format_io_log(i[0], o.generated_text, i[1], i[2])
        print(f"\n{io_log}")
