"""
Common functions used in all benchmarking scripts
"""
import json
import random
import asyncio
from typing import List, Tuple, Optional
from pathlib import Path
from transformers import PreTrainedTokenizerBase

from vllm import LLM, SamplingParams
from vllm.outputs import RequestOutput
from vllm.transformers_utils.tokenizer import get_tokenizer
from neuralmagic.tools.call_cmd import call_cmd
from neuralmagic.benchmarks.datasets_registry import SHAREGPT_PATH, SHAREGPT_DOWNLOAD_STR
from neuralmagic.benchmarks.scripts.backend_request_func import RequestFuncInput, async_request_vllm


def num_available_gpus() -> int:
    import torch
    return torch.cuda.device_count()


def get_benchmarking_context() -> dict:
    """
    Return the current python version, pytorch version and CUDA version as a dict
    """
    import sys
    import torch

    cuda_devices = [
        torch.cuda.get_device_properties(dev_idx)
        for dev_idx in range(torch.cuda.device_count())
    ]

    cuda_device_names = [cuda_device.name for cuda_device in cuda_devices]

    return {
        "python_version": f"{sys.version}",
        "torch_version": f"{torch.__version__}",
        "torch_cuda_version": f"{torch.version.cuda}",
        "cuda_devices": f"{cuda_devices}",
        "cuda_device_names": f"{cuda_device_names}"
    }


def remove_special_tokens_and_decode(
        prompt_ids: list[int], tokenizer: PreTrainedTokenizerBase) -> str:
    # Remove special tokens from prompt ids
    prompt_ids = list(
        filter(lambda id: id not in tokenizer.all_special_ids, prompt_ids))
    return tokenizer.decode(prompt_ids)


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
        prompt = remove_special_tokens_and_decode(prompt_ids, tokenizer)

        sampled_requests.append((prompt, num_input_tokens, num_output_tokens))

    assert len(sampled_requests) == num_requests
    return sampled_requests


def warmup_requests(tokenizer: PreTrainedTokenizerBase,
                    num_requests: int = 1000,
                    num_input_tokens: int = 128,
                    num_output_tokens: int = 1) -> List[Tuple[str, int, int]]:
    """
    Given a tokenizer, generate `num_requests` requests that would be used for vllm engine warmup 
    """
    words = list(tokenizer.get_vocab().keys())
    requests = []
    for _ in range(num_requests):
        # We make up random prompts for warmups in order to avoid the effects of
        # prefix caching during actual benchmarking.
        prompt = " ".join(random.choices(words, k=num_input_tokens))
        prompt_ids = tokenizer(prompt).input_ids
        prompt_ids = prompt_ids[:num_input_tokens]
        prompt = remove_special_tokens_and_decode(prompt_ids, tokenizer)
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
            sampling_params=sampling_params,
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
                    async_request_vllm(request_func_input=request_func_input)))
        _ = await asyncio.gather(*tasks)

    tokenizer = get_tokenizer(model)
    requests = warmup_requests(tokenizer,
                               num_requests=num_prompts,
                               num_input_tokens=num_input_tokens,
                               num_output_tokens=num_output_tokens)
    asyncio.run(process_requests(requests))


def instantiate_benchmark_results_dict(benchmarking_script_name: str,
                                       tensor_parallel_size: int, model: str,
                                       tokenizer: Optional[str],
                                       dataset: Optional[str]) -> dict:
    """
    instantiate_benchmark_results_dict populates an empty dict with all the must-have
    key-value pairs. These are the key-value pairs that the scripts that process
    the benchmark results rely on.
    """
    result_dict = {}
    result_dict['script_name'] = benchmarking_script_name
    result_dict['benchmarking_context'] = get_benchmarking_context()
    result_dict['tensor_parallel_size'] = tensor_parallel_size
    result_dict['model'] = model
    result_dict['tokenizer'] = tokenizer if tokenizer is not None else model
    result_dict['dataset'] = dataset if dataset is not None else "synthetic"

    return result_dict


def print_benchmark_io(results: List[RequestOutput]) -> None:
    for result in results:
        output = result.outputs[0]
        print(
            f"\n\n inputs({len(result.prompt_token_ids)}): {result.prompt}\n output({len(output.token_ids)}): {output.text}"
        )
