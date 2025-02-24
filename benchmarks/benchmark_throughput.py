# SPDX-License-Identifier: Apache-2.0
"""Benchmark offline inference throughput."""
import argparse
import dataclasses
import json
import os
import random
import time
from functools import cache
from typing import Any, Dict, List, Optional, Tuple

import torch
import uvloop
from benchmark_utils import convert_to_pytorch_benchmark_format, write_to_json
from PIL import Image
from tqdm import tqdm
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          PreTrainedTokenizerBase)

from vllm.engine.arg_utils import AsyncEngineArgs, EngineArgs
from vllm.entrypoints.openai.api_server import (
    build_async_engine_client_from_engine_args)
from vllm.inputs import TextPrompt
from vllm.lora.request import LoRARequest
from vllm.lora.utils import get_adapter_absolute_path
from vllm.multimodal import MultiModalDataDict
from vllm.sampling_params import BeamSearchParams
from vllm.transformers_utils.tokenizer import AnyTokenizer, get_lora_tokenizer
from vllm.utils import FlexibleArgumentParser, merge_async_iterators


@dataclasses.dataclass
class SampleRequest:
    """A class representing a single inference request for benchmarking.

    Attributes:
        prompt: The input text prompt for the model.
        prompt_len: The length of the prompt in tokens.
        expected_output_len: The expected length of the output in tokens.
        multi_modal_data: Optional dictionary containing multi-modal data (e.g.
            images).
        lora_request: Optional LoRARequest specifying the LoRA to use. 
    """
    prompt: str
    prompt_len: int
    expected_output_len: int
    multi_modal_data: Optional[MultiModalDataDict] = None
    lora_request: Optional[LoRARequest] = None


def _get_prompt_for_image_model(question: str, *, model: str) -> str:
    """Prepend and append special tokens around the question to form a prompt.

    Args:
        question: The input question text to wrap with special tokens
        model: The name of the model being used, to determine which special
            tokens to add

    Returns:
        The formatted prompt string with appropriate special tokens for the
            model

    Raises:
        ValueError: If an unsupported model name is provided
    """
    model = model.lower()
    if "pixtral" in model:
        return f"<s>[INST]{question}\n[IMG][/INST]"
    raise ValueError(f"Unsupported model {model}")


@cache
def lora_path_on_disk(lora_path: str) -> str:
    return get_adapter_absolute_path(lora_path)


lora_tokenizer_cache: Dict[int, AnyTokenizer] = {}


def get_random_lora_request(
        args: argparse.Namespace
) -> Tuple[LoRARequest, Optional[AnyTokenizer]]:
    global lora_tokenizer_cache
    lora_id = random.randint(1, args.max_loras)
    lora_request = LoRARequest(lora_name=str(lora_id),
                               lora_int_id=lora_id,
                               lora_path=lora_path_on_disk(args.lora_path))
    if lora_id not in lora_tokenizer_cache:
        lora_tokenizer_cache[lora_id] = get_lora_tokenizer(lora_request)
    return lora_request, lora_tokenizer_cache[lora_id]


def sample_requests(tokenizer: PreTrainedTokenizerBase,
                    args: argparse.Namespace) -> List[SampleRequest]:

    dataset_path: str = args.dataset
    num_requests: int = args.num_prompts
    fixed_output_len: Optional[int] = args.output_len
    model: str = args.model
    if fixed_output_len is not None and fixed_output_len < 4:
        raise ValueError("output_len too small")

    # Load the dataset.
    with open(dataset_path) as f:
        dataset = json.load(f)
    # Filter out the conversations with less than 2 turns.
    dataset = [data for data in dataset if len(data["conversations"]) >= 2]
    # Shuffle the dataset.
    random.shuffle(dataset)

    # Filter out sequences that are too long or too short
    filtered_dataset: List[SampleRequest] = []
    for data in tqdm(dataset,
                     total=len(filtered_dataset),
                     desc="sampling requests"):
        if len(filtered_dataset) == num_requests:
            break

        # Only keep the first two turns of each conversation.
        prompt = data["conversations"][0]["value"]
        completion = data["conversations"][1]["value"]

        multi_modal_data: Optional[MultiModalDataDict] = None
        if "image" in data:
            multi_modal_data = multi_modal_data or {}
            image_path = data["image"]
            # TODO(vllm-project/vllm/issues/9778): Support multiple images.
            assert isinstance(image_path,
                              str), "Only support single image input"
            try:
                multi_modal_data["image"] = Image.open(image_path).convert(
                    "RGB")
            except FileNotFoundError:
                # Ignore datapoint where asset is missing
                continue
            prompt = _get_prompt_for_image_model(question=prompt, model=model)

        request_tokenizer = tokenizer
        lora_request: Optional[LoRARequest] = None
        if args.enable_lora:
            lora_request, lora_tokenizer = get_random_lora_request(args)
            if lora_tokenizer:
                request_tokenizer = lora_tokenizer

        # Tokenize the prompts and completions.
        prompt_token_ids = request_tokenizer(prompt).input_ids
        completion_token_ids = request_tokenizer(completion).input_ids
        prompt_len = len(prompt_token_ids)
        output_len = len(completion_token_ids
                         ) if fixed_output_len is None else fixed_output_len
        if prompt_len < 4 or output_len < 4:
            # Prune too short sequences.
            continue
        if prompt_len > 1024 or prompt_len + output_len > 2048:
            # Prune too long sequences.
            continue
        filtered_dataset.append(
            SampleRequest(prompt=prompt,
                          prompt_len=prompt_len,
                          expected_output_len=output_len,
                          multi_modal_data=multi_modal_data,
                          lora_request=lora_request))

    return filtered_dataset


def run_vllm(
    requests: List[SampleRequest],
    n: int,
    engine_args: EngineArgs,
) -> float:
    from vllm import LLM, SamplingParams
    llm = LLM(**dataclasses.asdict(engine_args))
    assert all(
        llm.llm_engine.model_config.max_model_len >= (
            request.prompt_len + request.expected_output_len)
        for request in requests), (
            "Please ensure that max_model_len is greater than the sum of"
            " prompt_len and expected_output_len for all requests.")
    # Add the requests to the engine.
    prompts: List[TextPrompt] = []
    sampling_params: List[SamplingParams] = []
    for request in requests:
        prompts.append(
            TextPrompt(prompt=request.prompt,
                       multi_modal_data=request.multi_modal_data))
        sampling_params.append(
            SamplingParams(
                n=n,
                temperature=1.0,
                top_p=1.0,
                ignore_eos=True,
                max_tokens=request.expected_output_len,
            ))
    lora_requests: Optional[List[LoRARequest]] = None
    if engine_args.enable_lora:
        lora_requests = [request.lora_request for request in requests]

    use_beam_search = False

    if not use_beam_search:
        start = time.perf_counter()
        llm.generate(prompts,
                     sampling_params,
                     lora_request=lora_requests,
                     use_tqdm=True)
        end = time.perf_counter()
    else:
        assert lora_requests is None, "BeamSearch API does not support LoRA"
        prompts = [request.prompt for request in requests]
        # output_len should be the same for all requests.
        output_len = requests[0][2]
        for request in requests:
            assert request.expected_output_len == output_len
        start = time.perf_counter()
        llm.beam_search(
            prompts,
            BeamSearchParams(
                beam_width=n,
                max_tokens=output_len,
                ignore_eos=True,
            ))
        end = time.perf_counter()
    return end - start


async def run_vllm_async(
    requests: List[SampleRequest],
    n: int,
    engine_args: AsyncEngineArgs,
    disable_frontend_multiprocessing: bool = False,
) -> float:
    from vllm import SamplingParams

    async with build_async_engine_client_from_engine_args(
            engine_args, disable_frontend_multiprocessing) as llm:
        assert all(
            llm.model_config.max_model_len >= (request.prompt_len +
                                               request.expected_output_len)
            for request in requests), (
                "Please ensure that max_model_len is greater than the sum of"
                " prompt_len and expected_output_len for all requests.")

        # Add the requests to the engine.
        prompts: List[TextPrompt] = []
        sampling_params: List[SamplingParams] = []
        lora_requests: List[Optional[LoRARequest]] = []
        for request in requests:
            prompts.append(
                TextPrompt(prompt=request.prompt,
                           multi_modal_data=request.multi_modal_data))
            sampling_params.append(
                SamplingParams(
                    n=n,
                    temperature=1.0,
                    top_p=1.0,
                    ignore_eos=True,
                    max_tokens=request.expected_output_len,
                ))
            lora_requests.append(request.lora_request)

        generators = []
        start = time.perf_counter()
        for i, (prompt, sp,
                lr) in enumerate(zip(prompts, sampling_params, lora_requests)):
            generator = llm.generate(prompt,
                                     sp,
                                     lora_request=lr,
                                     request_id=f"test{i}")
            generators.append(generator)
        all_gens = merge_async_iterators(*generators)
        async for i, res in all_gens:
            pass
        end = time.perf_counter()
        return end - start


def run_hf(
    requests: List[SampleRequest],
    model: str,
    tokenizer: PreTrainedTokenizerBase,
    n: int,
    max_batch_size: int,
    trust_remote_code: bool,
) -> float:
    llm = AutoModelForCausalLM.from_pretrained(
        model, torch_dtype=torch.float16, trust_remote_code=trust_remote_code)
    if llm.config.model_type == "llama":
        # To enable padding in the HF backend.
        tokenizer.pad_token = tokenizer.eos_token
    llm = llm.cuda()

    pbar = tqdm(total=len(requests))
    start = time.perf_counter()
    batch: List[str] = []
    max_prompt_len = 0
    max_output_len = 0
    for i in range(len(requests)):
        prompt, prompt_len, output_len = requests[i]
        # Add the prompt to the batch.
        batch.append(prompt)
        max_prompt_len = max(max_prompt_len, prompt_len)
        max_output_len = max(max_output_len, output_len)
        if len(batch) < max_batch_size and i != len(requests) - 1:
            # Check if we can add more requests to the batch.
            _, next_prompt_len, next_output_len = requests[i + 1]
            if (max(max_prompt_len, next_prompt_len) +
                    max(max_output_len, next_output_len)) <= 2048:
                # We can add more requests to the batch.
                continue

        # Generate the sequences.
        input_ids = tokenizer(batch, return_tensors="pt",
                              padding=True).input_ids
        llm_outputs = llm.generate(
            input_ids=input_ids.cuda(),
            do_sample=True,
            num_return_sequences=n,
            temperature=1.0,
            top_p=1.0,
            use_cache=True,
            max_new_tokens=max_output_len,
        )
        # Include the decoding time.
        tokenizer.batch_decode(llm_outputs, skip_special_tokens=True)
        pbar.update(len(batch))

        # Clear the batch.
        batch = []
        max_prompt_len = 0
        max_output_len = 0
    end = time.perf_counter()
    return end - start


def run_mii(
    requests: List[SampleRequest],
    model: str,
    tensor_parallel_size: int,
    output_len: int,
) -> float:
    from mii import client, serve
    llm = serve(model, tensor_parallel=tensor_parallel_size)
    prompts = [request.prompt for request in requests]

    start = time.perf_counter()
    llm.generate(prompts, max_new_tokens=output_len)
    end = time.perf_counter()
    client = client(model)
    client.terminate_server()
    return end - start


def save_to_pytorch_benchmark_format(args: argparse.Namespace,
                                     results: Dict[str, Any]) -> None:
    pt_records = convert_to_pytorch_benchmark_format(
        args=args,
        metrics={
            "requests_per_second": [results["requests_per_second"]],
            "tokens_per_second": [results["tokens_per_second"]],
        },
        extra_info={
            k: results[k]
            for k in ["elapsed_time", "num_requests", "total_num_tokens"]
        })
    if pt_records:
        # Don't use json suffix here as we don't want CI to pick it up
        pt_file = f"{os.path.splitext(args.output_json)[0]}.pytorch.json"
        write_to_json(pt_file, pt_records)


def main(args: argparse.Namespace):
    print(args)
    random.seed(args.seed)

    # Sample the requests.
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer, trust_remote_code=args.trust_remote_code)
    if args.dataset is None:
        vocab_size = tokenizer.vocab_size
        requests = []
        for _ in range(args.num_prompts):

            request_tokenizer = tokenizer
            lora_request: Optional[LoRARequest] = None
            if args.enable_lora:
                lora_request, lora_tokenizer = get_random_lora_request(args)
                if lora_tokenizer:
                    request_tokenizer = lora_tokenizer

            # Synthesize a prompt with the given input length.
            candidate_ids = [
                random.randint(0, vocab_size - 1)
                for _ in range(args.input_len)
            ]
            # As tokenizer may add additional tokens like BOS, we need to try
            # different lengths to get the desired input length.
            for _ in range(5):  # Max attempts to correct
                candidate_prompt = request_tokenizer.decode(candidate_ids)
                tokenized_len = len(request_tokenizer.encode(candidate_prompt))

                if tokenized_len == args.input_len:
                    break

                # Adjust length based on difference
                diff = args.input_len - tokenized_len
                if diff > 0:
                    candidate_ids.extend([
                        random.randint(100, vocab_size - 100)
                        for _ in range(diff)
                    ])
                else:
                    candidate_ids = candidate_ids[:diff]
            requests.append(
                SampleRequest(prompt=candidate_prompt,
                              prompt_len=args.input_len,
                              expected_output_len=args.output_len,
                              lora_request=lora_request))
    else:
        requests = sample_requests(tokenizer, args)

    is_multi_modal = any(request.multi_modal_data is not None
                         for request in requests)
    if args.backend == "vllm":
        if args.async_engine:
            elapsed_time = uvloop.run(
                run_vllm_async(
                    requests,
                    args.n,
                    AsyncEngineArgs.from_cli_args(args),
                    args.disable_frontend_multiprocessing,
                ))
        else:
            elapsed_time = run_vllm(requests, args.n,
                                    EngineArgs.from_cli_args(args))
    elif args.backend == "hf":
        assert args.tensor_parallel_size == 1
        elapsed_time = run_hf(requests, args.model, tokenizer, args.n,
                              args.hf_max_batch_size, args.trust_remote_code)
    elif args.backend == "mii":
        elapsed_time = run_mii(requests, args.model, args.tensor_parallel_size,
                               args.output_len)
    else:
        raise ValueError(f"Unknown backend: {args.backend}")
    total_num_tokens = sum(request.prompt_len + request.expected_output_len
                           for request in requests)
    total_output_tokens = sum(request.expected_output_len
                              for request in requests)
    if is_multi_modal:
        print("\033[91mWARNING\033[0m: Multi-modal request detected. The "
              "following metrics are not accurate because image tokens are not"
              " counted. See vllm-project/vllm/issues/9778 for details.")
        # TODO(vllm-project/vllm/issues/9778): Count molti-modal token length.
    print(f"Throughput: {len(requests) / elapsed_time:.2f} requests/s, "
          f"{total_num_tokens / elapsed_time:.2f} total tokens/s, "
          f"{total_output_tokens / elapsed_time:.2f} output tokens/s")

    # Output JSON results if specified
    if args.output_json:
        results = {
            "elapsed_time": elapsed_time,
            "num_requests": len(requests),
            "total_num_tokens": total_num_tokens,
            "requests_per_second": len(requests) / elapsed_time,
            "tokens_per_second": total_num_tokens / elapsed_time,
        }
        with open(args.output_json, "w") as f:
            json.dump(results, f, indent=4)
        save_to_pytorch_benchmark_format(args, results)


if __name__ == "__main__":
    parser = FlexibleArgumentParser(description="Benchmark the throughput.")
    parser.add_argument("--backend",
                        type=str,
                        choices=["vllm", "hf", "mii"],
                        default="vllm")
    parser.add_argument("--dataset",
                        type=str,
                        default=None,
                        help="Path to the dataset. The dataset is expected to "
                        "be a json in form of List[Dict[..., conversations: "
                        "List[Dict[..., value: <prompt_or_response>]]]]")
    parser.add_argument("--input-len",
                        type=int,
                        default=None,
                        help="Input prompt length for each request")
    parser.add_argument("--output-len",
                        type=int,
                        default=None,
                        help="Output length for each request. Overrides the "
                        "output length from the dataset.")
    parser.add_argument("--n",
                        type=int,
                        default=1,
                        help="Number of generated sequences per prompt.")
    parser.add_argument("--num-prompts",
                        type=int,
                        default=1000,
                        help="Number of prompts to process.")
    parser.add_argument("--hf-max-batch-size",
                        type=int,
                        default=None,
                        help="Maximum batch size for HF backend.")
    parser.add_argument(
        '--output-json',
        type=str,
        default=None,
        help='Path to save the throughput results in JSON format.')
    parser.add_argument("--async-engine",
                        action='store_true',
                        default=False,
                        help="Use vLLM async engine rather than LLM class.")
    parser.add_argument("--disable-frontend-multiprocessing",
                        action='store_true',
                        default=False,
                        help="Disable decoupled async engine frontend.")
    # LoRA
    parser.add_argument(
        "--lora-path",
        type=str,
        default=None,
        help="Path to the lora adapters to use. This can be an absolute path, "
        "a relative path, or a Hugging Face model identifier.")

    parser = AsyncEngineArgs.add_cli_args(parser)
    args = parser.parse_args()
    if args.tokenizer is None:
        args.tokenizer = args.model
    if args.dataset is None:
        assert args.input_len is not None
        assert args.output_len is not None
    else:
        assert args.input_len is None
    if args.enable_lora:
        assert args.lora_path is not None

    if args.backend == "vllm":
        if args.hf_max_batch_size is not None:
            raise ValueError("HF max batch size is only for HF backend.")
    elif args.backend == "hf":
        if args.hf_max_batch_size is None:
            raise ValueError("HF max batch size is required for HF backend.")
        if args.quantization is not None:
            raise ValueError("Quantization is only for vLLM backend.")
        if args.enable_lora is not None:
            raise ValueError("LoRA benchmarking is only supported for vLLM"
                             " backend")
    elif args.backend == "mii":
        if args.dtype != "auto":
            raise ValueError("dtype must be auto for MII backend.")
        if args.n != 1:
            raise ValueError("n must be 1 for MII backend.")
        if args.quantization is not None:
            raise ValueError("Quantization is only for vLLM backend.")
        if args.hf_max_batch_size is not None:
            raise ValueError("HF max batch size is only for HF backend.")
        if args.tokenizer != args.model:
            raise ValueError("Tokenizer must be the same as the model for MII "
                             "backend.")
        if args.enable_lora is not None:
            raise ValueError("LoRA benchmarking is only supported for vLLM"
                             " backend")
    main(args)
