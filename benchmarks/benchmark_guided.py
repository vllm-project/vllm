"""Benchmark guided decoding throughput."""
import argparse
import dataclasses
import json
import random
import time
from typing import List

import datasets
import uvloop
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from vllm.engine.arg_utils import AsyncEngineArgs, EngineArgs
from vllm.entrypoints.openai.api_server import (
    build_async_engine_client_from_engine_args)
from vllm.sampling_params import GuidedDecodingParams
from vllm.utils import FlexibleArgumentParser, merge_async_iterators

SCHEMA = {
    "$schema":
    "https://json-schema.org/draft/2020-12/schema",
    "title":
    "User Profile",
    "type":
    "object",
    "properties": {
        "userId": {
            "type": "string",
            "description": "Unique identifier for the user."
        },
        "personalInfo": {
            "type": "object",
            "properties": {
                "firstName": {
                    "type": "string",
                    "description": "The user's first name."
                },
                "lastName": {
                    "type": "string",
                    "description": "The user's last name."
                },
                "age": {
                    "type": "integer",
                    "minimum": 0,
                    "description": "The user's age."
                },
                "phoneNumbers": {
                    "type":
                    "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "type": {
                                "type": "string",
                                "enum": ["home", "work", "mobile"],
                                "description": "Type of phone number."
                            },
                            "number": {
                                "type": "string",
                                "pattern": "^\\+?[1-9]\\d{1,14}$",
                                "description": "Phone number in E.164 format."
                            }
                        },
                        "required": ["type", "number"]
                    },
                    "description":
                    "List of phone numbers associated with the user."
                }
            },
            "required": ["firstName", "lastName"]
        },
        "address": {
            "type": "object",
            "properties": {
                "street": {
                    "type": "string",
                    "description": "Street address."
                },
                "city": {
                    "type": "string",
                    "description": "City name."
                },
                "state": {
                    "type": "string",
                    "description": "State or province."
                },
                "postalCode": {
                    "type": "string",
                    "pattern": "^\\d{5}(-\\d{4})?$",
                    "description": "Postal code."
                },
                "country": {
                    "type": "string",
                    "description": "Country name."
                }
            },
            "required": ["street", "city", "state", "postalCode", "country"]
        },
        "preferences": {
            "type": "object",
            "properties": {
                "newsletterSubscribed": {
                    "type":
                    "boolean",
                    "description":
                    "Indicates if the user is subscribed to the newsletter."
                },
                "favoriteCategories": {
                    "type": "array",
                    "items": {
                        "type": "string"
                    },
                    "description": "List of user's favorite categories."
                }
            },
            "required": ["newsletterSubscribed"]
        },
        "accountStatus": {
            "type": "string",
            "enum": ["active", "inactive", "suspended"],
            "description": "Current status of the user's account."
        },
        "registrationDate": {
            "type": "string",
            "format": "date-time",
            "description": "ISO 8601 formatted date-time of user registration."
        }
    },
    "required":
    ["userId", "personalInfo", "address", "accountStatus", "registrationDate"]
}


@dataclasses.dataclass
class SampleRequest:
    """A class representing a single inference request for benchmarking.

    Attributes:
        prompt: The input text prompt for the model.
        multi_modal_data: Optional dictionary containing multi-modal data (e.g.
            images).
        prompt_len: The length of the prompt in tokens.
        expected_output_len: The expected length of the output in tokens.
    """
    prompt: str
    prompt_len: int
    expected_output_len: int
    schema: dict
    completion: str = None


def run_vllm(requests: List[SampleRequest],
             engine_args: EngineArgs,
             n: int,
             guided_decoding: bool = False,
             warmup: bool = False,
             result_file_name: str = None) -> float:
    from vllm import LLM, SamplingParams
    llm = LLM(**vars(engine_args))

    # Add the requests to the engine.
    prompts: List[str] = []
    sampling_params: List[SamplingParams] = []
    for request in requests:
        prompts.append(request.prompt)
        sampling_params.append(
            SamplingParams(
                n=n,
                temperature=1.0,
                top_p=1.0,
                ignore_eos=True,
                max_tokens=request.expected_output_len,
                guided_decoding=GuidedDecodingParams(
                    json=request.schema) if guided_decoding else None,
            ))

    start = time.perf_counter()
    outputs = llm.generate(prompts, sampling_params, use_tqdm=True)
    ret = []
    for output, request in zip(outputs, requests):
        generated_text = output.outputs[0].text
        ret.append({
            "generated": generated_text,
            "expected": request.completion
        })
    end = time.perf_counter()
    # save ret list into a json
    if result_file_name:
        with open(result_file_name, 'w') as f:
            json.dump(ret, f, indent=4)
            f.write("\n")
    return end - start


async def run_vllm_async(requests: List[SampleRequest],
                         engine_args: AsyncEngineArgs,
                         n: int,
                         guided_decoding: bool = False,
                         warmup: bool = False,
                         disable_frontend_multiprocessing: bool = False,
                         result_file_name: str = None) -> float:
    from vllm import SamplingParams

    async with build_async_engine_client_from_engine_args(
            engine_args, disable_frontend_multiprocessing) as llm:

        # Add the requests to the engine.
        prompts: List[str] = []
        sampling_params: List[SamplingParams] = []
        if warmup:
            print("Running warmup prompt, for the first 5")
            # We setup the first 5 requests to warmup FSM
            warmup_requests = requests[:5]
            requests = requests[5:]
            for request in warmup_requests:
                prompts.append(request.prompt)
                sampling_params.append(
                    SamplingParams(
                        n=n,
                        temperature=1.0,
                        top_p=1.0,
                        ignore_eos=True,
                        max_tokens=request.expected_output_len,
                        guided_decoding=GuidedDecodingParams(
                            json=request.schema) if guided_decoding else None,
                    ))
            generators = []
            for i, (prompt, sp) in enumerate(zip(prompts, sampling_params)):
                generator = llm.generate(prompt, sp, request_id=f"test{i}")
                generators.append(generator)
            all_gens = merge_async_iterators(*generators)
            async for i, res in all_gens:
                pass

        prompts = []
        sampling_params = []
        for request in requests:
            prompts.append(request.prompt)
            sampling_params.append(
                SamplingParams(
                    n=n,
                    temperature=1.0,
                    top_p=1.0,
                    ignore_eos=True,
                    max_tokens=request.expected_output_len,
                    guided_decoding=GuidedDecodingParams(
                        json=request.schema) if guided_decoding else None,
                ))

        generators = []
        start = time.perf_counter()
        for i, (prompt, sp) in enumerate(zip(prompts, sampling_params)):
            generator = llm.generate(prompt, sp, request_id=f"test{i}")
            generators.append(generator)
        all_gens = merge_async_iterators(*generators)
        ret = []
        async for i, res in all_gens:
            generated_text = res.outputs[0].text
            ret.append(generated_text)
        end = time.perf_counter()
        if result_file_name:
            with open(result_file_name, 'w') as f:
                json.dump(ret, f, indent=4)
                f.write("\n")
        return end - start


def sample_requests(tokenizer: PreTrainedTokenizerBase,
                    args: argparse.Namespace) -> List[SampleRequest]:
    if args.dataset == 'single_schema':
        prompt = f"Generate an example of a user profile given the following schema: {json.dumps(SCHEMA)}"  # noqa: E501
        input_len = len(tokenizer(prompt).input_ids)
        print(f"Input length of the prompt: {input_len} tokens")
        requests = [
            SampleRequest(prompt=prompt,
                          prompt_len=input_len,
                          expected_output_len=args.output_len,
                          schema=SCHEMA) for _ in range(args.num_prompts)
        ]

    elif args.dataset == "xgrammar_bench":
        requests: List[SampleRequest] = []
        dataset = datasets.load_dataset("NousResearch/json-mode-eval",
                                        split="train")
        print(f"dataset has {len(dataset)} entries")
        len_dataset = len(dataset)
        for data_point_idx in range(args.num_prompts):
            idx = data_point_idx
            while idx >= len_dataset:
                idx -= len_dataset
            schema = dataset["schema"][idx]
            prompt = tokenizer.apply_chat_template(dataset["prompt"][idx],
                                                   tokenize=False)
            input_len = len(tokenizer(prompt).input_ids)
            completion = dataset["completion"][idx]

            requests.append(
                SampleRequest(prompt=prompt,
                              prompt_len=input_len,
                              expected_output_len=args.output_len,
                              schema=schema,
                              completion=completion))

    return requests


def main(args: argparse.Namespace):
    print(args)
    random.seed(args.seed)

    if args.save_results:
        result_file_name = 'guided' if args.guided_decoding else 'no_guided'
        result_file_name += f"_{args.model.split('/')[-1]}"
        result_file_name += f"_{args.dataset}"
        result_file_name += f"_{args.num_prompts}"
        result_file_name += f"_out{args.output_len}"
        result_file_name += f"_async{args.async_engine}"
        result_file_name += f"_warmup{args.warmup}"
        result_file_name += f"_chunkedprefill{args.enable_chunked_prefill}"
        result_file_name += ".txt"
    else:
        result_file_name = None

    # Synthesize a prompt with the given input length.
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer, trust_remote_code=args.trust_remote_code)
    requests = sample_requests(tokenizer, args)

    if args.async_engine:
        engine_args = AsyncEngineArgs.from_cli_args(args)
        elapsed_time = uvloop.run(
            run_vllm_async(
                requests,
                engine_args,
                args.n,
                args.guided_decoding,
                args.warmup,
                args.disable_frontend_multiprocessing,
                result_file_name,
            ))
    else:
        engine_args = EngineArgs.from_cli_args(args)
        elapsed_time = run_vllm(
            requests,
            engine_args,
            args.n,
            args.guided_decoding,
            args.warmup,
            result_file_name,
        )

    total_num_tokens = sum(request.prompt_len + request.expected_output_len
                           for request in requests)
    total_output_tokens = sum(request.expected_output_len
                              for request in requests)
    print(f"Throughput: {len(requests) / elapsed_time:.2f} requests/s, "
          f"{total_num_tokens / elapsed_time:.2f} total tokens/s, "
          f"{total_output_tokens / elapsed_time:.2f} output tokens/s")

    # Output JSON results if specified
    if args.output_json or result_file_name:
        results = {
            "elapsed_time":
            elapsed_time,
            "num_requests":
            len(requests),
            "total_num_tokens":
            total_num_tokens,
            "total_output_tokens":
            total_output_tokens,
            "requests_per_second":
            len(requests) / elapsed_time,
            "tokens_per_second":
            f"{total_num_tokens / elapsed_time:.2f}",
            "output_tokens_per_second":
            f"{total_output_tokens / elapsed_time:.2f}",
        }
        if args.output_json:
            with open(args.output_json, "w") as f:
                json.dump(results, f, indent=4)
        elif result_file_name:
            with open(result_file_name, "a") as f:
                json.dump(results, f, indent=4)


if __name__ == "__main__":
    parser = FlexibleArgumentParser(description="Benchmark guided decoding.")
    parser = AsyncEngineArgs.add_cli_args(parser)

    parser.add_argument("--output-len",
                        type=int,
                        help="Output length for each request. Overrides the "
                        "output length from the dataset.")
    parser.add_argument("--dataset",
                        choices=['single_schema', 'xgrammar_bench'])
    parser.add_argument("--n",
                        type=int,
                        default=1,
                        help="Number of generated sequences per prompt.")
    parser.add_argument("--num-prompts",
                        type=int,
                        default=10,
                        help="Number of prompts to process.")
    parser.add_argument(
        '--output-json',
        type=str,
        default=None,
        help='Path to save the throughput results in JSON format.')
    parser.add_argument("--async-engine",
                        action='store_true',
                        default=False,
                        help="Use vLLM async engine rather than LLM class.")
    parser.add_argument("--guided-decoding",
                        action='store_true',
                        default=False,
                        help="Whether to enable JSON decoding or not.")
    parser.add_argument("--disable-frontend-multiprocessing",
                        action='store_true',
                        default=False,
                        help="Disable decoupled async engine frontend.")
    parser.add_argument("--warmup",
                        action="store_true",
                        default=False,
                        help="Run warmup prompts before benchmark.")
    parser.add_argument("--save-results",
                        action="store_true",
                        default=False,
                        help="save output results.")
    args = parser.parse_args()
    if args.tokenizer is None:
        args.tokenizer = args.model
    main(args)
