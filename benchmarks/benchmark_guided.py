# SPDX-License-Identifier: Apache-2.0
"""Benchmark guided decoding throughput."""
import argparse
import dataclasses
import json
import os
import random
import time
from typing import List

import datasets
import pandas as pd
import uvloop
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from vllm.engine.arg_utils import AsyncEngineArgs, EngineArgs
from vllm.entrypoints.openai.api_server import (
    build_async_engine_client_from_engine_args)
from vllm.sampling_params import GuidedDecodingParams
from vllm.utils import FlexibleArgumentParser, merge_async_iterators


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
    structure_type: str = 'json'
    completion: str = None


def run_vllm(requests: List[SampleRequest],
             engine_args: EngineArgs,
             n: int,
             guided_decoding_rate: float = 1.0,
             warmup: bool = False) -> float:
    from vllm import LLM, SamplingParams
    llm = LLM(**vars(engine_args))
    assert all(
        llm.llm_engine.model_config.max_model_len >= (
            request.prompt_len + request.expected_output_len)
        for request in requests), (
            "Please ensure that max_model_len is greater than the sum of"
            " prompt_len and expected_output_len for all requests.")

    # Add the requests to the engine.
    prompts: List[str] = []
    sampling_params: List[SamplingParams] = []
    # create a list containing random selected true or false
    guided_decoding_req_idx = random.sample(
        range(len(requests)), int(len(requests) * guided_decoding_rate))

    if warmup:
        print(">>>>> Running warmup prompt, for the first 5")
        # We setup the first 5 requests to warmup FSM
        # if using xgrammar dataset, we will skip warmup
        warmup_requests = requests[:5]
        for i, request in enumerate(warmup_requests):
            prompts.append(request.prompt)
            sampling_params.append(
                SamplingParams(
                    n=n,
                    temperature=1.0,
                    top_p=1.0,
                    ignore_eos=True,
                    max_tokens=request.expected_output_len,
                    guided_decoding=GuidedDecodingParams(json=request.schema)
                    if guided_decoding_rate > 0 else None,
                ))
        llm.generate(prompts, sampling_params, use_tqdm=False)

    print(">>>>> Benchmark started...")
    prompts = []
    sampling_params = []
    for i, request in enumerate(requests):
        prompts.append(request.prompt)
        sampling_params.append(
            SamplingParams(
                n=n,
                temperature=1.0,
                top_p=1.0,
                ignore_eos=True,
                max_tokens=request.expected_output_len,
                guided_decoding=GuidedDecodingParams(
                    **{request.structure_type: request.schema})
                if i in guided_decoding_req_idx else None,
            ))

    start = time.perf_counter()
    outputs = llm.generate(prompts, sampling_params, use_tqdm=False)
    ret = []
    for output, request in zip(outputs, requests):
        generated_text = output.outputs[0].text
        ret.append({
            "generated": generated_text,
            "expected": request.completion
        })
    end = time.perf_counter()
    return end - start, ret


async def run_vllm_async(
        requests: List[SampleRequest],
        engine_args: AsyncEngineArgs,
        n: int,
        guided_decoding_rate: float = 1.0,
        warmup: bool = False,
        disable_frontend_multiprocessing: bool = False) -> float:
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
        prompts: List[str] = []
        sampling_params: List[SamplingParams] = []
        guided_decoding_req_idx = random.sample(
            range(len(requests)), int(len(requests) * guided_decoding_rate))

        if warmup:
            print(">>>>>> Running warmup prompt, for the first 5")
            # We setup the first 5 requests to warmup FSM
            # if using xgrammar dataset, we will skip warmup
            warmup_requests = requests[:5]
            for i, request in enumerate(warmup_requests):
                prompts.append(request.prompt)
                sampling_params.append(
                    SamplingParams(
                        n=n,
                        temperature=1.0,
                        top_p=1.0,
                        ignore_eos=True,
                        max_tokens=request.expected_output_len,
                        guided_decoding=GuidedDecodingParams(
                            json=request.schema)
                        if guided_decoding_rate > 0 else None,
                    ))
            generators = []
            for i, (prompt, sp) in enumerate(zip(prompts, sampling_params)):
                generator = llm.generate(prompt, sp, request_id=f"test{i}")
                generators.append(generator)
            all_gens = merge_async_iterators(*generators)
            async for i, res in all_gens:
                pass

        print(">>>>> Benchmark started...")
        prompts = []
        sampling_params = []
        for i, request in enumerate(requests):
            prompts.append(request.prompt)
            sampling_params.append(
                SamplingParams(
                    n=n,
                    temperature=1.0,
                    top_p=1.0,
                    ignore_eos=True,
                    max_tokens=request.expected_output_len,
                    guided_decoding=GuidedDecodingParams(json=request.schema)
                    if i in guided_decoding_req_idx else None,
                ))

        generators = []
        start_time = []
        latencies = []
        start = time.perf_counter()
        for i, (prompt, sp) in enumerate(zip(prompts, sampling_params)):
            generator = llm.generate(prompt, sp, request_id=f"test{i}")
            generators.append(generator)
            start_time.append(time.perf_counter())
            latencies.append([])
        all_gens = merge_async_iterators(*generators)
        generated_texts = [''] * len(requests)
        async for i, res in all_gens:
            generated_texts[i] = res.outputs[0].text
            lat = time.perf_counter() - start_time[i]
            latencies[i].append(lat)
        ret = [{
            'generated': gt,
            'expected': req.completion
        } for gt, req in zip(generated_texts, requests)]
        end = time.perf_counter()
        first_latency = pd.Series([lat[0] * 1000 for lat in latencies])
        next_latency = pd.Series([(lat[-1] - lat[0]) / len(lat[1:]) * 1000
                                  for lat in latencies])
        return end - start, ret, (first_latency, next_latency)


def sample_requests(tokenizer: PreTrainedTokenizerBase,
                    args: argparse.Namespace) -> List[SampleRequest]:
    if args.dataset == 'json':
        if args.json_schema_path is None:
            dir_path = os.path.dirname(os.path.realpath(__file__))
            args.json_schema_path = os.path.join(dir_path,
                                                 "structured_schemas",
                                                 "structured_schema_1.json")
        with open(args.json_schema_path) as f:
            schema = json.load(f)
        prompt = f"Generate an example of a user profile given the following schema: {json.dumps(schema)}"  # noqa: E501
        input_len = len(tokenizer(prompt).input_ids)
        print(f"Input length of the prompt: {input_len} tokens")
        requests = [
            SampleRequest(prompt=prompt,
                          prompt_len=input_len,
                          expected_output_len=args.output_len,
                          schema=schema,
                          structure_type=args.structure_type)
            for _ in range(args.num_prompts)
        ]

    elif args.dataset == "grammar":
        schema = """
            ?start: select_statement

            ?select_statement: "SELECT " column_list " FROM " table_name

            ?column_list: column_name ("," column_name)*

            ?table_name: identifier

            ?column_name: identifier

            ?identifier: /[a-zA-Z_][a-zA-Z0-9_]*/
        """
        prompt = "Generate an SQL query to show the 'username' \
            and 'email' from the 'users' table."

        input_len = len(tokenizer(prompt).input_ids)
        print(f"Input length of the prompt: {input_len} tokens")
        requests = [
            SampleRequest(prompt=prompt,
                          prompt_len=input_len,
                          expected_output_len=args.output_len,
                          schema=schema,
                          structure_type=args.structure_type)
            for _ in range(args.num_prompts)
        ]

    elif args.dataset == "regex":
        regex = r"\w+@\w+\.com\n"
        args.regex = regex
        prompt = "Generate an email address for Alan Turing, \
            who works in Enigma. End in .com and new line. \
                Example result: alan.turing@enigma.com\n"

        input_len = len(tokenizer(prompt).input_ids)
        print(f"Input length of the prompt: {input_len} tokens")
        requests = [
            SampleRequest(prompt=prompt,
                          prompt_len=input_len,
                          expected_output_len=args.output_len,
                          schema=regex,
                          structure_type=args.structure_type)
            for _ in range(args.num_prompts)
        ]

    elif args.dataset == "choice":
        choice = ["Positive", "Negative"]
        args.choice = choice
        prompt = "Classify this sentiment: vLLM is wonderful!"
        input_len = len(tokenizer(prompt).input_ids)
        print(f"Input length of the prompt: {input_len} tokens")
        requests = [
            SampleRequest(prompt=prompt,
                          prompt_len=input_len,
                          expected_output_len=args.output_len,
                          schema=choice,
                          structure_type=args.structure_type)
            for _ in range(args.num_prompts)
        ]

    elif args.dataset == "xgrammar_bench":
        args.warmup = False
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


def evaluate(ret, args):

    def _eval_correctness_json(expected, actual):
        # extract json string from string using regex
        import re
        actual = actual.replace('\n', '').replace(' ', '').strip()
        try:
            actual = re.search(r'\{.*\}', actual).group()
            actual = json.loads(actual)
        except Exception:
            return False

        return True

    def _eval_correctness_choice(expected, actual):
        return actual in args.choice

    def _eval_correctness_regex(expected, actual):
        import re
        return re.match(args.regex, actual) is not None

    def _eval_correctness(expected, actual):
        if args.structure_type == 'json':
            return _eval_correctness_json(expected, actual)
        elif args.structure_type == 'regex':
            return _eval_correctness_regex(expected, actual)
        elif args.structure_type == 'choice':
            return _eval_correctness_choice(expected, actual)
        else:
            return None

    scores = []
    for res in ret:
        score = _eval_correctness(res['expected'], res['generated'])
        res['correctness'] = score
        scores.append(score)

    not_none_scores = [score for score in scores if score is not None]

    return (sum(not_none_scores) / len(not_none_scores) *
            100) if len(not_none_scores) > 0 else None


def main(args: argparse.Namespace):
    print(args)
    random.seed(args.seed)

    # async engine is working for 'regex', 'choice' and 'grammar'
    if args.dataset == 'grammar':
        args.structure_type = 'grammar'
        args.async_engine = False
    elif args.dataset == 'regex':
        args.structure_type = 'regex'
        args.async_engine = False
    elif args.dataset == 'choice':
        args.structure_type = 'choice'
        args.async_engine = False
    else:
        args.structure_type = 'json'

    if args.no_guided_decoding:
        args.guided_decoding_ratio = 0
    if args.save_results:
        result_file_name = f'{args.guided_decoding_ratio}guided'
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
        elapsed_time, ret, (first_latency, next_latency) = uvloop.run(
            run_vllm_async(requests, engine_args, args.n,
                           args.guided_decoding_ratio, args.warmup,
                           args.disable_frontend_multiprocessing))
    else:
        engine_args = EngineArgs.from_cli_args(args)
        elapsed_time, ret = run_vllm(requests, engine_args, args.n,
                                     args.guided_decoding_ratio, args.warmup)
        first_latency, next_latency = None, None

    score = evaluate(ret, args)
    total_num_tokens = sum(request.prompt_len + request.expected_output_len
                           for request in requests)
    total_output_tokens = sum(request.expected_output_len
                              for request in requests)
    if first_latency is not None:
        latency_breakdown = "\nFirst token latency(msecs):\n"
        latency_breakdown += f"{first_latency.describe()}"
        latency_breakdown += "\nNext token latency(msecs):\n"
        latency_breakdown += f"{next_latency.describe()}"
    print(
        f"Throughput: {len(requests) / elapsed_time:.2f} requests/s, "
        f"{total_num_tokens / elapsed_time:.2f} total tokens/s, "
        f"{total_output_tokens / elapsed_time:.2f} output tokens/s",
        f"Correct rate is {score} %",
        f"{latency_breakdown if first_latency is not None else ''}")

    # Output JSON results if specified
    if args.output_json or result_file_name:
        results = {
            "elapsed_time": elapsed_time,
            "num_requests": len(requests),
            "total_num_tokens": total_num_tokens,
            "total_output_tokens": total_output_tokens,
            "requests_per_second": len(requests) / elapsed_time,
            "tokens_per_second": f"{total_num_tokens / elapsed_time:.2f}",
            "output_tokens_per_second":
            f"{total_output_tokens / elapsed_time:.2f}",
            "correct_rate(%)": score
        }
        results = {"outputs": ret, **results}
        if first_latency is not None:
            results["first_token_latency(msecs)"] = first_latency.describe(
            ).to_dict()
            results["next_token_latency(msecs)"] = next_latency.describe(
            ).to_dict()
        if args.output_json:
            with open(args.output_json, "w") as f:
                json.dump(results, f, indent=4)
        elif result_file_name:
            with open(result_file_name, "w") as f:
                json.dump(results, f, indent=4)


if __name__ == "__main__":
    parser = FlexibleArgumentParser(description="Benchmark guided decoding.")
    parser = AsyncEngineArgs.add_cli_args(parser)

    parser.add_argument("--output-len",
                        type=int,
                        default=512,
                        help="Output length for each request. Overrides the "
                        "output length from the dataset.")
    parser.add_argument(
        "--dataset",
        default='json',
        choices=['json', 'grammar', 'regex', 'choice', 'xgrammar_bench'])
    parser.add_argument("--json_schema_path",
                        type=str,
                        default=None,
                        help="Path to json schema.")
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
    parser.add_argument("--no-guided-decoding",
                        action='store_true',
                        default=False,
                        help="Whether to disable JSON decoding or not.")
    parser.add_argument("--guided-decoding-ratio",
                        type=float,
                        default=1.0,
                        help="Ratio of Guided Decoding requests")
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
