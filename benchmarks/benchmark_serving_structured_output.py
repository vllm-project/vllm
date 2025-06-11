# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
r"""Benchmark online serving throughput with structured outputs.

On the server side, run one of the following commands:
    (vLLM OpenAI API server)
    vllm serve <your_model> --disable-log-requests

On the client side, run:
    python benchmarks/benchmark_serving_structured_output.py \
        --backend <backend> \
        --model <your_model> \
        --dataset json \
        --structured-output-ratio 1.0 \
        --request-rate 10 \
        --num-prompts 1000

    when using tgi backend, add
        --endpoint /generate_stream
    to the end of the command above.
"""

import argparse
import asyncio
import copy
import dataclasses
import json
import os
import random
import time
import uuid
import warnings
from collections.abc import AsyncGenerator
from dataclasses import dataclass
from typing import Optional

import datasets
import numpy as np
import pandas as pd
from tqdm.asyncio import tqdm
from transformers import PreTrainedTokenizerBase

from backend_request_func import (
    ASYNC_REQUEST_FUNCS,
    RequestFuncInput,
    RequestFuncOutput,
)

try:
    from vllm.transformers_utils.tokenizer import get_tokenizer
except ImportError:
    from backend_request_func import get_tokenizer

try:
    from vllm.utils import FlexibleArgumentParser
except ImportError:
    from argparse import ArgumentParser as FlexibleArgumentParser

from vllm.v1.structured_output.backend_xgrammar import (
    has_xgrammar_unsupported_json_features,
)

MILLISECONDS_TO_SECONDS_CONVERSION = 1000


@dataclass
class BenchmarkMetrics:
    completed: int
    total_input: int
    total_output: int
    request_throughput: float
    request_goodput: float
    output_throughput: float
    total_token_throughput: float
    mean_ttft_ms: float
    median_ttft_ms: float
    std_ttft_ms: float
    percentiles_ttft_ms: list[tuple[float, float]]
    mean_tpot_ms: float
    median_tpot_ms: float
    std_tpot_ms: float
    percentiles_tpot_ms: list[tuple[float, float]]
    mean_itl_ms: float
    median_itl_ms: float
    std_itl_ms: float
    percentiles_itl_ms: list[tuple[float, float]]
    # E2EL stands for end-to-end latency per request.
    # It is the time taken on the client side from sending
    # a request to receiving a complete response.
    mean_e2el_ms: float
    median_e2el_ms: float
    std_e2el_ms: float
    percentiles_e2el_ms: list[tuple[float, float]]


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
    structure_type: str
    completion: str = None


def sample_requests(
    tokenizer: PreTrainedTokenizerBase, args: argparse.Namespace
) -> list[SampleRequest]:
    if args.dataset == "json" or args.dataset == "json-unique":
        if args.json_schema_path is None:
            dir_path = os.path.dirname(os.path.realpath(__file__))
            args.json_schema_path = os.path.join(
                dir_path, "structured_schemas", "structured_schema_1.json"
            )
        json_schemas = []
        with open(args.json_schema_path) as f:
            schema = json.load(f)

        if args.dataset == "json-unique":
            json_schemas = [copy.deepcopy(schema) for _ in range(args.num_prompts)]
            for i in range(len(json_schemas)):
                if "properties" not in json_schemas[i]:
                    json_schemas[i]["properties"] = {}
                json_schemas[i]["properties"][f"__optional_field_{uuid.uuid4()}"] = {
                    "type": "string",
                    "description": "An unique optional field to avoid cached schemas",
                }
        else:
            json_schemas = [schema] * args.num_prompts

        def gen_prompt(index: int):
            return f"Generate an example of a brief user profile given the following schema: {json.dumps(get_schema(index))}"  # noqa: E501

        def get_schema(index: int):
            return json_schemas[index % len(json_schemas)]

        requests = [
            SampleRequest(
                prompt=gen_prompt(i),
                prompt_len=len(tokenizer(gen_prompt(i)).input_ids),
                expected_output_len=args.output_len,
                schema=get_schema(i),
                structure_type=args.structure_type,
            )
            for i in range(args.num_prompts)
        ]

    elif args.dataset == "grammar":
        schema = """
        root ::= select_statement

        select_statement ::= "SELECT " column " from " table " where " condition

        column ::= "col_1 " | "col_2 "

        table ::= "table_1 " | "table_2 "

        condition ::= column "= " number

        number ::= "1 " | "2 "
        """
        prompt = "Generate an SQL query to show the 'username' \
            and 'email' from the 'users' table."

        input_len = len(tokenizer(prompt).input_ids)
        print(f"Input length of the prompt: {input_len} tokens")
        requests = [
            SampleRequest(
                prompt=prompt,
                prompt_len=input_len,
                expected_output_len=args.output_len,
                schema=schema,
                structure_type=args.structure_type,
            )
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
            SampleRequest(
                prompt=prompt,
                prompt_len=input_len,
                expected_output_len=args.output_len,
                schema=regex,
                structure_type=args.structure_type,
            )
            for _ in range(args.num_prompts)
        ]

    elif args.dataset == "choice":
        choice = ["Positive", "Negative"]
        args.choice = choice
        prompt = "Classify this sentiment: vLLM is wonderful!"
        input_len = len(tokenizer(prompt).input_ids)
        print(f"Input length of the prompt: {input_len} tokens")
        requests = [
            SampleRequest(
                prompt=prompt,
                prompt_len=input_len,
                expected_output_len=args.output_len,
                schema=choice,
                structure_type=args.structure_type,
            )
            for _ in range(args.num_prompts)
        ]

    elif args.dataset == "xgrammar_bench":
        requests: list[SampleRequest] = []
        dataset = datasets.load_dataset("NousResearch/json-mode-eval", split="train")
        full_dataset_len = len(dataset)

        def _filter_func(item):
            import json

            schema = json.loads(item["schema"])
            return not has_xgrammar_unsupported_json_features(schema)

        dataset = dataset.filter(_filter_func)
        num_filtered_out = full_dataset_len - len(dataset)
        print(
            f"dataset has {len(dataset)} entries after filtering "
            f"out {num_filtered_out} entries with unsupported features"
        )
        len_dataset = len(dataset)
        for data_point_idx in range(args.num_prompts):
            idx = data_point_idx
            while idx >= len_dataset:
                idx -= len_dataset
            schema = dataset["schema"][idx]
            prompt = tokenizer.apply_chat_template(
                dataset["prompt"][idx], tokenize=False, add_generation_prompt=True
            )
            input_len = len(tokenizer(prompt).input_ids)
            completion = dataset["completion"][idx]

            requests.append(
                SampleRequest(
                    prompt=prompt,
                    prompt_len=input_len,
                    expected_output_len=args.output_len,
                    schema=schema,
                    structure_type=args.structure_type,
                    completion=completion,
                )
            )

    return requests


async def get_request(
    input_requests: list[SampleRequest],
    request_rate: float,
    burstiness: float = 1.0,
) -> AsyncGenerator[tuple[int, SampleRequest], None]:
    """
    Asynchronously generates requests at a specified rate
    with OPTIONAL burstiness.

    Args:
        input_requests:
            A list of input requests, each represented as a tuple.
        request_rate:
            The rate at which requests are generated (requests/s).
        burstiness (optional):
            The burstiness factor of the request generation.
            Only takes effect when request_rate is not inf.
            Default value is 1, which follows a Poisson process.
            Otherwise, the request intervals follow a gamma distribution.
            A lower burstiness value (0 < burstiness < 1) results
            in more bursty requests, while a higher burstiness value
            (burstiness > 1) results in a more uniform arrival of requests.
    """
    input_requests = iter(input_requests)

    # Calculate scale parameter theta to maintain the desired request_rate.
    assert burstiness > 0, (
        f"A positive burstiness factor is expected, but given {burstiness}."
    )
    theta = 1.0 / (request_rate * burstiness)

    for i, request in enumerate(input_requests):
        yield i, request

        if request_rate == float("inf"):
            # If the request rate is infinity, then we don't need to wait.
            continue

        # Sample the request interval from the gamma distribution.
        # If burstiness is 1, it follows exponential distribution.
        interval = np.random.gamma(shape=burstiness, scale=theta)
        # The next request will be sent after the interval.
        await asyncio.sleep(interval)


def calculate_metrics(
    input_requests: list[tuple[str, int, int]],
    outputs: list[RequestFuncOutput],
    dur_s: float,
    tokenizer: PreTrainedTokenizerBase,
    selected_percentile_metrics: list[str],
    selected_percentiles: list[float],
    goodput_config_dict: Optional[dict[str, float]] = None,
) -> tuple[BenchmarkMetrics, list[int]]:
    actual_output_lens: list[int] = []
    total_input = 0
    completed = 0
    good_completed = 0
    itls: list[float] = []
    tpots: list[float] = []
    all_tpots: list[float] = []
    ttfts: list[float] = []
    e2els: list[float] = []
    for i in range(len(outputs)):
        if outputs[i].success:
            # We use the tokenizer to count the number of output tokens for all
            # serving backends instead of looking at len(outputs[i].itl) since
            # multiple output tokens may be bundled together
            # Note : this may inflate the output token count slightly
            output_len = len(
                tokenizer(outputs[i].generated_text, add_special_tokens=False).input_ids
            )
            actual_output_lens.append(output_len)
            total_input += input_requests[i].prompt_len
            tpot = 0
            if output_len > 1:
                latency_minus_ttft = outputs[i].latency - outputs[i].ttft
                tpot = latency_minus_ttft / (output_len - 1)
                tpots.append(tpot)
            outputs[i].tpot = tpot
            # Note: if output_len <= 1, we regard tpot as 0 for goodput
            all_tpots.append(tpot)
            itls += outputs[i].itl
            ttfts.append(outputs[i].ttft)
            e2els.append(outputs[i].latency)
            completed += 1
        else:
            actual_output_lens.append(0)

    if goodput_config_dict:
        valid_metrics = []
        slo_values = []

        if "ttft" in goodput_config_dict:
            valid_metrics.append(ttfts)
            slo_values.append(
                goodput_config_dict["ttft"] / MILLISECONDS_TO_SECONDS_CONVERSION
            )
        if "tpot" in goodput_config_dict:
            valid_metrics.append(all_tpots)
            slo_values.append(
                goodput_config_dict["tpot"] / MILLISECONDS_TO_SECONDS_CONVERSION
            )
        if "e2el" in goodput_config_dict:
            valid_metrics.append(e2els)
            slo_values.append(
                goodput_config_dict["e2el"] / MILLISECONDS_TO_SECONDS_CONVERSION
            )

        for req_metric in zip(*valid_metrics):
            is_good_req = all([s >= r for s, r in zip(slo_values, req_metric)])
            if is_good_req:
                good_completed += 1

    if completed == 0:
        warnings.warn(
            "All requests failed. This is likely due to a misconfiguration "
            "on the benchmark arguments.",
            stacklevel=2,
        )
    metrics = BenchmarkMetrics(
        completed=completed,
        total_input=total_input,
        total_output=sum(actual_output_lens),
        request_throughput=completed / dur_s,
        request_goodput=good_completed / dur_s,
        output_throughput=sum(actual_output_lens) / dur_s,
        total_token_throughput=(total_input + sum(actual_output_lens)) / dur_s,
        mean_ttft_ms=np.mean(ttfts or 0)
        * 1000,  # ttfts is empty if streaming is not supported by backend
        std_ttft_ms=np.std(ttfts or 0) * 1000,
        median_ttft_ms=np.median(ttfts or 0) * 1000,
        percentiles_ttft_ms=[
            (p, np.percentile(ttfts or 0, p) * 1000) for p in selected_percentiles
        ],
        mean_tpot_ms=np.mean(tpots or 0) * 1000,
        std_tpot_ms=np.std(tpots or 0) * 1000,
        median_tpot_ms=np.median(tpots or 0) * 1000,
        percentiles_tpot_ms=[
            (p, np.percentile(tpots or 0, p) * 1000) for p in selected_percentiles
        ],
        mean_itl_ms=np.mean(itls or 0) * 1000,
        std_itl_ms=np.std(itls or 0) * 1000,
        median_itl_ms=np.median(itls or 0) * 1000,
        percentiles_itl_ms=[
            (p, np.percentile(itls or 0, p) * 1000) for p in selected_percentiles
        ],
        mean_e2el_ms=np.mean(e2els or 0) * 1000,
        std_e2el_ms=np.std(e2els or 0) * 1000,
        median_e2el_ms=np.median(e2els or 0) * 1000,
        percentiles_e2el_ms=[
            (p, np.percentile(e2els or 0, p) * 1000) for p in selected_percentiles
        ],
    )

    return metrics, actual_output_lens


async def benchmark(
    backend: str,
    api_url: str,
    base_url: str,
    model_id: str,
    tokenizer: PreTrainedTokenizerBase,
    input_requests: list[SampleRequest],
    request_rate: float,
    burstiness: float,
    disable_tqdm: bool,
    profile: bool,
    selected_percentile_metrics: list[str],
    selected_percentiles: list[str],
    ignore_eos: bool,
    max_concurrency: Optional[int],
    structured_output_ratio: float,
    goodput_config_dict: Optional[dict[str, float]] = None,
):
    if backend in ASYNC_REQUEST_FUNCS:
        request_func = ASYNC_REQUEST_FUNCS[backend]
    else:
        raise ValueError(f"Unknown backend: {backend}")

    def prepare_extra_body(request) -> dict:
        extra_body = {}
        # Add the schema to the extra_body
        extra_body[request.structure_type] = request.schema
        return extra_body

    print("Starting initial single prompt test run...")
    structured_output_req_idx = random.sample(
        range(len(input_requests)), int(len(input_requests) * structured_output_ratio)
    )

    test_request = input_requests[0]
    test_req_extra_body = (
        prepare_extra_body(test_request) if 0 in structured_output_req_idx else None
    )
    test_input = RequestFuncInput(
        model=model_id,
        prompt=test_request.prompt,
        api_url=api_url,
        prompt_len=test_request.prompt_len,
        output_len=test_request.expected_output_len,
        ignore_eos=ignore_eos,
        extra_body=test_req_extra_body,
    )
    test_output = await request_func(request_func_input=test_input)
    if not test_output.success:
        raise ValueError(
            "Initial test run failed - Please make sure benchmark arguments "
            f"are correctly specified. Error: {test_output.error}"
        )
    else:
        print("Initial test run completed. Starting main benchmark run...")

    if profile:
        print("Starting profiler...")
        profile_input = RequestFuncInput(
            model=model_id,
            prompt=test_request.prompt,
            api_url=base_url + "/start_profile",
            prompt_len=test_request.prompt_len,
            output_len=test_request.expected_output_len,
            ignore_eos=ignore_eos,
            extra_body=test_req_extra_body,
        )
        profile_output = await request_func(request_func_input=profile_input)
        if profile_output.success:
            print("Profiler started")

    distribution = "Poisson process" if burstiness == 1.0 else "Gamma distribution"

    print(f"Traffic request rate: {request_rate}")
    print(f"Burstiness factor: {burstiness} ({distribution})")
    print(f"Maximum request concurrency: {max_concurrency}")

    pbar = None if disable_tqdm else tqdm(total=len(input_requests))

    # This can be used once the minimum Python version is 3.10 or higher,
    # and it will simplify the code in limited_request_func.
    #    semaphore = (asyncio.Semaphore(max_concurrency)
    #                 if max_concurrency else contextlib.nullcontext())
    semaphore = asyncio.Semaphore(max_concurrency) if max_concurrency else None

    async def limited_request_func(request_func_input, pbar):
        if semaphore is None:
            return await request_func(request_func_input=request_func_input, pbar=pbar)
        async with semaphore:
            return await request_func(request_func_input=request_func_input, pbar=pbar)

    benchmark_start_time = time.perf_counter()
    tasks: list[asyncio.Task] = []
    expected: list[str] = []
    async for i, request in get_request(input_requests, request_rate, burstiness):
        extra_body = (
            prepare_extra_body(request) if i in structured_output_req_idx else None
        )
        request_func_input = RequestFuncInput(
            model=model_id,
            prompt=request.prompt,
            api_url=api_url,
            prompt_len=request.prompt_len,
            output_len=request.expected_output_len,
            ignore_eos=ignore_eos,
            extra_body=extra_body,
        )
        expected.append(request.completion)
        tasks.append(
            asyncio.create_task(
                limited_request_func(request_func_input=request_func_input, pbar=pbar)
            )
        )
    outputs: list[RequestFuncOutput] = await asyncio.gather(*tasks)

    if profile:
        print("Stopping profiler...")
        profile_input = RequestFuncInput(
            model=model_id,
            prompt=test_request.prompt,
            api_url=base_url + "/stop_profile",
            prompt_len=test_request.prompt_len,
            output_len=test_request.expected_output_len,
            extra_body={test_request.structure_type: test_request.schema},
        )
        profile_output = await request_func(request_func_input=profile_input)
        if profile_output.success:
            print("Profiler stopped")

    if pbar is not None:
        pbar.close()

    benchmark_duration = time.perf_counter() - benchmark_start_time

    metrics, actual_output_lens = calculate_metrics(
        input_requests=input_requests,
        outputs=outputs,
        dur_s=benchmark_duration,
        tokenizer=tokenizer,
        selected_percentile_metrics=selected_percentile_metrics,
        selected_percentiles=selected_percentiles,
        goodput_config_dict=goodput_config_dict,
    )

    print("{s:{c}^{n}}".format(s=" Serving Benchmark Result ", n=50, c="="))
    print("{:<40} {:<10}".format("Successful requests:", metrics.completed))
    print("{:<40} {:<10.2f}".format("Benchmark duration (s):", benchmark_duration))
    print("{:<40} {:<10}".format("Total input tokens:", metrics.total_input))
    print("{:<40} {:<10}".format("Total generated tokens:", metrics.total_output))
    print(
        "{:<40} {:<10.2f}".format(
            "Request throughput (req/s):", metrics.request_throughput
        )
    )
    if goodput_config_dict:
        print(
            "{:<40} {:<10.2f}".format(
                "Request goodput (req/s):", metrics.request_goodput
            )
        )
    print(
        "{:<40} {:<10.2f}".format(
            "Output token throughput (tok/s):", metrics.output_throughput
        )
    )
    print(
        "{:<40} {:<10.2f}".format(
            "Total Token throughput (tok/s):", metrics.total_token_throughput
        )
    )

    result = {
        "duration": benchmark_duration,
        "completed": metrics.completed,
        "total_input_tokens": metrics.total_input,
        "total_output_tokens": metrics.total_output,
        "request_throughput": metrics.request_throughput,
        "output_throughput": metrics.output_throughput,
        "total_token_throughput": metrics.total_token_throughput,
        "ttft_description": pd.Series([output.ttft for output in outputs])
        .describe()
        .to_dict(),
        "tpot_description": pd.Series([output.tpot for output in outputs])
        .describe()
        .to_dict(),
        "input_lens": [output.prompt_len for output in outputs],
        "output_lens": actual_output_lens,
        "ttfts": [output.ttft for output in outputs],
        "itls": [output.itl for output in outputs],
        "errors": [output.error for output in outputs],
    }

    ret = [
        {"generated": output.generated_text, "expected": gt}
        for output, gt in zip(outputs, expected)
    ]

    def process_one_metric(
        # E.g., "ttft"
        metric_attribute_name: str,
        # E.g., "TTFT"
        metric_name: str,
        # E.g., "Time to First Token"
        metric_header: str,
    ):
        # This function prints and adds statistics of the specified
        # metric.
        if metric_attribute_name not in selected_percentile_metrics:
            return
        print("{s:{c}^{n}}".format(s=metric_header, n=50, c="-"))
        print(
            "{:<40} {:<10.2f}".format(
                f"Mean {metric_name} (ms):",
                getattr(metrics, f"mean_{metric_attribute_name}_ms"),
            )
        )
        print(
            "{:<40} {:<10.2f}".format(
                f"Median {metric_name} (ms):",
                getattr(metrics, f"median_{metric_attribute_name}_ms"),
            )
        )
        result[f"mean_{metric_attribute_name}_ms"] = getattr(
            metrics, f"mean_{metric_attribute_name}_ms"
        )
        result[f"median_{metric_attribute_name}_ms"] = getattr(
            metrics, f"median_{metric_attribute_name}_ms"
        )
        result[f"std_{metric_attribute_name}_ms"] = getattr(
            metrics, f"std_{metric_attribute_name}_ms"
        )
        for p, value in getattr(metrics, f"percentiles_{metric_attribute_name}_ms"):
            p_word = str(int(p)) if int(p) == p else str(p)
            print("{:<40} {:<10.2f}".format(f"P{p_word} {metric_name} (ms):", value))
            result[f"p{p_word}_{metric_attribute_name}_ms"] = value

    process_one_metric("ttft", "TTFT", "Time to First Token")
    process_one_metric("tpot", "TPOT", "Time per Output Token (excl. 1st token)")
    process_one_metric("itl", "ITL", "Inter-token Latency")
    process_one_metric("e2el", "E2EL", "End-to-end Latency")

    print("=" * 50)

    return result, ret


def evaluate(ret, args):
    def _eval_correctness_json(expected, actual):
        # extract json string from string using regex
        import regex as re

        actual = actual.replace("\n", "").replace(" ", "").strip()
        try:
            actual = re.search(r"\{.*\}", actual).group()
            actual = json.loads(actual)
        except Exception:
            return False

        return True

    def _eval_correctness_choice(expected, actual):
        return actual in args.choice

    def _eval_correctness_regex(expected, actual):
        import regex as re

        return re.match(args.regex, actual) is not None

    def _eval_correctness(expected, actual):
        if args.structure_type == "guided_json":
            return _eval_correctness_json(expected, actual)
        elif args.structure_type == "guided_regex":
            return _eval_correctness_regex(expected, actual)
        elif args.structure_type == "guided_choice":
            return _eval_correctness_choice(expected, actual)
        else:
            return None

    scores = []
    for res in ret:
        score = _eval_correctness(res["expected"], res["generated"])
        res["correctness"] = score
        scores.append(score)

    not_none_scores = [score for score in scores if score is not None]

    return (
        (sum(not_none_scores) / len(not_none_scores) * 100)
        if len(not_none_scores) > 0
        else None
    )


def parse_goodput(slo_pairs):
    goodput_config_dict = {}
    try:
        for slo_pair in slo_pairs:
            slo_name, slo_val = slo_pair.split(":")
            goodput_config_dict[slo_name] = float(slo_val)
    except ValueError as err:
        raise argparse.ArgumentTypeError(
            "Invalid format found for service level objectives. "
            'Specify service level objectives for goodput as "KEY:VALUE" '
            "pairs, where the key is a metric name, and the value is a "
            "number in milliseconds."
        ) from err
    return goodput_config_dict


def check_goodput_args(args):
    goodput_config_dict = {}
    VALID_NAMES = ["ttft", "tpot", "e2el"]
    if args.goodput:
        goodput_config_dict = parse_goodput(args.goodput)
        for slo_name, slo_val in goodput_config_dict.items():
            if slo_name not in VALID_NAMES:
                raise ValueError(
                    f"Invalid metric name found, {slo_name}: {slo_val}. "
                    "The service level objective name should be one of "
                    f"{str(VALID_NAMES)}. "
                )
            if slo_val < 0:
                raise ValueError(
                    f"Invalid value found, {slo_name}: {slo_val}. "
                    "The service level objective value should be "
                    "non-negative."
                )
    return goodput_config_dict


def main(args: argparse.Namespace):
    print(args)
    random.seed(args.seed)
    np.random.seed(args.seed)

    backend = args.backend
    model_id = args.model
    tokenizer_id = args.tokenizer if args.tokenizer is not None else args.model

    if args.base_url is not None:
        api_url = f"{args.base_url}{args.endpoint}"
        base_url = f"{args.base_url}"
    else:
        api_url = f"http://{args.host}:{args.port}{args.endpoint}"
        base_url = f"http://{args.host}:{args.port}"

    tokenizer = get_tokenizer(
        tokenizer_id,
        trust_remote_code=args.trust_remote_code,
        tokenizer_mode=args.tokenizer_mode,
    )

    if args.dataset == "grammar":
        args.structure_type = "guided_grammar"
    elif args.dataset == "regex":
        args.structure_type = "guided_regex"
    elif args.dataset == "choice":
        args.structure_type = "guided_choice"
    else:
        args.structure_type = "guided_json"

    if args.no_structured_output:
        args.structured_output_ratio = 0
    if args.save_results:
        result_file_name = f"{args.structured_output_ratio}guided"
        result_file_name += f"_{backend}"
        result_file_name += f"_{args.request_rate}qps"
        result_file_name += f"_{args.model.split('/')[-1]}"
        result_file_name += f"_{args.dataset}"
        result_file_name += f"_{args.num_prompts}"
        result_file_name += f"_out{args.output_len}"
        result_file_name += ".txt"
    else:
        result_file_name = None

    input_requests = sample_requests(tokenizer, args)

    goodput_config_dict = check_goodput_args(args)

    benchmark_result, ret = asyncio.run(
        benchmark(
            backend=backend,
            api_url=api_url,
            base_url=base_url,
            model_id=model_id,
            tokenizer=tokenizer,
            input_requests=input_requests,
            request_rate=args.request_rate,
            burstiness=args.burstiness,
            disable_tqdm=args.disable_tqdm,
            profile=args.profile,
            selected_percentile_metrics=args.percentile_metrics.split(","),
            selected_percentiles=[float(p) for p in args.metric_percentiles.split(",")],
            ignore_eos=args.ignore_eos,
            max_concurrency=args.max_concurrency,
            structured_output_ratio=args.structured_output_ratio,
            goodput_config_dict=goodput_config_dict,
        )
    )

    # Save config and results to json
    score = evaluate(ret, args)
    print("correct_rate(%)", score, "\n")
    if args.save_results:
        results = {
            "backend": backend,
            "model_id": model_id,
            "tokenizer_id": tokenizer_id,
            "num_prompts": args.num_prompts,
            "request_rate": args.request_rate
            if args.request_rate < float("inf")
            else "inf",
            "burstiness": args.burstiness,
            "max_concurrency": args.max_concurrency,
            "correct_rate(%)": score,
        }
        results = {"outputs": ret, **results, **benchmark_result}

        # Save to file
        if args.result_filename:
            result_file_name = args.result_filename
        if args.result_dir:
            result_file_name = os.path.join(args.result_dir, result_file_name)
        with open(result_file_name, "w", encoding="utf-8") as outfile:
            json.dump(results, outfile, indent=4)


if __name__ == "__main__":
    parser = FlexibleArgumentParser(
        description="Benchmark the online serving throughput."
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="vllm",
        choices=list(ASYNC_REQUEST_FUNCS.keys()),
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default=None,
        help="Server or API base url if not using http host and port.",
    )
    # Use 127.0.0.1 here instead of localhost to force the use of ipv4
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--endpoint",
        type=str,
        default="/v1/completions",
        help="API endpoint.",
    )
    parser.add_argument(
        "--dataset",
        default="json",
        choices=["json", "json-unique", "grammar", "regex", "choice", "xgrammar_bench"],
    )
    parser.add_argument(
        "--json-schema-path", type=str, default=None, help="Path to json schema."
    )
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=None,
        help="Maximum number of concurrent requests. This can be used "
        "to help simulate an environment where a higher level component "
        "is enforcing a maximum number of concurrent requests. While the "
        "--request-rate argument controls the rate at which requests are "
        "initiated, this argument will control how many are actually allowed "
        "to execute at a time. This means that when used in combination, the "
        "actual request rate may be lower than specified with --request-rate, "
        "if the server is not processing requests fast enough to keep up.",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Name of the model.",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        help="Name or path of the tokenizer, if not using the default tokenizer.",  # noqa: E501
    )
    parser.add_argument(
        "--tokenizer-mode",
        type=str,
        default="auto",
        help="Name or path of the tokenizer, if not using the default tokenizer.",  # noqa: E501
    )
    parser.add_argument(
        "--num-prompts",
        type=int,
        default=1000,
        help="Number of prompts to process.",
    )
    parser.add_argument(
        "--output-len",
        type=int,
        default=128,
        help="Number of output tokens.",
    )
    parser.add_argument(
        "--request-rate",
        type=float,
        default=float("inf"),
        help="Number of requests per second. If this is inf, "
        "then all the requests are sent at time 0. "
        "Otherwise, we use Poisson process or gamma distribution "
        "to synthesize the request arrival times.",
    )
    parser.add_argument(
        "--burstiness",
        type=float,
        default=1.0,
        help="Burstiness factor of the request generation. "
        "Only take effect when request_rate is not inf. "
        "Default value is 1, which follows Poisson process. "
        "Otherwise, the request intervals follow a gamma distribution. "
        "A lower burstiness value (0 < burstiness < 1) results in more "
        "bursty requests. A higher burstiness value (burstiness > 1) "
        "results in a more uniform arrival of requests.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Trust remote code from huggingface",
    )
    parser.add_argument(
        "--disable-tqdm",
        action="store_true",
        help="Specify to disable tqdm progress bar.",
    )
    parser.add_argument(
        "--save-results",
        action="store_true",
        help="Specify to save benchmark results to a json file",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Use Torch Profiler. The endpoint must be launched with "
        "VLLM_TORCH_PROFILER_DIR to enable profiler.",
    )
    parser.add_argument(
        "--result-dir",
        type=str,
        default=None,
        help="Specify directory to save benchmark json results."
        "If not specified, results are saved in the current directory.",
    )
    parser.add_argument(
        "--result-filename",
        type=str,
        default=None,
        help="Specify the filename to save benchmark json results."
        "If not specified, results will be saved in "
        "{backend}-{args.request_rate}qps-{base_model_id}-{current_dt}.json"
        " format.",
    )
    parser.add_argument(
        "--ignore-eos",
        action="store_true",
        help="Set ignore_eos flag when sending the benchmark request."
        "Warning: ignore_eos is not supported in deepspeed_mii and tgi.",
    )
    parser.add_argument(
        "--percentile-metrics",
        type=str,
        default="ttft,tpot,itl",
        help="Comma-separated list of selected metrics to report percentils. "
        "This argument specifies the metrics to report percentiles. "
        'Allowed metric names are "ttft", "tpot", "itl", "e2el". '
        'Default value is "ttft,tpot,itl".',
    )
    parser.add_argument(
        "--metric-percentiles",
        type=str,
        default="99",
        help="Comma-separated list of percentiles for selected metrics. "
        'To report 25-th, 50-th, and 75-th percentiles, use "25,50,75". '
        'Default value is "99". '
        'Use "--percentile-metrics" to select metrics.',
    )
    parser.add_argument(
        "--goodput",
        nargs="+",
        required=False,
        help='Specify service level objectives for goodput as "KEY:VALUE" '
        "pairs, where the key is a metric name, and the value is in "
        'milliseconds. Multiple "KEY:VALUE" pairs can be provided, '
        "separated by spaces. Allowed request level metric names are "
        '"ttft", "tpot", "e2el". For more context on the definition of '
        "goodput, refer to DistServe paper: https://arxiv.org/pdf/2401.09670 "
        "and the blog: https://hao-ai-lab.github.io/blogs/distserve",
    )

    parser.add_argument(
        "--no-structured-output",
        action="store_true",
        default=False,
        help="Whether to disable JSON decoding or not.",
    )
    parser.add_argument(
        "--structured-output-ratio",
        type=float,
        default=1.0,
        help="Ratio of Structured Outputs requests",
    )

    args = parser.parse_args()
    main(args)
