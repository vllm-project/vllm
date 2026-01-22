# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
r"""Benchmark online serving throughput.

On the server side, run one of the following commands
to launch the vLLM OpenAI API server:
    vllm serve <your_model> <engine arguments>

On the client side, run:
    vllm bench serve \
        --backend <backend or endpoint type. Default 'openai'> \
        --label <benchmark result label. Default using backend> \
        --model <your_model. Optional, defaults to first model from server> \
        --dataset-name <dataset_name. Default 'random'> \
        --input-len <general input length. Optional, maps to dataset-specific args> \
        --output-len <general output length. Optional, maps to dataset-specific args> \
        --request-rate <request_rate. Default inf> \
        --num-prompts <num_prompts. Default 1000>
"""

import argparse
import asyncio
import importlib.util
import json
import os
import random
import shutil
import uuid
from datetime import datetime
from typing import Any

import aiohttp
import numpy as np

from vllm.benchmarks.benchmark import TaskType
from vllm.benchmarks.datasets import add_dataset_parser
from vllm.benchmarks.lib.endpoint_request_func import (
    OPENAI_COMPATIBLE_BACKENDS,
)
from vllm.benchmarks.lib.utils import convert_to_pytorch_benchmark_format, write_to_json
from vllm.tokenizers import get_tokenizer
from vllm.utils.gc_utils import freeze_gc_heap
from vllm.utils.network_utils import join_host_port

MILLISECONDS_TO_SECONDS_CONVERSION = 1000

TERM_PLOTLIB_AVAILABLE = (importlib.util.find_spec("termplotlib") is not None) and (
    shutil.which("gnuplot") is not None
)


async def get_first_model_from_server(
    base_url: str, headers: dict | None = None
) -> tuple[str, str]:
    """Fetch the first model from the server's /v1/models endpoint."""
    models_url = f"{base_url}/v1/models"
    async with aiohttp.ClientSession() as session:
        try:
            async with session.get(models_url, headers=headers) as response:
                response.raise_for_status()
                data = await response.json()
                if "data" in data and len(data["data"]) > 0:
                    return data["data"][0]["id"], data["data"][0]["root"]
                else:
                    raise ValueError(
                        f"No models found on the server at {base_url}. "
                        "Make sure the server is running and has models loaded."
                    )
        except (aiohttp.ClientError, json.JSONDecodeError) as e:
            raise RuntimeError(
                f"Failed to fetch models from server at {models_url}. "
                "Check that:\n"
                "1. The server is running\n"
                "2. The server URL is correct\n"
                f"Error: {e}"
            ) from e


def check_goodput_args(args):
    # Check and parse goodput arguments
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


def save_to_pytorch_benchmark_format(
    args: argparse.Namespace, results: dict[str, Any], file_name: str
) -> None:
    metrics = [
        "median_ttft_ms",
        "mean_ttft_ms",
        "std_ttft_ms",
        "p99_ttft_ms",
        "mean_tpot_ms",
        "median_tpot_ms",
        "std_tpot_ms",
        "p99_tpot_ms",
        "median_itl_ms",
        "mean_itl_ms",
        "std_itl_ms",
        "p99_itl_ms",
    ]
    # These raw data might be useful, but they are rather big. They can be added
    # later if needed
    ignored_metrics = ["ttfts", "itls", "generated_texts", "errors"]
    pt_records = convert_to_pytorch_benchmark_format(
        args=args,
        metrics={k: [results[k]] for k in metrics if k in results},
        extra_info={
            k: results[k]
            for k in results
            if k not in metrics and k not in ignored_metrics
        },
    )
    if pt_records:
        # Don't use json suffix here as we don't want CI to pick it up
        pt_file = f"{os.path.splitext(file_name)[0]}.pytorch.json"
        write_to_json(pt_file, pt_records)


def add_cli_args(parser: argparse.ArgumentParser):
    add_dataset_parser(parser)
    parser.add_argument(
        "--label",
        type=str,
        default=None,
        help="The label (prefix) of the benchmark results. If not specified, "
        "the value of '--backend' will be used as the label.",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="openai",
        help="The type of backend or endpoint to use for the benchmark.",
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
        "--header",
        metavar="KEY=VALUE",
        nargs="*",
        help="Key-value pairs (e.g, --header x-additional-info=0.3.3) "
        "for headers to be passed with each request. These headers override "
        "per backend constants and values set via environment variable, and "
        "will be overridden by other arguments (such as request ids).",
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
        required=False,
        default=None,
        help="Name of the model. If not specified, will fetch the first model "
        "from the server's /v1/models endpoint.",
    )
    parser.add_argument(
        "--input-len",
        type=int,
        default=None,
        help="General input length for datasets. Maps to dataset-specific "
        "input length arguments (e.g., --random-input-len, --sonnet-input-len). "
        "If not specified, uses dataset defaults.",
    )
    parser.add_argument(
        "--output-len",
        type=int,
        default=None,
        help="General output length for datasets. Maps to dataset-specific "
        "output length arguments (e.g., --random-output-len, --sonnet-output-len). "
        "If not specified, uses dataset defaults.",
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
        help="""Tokenizer mode:\n
        - "auto" will use the tokenizer from `mistral_common` for Mistral models
        if available, otherwise it will use the "hf" tokenizer.\n
        - "hf" will use the fast tokenizer if available.\n
        - "slow" will always use the slow tokenizer.\n
        - "mistral" will always use the tokenizer from `mistral_common`.\n
        - "deepseek_v32" will always use the tokenizer from `deepseek_v32`.\n
        - Other custom values can be supported via plugins.""",
    )
    parser.add_argument("--use-beam-search", action="store_true")
    parser.add_argument(
        "--logprobs",
        type=int,
        default=None,
        help=(
            "Number of logprobs-per-token to compute & return as part of "
            "the request. If unspecified, then either (1) if beam search "
            "is disabled, no logprobs are computed & a single dummy "
            "logprob is returned for each token; or (2) if beam search "
            "is enabled 1 logprob per token is computed"
        ),
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
        "--num-warmups",
        type=int,
        default=0,
        help="Number of warmup requests.",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Use vLLM Profiling. --profiler-config must be provided on the server.",
    )
    parser.add_argument(
        "--save-result",
        action="store_true",
        help="Specify to save benchmark results to a json file",
    )
    parser.add_argument(
        "--save-detailed",
        action="store_true",
        help="When saving the results, whether to include per request "
        "information such as response, error, ttfts, tpots, etc.",
    )
    parser.add_argument(
        "--append-result",
        action="store_true",
        help="Append the benchmark result to the existing json file.",
    )
    parser.add_argument(
        "--metadata",
        metavar="KEY=VALUE",
        nargs="*",
        help="Key-value pairs (e.g, --metadata version=0.3.3 tp=1) "
        "for metadata of this run to be saved in the result JSON file "
        "for record keeping purposes.",
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
        "{label}-{args.request_rate}qps-{base_model_id}-{current_dt}.json"  # noqa
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
        default=None,
        help="Comma-separated list of selected metrics to report percentiles. "
        "This argument specifies the metrics to report percentiles. "
        'Allowed metric names are "ttft", "tpot", "itl", "e2el". '
        'If not specified, defaults to "ttft,tpot,itl" for generative models '
        'and "e2el" for pooling models.',
    )
    parser.add_argument(
        "--metric-percentiles",
        type=str,
        default="99",
        help="Comma-separated list of percentiles for selected metrics. "
        'To report 25-th, 50-th, and 75-th percentiles, use "25,50,75". '
        'Default value is "99".'
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
        "--request-id-prefix",
        type=str,
        required=False,
        default=f"bench-{uuid.uuid4().hex[:8]}-",
        help="Specify the prefix of request id.",
    )

    sampling_group = parser.add_argument_group("sampling parameters")
    sampling_group.add_argument(
        "--top-p",
        type=float,
        default=None,
        help="Top-p sampling parameter. Only has effect on openai-compatible backends.",
    )
    sampling_group.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Top-k sampling parameter. Only has effect on openai-compatible backends.",
    )
    sampling_group.add_argument(
        "--min-p",
        type=float,
        default=None,
        help="Min-p sampling parameter. Only has effect on openai-compatible backends.",
    )
    sampling_group.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Temperature sampling parameter. Only has effect on "
        "openai-compatible backends. If not specified, default to greedy "
        "decoding (i.e. temperature==0.0).",
    )
    sampling_group.add_argument(
        "--frequency-penalty",
        type=float,
        default=None,
        help="Frequency penalty sampling parameter. Only has effect on "
        "openai-compatible backends.",
    )
    sampling_group.add_argument(
        "--presence-penalty",
        type=float,
        default=None,
        help="Presence penalty sampling parameter. Only has effect on "
        "openai-compatible backends.",
    )
    sampling_group.add_argument(
        "--repetition-penalty",
        type=float,
        default=None,
        help="Repetition penalty sampling parameter. Only has effect on "
        "openai-compatible backends.",
    )

    parser.add_argument(
        "--served-model-name",
        type=str,
        default=None,
        help="The model name used in the API. "
        "If not specified, the model name will be the "
        "same as the `--model` argument. ",
    )

    parser.add_argument(
        "--lora-modules",
        nargs="+",
        default=None,
        help="A subset of LoRA module names passed in when "
        "launching the server. For each request, the "
        "script chooses a LoRA module at random.",
    )

    parser.add_argument(
        "--ramp-up-strategy",
        type=str,
        default=None,
        choices=["linear", "exponential"],
        help="The ramp-up strategy. This would be used to "
        "ramp up the request rate from initial RPS to final "
        "RPS rate (specified by --ramp-up-start-rps and "
        "--ramp-up-end-rps.) over the duration of the benchmark.",
    )
    parser.add_argument(
        "--ramp-up-start-rps",
        type=int,
        default=None,
        help="The starting request rate for ramp-up (RPS). "
        "Needs to be specified when --ramp-up-strategy is used.",
    )
    parser.add_argument(
        "--ramp-up-end-rps",
        type=int,
        default=None,
        help="The ending request rate for ramp-up (RPS). "
        "Needs to be specified when --ramp-up-strategy is used.",
    )
    parser.add_argument(
        "--ready-check-timeout-sec",
        type=int,
        default=0,
        help="Maximum time to wait for the endpoint to become ready "
        "in seconds. Ready check will be skipped by default.",
    )

    parser.add_argument(
        "--extra-body",
        help="A JSON string representing extra body parameters to include "
        "in each request."
        'Example: \'{"chat_template_kwargs":{"enable_thinking":false}}\'',
        type=json.loads,
        default=None,
    )

    parser.add_argument(
        "--benchmark-cls",
        help="The benchmark class to use. If not specified, the default "
        "benchmark class will be used. "
        "Example: vllm.benchmarks.benchmarks.ServingBenchmark",
        type=str,
        default=None,
    )


def main(args: argparse.Namespace) -> dict[str, Any]:
    return asyncio.run(main_async(args))


def load_class(class_path: str):
    module_path, cls_name = class_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, cls_name)


async def main_async(args: argparse.Namespace) -> dict[str, Any]:
    print(args)
    random.seed(args.seed)
    np.random.seed(args.seed)

    # Validate ramp-up arguments
    if args.ramp_up_strategy is not None:
        if args.request_rate != float("inf"):
            raise ValueError(
                "When using ramp-up, do not specify --request-rate. "
                "The request rate will be controlled by ramp-up parameters. "
                "Please remove the --request-rate argument."
            )
        if args.ramp_up_start_rps is None or args.ramp_up_end_rps is None:
            raise ValueError(
                "When using --ramp-up-strategy, both --ramp-up-start-rps and "
                "--ramp-up-end-rps must be specified"
            )
        if args.ramp_up_start_rps < 0 or args.ramp_up_end_rps < 0:
            raise ValueError("Ramp-up start and end RPS must be non-negative")
        if args.ramp_up_start_rps > args.ramp_up_end_rps:
            raise ValueError("Ramp-up start RPS must be less than end RPS")
        if args.ramp_up_strategy == "exponential" and args.ramp_up_start_rps == 0:
            raise ValueError("For exponential ramp-up, the start RPS cannot be 0.")

    label = args.label

    if args.base_url is not None:
        api_url = f"{args.base_url}{args.endpoint}"
        base_url = f"{args.base_url}"
    else:
        host_port = join_host_port(args.host, args.port)
        api_url = f"http://{host_port}{args.endpoint}"
        base_url = f"http://{host_port}"

    # Headers
    headers = None
    if args.header:
        headers = {}
        for item in args.header:
            if "=" in item:
                kvstring = item.split("=", 1)
                headers[kvstring[0].strip()] = kvstring[1].strip()
            else:
                raise ValueError("Invalid header format. Please use KEY=VALUE format.")

    # Fetch model from server if not specified
    if args.model is None:
        print("Model not specified, fetching first model from server...")
        model_name, model_id = await get_first_model_from_server(base_url, headers)
        print(f"First model name: {model_name}, first model id: {model_id}")
    else:
        model_name = args.served_model_name
        model_id = args.model

    if args.benchmark_cls:
        BenchmarkCls = load_class(args.benchmark_cls)
    else:
        from vllm.benchmarks.benchmark import ServingBenchmark

        BenchmarkCls = ServingBenchmark

    tokenizer_id = args.tokenizer if args.tokenizer is not None else model_id
    tokenizer_mode = args.tokenizer_mode

    tokenizer = get_tokenizer(
        tokenizer_id,
        tokenizer_mode=tokenizer_mode,
        trust_remote_code=args.trust_remote_code,
    )

    if args.dataset_name is None:
        raise ValueError(
            "Please specify '--dataset-name' and the corresponding "
            "'--dataset-path' if required."
        )

    # Map general --input-len and --output-len to all dataset-specific arguments
    if args.input_len is not None:
        args.random_input_len = args.input_len
        args.sonnet_input_len = args.input_len

    if args.output_len is not None:
        args.random_output_len = args.output_len
        args.sonnet_output_len = args.output_len
        args.sharegpt_output_len = args.output_len
        args.custom_output_len = args.output_len
        args.hf_output_len = args.output_len
        args.spec_bench_output_len = args.output_len
        args.prefix_repetition_output_len = args.output_len

    # when using random datasets, default to ignoring EOS
    # so generation runs to the requested length
    if (
        args.dataset_name in ("random", "random-mm")
        and args.backend in BenchmarkCls.get_openai_compatible_backends()
    ):
        args.ignore_eos = True

    # Load the dataset.
    input_requests = BenchmarkCls.get_samples(args, tokenizer)
    goodput_config_dict = check_goodput_args(args)

    backend = args.backend
    task_type = (
        TaskType.POOLING
        if "embeddings" in backend or "rerank" in backend
        else TaskType.GENERATION
    )

    # Collect the sampling parameters.
    if task_type == TaskType.GENERATION:
        sampling_params = {
            k: v
            for k, v in {
                "top_p": args.top_p,
                "top_k": args.top_k,
                "min_p": args.min_p,
                "temperature": args.temperature,
                "frequency_penalty": args.frequency_penalty,
                "presence_penalty": args.presence_penalty,
                "repetition_penalty": args.repetition_penalty,
            }.items()
            if v is not None
        }

        # Sampling parameters are only supported by openai-compatible backend.
        if sampling_params and args.backend not in OPENAI_COMPATIBLE_BACKENDS:
            raise ValueError(
                "Sampling parameters are only supported by openai-compatible backends."
            )

        if "temperature" not in sampling_params:
            sampling_params["temperature"] = 0.0  # Default to greedy decoding.

        default_percentile_metrics = "ttft,tpot,itl"
    else:
        sampling_params = {}
        default_percentile_metrics = "e2el"

    extra_body = args.extra_body or {}
    extra_body = {**sampling_params, **extra_body}

    percentile_metrics: str = args.percentile_metrics or default_percentile_metrics

    # Avoid GC processing "static" data - reduce pause times.
    freeze_gc_heap()

    benchmark = BenchmarkCls(
        task_type=task_type,
        endpoint_type=backend,
        api_url=api_url,
        base_url=base_url,
        model_id=model_id,
        model_name=model_name,
        tokenizer=tokenizer,
        input_requests=input_requests,
        logprobs=args.logprobs,
        request_rate=args.request_rate,
        burstiness=args.burstiness,
        disable_tqdm=args.disable_tqdm,
        num_warmups=args.num_warmups,
        profile=args.profile,
        selected_percentile_metrics=percentile_metrics.split(","),
        selected_percentiles=[float(p) for p in args.metric_percentiles.split(",")],
        ignore_eos=args.ignore_eos,
        goodput_config_dict=goodput_config_dict,
        max_concurrency=args.max_concurrency,
        lora_modules=args.lora_modules,
        extra_headers=headers,
        extra_body=extra_body,
        ramp_up_strategy=args.ramp_up_strategy,
        ramp_up_start_rps=args.ramp_up_start_rps,
        ramp_up_end_rps=args.ramp_up_end_rps,
        ready_check_timeout_sec=args.ready_check_timeout_sec,
    )
    benchmark_result = await benchmark.run()

    # Save config and results to json
    result_json: dict[str, Any] = {}

    # Setup
    current_dt = datetime.now().strftime("%Y%m%d-%H%M%S")
    result_json["date"] = current_dt
    result_json["endpoint_type"] = args.backend  # for backward compatibility
    result_json["backend"] = args.backend
    result_json["label"] = label
    result_json["model_id"] = model_id
    result_json["tokenizer_id"] = tokenizer_id
    result_json["num_prompts"] = args.num_prompts

    # Metadata
    if args.metadata:
        for item in args.metadata:
            if "=" in item:
                kvstring = item.split("=", 1)
                result_json[kvstring[0].strip()] = kvstring[1].strip()
            else:
                raise ValueError(
                    "Invalid metadata format. Please use KEY=VALUE format."
                )

    # Traffic
    result_json["request_rate"] = (
        args.request_rate if args.request_rate < float("inf") else "inf"
    )
    result_json["burstiness"] = args.burstiness
    result_json["max_concurrency"] = args.max_concurrency

    if args.ramp_up_strategy is not None:
        result_json["ramp_up_strategy"] = args.ramp_up_strategy
        result_json["ramp_up_start_rps"] = args.ramp_up_start_rps
        result_json["ramp_up_end_rps"] = args.ramp_up_end_rps

    # Merge with benchmark result
    result_json = {**result_json, **benchmark_result}

    if not args.save_detailed:
        # Remove fields with too many data points
        for field in [
            "input_lens",
            "output_lens",
            "ttfts",
            "itls",
            "generated_texts",
            "errors",
        ]:
            if field in result_json:
                del result_json[field]
            if field in benchmark_result:
                del benchmark_result[field]

        # Save to file
    if args.save_result or args.append_result:
        base_model_id = model_id.split("/")[-1]
        max_concurrency_str = (
            f"-concurrency{args.max_concurrency}"
            if args.max_concurrency is not None
            else ""
        )
        label = label or args.backend
        if args.ramp_up_strategy is not None:
            file_name = f"{label}-ramp-up-{args.ramp_up_strategy}-{args.ramp_up_start_rps}qps-{args.ramp_up_end_rps}qps{max_concurrency_str}-{base_model_id}-{current_dt}.json"  # noqa
        else:
            file_name = f"{label}-{args.request_rate}qps{max_concurrency_str}-{base_model_id}-{current_dt}.json"  # noqa
        if args.result_filename:
            file_name = args.result_filename
        if args.result_dir:
            os.makedirs(args.result_dir, exist_ok=True)
            file_name = os.path.join(args.result_dir, file_name)
        with open(
            file_name, mode="a+" if args.append_result else "w", encoding="utf-8"
        ) as outfile:
            # Append a newline.
            if args.append_result and outfile.tell() != 0:
                outfile.write("\n")
            json.dump(result_json, outfile)
        save_to_pytorch_benchmark_format(args, result_json, file_name)

    return result_json
